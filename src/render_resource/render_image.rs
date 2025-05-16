use crate::render_resource::render_buffer::{RenderBuffer, RenderBufferAllocatorRef};
use crate::vk_context::device::WrappedDeviceRef;
use anyhow::{Result, anyhow, bail};
use ash::vk::{
    AccessFlags, BufferImageCopy, BufferUsageFlags, DependencyFlags, DeviceMemory, DeviceSize, Extent3D, Format, Image, ImageAspectFlags, ImageCopy, ImageCreateInfo, ImageLayout, ImageMemoryBarrier,
    ImageSubresourceLayers, ImageSubresourceRange, ImageTiling, ImageType, ImageUsageFlags, ImageView, ImageViewCreateInfo, ImageViewType, MemoryAllocateInfo, MemoryPropertyFlags, Offset3D,
    PipelineStageFlags, SampleCountFlags, SharingMode,
};
use core::slice;
use gpu_allocator::MemoryLocation;
use std::ops::Deref;
use std::sync::Arc;

#[derive(Copy, Clone, Debug)]
pub struct ImageDesc {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mip_level: u32,
    pub format: Format,
    pub tiling: ImageTiling,
    pub aspect_flags: ImageAspectFlags,
    pub usage: ImageUsageFlags,
}

impl ImageDesc {
    pub fn default_2d(width: u32, height: u32, format: Format, usage: ImageUsageFlags) -> Self {
        Self {
            width,
            height,
            depth: 1,
            mip_level: 1,
            format,
            tiling: ImageTiling::OPTIMAL,
            aspect_flags: ImageAspectFlags::COLOR,
            usage,
        }
    }

    pub fn mip_levels(mut self, mip_levels: u32) -> Self {
        self.mip_level = mip_levels;
        self
    }

    pub fn tiling(mut self, tiling: ImageTiling) -> Self {
        self.tiling = tiling;
        self
    }

    pub fn aspect_flags(mut self, aspect_flags: ImageAspectFlags) -> Self {
        self.aspect_flags = aspect_flags;
        self
    }

    pub fn image_type(&self) -> ImageType {
        if self.depth > 1 { ImageType::TYPE_3D } else { ImageType::TYPE_2D }
    }

    pub fn image_view_type(&self) -> ImageViewType {
        if self.depth > 1 { ImageViewType::TYPE_3D } else { ImageViewType::TYPE_2D }
    }
}

#[derive(Clone)]
pub struct RenderImage {
    device: WrappedDeviceRef,

    pub desc: ImageDesc,
    pub image: Image,
    pub image_view: ImageView,
    pub image_memory: DeviceMemory,
    pub current_layout: ImageLayout,
}

impl RenderImage {
    pub fn new(device: WrappedDeviceRef, desc: ImageDesc, image: Image, image_view: ImageView, image_memory: DeviceMemory, current_layout: ImageLayout) -> RenderImage {
        RenderImage {
            device,
            desc,
            image,
            image_view,
            image_memory,
            current_layout,
        }
    }
}

impl Drop for RenderImage {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image(self.image, None);
            self.device.destroy_image_view(self.image_view, None);
            self.device.free_memory(self.image_memory, None);
        }
    }
}

#[derive(Clone)]
pub struct ImageAllocatorRef(Arc<ImageAllocator>);

impl From<ImageAllocator> for ImageAllocatorRef {
    fn from(manager: ImageAllocator) -> Self {
        ImageAllocatorRef(Arc::new(manager))
    }
}

impl Deref for ImageAllocatorRef {
    type Target = Arc<ImageAllocator>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct ImageAllocator {
    device: WrappedDeviceRef,
    buffer_allocator: RenderBufferAllocatorRef,
}

impl ImageAllocator {
    pub fn new(device: WrappedDeviceRef, buffer_allocator: RenderBufferAllocatorRef) -> Self {
        ImageAllocator { device, buffer_allocator }
    }

    pub fn allocate(&self, desc: ImageDesc, properties: MemoryPropertyFlags) -> Result<RenderImage> {
        let (image, image_memory) = self.allocate_image(desc, properties)?;
        let image_view = self.create_image_view(desc, image)?;

        Ok(RenderImage::new(self.device.clone(), desc, image, image_view, image_memory, ImageLayout::UNDEFINED))
    }

    pub fn upload_from_buffer(&self, buffer: &RenderBuffer, image: &RenderImage) -> Result<()> {
        let region = BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(ImageSubresourceLayers::default().aspect_mask(image.desc.aspect_flags).mip_level(0).base_array_layer(0).layer_count(1))
            .image_offset(Offset3D::default().x(0).y(0).z(0))
            .image_extent(Extent3D::default().width(image.desc.width).height(image.desc.height).depth(image.desc.depth));

        self.device.single_time_command(|cmd_buf| unsafe {
            self.device
                .cmd_copy_buffer_to_image(cmd_buf, buffer.buffer, image.image, ImageLayout::TRANSFER_DST_OPTIMAL, slice::from_ref(&region))
        })?;

        Ok(())
    }

    pub fn allocate_from_pixels(&self, width: u32, height: u32, pixels: &[u8]) -> Result<RenderImage> {
        if pixels.len() != ((width * height * 4) as usize) {
            return Err(anyhow!("Pixel array size {} mismatch with width {} and height {}", pixels.len(), width, height));
        }

        let staging_buffer = self.buffer_allocator.allocate(pixels.len() as DeviceSize, BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu)?;

        self.buffer_allocator.upload_data(&staging_buffer, pixels)?;

        let desc = ImageDesc::default_2d(width, height, Format::R8G8B8A8_UNORM, ImageUsageFlags::TRANSFER_DST | ImageUsageFlags::SAMPLED);

        let mut image = self.allocate(desc, MemoryPropertyFlags::DEVICE_LOCAL)?;

        self.transition_layout(&mut image, ImageLayout::TRANSFER_DST_OPTIMAL)?;
        self.upload_from_buffer(&staging_buffer, &image)?;
        self.transition_layout(&mut image, ImageLayout::SHADER_READ_ONLY_OPTIMAL)?;

        Ok(image)
    }

    pub fn transition_layout(&self, image: &mut RenderImage, new_layout: ImageLayout) -> Result<()> {
        if image.current_layout == new_layout {
            return Ok(());
        }

        let (src_access_mask, src_stage) = match image.current_layout {
            ImageLayout::UNDEFINED => (AccessFlags::HOST_WRITE, PipelineStageFlags::HOST),
            ImageLayout::PREINITIALIZED => (AccessFlags::HOST_WRITE, PipelineStageFlags::HOST),
            ImageLayout::TRANSFER_SRC_OPTIMAL => (AccessFlags::TRANSFER_READ, PipelineStageFlags::TRANSFER),
            ImageLayout::TRANSFER_DST_OPTIMAL => (AccessFlags::TRANSFER_WRITE, PipelineStageFlags::TRANSFER),
            ImageLayout::SHADER_READ_ONLY_OPTIMAL => (AccessFlags::HOST_WRITE, PipelineStageFlags::HOST),
            ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (AccessFlags::COLOR_ATTACHMENT_WRITE, PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT),
            ImageLayout::GENERAL => (AccessFlags::HOST_WRITE, PipelineStageFlags::HOST),
            _ => return Err(anyhow!("Unsupported layout transition")),
        };

        let (dst_access_mask, dst_stage) = match new_layout {
            ImageLayout::TRANSFER_SRC_OPTIMAL => (AccessFlags::TRANSFER_READ, PipelineStageFlags::TRANSFER),
            ImageLayout::TRANSFER_DST_OPTIMAL => (AccessFlags::TRANSFER_WRITE, PipelineStageFlags::TRANSFER),
            ImageLayout::SHADER_READ_ONLY_OPTIMAL => (AccessFlags::SHADER_READ, PipelineStageFlags::FRAGMENT_SHADER),
            ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (AccessFlags::COLOR_ATTACHMENT_WRITE, PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT),
            ImageLayout::GENERAL => (AccessFlags::SHADER_READ, PipelineStageFlags::FRAGMENT_SHADER),
            _ => return Err(anyhow!("Unsupported layout transition")),
        };

        let subresource_range = ImageSubresourceRange::default()
            .aspect_mask(image.desc.aspect_flags)
            .base_array_layer(0)
            .layer_count(1)
            .base_mip_level(0)
            .level_count(image.desc.mip_level);

        let barrier = ImageMemoryBarrier::default()
            .old_layout(image.current_layout)
            .new_layout(new_layout)
            .src_queue_family_index(self.device.queue_family_index)
            .dst_queue_family_index(self.device.queue_family_index)
            .image(image.image)
            .subresource_range(subresource_range)
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask);

        self.device.single_time_command(|cmd_buf| unsafe {
            self.device
                .cmd_pipeline_barrier(cmd_buf, src_stage, dst_stage, DependencyFlags::empty(), &[], &[], slice::from_ref(&barrier))
        })?;

        image.current_layout = new_layout;

        Ok(())
    }

    pub fn copy_image(&self, src_image: &RenderImage, dst_image: &RenderImage, mip_level: Option<u32>) -> Result<()> {
        if src_image.desc.aspect_flags != dst_image.desc.aspect_flags {
            return Err(anyhow!("Image aspect flags mismatch"));
        }

        if src_image.desc.width != dst_image.desc.width || src_image.desc.height != dst_image.desc.height || src_image.desc.depth != dst_image.desc.depth {
            return Err(anyhow!("Image extend mismatch"));
        }

        let subresource_range = ImageSubresourceLayers::default()
            .aspect_mask(src_image.desc.aspect_flags)
            .mip_level(mip_level.unwrap_or(0))
            .base_array_layer(0)
            .layer_count(1);

        let extent = Extent3D::default().width(src_image.desc.width).height(src_image.desc.height).depth(src_image.desc.depth);

        let region = ImageCopy::default().src_subresource(subresource_range).dst_subresource(subresource_range).extent(extent);

        self.device.single_time_command(|cmd_buf| unsafe {
            self.device
                .cmd_copy_image(cmd_buf, src_image.image, src_image.current_layout, dst_image.image, dst_image.current_layout, slice::from_ref(&region));
        })?;

        Ok(())
    }

    pub fn acquire_pixels(&self, image: &mut RenderImage, mip_level: Option<u32>) -> Result<Vec<[f32; 4]>> {
        if image.desc.aspect_flags != ImageAspectFlags::COLOR {
            bail!("Only images with color aspect flag supported");
        }

        let pixel_size = match image.desc.format {
            Format::R32G32B32A32_SFLOAT => 4 * size_of::<f32>(),
            Format::R8G8B8A8_UNORM => 4 * size_of::<u32>(),
            Format::R8G8B8A8_SRGB => 4 * size_of::<u32>(),
            _ => bail!("Unsupported image format: {:?}", image.desc.format),
        };

        let staging_size = (image.desc.width * image.desc.height * pixel_size as u32) as DeviceSize;
        if staging_size == 0 {
            return Ok(Vec::new());
        }

        let staging_buffer = self.buffer_allocator.allocate(staging_size, BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuToCpu)?;

        if image.current_layout != ImageLayout::TRANSFER_SRC_OPTIMAL {
            self.transition_layout(image, ImageLayout::TRANSFER_SRC_OPTIMAL)?
        }

        let image_subresource = ImageSubresourceLayers::default()
            .aspect_mask(image.desc.aspect_flags)
            .mip_level(mip_level.unwrap_or(0))
            .base_array_layer(0)
            .layer_count(1);

        let extent = Extent3D::default().width(image.desc.width).height(image.desc.height).depth(image.desc.depth);

        let buffer_image_copy = BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(image_subresource)
            .image_extent(extent);

        self.device.single_time_command(|cmd_buf| unsafe {
            self.device
                .cmd_copy_image_to_buffer(cmd_buf, image.image, ImageLayout::TRANSFER_SRC_OPTIMAL, staging_buffer.buffer, slice::from_ref(&buffer_image_copy));
        })?;

        let pixels = match image.desc.format {
            Format::R32G32B32A32_SFLOAT => self.buffer_allocator.download_data::<[f32; 4]>(&staging_buffer)?,
            Format::R8G8B8A8_UNORM => self
                .buffer_allocator
                .download_data::<[u8; 4]>(&staging_buffer)?
                .iter()
                .map(|pixel| {
                    let r = pixel[0] as f32 / 255.0;
                    let g = pixel[1] as f32 / 255.0;
                    let b = pixel[2] as f32 / 255.0;
                    let a = pixel[3] as f32 / 255.0;

                    [r, g, b, a]
                })
                .collect(),
            Format::R8G8B8A8_SRGB => self
                .buffer_allocator
                .download_data::<[u8; 4]>(&staging_buffer)?
                .iter()
                .map(|pixel| {
                    let r = pixel[0] as f32 / 255.0;
                    let g = pixel[1] as f32 / 255.0;
                    let b = pixel[2] as f32 / 255.0;
                    let a = pixel[3] as f32 / 255.0;

                    [r, g, b, a]
                })
                .collect(),
            _ => bail!("Unsupported image format: {:?}", image.desc.format),
        };

        Ok(pixels)
    }

    fn allocate_image(&self, desc: ImageDesc, properties: MemoryPropertyFlags) -> Result<(Image, DeviceMemory)> {
        unsafe {
            let image_info = ImageCreateInfo::default()
                .image_type(desc.image_type())
                .extent(Extent3D::default().width(desc.width).height(desc.height).depth(desc.depth))
                .mip_levels(desc.mip_level)
                .array_layers(1)
                .format(desc.format)
                .tiling(desc.tiling)
                .initial_layout(ImageLayout::UNDEFINED)
                .usage(desc.usage)
                .samples(SampleCountFlags::TYPE_1)
                .sharing_mode(SharingMode::EXCLUSIVE);

            let image = self.device.create_image(&image_info, None)?;

            let memory_requirement = self.device.get_image_memory_requirements(image);

            let allocate_info = MemoryAllocateInfo::default().allocation_size(memory_requirement.size).memory_type_index(
                self.device
                    .find_valid_memory_type(memory_requirement, properties)
                    .ok_or_else(|| anyhow!("Failed to find valid memory type."))?,
            );

            let image_memory = self.device.allocate_memory(&allocate_info, None)?;

            self.device.bind_image_memory(image, image_memory, 0)?;

            Ok((image, image_memory))
        }
    }

    fn create_image_view(&self, desc: ImageDesc, image: Image) -> Result<ImageView> {
        let subresource_range = ImageSubresourceRange::default()
            .aspect_mask(desc.aspect_flags)
            .base_array_layer(0)
            .layer_count(1)
            .base_mip_level(0)
            .level_count(desc.mip_level);

        let image_view_info = ImageViewCreateInfo::default()
            .image(image)
            .view_type(desc.image_view_type())
            .format(desc.format)
            .subresource_range(subresource_range);

        Ok(unsafe { self.device.create_image_view(&image_view_info, None) }?)
    }
}
