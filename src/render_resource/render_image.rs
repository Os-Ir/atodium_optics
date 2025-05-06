use crate::render_resource::render_buffer::{RenderBuffer, RenderBufferAllocatorRef};
use crate::vk_context::device::WrappedDeviceRef;
use anyhow::{Result, anyhow};
use ash::vk::{
    AccessFlags, BufferImageCopy, BufferUsageFlags, DependencyFlags, DescriptorImageInfo, DeviceMemory, DeviceSize, Extent3D, Format, Image, ImageAspectFlags, ImageCreateInfo, ImageLayout,
    ImageMemoryBarrier, ImageSubresourceLayers, ImageSubresourceRange, ImageTiling, ImageType, ImageUsageFlags, ImageView, ImageViewCreateInfo, ImageViewType, MemoryAllocateInfo, MemoryPropertyFlags,
    Offset3D, PipelineStageFlags, SampleCountFlags, Sampler, SharingMode,
};
use core::slice;
use gpu_allocator::MemoryLocation;
use std::ops::Deref;
use std::sync::Arc;

#[derive(Clone)]
pub struct RenderImage {
    device: WrappedDeviceRef,

    pub width: u32,
    pub height: u32,
    pub image: Image,
    pub image_view: ImageView,
    pub image_memory: DeviceMemory,
}

impl RenderImage {
    pub fn create(device: WrappedDeviceRef, width: u32, height: u32, image: Image, image_view: ImageView, image_memory: DeviceMemory) -> RenderImage {
        RenderImage {
            device,
            width,
            height,
            image,
            image_view,
            image_memory,
        }
    }

    pub fn image_info(&self, sampler: Sampler, image_layout: Option<ImageLayout>) -> DescriptorImageInfo {
        DescriptorImageInfo::default()
            .image_layout(image_layout.unwrap_or(ImageLayout::SHADER_READ_ONLY_OPTIMAL))
            .image_view(self.image_view)
            .sampler(sampler)
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
pub struct ImageManagerRef(Arc<ImageManager>);

impl From<ImageManager> for ImageManagerRef {
    fn from(manager: ImageManager) -> Self {
        ImageManagerRef(Arc::new(manager))
    }
}

impl Deref for ImageManagerRef {
    type Target = Arc<ImageManager>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct ImageManager {
    device: WrappedDeviceRef,
    buffer_allocator: RenderBufferAllocatorRef,
}

impl ImageManager {
    pub fn new(device: WrappedDeviceRef, buffer_allocator: RenderBufferAllocatorRef) -> Self {
        ImageManager { device, buffer_allocator }
    }

    pub fn allocate_image(&self, width: u32, height: u32, format: Format, tiling: ImageTiling, usage: ImageUsageFlags, properties: MemoryPropertyFlags) -> Result<(Image, DeviceMemory)> {
        unsafe {
            let image_info = ImageCreateInfo::default()
                .image_type(ImageType::TYPE_2D)
                .extent(Extent3D::default().width(width).height(height).depth(1))
                .mip_levels(1)
                .array_layers(1)
                .format(format)
                .tiling(tiling)
                .initial_layout(ImageLayout::UNDEFINED)
                .usage(usage)
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

    pub fn create_image_view(&self, image: Image, format: Format, aspect_mask: ImageAspectFlags) -> Result<ImageView> {
        let image_view_info = ImageViewCreateInfo::default()
            .image(image)
            .view_type(ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        Ok(unsafe { self.device.create_image_view(&image_view_info, None) }?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_render_image(
        &self,
        width: u32,
        height: u32,
        format: Format,
        tiling: ImageTiling,
        usage: ImageUsageFlags,
        properties: MemoryPropertyFlags,
        aspect_mask: ImageAspectFlags,
    ) -> Result<RenderImage> {
        let (image, image_memory) = self.allocate_image(width, height, format, tiling, usage, properties)?;
        let image_view = self.create_image_view(image, format, aspect_mask)?;

        Ok(RenderImage::create(self.device.clone(), width, height, image, image_view, image_memory))
    }

    pub fn transition_image_layout(&self, image: Image, old_layout: ImageLayout, new_layout: ImageLayout) -> Result<()> {
        let mut barrier = ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(self.device.queue_family_index)
            .dst_queue_family_index(self.device.queue_family_index)
            .image(image)
            .subresource_range(
                ImageSubresourceRange::default()
                    .aspect_mask(ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let (src_stage, dst_stage) = match (old_layout, new_layout) {
            (ImageLayout::UNDEFINED, ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL) => {
                barrier = barrier
                    .src_access_mask(AccessFlags::empty())
                    .dst_access_mask(AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ | AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE);

                (PipelineStageFlags::TOP_OF_PIPE, PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            }
            (ImageLayout::UNDEFINED, ImageLayout::TRANSFER_DST_OPTIMAL) => {
                barrier = barrier.src_access_mask(AccessFlags::empty()).dst_access_mask(AccessFlags::TRANSFER_WRITE);

                (PipelineStageFlags::TOP_OF_PIPE, PipelineStageFlags::TRANSFER)
            }
            (ImageLayout::TRANSFER_DST_OPTIMAL, ImageLayout::SHADER_READ_ONLY_OPTIMAL) => {
                barrier = barrier.src_access_mask(AccessFlags::TRANSFER_WRITE).dst_access_mask(AccessFlags::SHADER_READ);

                (PipelineStageFlags::TRANSFER, PipelineStageFlags::FRAGMENT_SHADER)
            }
            _ => return Err(anyhow!("Unsupported layout transition")),
        };

        self.device.single_time_command(|cmd_buf| unsafe {
            self.device
                .cmd_pipeline_barrier(cmd_buf, src_stage, dst_stage, DependencyFlags::empty(), &[], &[], slice::from_ref(&barrier))
        })?;

        Ok(())
    }

    pub fn copy_buffer_to_image(&self, buffer: &RenderBuffer, image: Image, width: u32, height: u32) -> Result<()> {
        let region = BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(ImageSubresourceLayers::default().aspect_mask(ImageAspectFlags::COLOR).mip_level(0).base_array_layer(0).layer_count(1))
            .image_offset(Offset3D::default().x(0).y(0).z(0))
            .image_extent(Extent3D::default().width(width).height(height).depth(1));
        let regions = [region];

        self.device
            .single_time_command(|cmd_buf| unsafe { self.device.cmd_copy_buffer_to_image(cmd_buf, buffer.buffer, image, ImageLayout::TRANSFER_DST_OPTIMAL, &regions) })?;

        Ok(())
    }

    pub fn create_image_from_pixels(&self, width: u32, height: u32, pixels: &[u8]) -> Result<RenderImage> {
        if pixels.len() != ((width * height * 4) as usize) {
            return Err(anyhow!("Pixel array size {} mismatch with width {} and height {}", pixels.len(), width, height));
        }

        let staging_buffer = self.buffer_allocator.allocate(pixels.len() as DeviceSize, BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu)?;

        self.buffer_allocator.upload_data(&staging_buffer, pixels)?;

        let (image, image_memory) = self.allocate_image(
            width,
            height,
            Format::R8G8B8A8_UNORM,
            ImageTiling::OPTIMAL,
            ImageUsageFlags::TRANSFER_DST | ImageUsageFlags::SAMPLED,
            MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        self.transition_image_layout(image, ImageLayout::UNDEFINED, ImageLayout::TRANSFER_DST_OPTIMAL)?;
        self.copy_buffer_to_image(&staging_buffer, image, width, height)?;
        self.transition_image_layout(image, ImageLayout::TRANSFER_DST_OPTIMAL, ImageLayout::SHADER_READ_ONLY_OPTIMAL)?;

        let image_view = self.create_image_view(image, Format::R8G8B8A8_UNORM, ImageAspectFlags::COLOR)?;

        Ok(RenderImage {
            device: self.device.clone(),
            width,
            height,
            image,
            image_view,
            image_memory,
        })
    }
}
