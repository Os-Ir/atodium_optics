use crate::render_resource::render_image::{ImageAllocator, RenderImage};
use crate::vk_context::device::{WrappedDevice, WrappedDeviceRef};
use anyhow::Result;
use ash::prelude::VkResult;
use ash::vk::{BorderColor, CompareOp, DescriptorImageInfo, Filter, ImageLayout, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode};

#[derive(Clone)]
pub struct Texture {
    device: WrappedDeviceRef,

    pub render_image: RenderImage,
    pub sampler: Sampler,
    pub descriptor_info: DescriptorImageInfo,
}

impl Drop for Texture {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_sampler(self.sampler, None);
        }
    }
}

impl Texture {
    pub fn new(device: WrappedDeviceRef, render_image: RenderImage, sampler: Sampler, descriptor_info: DescriptorImageInfo) -> Self {
        Self {
            device,
            render_image,
            sampler,
            descriptor_info,
        }
    }

    pub fn from_pixels(device: WrappedDeviceRef, image_allocator: &ImageAllocator, width: u32, height: u32, pixels: &[u8]) -> Result<Self> {
        let render_image = image_allocator.allocate_from_pixels(width, height, pixels)?;

        let sampler = Self::default_sampler(&device, 1)?;

        let descriptor_info = DescriptorImageInfo {
            image_layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            image_view: render_image.image_view,
            sampler,
        };

        Ok(Self::new(device.clone(), render_image, sampler, descriptor_info))
    }

    pub fn default_sampler(device: &WrappedDevice, mip_levels: u32) -> VkResult<Sampler> {
        let sampler_info = SamplerCreateInfo {
            mag_filter: Filter::LINEAR,
            min_filter: Filter::LINEAR,
            mipmap_mode: SamplerMipmapMode::LINEAR,
            address_mode_u: SamplerAddressMode::MIRRORED_REPEAT,
            address_mode_v: SamplerAddressMode::MIRRORED_REPEAT,
            address_mode_w: SamplerAddressMode::MIRRORED_REPEAT,
            max_anisotropy: 1.0,
            border_color: BorderColor::FLOAT_OPAQUE_WHITE,
            compare_op: CompareOp::NEVER,
            min_lod: 0.0,
            max_lod: mip_levels as f32,
            ..Default::default()
        };

        unsafe { device.create_sampler(&sampler_info, None) }
    }
}
