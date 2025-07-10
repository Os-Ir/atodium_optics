use crate::memory::render_buffer::RenderBuffer;
use crate::memory::render_image::RenderImage;
use crate::render::device::WrappedDeviceRef;
use crate::render::pipeline::WrappedPipeline;
use crate::render::shader_reflection::BindingMap;
use crate::rt::tlas::Tlas;
use anyhow::{anyhow, Result};
use ash::vk::{
    CommandBuffer, DescriptorBufferInfo, DescriptorImageInfo, DescriptorPool, DescriptorPoolCreateFlags, DescriptorPoolCreateInfo, DescriptorPoolSize, DescriptorSet, DescriptorSetAllocateInfo,
    DescriptorType, ImageLayout, Sampler, WriteDescriptorSet, WriteDescriptorSetAccelerationStructureKHR,
};
use std::collections::HashMap;
use std::slice;

pub fn map_rspirv_descriptor_type(rspirv_type: rspirv_reflect::DescriptorType) -> DescriptorType {
    match rspirv_type {
        rspirv_reflect::DescriptorType::SAMPLER => DescriptorType::SAMPLER,
        rspirv_reflect::DescriptorType::COMBINED_IMAGE_SAMPLER => DescriptorType::COMBINED_IMAGE_SAMPLER,
        rspirv_reflect::DescriptorType::SAMPLED_IMAGE => DescriptorType::SAMPLED_IMAGE,
        rspirv_reflect::DescriptorType::STORAGE_IMAGE => DescriptorType::STORAGE_IMAGE,
        rspirv_reflect::DescriptorType::UNIFORM_TEXEL_BUFFER => DescriptorType::UNIFORM_TEXEL_BUFFER,
        rspirv_reflect::DescriptorType::STORAGE_TEXEL_BUFFER => DescriptorType::STORAGE_TEXEL_BUFFER,
        rspirv_reflect::DescriptorType::UNIFORM_BUFFER => DescriptorType::UNIFORM_BUFFER,
        rspirv_reflect::DescriptorType::STORAGE_BUFFER => DescriptorType::STORAGE_BUFFER,
        rspirv_reflect::DescriptorType::UNIFORM_BUFFER_DYNAMIC => DescriptorType::UNIFORM_BUFFER_DYNAMIC,
        rspirv_reflect::DescriptorType::STORAGE_BUFFER_DYNAMIC => DescriptorType::STORAGE_BUFFER_DYNAMIC,
        rspirv_reflect::DescriptorType::INPUT_ATTACHMENT => DescriptorType::INPUT_ATTACHMENT,
        rspirv_reflect::DescriptorType::INLINE_UNIFORM_BLOCK_EXT => DescriptorType::INLINE_UNIFORM_BLOCK_EXT,
        rspirv_reflect::DescriptorType::ACCELERATION_STRUCTURE_KHR => DescriptorType::ACCELERATION_STRUCTURE_KHR,
        rspirv_reflect::DescriptorType::ACCELERATION_STRUCTURE_NV => DescriptorType::ACCELERATION_STRUCTURE_NV,
        _ => unreachable!(),
    }
}

pub struct WrappedDescriptorSet {
    device: WrappedDeviceRef,

    pub descriptor_set: DescriptorSet,
    pub descriptor_pool: DescriptorPool,
    pub binding_map: BindingMap,
}

pub enum DescriptorId {
    Name(String),
    Index(u32),
}

impl DescriptorId {
    pub fn get_binding(&self, binding_map: &BindingMap) -> Result<u32> {
        match self {
            DescriptorId::Name(name) => binding_map
                .get(name)
                .and_then(|shader_binding| Some(shader_binding.binding))
                .ok_or_else(|| anyhow!("Descriptor with name [ {} ] not founded", name)),
            DescriptorId::Index(binding) => Ok(*binding),
        }
    }
}

impl Drop for WrappedDescriptorSet {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}

impl WrappedDescriptorSet {
    pub fn new(device: WrappedDeviceRef, pipeline: &WrappedPipeline, descriptor_set_index: usize) -> Result<Self> {
        let layout = pipeline.descriptor_set_layouts[descriptor_set_index];

        let mut descriptor_pool_sizes: HashMap<DescriptorType, u32> = HashMap::new();
        pipeline.reflection.binding_map.values().for_each(|val| {
            *descriptor_pool_sizes.entry(map_rspirv_descriptor_type(val.info.ty)).or_insert(0) += 1;
        });

        let descriptor_pool_sizes: Vec<DescriptorPoolSize> = descriptor_pool_sizes.iter().map(|(&ty, &count)| DescriptorPoolSize::default().ty(ty).descriptor_count(count)).collect();

        let descriptor_pool_info = DescriptorPoolCreateInfo::default()
            .pool_sizes(&descriptor_pool_sizes)
            .flags(DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET | DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .max_sets(descriptor_pool_sizes.len() as u32);

        let descriptor_pool = unsafe { device.create_descriptor_pool(&descriptor_pool_info, None)? };

        let descriptor_allocate_info = DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(slice::from_ref(&layout));

        let descriptor_set = unsafe { device.allocate_descriptor_sets(&descriptor_allocate_info)? }[0];

        Ok(WrappedDescriptorSet {
            device,
            descriptor_set,
            descriptor_pool,
            binding_map: pipeline.reflection.binding_map.clone(),
        })
    }

    pub fn write_uniform_buffer(&self, descriptor_id: DescriptorId, buffer: &RenderBuffer) -> Result<()> {
        let buffer_info = DescriptorBufferInfo::default().offset(0).range(buffer.size).buffer(buffer.buffer);

        let binding = descriptor_id.get_binding(&self.binding_map)?;

        let descriptor_writes = WriteDescriptorSet::default()
            .dst_set(self.descriptor_set)
            .dst_binding(binding)
            .descriptor_type(DescriptorType::UNIFORM_BUFFER)
            .buffer_info(slice::from_ref(&buffer_info));

        unsafe { self.device.update_descriptor_sets(slice::from_ref(&descriptor_writes), &[]) };

        Ok(())
    }

    pub fn write_storage_buffer(&self, descriptor_id: DescriptorId, buffer: &RenderBuffer) -> Result<()> {
        let buffer_info = DescriptorBufferInfo::default().offset(0).range(buffer.size).buffer(buffer.buffer);

        let binding = descriptor_id.get_binding(&self.binding_map)?;

        let descriptor_writes = WriteDescriptorSet::default()
            .dst_set(self.descriptor_set)
            .dst_binding(binding)
            .descriptor_type(DescriptorType::STORAGE_BUFFER)
            .buffer_info(slice::from_ref(&buffer_info));

        unsafe { self.device.update_descriptor_sets(slice::from_ref(&descriptor_writes), &[]) };

        Ok(())
    }

    pub fn write_storage_image(&self, descriptor_id: DescriptorId, image: &RenderImage) -> Result<()> {
        let image_info = DescriptorImageInfo::default().image_layout(ImageLayout::GENERAL).image_view(image.image_view).sampler(Sampler::null());

        let binding = descriptor_id.get_binding(&self.binding_map)?;

        let descriptor_writes = WriteDescriptorSet::default()
            .dst_set(self.descriptor_set)
            .dst_binding(binding)
            .descriptor_type(DescriptorType::STORAGE_IMAGE)
            .image_info(slice::from_ref(&image_info));

        unsafe { self.device.update_descriptor_sets(slice::from_ref(&descriptor_writes), &[]) };

        Ok(())
    }

    pub fn write_tlas(&self, descriptor_id: DescriptorId, tlas: &Tlas) -> Result<()> {
        let binding = descriptor_id.get_binding(&self.binding_map)?;

        let mut descriptor_info = WriteDescriptorSetAccelerationStructureKHR::default().acceleration_structures(slice::from_ref(&tlas.handle));

        let descriptor_writes = WriteDescriptorSet::default()
            .dst_set(self.descriptor_set)
            .dst_binding(binding)
            .descriptor_type(DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1)
            .push_next(&mut descriptor_info);

        unsafe { self.device.update_descriptor_sets(slice::from_ref(&descriptor_writes), &[]) };

        Ok(())
    }

    pub fn bind(&self, cmd_buf: CommandBuffer, pipeline: &WrappedPipeline) {
        unsafe {
            self.device
                .cmd_bind_descriptor_sets(cmd_buf, pipeline.bind_point(), pipeline.pipeline_layout, 0, slice::from_ref(&self.descriptor_set), &[])
        };
    }
}
