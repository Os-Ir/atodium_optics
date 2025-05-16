use crate::vk_context::device::WrappedDevice;
use anyhow::Result;
use ash::vk::{
    DescriptorBindingFlags, DescriptorPoolCreateFlags, DescriptorPoolCreateInfo, DescriptorPoolSize, DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBinding,
    DescriptorSetLayoutBindingFlagsCreateInfo, DescriptorSetLayoutCreateFlags, DescriptorSetLayoutCreateInfo, DescriptorSetVariableDescriptorCountAllocateInfo, DescriptorType, ShaderStageFlags,
};
use std::slice;

pub const MAX_BINDLESS_DESCRIPTOR_COUNT: u32 = 512 * 400;

pub fn default_bindless_layout(device: &WrappedDevice) -> Result<DescriptorSetLayout> {
    let descriptor_set_layout_binding = vec![
        DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(MAX_BINDLESS_DESCRIPTOR_COUNT)
            .stage_flags(ShaderStageFlags::ALL),
        DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(DescriptorType::STORAGE_BUFFER)
            .descriptor_count(MAX_BINDLESS_DESCRIPTOR_COUNT)
            .stage_flags(ShaderStageFlags::ALL),
        DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(DescriptorType::STORAGE_BUFFER)
            .descriptor_count(MAX_BINDLESS_DESCRIPTOR_COUNT)
            .stage_flags(ShaderStageFlags::ALL),
        DescriptorSetLayoutBinding::default()
            .binding(3)
            .descriptor_type(DescriptorType::STORAGE_BUFFER)
            .descriptor_count(MAX_BINDLESS_DESCRIPTOR_COUNT)
            .stage_flags(ShaderStageFlags::ALL),
        DescriptorSetLayoutBinding::default()
            .binding(4)
            .descriptor_type(DescriptorType::STORAGE_BUFFER)
            .descriptor_count(MAX_BINDLESS_DESCRIPTOR_COUNT)
            .stage_flags(ShaderStageFlags::ALL),
        DescriptorSetLayoutBinding::default()
            .binding(5)
            .descriptor_type(DescriptorType::STORAGE_BUFFER)
            .descriptor_count(MAX_BINDLESS_DESCRIPTOR_COUNT)
            .stage_flags(ShaderStageFlags::ALL),
    ];

    let binding_flags: Vec<DescriptorBindingFlags> = vec![
        DescriptorBindingFlags::PARTIALLY_BOUND,
        DescriptorBindingFlags::PARTIALLY_BOUND,
        DescriptorBindingFlags::PARTIALLY_BOUND,
        DescriptorBindingFlags::PARTIALLY_BOUND,
        DescriptorBindingFlags::PARTIALLY_BOUND,
        DescriptorBindingFlags::PARTIALLY_BOUND | DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT,
    ];

    let mut binding_flags_create_info = DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);

    let descriptor_sets_layout_info = DescriptorSetLayoutCreateInfo::default()
        .bindings(&descriptor_set_layout_binding)
        .flags(DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
        .push_next(&mut binding_flags_create_info);

    let layout = unsafe { device.create_descriptor_set_layout(&descriptor_sets_layout_info, None)? };

    Ok(layout)
}

pub fn create_bindless_descriptor_set(device: &WrappedDevice, layout: DescriptorSetLayout) -> Result<DescriptorSet> {
    let pool_size = DescriptorPoolSize::default().ty(DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(MAX_BINDLESS_DESCRIPTOR_COUNT);

    let descriptor_pool_info = DescriptorPoolCreateInfo::default()
        .pool_sizes(slice::from_ref(&pool_size))
        .flags(DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
        .max_sets(1);

    let descriptor_pool = unsafe { device.handle.create_descriptor_pool(&descriptor_pool_info, None)? };

    let mut variable_descriptor_count_allocate_info = DescriptorSetVariableDescriptorCountAllocateInfo::default().descriptor_counts(slice::from_ref(&MAX_BINDLESS_DESCRIPTOR_COUNT));

    let descriptor_info = DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(slice::from_ref(&layout))
        .push_next(&mut variable_descriptor_count_allocate_info);

    let descriptor_set = unsafe { device.allocate_descriptor_sets(&descriptor_info) }?[0];

    Ok(descriptor_set)
}

pub fn default_bindless_descriptor_set(device: &WrappedDevice) -> Result<DescriptorSet> {
    create_bindless_descriptor_set(device, default_bindless_layout(device)?)
}
