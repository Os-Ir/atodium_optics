use anyhow::{Result, anyhow};
use ash::ext::debug_utils;
use ash::khr::{acceleration_structure, ray_tracing_pipeline};
use ash::vk;
use ash::vk::{
    ApplicationInfo, Bool32, CommandBuffer, CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel, CommandBufferUsageFlags, CommandPool, CommandPoolCreateFlags, CommandPoolCreateInfo,
    DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT, DebugUtilsMessengerCallbackDataEXT, DebugUtilsMessengerCreateInfoEXT, DebugUtilsMessengerEXT, DeviceCreateInfo,
    DeviceQueueCreateInfo, FenceCreateInfo, MemoryPropertyFlags, MemoryRequirements, PhysicalDevice, PhysicalDeviceAccelerationStructureFeaturesKHR, PhysicalDeviceFeatures, PhysicalDeviceFeatures2,
    PhysicalDeviceProperties, PhysicalDeviceRayTracingPipelineFeaturesKHR, PhysicalDeviceRayTracingPipelinePropertiesKHR, PhysicalDeviceVulkan12Features, PhysicalDeviceVulkan13Features,
    PresentModeKHR, QueueFlags, SubmitInfo, SurfaceCapabilitiesKHR, SurfaceFormatKHR, SurfaceKHR,
};
use log::{error, info};
use std::collections::HashSet;
use std::ffi::{CStr, CString};
use std::ops::Deref;
use std::os::raw::{c_char, c_void};
use std::slice;
use std::sync::{Arc, Mutex};

#[allow(unsafe_op_in_unsafe_fn)]
unsafe extern "system" fn vulkan_debug_callback(
    message_severity: DebugUtilsMessageSeverityFlagsEXT,
    message_type: DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> Bool32 {
    let severity = match message_severity {
        DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "[Verbose]",
        DebugUtilsMessageSeverityFlagsEXT::WARNING => "[Warning]",
        DebugUtilsMessageSeverityFlagsEXT::ERROR => "[Error]",
        DebugUtilsMessageSeverityFlagsEXT::INFO => "[Info]",
        _ => "[Unknown]",
    };

    let types = match message_type {
        DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
        DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
        DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
        _ => "[Unknown]",
    };

    let message = CStr::from_ptr((*p_callback_data).p_message);
    println!("[Debug]{}{}{:?}", severity, types, message);

    vk::FALSE
}

#[derive(Clone)]
pub struct WrappedDeviceRef(Arc<WrappedDevice>);

impl Deref for WrappedDeviceRef {
    type Target = Arc<WrappedDevice>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<WrappedDevice> for WrappedDeviceRef {
    fn from(w: WrappedDevice) -> Self {
        WrappedDeviceRef(Arc::new(w))
    }
}

pub struct WrappedDevice {
    pub app_name: String,
    pub app_version: u32,

    pub entry: ash::Entry,
    pub instance: ash::Instance,

    pub debug_instance: debug_utils::Instance,
    pub debug_messenger: DebugUtilsMessengerEXT,

    pub physical_device: PhysicalDevice,

    pub queue_family_index: u32,

    pub handle: ash::Device,
    pub graphic_queue: Mutex<vk::Queue>,

    pub single_time_command_pool: Mutex<CommandPool>,

    pub rt_pipeline_device: ray_tracing_pipeline::Device,
    pub acceleration_device: acceleration_structure::Device,

    pub rt_pipeline_properties: PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>,
    pub acceleration_structure_features: PhysicalDeviceAccelerationStructureFeaturesKHR<'static>,
}

impl WrappedDevice {
    pub const ANYHOW_PARSE: fn() -> anyhow::Error = || unreachable!();

    pub fn new(enable_validation: bool, validation_layers: &[&str], engine_name: &str, engine_version: u32, app_name: &str, app_version: u32, api_version: u32, device_extensions: &[&CStr]) -> Result<Self> {
        unsafe {
            let entry = ash::Entry::linked();
            let instance = create_instance(&entry, enable_validation, validation_layers, engine_name, engine_version, app_name, app_version, api_version)?;
            let (debug_instance, debug_messenger) = create_debug_messenger(&entry, &instance)?;
            let (physical_device, queue_family_index) = select_physical_device(&instance, device_extensions)?;
            let (handle, graphic_queue) = create_device(&instance, physical_device, queue_family_index, device_extensions)?;
            let single_time_command_pool = create_command_pool(&handle, queue_family_index)?;
            let (rt_pipeline_device, acceleration_device) = create_acceleration_context(&instance, &handle);
            let (rt_pipeline_properties, acceleration_structure_features) = acquire_rt_properties(&instance, physical_device);

            Ok(Self {
                app_name: app_name.into(),
                app_version,
                entry,
                instance,
                debug_instance,
                debug_messenger,
                physical_device,
                queue_family_index,
                handle,
                graphic_queue: Mutex::new(graphic_queue),
                single_time_command_pool: Mutex::new(single_time_command_pool),
                rt_pipeline_device,
                acceleration_device,
                rt_pipeline_properties,
                acceleration_structure_features,
            })
        }
    }

    pub fn single_time_command(&self, f: impl FnOnce(&WrappedDevice, CommandBuffer)) -> Result<()> {
        unsafe {
            let queue = self.graphic_queue.lock().expect("Graphic queue is poisoned");
            let command_pool = self.single_time_command_pool.lock().expect("Single time command pool is poisoned");

            let allocate_info = CommandBufferAllocateInfo::default()
                .command_pool(*command_pool)
                .level(CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let command_buffer = self.handle.allocate_command_buffers(&allocate_info)?[0];

            let begin_info = CommandBufferBeginInfo::default().flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            self.handle.begin_command_buffer(command_buffer, &begin_info)?;

            f(self, command_buffer);

            self.handle.end_command_buffer(command_buffer)?;

            let submit_info = SubmitInfo::default().command_buffers(slice::from_ref(&command_buffer));

            let fence_info = FenceCreateInfo::default();
            let fence = self.handle.create_fence(&fence_info, None)?;
            self.handle.reset_fences(slice::from_ref(&fence))?;

            self.handle.queue_submit(*queue, slice::from_ref(&submit_info), fence)?;

            self.handle.wait_for_fences(slice::from_ref(&fence), true, u64::MAX)?;
            self.handle.free_command_buffers(*command_pool, slice::from_ref(&command_buffer));
            self.handle.destroy_fence(fence, None);

            Ok(())
        }
    }

    pub fn find_valid_memory_type(&self, requirements: MemoryRequirements, properties: MemoryPropertyFlags) -> Option<u32> {
        let memory_properties = unsafe { self.instance.get_physical_device_memory_properties(self.physical_device) };

        memory_properties.memory_types[..memory_properties.memory_type_count as usize]
            .iter()
            .enumerate()
            .find(|(index, memory_type)| ((requirements.memory_type_bits & (1u32 << index)) != 0) && ((memory_type.property_flags & properties) == properties))
            .map(|(index, _)| index as u32)
    }
}

impl Deref for WrappedDevice {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

impl Drop for WrappedDevice {
    fn drop(&mut self) {
        unsafe {
            self.handle.device_wait_idle().unwrap();
            self.handle.destroy_command_pool(*self.single_time_command_pool.lock().unwrap(), None);
            self.handle.destroy_device(None);
            self.debug_instance.destroy_debug_utils_messenger(self.debug_messenger, None);
            self.instance.destroy_instance(None);
        }
        info!("Vulkan device cleaned up.");
    }
}

unsafe fn check_validation_layer_support(entry: &ash::Entry, validation_layers: &[&str]) -> bool {
    let layer_properties = if let Ok(properties) = entry.enumerate_instance_layer_properties() {
        properties
    } else {
        error!("Failed to enumerate instance layers properties.");
        return false;
    };

    if layer_properties.is_empty() {
        error!("No available layers.");
        return false;
    } else {
        info!("Instance available layers: ");
        for layer in layer_properties.iter() {
            let layer_name = crate::cstr_to_str_unchecked(&layer.layer_name);
            info!("\t{}", layer_name);
        }
        info!("----------------------------------------");
    }

    for required_layer_name in validation_layers.iter() {
        let mut layer_found = false;
        for layer_property in layer_properties.iter() {
            if (*required_layer_name) == crate::cstr_to_str_unchecked(&layer_property.layer_name) {
                layer_found = true;
                break;
            }
        }
        if !layer_found {
            return false;
        }
    }
    true
}

fn get_required_extensions() -> Vec<*const c_char> {
    vec![debug_utils::NAME.as_ptr()]
}

fn generate_debug_messenger_info() -> DebugUtilsMessengerCreateInfoEXT<'static> {
    DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(DebugUtilsMessageSeverityFlagsEXT::WARNING | DebugUtilsMessageSeverityFlagsEXT::ERROR)
        .message_type(DebugUtilsMessageTypeFlagsEXT::GENERAL | DebugUtilsMessageTypeFlagsEXT::PERFORMANCE | DebugUtilsMessageTypeFlagsEXT::VALIDATION)
        .pfn_user_callback(Some(vulkan_debug_callback))
}

fn find_queue_family_info(instance: &ash::Instance, physical_device: PhysicalDevice) -> Option<u32> {
    let queue_family_properties = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

    queue_family_properties.iter().enumerate().find_map(|(index, &queue_family_property)| {
        if queue_family_property.queue_flags.contains(QueueFlags::GRAPHICS) {
            Some(index as u32)
        } else {
            None
        }
    })
}

unsafe fn create_instance(entry: &ash::Entry, enable_validation: bool, validation_layers: &[&str], engine_name: &str, engine_version: u32, app_name: &str, app_version: u32, api_version: u32) -> Result<ash::Instance> {
    if enable_validation && !check_validation_layer_support(entry, validation_layers) {
        return Err(anyhow!("Validation layers are not available."));
    }

    let cstr_app_name = CString::new(app_name)?;
    let cstr_engine_name = CString::new(engine_name)?;

    let application_info = ApplicationInfo::default()
        .application_name(cstr_app_name.as_c_str())
        .engine_name(cstr_engine_name.as_c_str())
        .application_version(app_version)
        .engine_version(engine_version)
        .api_version(api_version);

    let validation_layers_ptr: Vec<*const c_char> = validation_layers.iter().map(|layer| layer.as_ptr() as *const c_char).collect();
    let extensions_pointer: Vec<*const c_char> = get_required_extensions();

    info!("Instance required layers:");
    for layer in validation_layers {
        info!("\t{}", layer);
    }
    info!("----------------------------------------");

    let mut debug_messenger_info = generate_debug_messenger_info();

    let mut create_info = vk::InstanceCreateInfo::default()
        .application_info(&application_info)
        .enabled_layer_names(&validation_layers_ptr)
        .enabled_extension_names(&extensions_pointer);

    if enable_validation {
        create_info = create_info.push_next(&mut debug_messenger_info);
    }

    let instance = entry.create_instance(&create_info, None)?;

    Ok(instance)
}

unsafe fn create_debug_messenger(entry: &ash::Entry, instance: &ash::Instance) -> Result<(debug_utils::Instance, DebugUtilsMessengerEXT)> {
    let debug_messenger_info = generate_debug_messenger_info();
    let instance = debug_utils::Instance::new(entry, instance);
    let messenger = instance.create_debug_utils_messenger(&debug_messenger_info, None)?;

    Ok((instance, messenger))
}

unsafe fn check_physical_device(physical_device: PhysicalDevice, instance: &ash::Instance, device_extensions: &[&CStr]) -> Option<u32> {
    unsafe {
        let queue_family_index = find_queue_family_info(instance, physical_device);

        let Some(queue_family_index) = queue_family_index else {
            return None;
        };

        let extension_properties: Vec<String> = instance
            .enumerate_device_extension_properties(physical_device)
            .ok()?
            .iter()
            .map(|property| crate::cstr_to_str_unchecked(&property.extension_name).to_string())
            .collect();

        let mut required_extensions: HashSet<String> = device_extensions.iter().map(|string| string.to_str().unwrap().to_string()).collect();
        required_extensions.retain(|required| !extension_properties.iter().any(|property| property == required));
        if !required_extensions.is_empty() {
            return None;
        }

        let features = instance.get_physical_device_features(physical_device);

        if features.sampler_anisotropy == vk::FALSE {
            return None;
        }

        Some(queue_family_index)
    }
}

unsafe fn select_physical_device(instance: &ash::Instance, device_extensions: &[&CStr]) -> Result<(PhysicalDevice, u32)> {
    let physical_devices = instance.enumerate_physical_devices()?;

    info!("Detected physical devices: ");

    let mut valid_physical_devices: Vec<(PhysicalDevice, PhysicalDeviceProperties, u32)> = physical_devices
        .iter()
        .filter_map(|&physical_device| {
            let properties = instance.get_physical_device_properties(physical_device);

            match check_physical_device(physical_device, instance, device_extensions) {
                Some(queue_family_index) => {
                    info!("\t{} | valid", crate::cstr_to_str_unchecked(&properties.device_name));
                    Some((physical_device, properties, queue_family_index))
                }
                None => {
                    info!("\t{} | invalid", crate::cstr_to_str_unchecked(&properties.device_name));
                    None
                }
            }
        })
        .collect();

    if valid_physical_devices.is_empty() {
        return Err(anyhow!("Failed to find suitable physical devices."));
    }

    let (physical_device, properties, queue_family_index) = valid_physical_devices.remove(0);

    info!("Selected physical devices: ");
    info!("\t{}", crate::cstr_to_str_unchecked(&properties.device_name));
    info!("----------------------------------------");

    Ok((physical_device, queue_family_index))
}

unsafe fn create_device(instance: &ash::Instance, physical_device: PhysicalDevice, queue_family_index: u32, device_extensions: &[&CStr]) -> Result<(ash::Device, vk::Queue)> {
    unsafe {
        let device_queue_info = DeviceQueueCreateInfo::default().queue_family_index(queue_family_index).queue_priorities(slice::from_ref(&1.0));

        let device_extensions_ptr = device_extensions.iter().map(|extension| extension.as_ptr()).collect::<Vec<_>>();

        let features = PhysicalDeviceFeatures::default().sampler_anisotropy(true);

        let mut ray_tracing_features = PhysicalDeviceRayTracingPipelineFeaturesKHR::default().ray_tracing_pipeline(true);

        let mut acceleration_structure_features = PhysicalDeviceAccelerationStructureFeaturesKHR::default().acceleration_structure(true);

        let mut vulkan_12_features = PhysicalDeviceVulkan12Features::default().descriptor_indexing(true).runtime_descriptor_array(true).buffer_device_address(true);

        let mut vulkan_13_features = PhysicalDeviceVulkan13Features::default().dynamic_rendering(true).synchronization2(true);

        let mut features = PhysicalDeviceFeatures2::default()
            .features(features)
            .push_next(&mut ray_tracing_features)
            .push_next(&mut acceleration_structure_features)
            .push_next(&mut vulkan_12_features)
            .push_next(&mut vulkan_13_features);

        let device_info = DeviceCreateInfo::default()
            .queue_create_infos(slice::from_ref(&device_queue_info))
            .enabled_extension_names(&device_extensions_ptr)
            .push_next(&mut features);

        let device = instance.create_device(physical_device, &device_info, None)?;

        let graphic_queue = device.get_device_queue(queue_family_index, 0);

        Ok((device, graphic_queue))
    }
}

unsafe fn create_command_pool(device: &ash::Device, queue_family: u32) -> Result<CommandPool> {
    unsafe {
        let pool_info = CommandPoolCreateInfo::default().queue_family_index(queue_family).flags(CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        Ok(device.create_command_pool(&pool_info, None)?)
    }
}

pub unsafe fn create_acceleration_context(instance: &ash::Instance, device: &ash::Device) -> (ray_tracing_pipeline::Device, acceleration_structure::Device) {
    let rt_pipeline_device = ray_tracing_pipeline::Device::new(instance, device);
    let acceleration_device = acceleration_structure::Device::new(instance, device);

    (rt_pipeline_device, acceleration_device)
}

pub fn acquire_rt_properties(
    instance: &ash::Instance,
    physical_device: PhysicalDevice,
) -> (PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>, PhysicalDeviceAccelerationStructureFeaturesKHR<'static>) {
    unsafe {
        let mut rt_pipeline_properties = PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut rt_pipeline_properties);
        instance.get_physical_device_properties2(physical_device, &mut properties2);

        let mut acceleration_structure_features = PhysicalDeviceAccelerationStructureFeaturesKHR::default();
        let mut features2 = PhysicalDeviceFeatures2::default().push_next(&mut acceleration_structure_features);
        instance.get_physical_device_features2(physical_device, &mut features2);

        (rt_pipeline_properties, acceleration_structure_features)
    }
}
