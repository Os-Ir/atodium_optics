use atodium_optics::render_resource::render_buffer::{RenderBufferAllocator, RenderBufferAllocatorRef};
use atodium_optics::vulkan_context::device::{WrappedDevice, WrappedDeviceRef};
use atodium_optics::vulkan_context::pipeline::{PipelineDesc, WrappedPipeline};
use atodium_optics::vulkan_context::{shader_compiler, API_VERSION, DEVICE_EXTENSIONS, VALIDATION_LAYERS};

fn setup_logger() {
    unsafe { std::env::set_var("RUST_LOG", "debug") };

    env_logger::init();
}

fn main() {
    setup_logger();

    let device: WrappedDeviceRef = WrappedDevice::new(true, &VALIDATION_LAYERS, "atodium_optics", 0, "atodium_optics_test", 0, API_VERSION, &DEVICE_EXTENSIONS).unwrap().into();
    let buffer_allocator: RenderBufferAllocatorRef = RenderBufferAllocator::new(device.clone()).unwrap().into();

    let include_structure = shader_compiler::load_shaders();
    let pipeline_desc = PipelineDesc::default().compute_path("render.comp.glsl".into());
    let pipeline = WrappedPipeline::new(device.clone(), &buffer_allocator, pipeline_desc, &include_structure, None).unwrap();
}
