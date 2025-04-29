use atodium_optics::vulkan_context::shader_compiler;

fn setup_logger() {
    unsafe { std::env::set_var("RUST_LOG", "debug") };

    env_logger::init();
}

fn main() {
    setup_logger();
    
    shader_compiler::load_shaders();
}
