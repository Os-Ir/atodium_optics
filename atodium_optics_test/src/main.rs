use atodium_optics;

fn setup_logger() {
    unsafe { std::env::set_var("RUST_LOG", "info") };

    env_logger::init();
}

fn main() {
    setup_logger();

    atodium_optics::test_rt_pipeline().unwrap();
}
