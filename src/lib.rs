use anyhow::{Result, anyhow};
use std::ffi::{CStr, c_char};
use std::mem;
use std::path::Path;

pub mod render_resource;
pub mod vulkan_context;

#[inline]
pub fn lib_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

#[inline]
pub unsafe fn cstr_to_str_unchecked(vk_str: &[c_char]) -> &str {
    CStr::from_ptr(vk_str.as_ptr()).to_str().unwrap()
}

#[inline]
pub fn cstr_to_str(vk_str: &[c_char]) -> Result<&str> {
    let nul_pos = vk_str.iter().position(|&c| c == 0);
    let valid_slice = nul_pos.map(|pos| &vk_str[..=pos]);

    match valid_slice {
        Some(s) => unsafe { Ok(CStr::from_bytes_with_nul(mem::transmute(s))?.to_str()?) },
        None => Err(anyhow!("Invalid UTF-8 sequence")),
    }
}
