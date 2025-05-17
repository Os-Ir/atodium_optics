use anyhow::anyhow;
use anyhow::{Result, bail};
use bytemuck::Pod;
use image::{ImageBuffer, ImageFormat};
use std::ffi::{CStr, c_char};
use std::path::Path;

#[inline]
pub fn lib_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn cstr_to_str_unchecked(vk_str: &[c_char]) -> &str {
    CStr::from_ptr(vk_str.as_ptr()).to_str().unwrap()
}

#[inline]
pub fn cstr_to_str(vk_str: &[c_char]) -> Result<&str> {
    let bytes: &[u8] = bytemuck::cast_slice(vk_str);

    let nul_pos = bytes.iter().position(|&c| c == 0).ok_or_else(|| anyhow!("Missing nul terminator"))?;

    Ok(CStr::from_bytes_with_nul(&bytes[..=nul_pos])?.to_str()?)
}

pub enum OutputFormat {
    Png,
    Hdr,
}

pub fn output_image<T: Pod>(path: &impl AsRef<Path>, width: u32, height: u32, pixels: &[T], output_format: OutputFormat) -> Result<()> {
    if (width * height) as usize * size_of::<[f32; 4]>() != pixels.len() * size_of::<T>() {
        bail!("Image dimensions does not match pixels length: {}", pixels.len());
    }

    let pixels = bytemuck::cast_slice::<T, [f32; 4]>(pixels);

    match output_format {
        OutputFormat::Png => {
            let image = ImageBuffer::from_fn(width, height, |x, y| {
                let idx = (y * width + x) as usize;

                let r = (pixels[idx][0] * 255.0) as u8;
                let g = (pixels[idx][1] * 255.0) as u8;
                let b = (pixels[idx][2] * 255.0) as u8;
                let a = (pixels[idx][3] * 255.0) as u8;

                image::Rgba([r, g, b, a])
            });

            image.save_with_format(path, ImageFormat::Png)?;
        }
        OutputFormat::Hdr => {
            let image = ImageBuffer::from_fn(width, height, |x, y| {
                let idx = (y * width + x) as usize;

                let r = pixels[idx][0];
                let g = pixels[idx][1];
                let b = pixels[idx][2];

                image::Rgb([r, g, b])
            });

            image.save_with_format(path, ImageFormat::Hdr)?;
        }
    }

    Ok(())
}
