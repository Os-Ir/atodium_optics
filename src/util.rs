use anyhow::anyhow;
use anyhow::{bail, Result};
use bytemuck::Pod;
use image::codecs::hdr::HdrEncoder;
use image::{ImageBuffer, ImageFormat};
use std::ffi::{c_char, CStr};
use std::fs::File;
use std::mem;
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
    if (width * height) as usize * mem::size_of::<[f32; 4]>() != pixels.len() * mem::size_of::<T>() {
        bail!("Image dimensions does not match pixels length: {}", pixels.len());
    }

    let pixels = bytemuck::cast_slice::<T, [f32; 4]>(pixels);

    match output_format {
        OutputFormat::Png => {
            let image = ImageBuffer::from_fn(width, height, |x, y| {
                let idx = (y * width + x) as usize;

                let r: u8 = (pixels[idx][0] * 255.0) as _;
                let g: u8 = (pixels[idx][1] * 255.0) as _;
                let b: u8 = (pixels[idx][2] * 255.0) as _;
                let a: u8 = (pixels[idx][3] * 255.0) as _;

                image::Rgba([r, g, b, a])
            });

            image.save_with_format(path, ImageFormat::Png)?;
        }
        OutputFormat::Hdr => {
            let pixels = pixels.iter().map(|pixel| image::Rgb([pixel[0], pixel[1], pixel[2]])).collect::<Vec<_>>();
            let mut file = File::create(path)?;
            let encoder = HdrEncoder::new(&mut file);
            encoder.encode(&pixels, width as usize, height as usize)?;
        }
    }

    Ok(())
}
