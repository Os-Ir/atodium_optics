#![feature(const_fn_floating_point_arithmetic)]

use spectrum_table::Gamut;
use std::fs::File;
use std::io::Write;

fn main() {
    let res = 64;
    let (scale, table) = spectrum_table::generate_spectrum_tables(Gamut::Srgb, res);

    write_to_file(
        "shaders/src/spectrum_table/srgb_to_spectrum_table.rs",
        res,
        &scale,
        &table,
        "SRGB_TO_SPECTRUM_SCALE",
        "SRGB_TO_SPECTRUM_TABLE",
    )
    .unwrap();
}

fn write_to_file(path: &str, res: usize, scale: &[f32], table: &[f32], scale_field_name: &str, table_field_name: &str) -> std::io::Result<()> {
    let mut file = File::create(path)?;

    write!(file, "pub const {}: [f32; {}] = [ ", scale_field_name, res)?;
    for (i, &num) in scale.iter().enumerate() {
        if i != 0 {
            file.write_all(b", ")?;
        }
        write!(file, "{:.9e}", num)?;
    }
    file.write_all(b" ];\n\n")?;

    write!(file, "pub const {}: [[[[[f32; 3]; {}]; {}]; {}]; 3] = [\n", table_field_name, res, res, res)?;
    let mut ptr = 0;
    for _ in 0..3 {
        file.write_all(b"[ ")?;
        for _ in 0..res {
            file.write_all(b"[ ")?;
            for _ in 0..res {
                file.write_all(b"[ ")?;
                for _ in 0..res {
                    write!(file, "[ {:.9e}, {:.9e}, {:.9e} ], ", table[ptr], table[ptr + 1], table[ptr + 2])?;
                    ptr += 3;
                }
                file.write_all(b"],\n")?;
            }
            file.write_all(b"],")?;
        }
        file.write_all(b"],")?;
    }
    file.write_all(b"];\n")?;

    Ok(())
}
