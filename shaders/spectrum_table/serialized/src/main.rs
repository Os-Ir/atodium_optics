use spectrum_table::Gamut;

fn main() {
    let res = 64;
    let (scale, table) = spectrum_table::generate_spectrum_tables(Gamut::Srgb, res);
}
