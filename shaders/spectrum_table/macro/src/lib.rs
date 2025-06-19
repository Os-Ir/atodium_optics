use proc_macro::TokenStream;
use quote::quote;
use spectrum_table::Gamut;
use syn::parse::{Parse, ParseStream};
use syn::{parse_macro_input, LitInt, LitStr, Token};

#[proc_macro]
pub fn rgb_to_spectrum_table(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as RgbToSpectrumTableInput);

    let gamut = parse_gamut(&input.gamut.value());

    let generated = generate_spectrum_tables(
        gamut,
        input.resolution.base10_parse().expect("Failed to parse resolution"),
        &input.scale_table_name.value(),
        &input.data_table_name.value(),
    );

    TokenStream::from(generated)
}

#[derive(Debug)]
struct RgbToSpectrumTableInput {
    gamut: LitStr,
    resolution: LitInt,
    scale_table_name: LitStr,
    data_table_name: LitStr,
}

impl Parse for RgbToSpectrumTableInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let gamut: LitStr = input.parse()?;
        input.parse::<Token![,]>()?;

        let resolution: LitInt = input.parse()?;
        input.parse::<Token![,]>()?;

        let scale_table_name: LitStr = input.parse()?;
        input.parse::<Token![,]>()?;

        let data_table_name: LitStr = input.parse()?;
        if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
        }

        Ok(RgbToSpectrumTableInput {
            gamut,
            resolution,
            scale_table_name,
            data_table_name,
        })
    }
}

fn parse_gamut(gamut: &str) -> Gamut {
    match gamut {
        "srgb" => Gamut::Srgb,
        "pro_photo_rgb" => Gamut::ProPhotoRgb,
        "aces2065_1" => Gamut::Aces2065_1,
        "rec2020" => Gamut::Rec2020,
        "ergb" => Gamut::Ergb,
        "xyz" => Gamut::Xyz,
        "dci_p3" => Gamut::DciP3,
        _ => panic!("Unknown gamut id: {}", gamut),
    }
}

fn generate_spectrum_tables(gamut: Gamut, res: usize, scale_table_name: &str, data_table_name: &str) -> proc_macro2::TokenStream {
    let tables = spectrum_table::init_tables(gamut);

    let mut scale = Vec::with_capacity(res);
    for k in 0..res {
        scale.push(spectrum_table::smooth_step(spectrum_table::smooth_step(k as f64 / (res - 1) as f64)) as f32);
    }

    let mut out = vec![0.0; 9 * res * res * res];

    for l in 0..3 {
        for j in 0..res {
            let y = j as f64 / (res - 1) as f64;

            for i in 0..res {
                let x = i as f64 / (res - 1) as f64;

                let mut coes = [0.0; 3];

                let start = res / 5;

                for k in start..res {
                    let b = scale[k] as f64;

                    let mut rgb = [0.0; 3];
                    rgb[l] = b;
                    rgb[(l + 1) % 3] = x * b;
                    rgb[(l + 2) % 3] = y * b;

                    spectrum_table::gauss_newton(&tables, rgb, &mut coes, 15).expect("Gauss-Newton optimization failed");

                    let c0 = 360.0;
                    let c1 = 1.0 / (830.0 - 360.0);

                    let a_in = coes[0];
                    let b_in = coes[1];
                    let c_in = coes[2];

                    let idx = ((l * res + k) * res + j) * res + i;

                    out[3 * idx] = (a_in * spectrum_table::sqr(c1)) as f32;
                    out[3 * idx + 1] = (b_in * c1 - 2.0 * a_in * c0 * spectrum_table::sqr(c1)) as f32;
                    out[3 * idx + 2] = (c_in - b_in * c0 * c1 + a_in * spectrum_table::sqr(c0) * spectrum_table::sqr(c1)) as f32;
                }

                for k in (0..start).rev() {
                    let b = scale[k] as f64;

                    let mut rgb = [0.0; 3];
                    rgb[l] = b;
                    rgb[(l + 1) % 3] = x * b;
                    rgb[(l + 2) % 3] = y * b;

                    spectrum_table::gauss_newton(&tables, rgb, &mut coes, 15).expect("Gauss-Newton optimization failed");

                    let c0 = 360.0;
                    let c1 = 1.0 / (830.0 - 360.0);

                    let a_in = coes[0];
                    let b_in = coes[1];
                    let c_in = coes[2];

                    let idx = ((l * res + k) * res + j) * res + i;

                    out[3 * idx] = (a_in * spectrum_table::sqr(c1)) as f32;
                    out[3 * idx + 1] = (b_in * c1 - 2.0 * a_in * c0 * spectrum_table::sqr(c1)) as f32;
                    out[3 * idx + 2] = (c_in - b_in * c0 * c1 + a_in * spectrum_table::sqr(c0) * spectrum_table::sqr(c1)) as f32;
                }
            }
        }
    }

    let scale_table = generate_scale_table(scale_table_name, &scale);
    let data_table = generate_data_table(data_table_name, res, &out);

    quote! {
        #scale_table
        #data_table
    }
}

fn generate_scale_table(name: &str, scale: &[f32]) -> proc_macro2::TokenStream {
    let len = scale.len();
    let values = scale.iter().map(|v| quote! { #v }).collect::<Vec<_>>();

    quote! {
        pub const #name: [f32; #len] = [
            #(#values),*
        ];
    }
}

fn generate_data_table(name: &str, res: usize, data: &[f32]) -> proc_macro2::TokenStream {
    let mut table = Vec::new();

    for l in 0..3 {
        let mut l_array = Vec::new();
        for k in 0..res {
            let mut k_array = Vec::new();
            for j in 0..res {
                let mut j_array = Vec::new();
                for i in 0..res {
                    let idx = ((l * res + k) * res + j) * res + i;
                    let values = [data[3 * idx], data[3 * idx + 1], data[3 * idx + 2]];

                    j_array.push(quote! { [#(#values),*] });
                }
                k_array.push(quote! { [#(#j_array),*] });
            }
            l_array.push(quote! { [#(#k_array),*] });
        }
        table.push(quote! { [#(#l_array),*] });
    }

    quote! {
        pub const #name: [[[[[f32; 3]; #res]; #res]; #res]; 3] = [
            #(#table),*
        ];
    }
}
