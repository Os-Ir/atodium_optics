use crate::camera::filter::FilmFilter;
use crate::spectrum::color::RgbColor;
use crate::spectrum::color_space::RgbColorSpace;
use crate::spectrum::{DenselySampledSpectrum, DiscreteSpectrum, ISpectrum, SampledSpectrum, SampledWavelengths, CIE_X_SPECTRUM, CIE_Y_SPECTRUM, CIE_Z_SPECTRUM, LAMBDA_DENSELY_COUNT, LAMBDA_MIN};
use core::array;
use core::ops::Deref;
use spirv_std::glam::{Mat3, UVec2, Vec2, Vec3, Vec4, Vec4Swizzles};
use spirv_std::Image;

pub const SWATCH_REFLECTANCE_COUNT: usize = 24;

pub const SWATCH_REFLECTANCES: [DiscreteSpectrum; SWATCH_REFLECTANCE_COUNT] = unsafe {
    [
        DiscreteSpectrum::from_interleaved::<36, 72>([
            380.0, 0.055, 390.0, 0.058, 400.0, 0.061, 410.0, 0.062, 420.0, 0.062, 430.0, 0.062, 440.0, 0.062, 450.0, 0.062, 460.0, 0.062, 470.0, 0.062, 480.0, 0.062, 490.0, 0.063, 500.0, 0.065,
            510.0, 0.070, 520.0, 0.076, 530.0, 0.079, 540.0, 0.081, 550.0, 0.084, 560.0, 0.091, 570.0, 0.103, 580.0, 0.119, 590.0, 0.134, 600.0, 0.143, 610.0, 0.147, 620.0, 0.151, 630.0, 0.158,
            640.0, 0.168, 650.0, 0.179, 660.0, 0.188, 670.0, 0.190, 680.0, 0.186, 690.0, 0.181, 700.0, 0.182, 710.0, 0.187, 720.0, 0.196, 730.0, 0.209,
        ]),
        DiscreteSpectrum::from_interleaved::<36, 72>([
            380.0, 0.117, 390.0, 0.143, 400.0, 0.175, 410.0, 0.191, 420.0, 0.196, 430.0, 0.199, 440.0, 0.204, 450.0, 0.213, 460.0, 0.228, 470.0, 0.251, 480.0, 0.280, 490.0, 0.309, 500.0, 0.329,
            510.0, 0.333, 520.0, 0.315, 530.0, 0.286, 540.0, 0.273, 550.0, 0.276, 560.0, 0.277, 570.0, 0.289, 580.0, 0.339, 590.0, 0.420, 600.0, 0.488, 610.0, 0.525, 620.0, 0.546, 630.0, 0.562,
            640.0, 0.578, 650.0, 0.595, 660.0, 0.612, 670.0, 0.625, 680.0, 0.638, 690.0, 0.656, 700.0, 0.678, 710.0, 0.700, 720.0, 0.717, 730.0, 0.734,
        ]),
        DiscreteSpectrum::from_interleaved::<36, 72>([
            380.0, 0.130, 390.0, 0.177, 400.0, 0.251, 410.0, 0.306, 420.0, 0.324, 430.0, 0.330, 440.0, 0.333, 450.0, 0.331, 460.0, 0.323, 470.0, 0.311, 480.0, 0.298, 490.0, 0.285, 500.0, 0.269,
            510.0, 0.250, 520.0, 0.231, 530.0, 0.214, 540.0, 0.199, 550.0, 0.185, 560.0, 0.169, 570.0, 0.157, 580.0, 0.149, 590.0, 0.145, 600.0, 0.142, 610.0, 0.141, 620.0, 0.141, 630.0, 0.141,
            640.0, 0.143, 650.0, 0.147, 660.0, 0.152, 670.0, 0.154, 680.0, 0.150, 690.0, 0.144, 700.0, 0.136, 710.0, 0.132, 720.0, 0.135, 730.0, 0.147,
        ]),
        DiscreteSpectrum::from_interleaved::<36, 72>([
            380.0, 0.051, 390.0, 0.054, 400.0, 0.056, 410.0, 0.057, 420.0, 0.058, 430.0, 0.059, 440.0, 0.060, 450.0, 0.061, 460.0, 0.062, 470.0, 0.063, 480.0, 0.065, 490.0, 0.067, 500.0, 0.075,
            510.0, 0.101, 520.0, 0.145, 530.0, 0.178, 540.0, 0.184, 550.0, 0.170, 560.0, 0.149, 570.0, 0.133, 580.0, 0.122, 590.0, 0.115, 600.0, 0.109, 610.0, 0.105, 620.0, 0.104, 630.0, 0.106,
            640.0, 0.109, 650.0, 0.112, 660.0, 0.114, 670.0, 0.114, 680.0, 0.112, 690.0, 0.112, 700.0, 0.115, 710.0, 0.120, 720.0, 0.125, 730.0, 0.130,
        ]),
        DiscreteSpectrum::from_interleaved::<36, 72>([
            380.0, 0.144, 390.0, 0.198, 400.0, 0.294, 410.0, 0.375, 420.0, 0.408, 430.0, 0.421, 440.0, 0.426, 450.0, 0.426, 460.0, 0.419, 470.0, 0.403, 480.0, 0.379, 490.0, 0.346, 500.0, 0.311,
            510.0, 0.281, 520.0, 0.254, 530.0, 0.229, 540.0, 0.214, 550.0, 0.208, 560.0, 0.202, 570.0, 0.194, 580.0, 0.193, 590.0, 0.200, 600.0, 0.214, 610.0, 0.230, 620.0, 0.241, 630.0, 0.254,
            640.0, 0.279, 650.0, 0.313, 660.0, 0.348, 670.0, 0.366, 680.0, 0.366, 690.0, 0.359, 700.0, 0.358, 710.0, 0.365, 720.0, 0.377, 730.0, 0.398,
        ]),
        DiscreteSpectrum::from_interleaved::<36, 72>([
            380.0, 0.136, 390.0, 0.179, 400.0, 0.247, 410.0, 0.297, 420.0, 0.320, 430.0, 0.337, 440.0, 0.355, 450.0, 0.381, 460.0, 0.419, 470.0, 0.466, 480.0, 0.510, 490.0, 0.546, 500.0, 0.567,
            510.0, 0.574, 520.0, 0.569, 530.0, 0.551, 540.0, 0.524, 550.0, 0.488, 560.0, 0.445, 570.0, 0.400, 580.0, 0.350, 590.0, 0.299, 600.0, 0.252, 610.0, 0.221, 620.0, 0.204, 630.0, 0.196,
            640.0, 0.191, 650.0, 0.188, 660.0, 0.191, 670.0, 0.199, 680.0, 0.212, 690.0, 0.223, 700.0, 0.232, 710.0, 0.233, 720.0, 0.229, 730.0, 0.229,
        ]),
        DiscreteSpectrum::from_interleaved::<36, 72>([
            380.0, 0.054, 390.0, 0.054, 400.0, 0.053, 410.0, 0.054, 420.0, 0.054, 430.0, 0.055, 440.0, 0.055, 450.0, 0.055, 460.0, 0.056, 470.0, 0.057, 480.0, 0.058, 490.0, 0.061, 500.0, 0.068,
            510.0, 0.089, 520.0, 0.125, 530.0, 0.154, 540.0, 0.174, 550.0, 0.199, 560.0, 0.248, 570.0, 0.335, 580.0, 0.444, 590.0, 0.538, 600.0, 0.587, 610.0, 0.595, 620.0, 0.591, 630.0, 0.587,
            640.0, 0.584, 650.0, 0.584, 660.0, 0.590, 670.0, 0.603, 680.0, 0.620, 690.0, 0.639, 700.0, 0.655, 710.0, 0.663, 720.0, 0.663, 730.0, 0.667,
        ]),
        DiscreteSpectrum::from_interleaved::<36, 72>([
            380.0, 0.122, 390.0, 0.164, 400.0, 0.229, 410.0, 0.286, 420.0, 0.327, 430.0, 0.361, 440.0, 0.388, 450.0, 0.400, 460.0, 0.392, 470.0, 0.362, 480.0, 0.316, 490.0, 0.260, 500.0, 0.209,
            510.0, 0.168, 520.0, 0.138, 530.0, 0.117, 540.0, 0.104, 550.0, 0.096, 560.0, 0.090, 570.0, 0.086, 580.0, 0.084, 590.0, 0.084, 600.0, 0.084, 610.0, 0.084, 620.0, 0.084, 630.0, 0.085,
            640.0, 0.090, 650.0, 0.098, 660.0, 0.109, 670.0, 0.123, 680.0, 0.143, 690.0, 0.169, 700.0, 0.205, 710.0, 0.244, 720.0, 0.287, 730.0, 0.332,
        ]),
        DiscreteSpectrum::from_interleaved::<36, 72>([
            380.0, 0.096, 390.0, 0.115, 400.0, 0.131, 410.0, 0.135, 420.0, 0.133, 430.0, 0.132, 440.0, 0.130, 450.0, 0.128, 460.0, 0.125, 470.0, 0.120, 480.0, 0.115, 490.0, 0.110, 500.0, 0.105,
            510.0, 0.100, 520.0, 0.095, 530.0, 0.093, 540.0, 0.092, 550.0, 0.093, 560.0, 0.096, 570.0, 0.108, 580.0, 0.156, 590.0, 0.265, 600.0, 0.399, 610.0, 0.500, 620.0, 0.556, 630.0, 0.579,
            640.0, 0.588, 650.0, 0.591, 660.0, 0.593, 670.0, 0.594, 680.0, 0.598, 690.0, 0.602, 700.0, 0.607, 710.0, 0.609, 720.0, 0.609, 730.0, 0.610,
        ]),
        DiscreteSpectrum::from_interleaved::<36, 72>([
            380.0, 0.092, 390.0, 0.116, 400.0, 0.146, 410.0, 0.169, 420.0, 0.178, 430.0, 0.173, 440.0, 0.158, 450.0, 0.139, 460.0, 0.119, 470.0, 0.101, 480.0, 0.087, 490.0, 0.075, 500.0, 0.066,
            510.0, 0.060, 520.0, 0.056, 530.0, 0.053, 540.0, 0.051, 550.0, 0.051, 560.0, 0.052, 570.0, 0.052, 580.0, 0.051, 590.0, 0.052, 600.0, 0.058, 610.0, 0.073, 620.0, 0.096, 630.0, 0.119,
            640.0, 0.141, 650.0, 0.166, 660.0, 0.194, 670.0, 0.227, 680.0, 0.265, 690.0, 0.309, 700.0, 0.355, 710.0, 0.396, 720.0, 0.436, 730.0, 0.478,
        ]),
        DiscreteSpectrum::from_interleaved::<36, 72>([
            380.0, 0.061, 390.0, 0.061, 400.0, 0.062, 410.0, 0.063, 420.0, 0.064, 430.0, 0.066, 440.0, 0.069, 450.0, 0.075, 460.0, 0.085, 470.0, 0.105, 480.0, 0.139, 490.0, 0.192, 500.0, 0.271,
            510.0, 0.376, 520.0, 0.476, 530.0, 0.531, 540.0, 0.549, 550.0, 0.546, 560.0, 0.528, 570.0, 0.504, 580.0, 0.471, 590.0, 0.428, 600.0, 0.381, 610.0, 0.347, 620.0, 0.327, 630.0, 0.318,
            640.0, 0.312, 650.0, 0.310, 660.0, 0.314, 670.0, 0.327, 680.0, 0.345, 690.0, 0.363, 700.0, 0.376, 710.0, 0.381, 720.0, 0.378, 730.0, 0.379,
        ]),
        DiscreteSpectrum::from_interleaved::<36, 72>([
            380.0, 0.063, 390.0, 0.063, 400.0, 0.063, 410.0, 0.064, 420.0, 0.064, 430.0, 0.064, 440.0, 0.065, 450.0, 0.066, 460.0, 0.067, 470.0, 0.068, 480.0, 0.071, 490.0, 0.076, 500.0, 0.087,
            510.0, 0.125, 520.0, 0.206, 530.0, 0.305, 540.0, 0.383, 550.0, 0.431, 560.0, 0.469, 570.0, 0.518, 580.0, 0.568, 590.0, 0.607, 600.0, 0.628, 610.0, 0.637, 620.0, 0.640, 630.0, 0.642,
            640.0, 0.645, 650.0, 0.648, 660.0, 0.651, 670.0, 0.653, 680.0, 0.657, 690.0, 0.664, 700.0, 0.673, 710.0, 0.680, 720.0, 0.684, 730.0, 0.688,
        ]),
        DiscreteSpectrum::from_interleaved::<36, 72>([
            380.0, 0.066, 390.0, 0.079, 400.0, 0.102, 410.0, 0.146, 420.0, 0.200, 430.0, 0.244, 440.0, 0.282, 450.0, 0.309, 460.0, 0.308, 470.0, 0.278, 480.0, 0.231, 490.0, 0.178, 500.0, 0.130,
            510.0, 0.094, 520.0, 0.070, 530.0, 0.054, 540.0, 0.046, 550.0, 0.042, 560.0, 0.039, 570.0, 0.038, 580.0, 0.038, 590.0, 0.038, 600.0, 0.038, 610.0, 0.039, 620.0, 0.039, 630.0, 0.040,
            640.0, 0.041, 650.0, 0.042, 660.0, 0.044, 670.0, 0.045, 680.0, 0.046, 690.0, 0.046, 700.0, 0.048, 710.0, 0.052, 720.0, 0.057, 730.0, 0.065,
        ]),
        DiscreteSpectrum::from_interleaved::<36, 72>([
            380.0, 0.052, 390.0, 0.053, 400.0, 0.054, 410.0, 0.055, 420.0, 0.057, 430.0, 0.059, 440.0, 0.061, 450.0, 0.066, 460.0, 0.075, 470.0, 0.093, 480.0, 0.125, 490.0, 0.178, 500.0, 0.246,
            510.0, 0.307, 520.0, 0.337, 530.0, 0.334, 540.0, 0.317, 550.0, 0.293, 560.0, 0.262, 570.0, 0.230, 580.0, 0.198, 590.0, 0.165, 600.0, 0.135, 610.0, 0.115, 620.0, 0.104, 630.0, 0.098,
            640.0, 0.094, 650.0, 0.092, 660.0, 0.093, 670.0, 0.097, 680.0, 0.102, 690.0, 0.108, 700.0, 0.113, 710.0, 0.115, 720.0, 0.114, 730.0, 0.114,
        ]),
        DiscreteSpectrum::from_interleaved::<36, 72>([
            380.0, 0.050, 390.0, 0.049, 400.0, 0.048, 410.0, 0.047, 420.0, 0.047, 430.0, 0.047, 440.0, 0.047, 450.0, 0.047, 460.0, 0.046, 470.0, 0.045, 480.0, 0.044, 490.0, 0.044, 500.0, 0.045,
            510.0, 0.046, 520.0, 0.047, 530.0, 0.048, 540.0, 0.049, 550.0, 0.050, 560.0, 0.054, 570.0, 0.060, 580.0, 0.072, 590.0, 0.104, 600.0, 0.178, 610.0, 0.312, 620.0, 0.467, 630.0, 0.581,
            640.0, 0.644, 650.0, 0.675, 660.0, 0.690, 670.0, 0.698, 680.0, 0.706, 690.0, 0.715, 700.0, 0.724, 710.0, 0.730, 720.0, 0.734, 730.0, 0.738,
        ]),
        DiscreteSpectrum::from_interleaved::<36, 72>([
            380.0, 0.058, 390.0, 0.054, 400.0, 0.052, 410.0, 0.052, 420.0, 0.053, 430.0, 0.054, 440.0, 0.056, 450.0, 0.059, 460.0, 0.067, 470.0, 0.081, 480.0, 0.107, 490.0, 0.152, 500.0, 0.225,
            510.0, 0.336, 520.0, 0.462, 530.0, 0.559, 540.0, 0.616, 550.0, 0.650, 560.0, 0.672, 570.0, 0.694, 580.0, 0.710, 590.0, 0.723, 600.0, 0.731, 610.0, 0.739, 620.0, 0.746, 630.0, 0.752,
            640.0, 0.758, 650.0, 0.764, 660.0, 0.769, 670.0, 0.771, 680.0, 0.776, 690.0, 0.782, 700.0, 0.790, 710.0, 0.796, 720.0, 0.799, 730.0, 0.804,
        ]),
        DiscreteSpectrum::from_interleaved::<36, 72>([
            380.0, 0.145, 390.0, 0.195, 400.0, 0.283, 410.0, 0.346, 420.0, 0.362, 430.0, 0.354, 440.0, 0.334, 450.0, 0.306, 460.0, 0.276, 470.0, 0.248, 480.0, 0.218, 490.0, 0.190, 500.0, 0.168,
            510.0, 0.149, 520.0, 0.127, 530.0, 0.107, 540.0, 0.100, 550.0, 0.102, 560.0, 0.104, 570.0, 0.109, 580.0, 0.137, 590.0, 0.200, 600.0, 0.290, 610.0, 0.400, 620.0, 0.516, 630.0, 0.615,
            640.0, 0.687, 650.0, 0.732, 660.0, 0.760, 670.0, 0.774, 680.0, 0.783, 690.0, 0.793, 700.0, 0.803, 710.0, 0.812, 720.0, 0.817, 730.0, 0.825,
        ]),
        DiscreteSpectrum::from_interleaved::<36, 72>([
            380.0, 0.108, 390.0, 0.141, 400.0, 0.192, 410.0, 0.236, 420.0, 0.261, 430.0, 0.286, 440.0, 0.317, 450.0, 0.353, 460.0, 0.390, 470.0, 0.426, 480.0, 0.446, 490.0, 0.444, 500.0, 0.423,
            510.0, 0.385, 520.0, 0.337, 530.0, 0.283, 540.0, 0.231, 550.0, 0.185, 560.0, 0.146, 570.0, 0.118, 580.0, 0.101, 590.0, 0.090, 600.0, 0.082, 610.0, 0.076, 620.0, 0.074, 630.0, 0.073,
            640.0, 0.073, 650.0, 0.074, 660.0, 0.076, 670.0, 0.077, 680.0, 0.076, 690.0, 0.075, 700.0, 0.073, 710.0, 0.072, 720.0, 0.074, 730.0, 0.079,
        ]),
        DiscreteSpectrum::from_interleaved::<36, 72>([
            380.0, 0.189, 390.0, 0.255, 400.0, 0.423, 410.0, 0.660, 420.0, 0.811, 430.0, 0.862, 440.0, 0.877, 450.0, 0.884, 460.0, 0.891, 470.0, 0.896, 480.0, 0.899, 490.0, 0.904, 500.0, 0.907,
            510.0, 0.909, 520.0, 0.911, 530.0, 0.910, 540.0, 0.911, 550.0, 0.914, 560.0, 0.913, 570.0, 0.916, 580.0, 0.915, 590.0, 0.916, 600.0, 0.914, 610.0, 0.915, 620.0, 0.918, 630.0, 0.919,
            640.0, 0.921, 650.0, 0.923, 660.0, 0.924, 670.0, 0.922, 680.0, 0.922, 690.0, 0.925, 700.0, 0.927, 710.0, 0.930, 720.0, 0.930, 730.0, 0.933,
        ]),
        DiscreteSpectrum::from_interleaved::<36, 72>([
            380.0, 0.171, 390.0, 0.232, 400.0, 0.365, 410.0, 0.507, 420.0, 0.567, 430.0, 0.583, 440.0, 0.588, 450.0, 0.590, 460.0, 0.591, 470.0, 0.590, 480.0, 0.588, 490.0, 0.588, 500.0, 0.589,
            510.0, 0.589, 520.0, 0.591, 530.0, 0.590, 540.0, 0.590, 550.0, 0.590, 560.0, 0.589, 570.0, 0.591, 580.0, 0.590, 590.0, 0.590, 600.0, 0.587, 610.0, 0.585, 620.0, 0.583, 630.0, 0.580,
            640.0, 0.578, 650.0, 0.576, 660.0, 0.574, 670.0, 0.572, 680.0, 0.571, 690.0, 0.569, 700.0, 0.568, 710.0, 0.568, 720.0, 0.566, 730.0, 0.566,
        ]),
        DiscreteSpectrum::from_interleaved::<36, 72>([
            380.0, 0.144, 390.0, 0.192, 400.0, 0.272, 410.0, 0.331, 420.0, 0.350, 430.0, 0.357, 440.0, 0.361, 450.0, 0.363, 460.0, 0.363, 470.0, 0.361, 480.0, 0.359, 490.0, 0.358, 500.0, 0.358,
            510.0, 0.359, 520.0, 0.360, 530.0, 0.360, 540.0, 0.361, 550.0, 0.361, 560.0, 0.360, 570.0, 0.362, 580.0, 0.362, 590.0, 0.361, 600.0, 0.359, 610.0, 0.358, 620.0, 0.355, 630.0, 0.352,
            640.0, 0.350, 650.0, 0.348, 660.0, 0.345, 670.0, 0.343, 680.0, 0.340, 690.0, 0.338, 700.0, 0.335, 710.0, 0.334, 720.0, 0.332, 730.0, 0.331,
        ]),
        DiscreteSpectrum::from_interleaved::<36, 72>([
            380.0, 0.105, 390.0, 0.131, 400.0, 0.163, 410.0, 0.180, 420.0, 0.186, 430.0, 0.190, 440.0, 0.193, 450.0, 0.194, 460.0, 0.194, 470.0, 0.192, 480.0, 0.191, 490.0, 0.191, 500.0, 0.191,
            510.0, 0.192, 520.0, 0.192, 530.0, 0.192, 540.0, 0.192, 550.0, 0.192, 560.0, 0.192, 570.0, 0.193, 580.0, 0.192, 590.0, 0.192, 600.0, 0.191, 610.0, 0.189, 620.0, 0.188, 630.0, 0.186,
            640.0, 0.184, 650.0, 0.182, 660.0, 0.181, 670.0, 0.179, 680.0, 0.178, 690.0, 0.176, 700.0, 0.174, 710.0, 0.173, 720.0, 0.172, 730.0, 0.171,
        ]),
        DiscreteSpectrum::from_interleaved::<36, 72>([
            380.0, 0.068, 390.0, 0.077, 400.0, 0.084, 410.0, 0.087, 420.0, 0.089, 430.0, 0.090, 440.0, 0.092, 450.0, 0.092, 460.0, 0.091, 470.0, 0.090, 480.0, 0.090, 490.0, 0.090, 500.0, 0.090,
            510.0, 0.090, 520.0, 0.090, 530.0, 0.090, 540.0, 0.090, 550.0, 0.090, 560.0, 0.090, 570.0, 0.090, 580.0, 0.090, 590.0, 0.089, 600.0, 0.089, 610.0, 0.088, 620.0, 0.087, 630.0, 0.086,
            640.0, 0.086, 650.0, 0.085, 660.0, 0.084, 670.0, 0.084, 680.0, 0.083, 690.0, 0.083, 700.0, 0.082, 710.0, 0.081, 720.0, 0.081, 730.0, 0.081,
        ]),
        DiscreteSpectrum::from_interleaved::<36, 72>([
            380.0, 0.031, 390.0, 0.032, 400.0, 0.032, 410.0, 0.033, 420.0, 0.033, 430.0, 0.033, 440.0, 0.033, 450.0, 0.033, 460.0, 0.032, 470.0, 0.032, 480.0, 0.032, 490.0, 0.032, 500.0, 0.032,
            510.0, 0.032, 520.0, 0.032, 530.0, 0.032, 540.0, 0.032, 550.0, 0.032, 560.0, 0.032, 570.0, 0.032, 580.0, 0.032, 590.0, 0.032, 600.0, 0.032, 610.0, 0.032, 620.0, 0.032, 630.0, 0.032,
            640.0, 0.032, 650.0, 0.032, 660.0, 0.032, 670.0, 0.032, 680.0, 0.032, 690.0, 0.032, 700.0, 0.032, 710.0, 0.032, 720.0, 0.032, 730.0, 0.033,
        ]),
    ]
};

#[derive(Clone, Copy)]
#[repr(C)]
pub struct PixelSensor {
    pub xyz_from_sensor_rgb: Mat3,
    curve_r: DenselySampledSpectrum,
    curve_g: DenselySampledSpectrum,
    curve_b: DenselySampledSpectrum,
    image_ratio: f32,
}

impl PixelSensor {
    pub fn new(r: &dyn ISpectrum, g: &dyn ISpectrum, b: &dyn ISpectrum, color_space: &RgbColorSpace, sensor_illuminant: &dyn ISpectrum, image_ratio: f32) -> Self {
        let curve_r = DenselySampledSpectrum::new::<LAMBDA_DENSELY_COUNT>(LAMBDA_MIN as _, r);
        let curve_g = DenselySampledSpectrum::new::<LAMBDA_DENSELY_COUNT>(LAMBDA_MIN as _, g);
        let curve_b = DenselySampledSpectrum::new::<LAMBDA_DENSELY_COUNT>(LAMBDA_MIN as _, b);

        let rgb_camera: [Vec3; SWATCH_REFLECTANCE_COUNT] = array::from_fn(|i| Self::project_reflectance(&SWATCH_REFLECTANCES[i], sensor_illuminant, &curve_r, &curve_g, &curve_b));

        let sensor_white_g = sensor_illuminant.inner_product_densely(&curve_g);
        let sensor_white_y = sensor_illuminant.inner_product_densely(&CIE_Y_SPECTRUM);

        let xyz_output: [Vec3; SWATCH_REFLECTANCE_COUNT] =
            array::from_fn(|i| sensor_white_y / sensor_white_g * Self::project_reflectance(&SWATCH_REFLECTANCES[i], &color_space.illuminant, &CIE_X_SPECTRUM, &CIE_Y_SPECTRUM, &CIE_Z_SPECTRUM));

        let xyz_from_sensor_rgb = Self::linear_least_square(&rgb_camera, &xyz_output);

        Self {
            xyz_from_sensor_rgb,
            curve_r,
            curve_g,
            curve_b,
            image_ratio,
        }
    }

    pub fn sensor_rgb(&self, mut luminance: SampledSpectrum, lambda: &SampledWavelengths) -> RgbColor {
        luminance = luminance.safe_div(lambda.pdf_spectrum());

        RgbColor::new(
            (self.curve_r.sample(lambda) * luminance).average(),
            (self.curve_g.sample(lambda) * luminance).average(),
            (self.curve_b.sample(lambda) * luminance).average(),
        ) * self.image_ratio
    }

    fn project_reflectance(reflect: &dyn ISpectrum, illuminant: &dyn ISpectrum, b1: &dyn ISpectrum, b2: &dyn ISpectrum, b3: &dyn ISpectrum) -> Vec3 {
        let mut result = Vec3::ZERO;
        let mut g_integral: f32 = 0.0;

        for add in 0..LAMBDA_DENSELY_COUNT {
            let lambda = LAMBDA_MIN + add as f32;

            g_integral += b2.get_value(lambda) * illuminant.get_value(lambda);

            result.x += b1.get_value(lambda) * reflect.get_value(lambda) * illuminant.get_value(lambda);
            result.y += b2.get_value(lambda) * reflect.get_value(lambda) * illuminant.get_value(lambda);
            result.z += b3.get_value(lambda) * reflect.get_value(lambda) * illuminant.get_value(lambda);
        }

        result / g_integral
    }

    fn linear_least_square<const N: usize>(a: &[Vec3; N], b: &[Vec3; N]) -> Mat3 {
        let mut at_a = [[0.0_f32; 3]; 3];
        let mut at_b = [[0.0_f32; 3]; 3];

        for i in 0..3 {
            for j in 0..3 {
                for k in 0..N {
                    at_a[i][j] += a[k][i] * a[k][j];
                    at_b[i][j] += a[k][i] * b[k][j];
                }
            }
        }

        let at_a = Mat3::from_cols_array_2d(&at_a).transpose();
        let at_b = Mat3::from_cols_array_2d(&at_b).transpose();

        (at_a.inverse() * at_b).transpose()
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VisibleSurface {
    albedo: SampledSpectrum,
    point: Vec3,
    partial_point_x: Vec3,
    partial_point_y: Vec3,
    geometry_normal: Vec3,
    shading_normal: Vec3,
    uv: Vec2,
    time: f32,
}

pub trait IFilm {
    fn add_sample(&mut self, point_film: UVec2, luminance: SampledSpectrum, lambda: &SampledWavelengths, surface: Option<VisibleSurface>, weight: f32);

    fn sample_bounds(&self) -> (Vec2, Vec2);

    fn use_visible_surface(&self) -> bool;

    fn add_splat(&mut self, point: Vec2, luminance: SampledSpectrum, lambda: &SampledWavelengths);

    fn sample_wavelengths(&self, u: f32) -> SampledWavelengths;

    fn full_resolution(&self) -> UVec2;

    fn get_pixel_rgb(&self, point_film: UVec2, splat_scale: f32) -> RgbColor;

    fn get_filter(&self) -> FilmFilter;
}

#[derive(Clone)]
pub enum Film {
    Rgb(RgbFilm),
}

impl Deref for Film {
    type Target = dyn IFilm;

    fn deref(&self) -> &Self::Target {
        match self {
            Film::Rgb(film) => film,
        }
    }
}

#[derive(Clone)]
#[repr(C)]
pub struct FilmBase {
    pub full_resolution: UVec2,
    pub pixel_bounds_min: Vec2,
    pub pixel_bounds_max: Vec2,
    pub filter: FilmFilter,
    pub diagonal: f32,
    pub sensor: PixelSensor,
}

#[derive(Clone)]
#[repr(C)]
pub struct RgbFilm {
    base: FilmBase,
    color_space: RgbColorSpace,
    max_component_value: f32,
    filter_integral: f32,
    output_rgb_from_sensor_rgb: Mat3,
    pixels_packed_rgb_weight_sum: Image!(2D, format = rgba32f, sampled = false),
    pixels_rgb_splat: Image!(2D, format = rgba32f, sampled = false),
}

impl Deref for RgbFilm {
    type Target = FilmBase;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl IFilm for RgbFilm {
    fn add_sample(&mut self, point_film: UVec2, luminance: SampledSpectrum, lambda: &SampledWavelengths, _: Option<VisibleSurface>, weight: f32) {
        let mut rgb = self.sensor.sensor_rgb(luminance, &lambda);

        let m = rgb.r.max(rgb.g).max(rgb.b);
        if m > self.max_component_value {
            rgb *= self.max_component_value / m;
        }

        let mut rgb_weight: Vec4 = self.pixels_packed_rgb_weight_sum.read(point_film);

        for i in 0..3 {
            rgb_weight[i] += weight * rgb[i];
        }
        rgb_weight[3] += weight;

        unsafe { self.pixels_packed_rgb_weight_sum.write(point_film, rgb_weight) };
    }

    fn sample_bounds(&self) -> (Vec2, Vec2) {
        let radius = self.filter.radius();

        let min = self.pixel_bounds_min - radius + Vec2::new(0.5, 0.5);
        let max = self.pixel_bounds_max + radius - Vec2::new(0.5, 0.5);

        (min, max)
    }

    fn use_visible_surface(&self) -> bool {
        false
    }

    fn add_splat(&mut self, point: Vec2, luminance: SampledSpectrum, lambda: &SampledWavelengths) {
        let mut rgb = self.sensor.sensor_rgb(luminance, &lambda);

        let m = rgb.r.max(rgb.g).max(rgb.b);
        if m > self.max_component_value {
            rgb *= self.max_component_value / m;
        }

        let point_discrete = point + Vec2::new(0.5, 0.5);
        let radius = self.filter.radius();

        let mut splat_bound_min = (point_discrete - radius).floor();
        let mut splat_bound_max = (point_discrete + radius).floor() + Vec2::new(1.0, 1.0);

        splat_bound_min = splat_bound_min.max(self.pixel_bounds_min);
        splat_bound_max = splat_bound_max.min(self.pixel_bounds_max);

        [splat_bound_min, splat_bound_max].iter().for_each(|&pi| {
            let wt = self.filter.evaluate(point - pi - Vec2::new(0.5, 0.5));

            if wt != 0.0 {
                todo!()
            }
        })
    }

    fn sample_wavelengths(&self, u: f32) -> SampledWavelengths {
        SampledWavelengths::sample_visible(u)
    }

    fn full_resolution(&self) -> UVec2 {
        self.full_resolution
    }

    fn get_pixel_rgb(&self, point_film: UVec2, splat_scale: f32) -> RgbColor {
        let rgb_weight: Vec4 = self.pixels_packed_rgb_weight_sum.read(point_film);
        let rgb_splat: Vec4 = self.pixels_rgb_splat.read(point_film);
        let rgb_splat = rgb_splat.xyz();

        let mut rgb = RgbColor::new(rgb_weight.x, rgb_weight.y, rgb_weight.z);
        let weight_sum = rgb_weight.w;

        if weight_sum != 0.0 {
            rgb /= weight_sum;
        }

        for i in 0..3 {
            rgb[i] += splat_scale / self.filter_integral * rgb_splat[i];
        }

        rgb
    }

    fn get_filter(&self) -> FilmFilter {
        self.filter
    }
}
