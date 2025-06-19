#![feature(const_fn_floating_point_arithmetic)]

use std::sync::{Arc, Mutex};
use std::thread;

const CIE_LAMBDA_MIN: f64 = 360.0;
const CIE_LAMBDA_MAX: f64 = 830.0;
const CIE_SAMPLES: usize = 95;

#[rustfmt::skip]
const CIE_X: [f64; CIE_SAMPLES] = [
    0.000129900000, 0.000232100000, 0.000414900000, 0.000741600000, 0.001368000000,
    0.002236000000, 0.004243000000, 0.007650000000, 0.014310000000, 0.023190000000,
    0.043510000000, 0.077630000000, 0.134380000000, 0.214770000000, 0.283900000000,
    0.328500000000, 0.348280000000, 0.348060000000, 0.336200000000, 0.318700000000,
    0.290800000000, 0.251100000000, 0.195360000000, 0.142100000000, 0.095640000000,
    0.057950010000, 0.032010000000, 0.014700000000, 0.004900000000, 0.002400000000,
    0.009300000000, 0.029100000000, 0.063270000000, 0.109600000000, 0.165500000000,
    0.225749900000, 0.290400000000, 0.359700000000, 0.433449900000, 0.512050100000,
    0.594500000000, 0.678400000000, 0.762100000000, 0.842500000000, 0.916300000000,
    0.978600000000, 1.026300000000, 1.056700000000, 1.062200000000, 1.045600000000,
    1.002600000000, 0.938400000000, 0.854449900000, 0.751400000000, 0.642400000000,
    0.541900000000, 0.447900000000, 0.360800000000, 0.283500000000, 0.218700000000,
    0.164900000000, 0.121200000000, 0.087400000000, 0.063600000000, 0.046770000000,
    0.032900000000, 0.022700000000, 0.015840000000, 0.011359160000, 0.008110916000,
    0.005790346000, 0.004109457000, 0.002899327000, 0.002049190000, 0.001439971000,
    0.000999949300, 0.000690078600, 0.000476021300, 0.000332301100, 0.000234826100,
    0.000166150500, 0.000117413000, 0.000083075270, 0.000058706520, 0.000041509940,
    0.000029353260, 0.000020673830, 0.000014559770, 0.000010253980, 0.000007221456,
    0.000005085868, 0.000003581652, 0.000002522525, 0.000001776509, 0.000001251141,
];

#[rustfmt::skip]
const CIE_Y: [f64; CIE_SAMPLES] = [
    0.000003917000, 0.000006965000, 0.000012390000, 0.000022020000, 0.000039000000,
    0.000064000000, 0.000120000000, 0.000217000000, 0.000396000000, 0.000640000000,
    0.001210000000, 0.002180000000, 0.004000000000, 0.007300000000, 0.011600000000,
    0.016840000000, 0.023000000000, 0.029800000000, 0.038000000000, 0.048000000000,
    0.060000000000, 0.073900000000, 0.090980000000, 0.112600000000, 0.139020000000,
    0.169300000000, 0.208020000000, 0.258600000000, 0.323000000000, 0.407300000000,
    0.503000000000, 0.608200000000, 0.710000000000, 0.793200000000, 0.862000000000,
    0.914850100000, 0.954000000000, 0.980300000000, 0.994950100000, 1.000000000000,
    0.995000000000, 0.978600000000, 0.952000000000, 0.915400000000, 0.870000000000,
    0.816300000000, 0.757000000000, 0.694900000000, 0.631000000000, 0.566800000000,
    0.503000000000, 0.441200000000, 0.381000000000, 0.321000000000, 0.265000000000,
    0.217000000000, 0.175000000000, 0.138200000000, 0.107000000000, 0.081600000000,
    0.061000000000, 0.044580000000, 0.032000000000, 0.023200000000, 0.017000000000,
    0.011920000000, 0.008210000000, 0.005723000000, 0.004102000000, 0.002929000000,
    0.002091000000, 0.001484000000, 0.001047000000, 0.000740000000, 0.000520000000,
    0.000361100000, 0.000249200000, 0.000171900000, 0.000120000000, 0.000084800000,
    0.000060000000, 0.000042400000, 0.000030000000, 0.000021200000, 0.000014990000,
    0.000010600000, 0.000007465700, 0.000005257800, 0.000003702900, 0.000002607800,
    0.000001836600, 0.000001293400, 0.000000910930, 0.000000641530, 0.000000451810,
];

#[rustfmt::skip]
const CIE_Z: [f64; CIE_SAMPLES] = [
    0.000606100000, 0.001086000000, 0.001946000000, 0.003486000000, 0.006450001000,
    0.010549990000, 0.020050010000, 0.036210000000, 0.067850010000, 0.110200000000,
    0.207400000000, 0.371300000000, 0.645600000000, 1.039050100000, 1.385600000000,
    1.622960000000, 1.747060000000, 1.782600000000, 1.772110000000, 1.744100000000,
    1.669200000000, 1.528100000000, 1.287640000000, 1.041900000000, 0.812950100000,
    0.616200000000, 0.465180000000, 0.353300000000, 0.272000000000, 0.212300000000,
    0.158200000000, 0.111700000000, 0.078249990000, 0.057250010000, 0.042160000000,
    0.029840000000, 0.020300000000, 0.013400000000, 0.008749999000, 0.005749999000,
    0.003900000000, 0.002749999000, 0.002100000000, 0.001800000000, 0.001650001000,
    0.001400000000, 0.001100000000, 0.001000000000, 0.000800000000, 0.000600000000,
    0.000340000000, 0.000240000000, 0.000190000000, 0.000100000000, 0.000049999990,
    0.000030000000, 0.000020000000, 0.000010000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
];

const fn cie_d65_n(x: f64) -> f64 {
    x / 10566.864005283874576
}

#[rustfmt::skip]
const CIE_D65: [f64; CIE_SAMPLES] = [
    cie_d65_n(46.6383), cie_d65_n(49.3637), cie_d65_n(52.0891), cie_d65_n(51.0323), cie_d65_n(49.9755),
    cie_d65_n(52.3118), cie_d65_n(54.6482), cie_d65_n(68.7015), cie_d65_n(82.7549), cie_d65_n(87.1204),
    cie_d65_n(91.486) , cie_d65_n(92.4589), cie_d65_n(93.4318), cie_d65_n(90.057) , cie_d65_n(86.6823),
    cie_d65_n(95.7736), cie_d65_n(104.865), cie_d65_n(110.936), cie_d65_n(117.008), cie_d65_n(117.41) ,
    cie_d65_n(117.812), cie_d65_n(116.336), cie_d65_n(114.861), cie_d65_n(115.392), cie_d65_n(115.923),
    cie_d65_n(112.367), cie_d65_n(108.811), cie_d65_n(109.082), cie_d65_n(109.354), cie_d65_n(108.578),
    cie_d65_n(107.802), cie_d65_n(106.296), cie_d65_n(104.79) , cie_d65_n(106.239), cie_d65_n(107.689),
    cie_d65_n(106.047), cie_d65_n(104.405), cie_d65_n(104.225), cie_d65_n(104.046), cie_d65_n(102.023),
    cie_d65_n(100.0)  , cie_d65_n(98.1671), cie_d65_n(96.3342), cie_d65_n(96.0611), cie_d65_n(95.788) ,
    cie_d65_n(92.2368), cie_d65_n(88.6856), cie_d65_n(89.3459), cie_d65_n(90.0062), cie_d65_n(89.8026),
    cie_d65_n(89.5991), cie_d65_n(88.6489), cie_d65_n(87.6987), cie_d65_n(85.4936), cie_d65_n(83.2886),
    cie_d65_n(83.4939), cie_d65_n(83.6992), cie_d65_n(81.863) , cie_d65_n(80.0268), cie_d65_n(80.1207),
    cie_d65_n(80.2146), cie_d65_n(81.2462), cie_d65_n(82.2778), cie_d65_n(80.281) , cie_d65_n(78.2842),
    cie_d65_n(74.0027), cie_d65_n(69.7213), cie_d65_n(70.6652), cie_d65_n(71.6091), cie_d65_n(72.979) ,
    cie_d65_n(74.349) , cie_d65_n(67.9765), cie_d65_n(61.604) , cie_d65_n(65.7448), cie_d65_n(69.8856),
    cie_d65_n(72.4863), cie_d65_n(75.087) , cie_d65_n(69.3398), cie_d65_n(63.5927), cie_d65_n(55.0054),
    cie_d65_n(46.4182), cie_d65_n(56.6118), cie_d65_n(66.8054), cie_d65_n(65.0941), cie_d65_n(63.3828),
    cie_d65_n(63.8434), cie_d65_n(64.304) , cie_d65_n(61.8779), cie_d65_n(59.4519), cie_d65_n(55.7054),
    cie_d65_n(51.959) , cie_d65_n(54.6998), cie_d65_n(57.4406), cie_d65_n(58.8765), cie_d65_n(60.3125),
];

const fn cie_e_n(x: f64) -> f64 {
    x / 106.8
}

#[rustfmt::skip]
const CIE_E: [f64; CIE_SAMPLES] = [
    cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0),
    cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0),
    cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0),
    cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0),
    cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0),
    cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0),
    cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0),
    cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0),
    cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0),
    cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0),
    cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0),
    cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0),
    cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0),
    cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0),
    cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0),
    cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0),
    cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0),
    cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0),
    cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0), cie_e_n(1.0),
];

const fn cie_d50_n(x: f64) -> f64 {
    x / 10503.2
}

#[rustfmt::skip]
const CIE_D50: [f64; CIE_SAMPLES] = [
    cie_d50_n(23.942000) , cie_d50_n(25.451000) , cie_d50_n(26.961000) , cie_d50_n(25.724000) , cie_d50_n(24.488000) ,
    cie_d50_n(27.179000) , cie_d50_n(29.871000) , cie_d50_n(39.589000) , cie_d50_n(49.308000) , cie_d50_n(52.910000) ,
    cie_d50_n(56.513000) , cie_d50_n(58.273000) , cie_d50_n(60.034000) , cie_d50_n(58.926000) , cie_d50_n(57.818000) ,
    cie_d50_n(66.321000) , cie_d50_n(74.825000) , cie_d50_n(81.036000) , cie_d50_n(87.247000) , cie_d50_n(88.930000) ,
    cie_d50_n(90.612000) , cie_d50_n(90.990000) , cie_d50_n(91.368000) , cie_d50_n(93.238000) , cie_d50_n(95.109000) ,
    cie_d50_n(93.536000) , cie_d50_n(91.963000) , cie_d50_n(93.843000) , cie_d50_n(95.724000) , cie_d50_n(96.169000) ,
    cie_d50_n(96.613000) , cie_d50_n(96.871000) , cie_d50_n(97.129000) , cie_d50_n(99.614000) , cie_d50_n(102.099000),
    cie_d50_n(101.427000), cie_d50_n(100.755000), cie_d50_n(101.536000), cie_d50_n(102.317000), cie_d50_n(101.159000),
    cie_d50_n(100.000000), cie_d50_n(98.868000) , cie_d50_n(97.735000) , cie_d50_n(98.327000) , cie_d50_n(98.918000) ,
    cie_d50_n(96.208000) , cie_d50_n(93.499000) , cie_d50_n(95.593000) , cie_d50_n(97.688000) , cie_d50_n(98.478000) ,
    cie_d50_n(99.269000) , cie_d50_n(99.155000) , cie_d50_n(99.042000) , cie_d50_n(97.382000) , cie_d50_n(95.722000) ,
    cie_d50_n(97.290000) , cie_d50_n(98.857000) , cie_d50_n(97.262000) , cie_d50_n(95.667000) , cie_d50_n(96.929000) ,
    cie_d50_n(98.190000) , cie_d50_n(100.597000), cie_d50_n(103.003000), cie_d50_n(101.068000), cie_d50_n(99.133000) ,
    cie_d50_n(93.257000) , cie_d50_n(87.381000) , cie_d50_n(89.492000) , cie_d50_n(91.604000) , cie_d50_n(92.246000) ,
    cie_d50_n(92.889000) , cie_d50_n(84.872000) , cie_d50_n(76.854000) , cie_d50_n(81.683000) , cie_d50_n(86.511000) ,
    cie_d50_n(89.546000) , cie_d50_n(92.580000) , cie_d50_n(85.405000) , cie_d50_n(78.230000) , cie_d50_n(67.961000) ,
    cie_d50_n(57.692000) , cie_d50_n(70.307000) , cie_d50_n(82.923000) , cie_d50_n(80.599000) , cie_d50_n(78.274000) ,
    cie_d50_n(0.0)       , cie_d50_n(0.0)       , cie_d50_n(0.0)       , cie_d50_n(0.0)       , cie_d50_n(0.0)       ,
    cie_d50_n(0.0)       , cie_d50_n(0.0)       , cie_d50_n(0.0)       , cie_d50_n(0.0)       , cie_d50_n(0.0)       ,
];

const fn cie_d60_n(x: f64) -> f64 {
    x / 10536.3
}

#[rustfmt::skip]
const CIE_D60: [f64; CIE_SAMPLES] = [
    cie_d60_n(38.683115) , cie_d60_n(41.014457) , cie_d60_n(42.717548) , cie_d60_n(42.264182) , cie_d60_n(41.454941) ,
    cie_d60_n(41.763698) , cie_d60_n(46.605319) , cie_d60_n(59.226938) , cie_d60_n(72.278594) , cie_d60_n(78.231500) ,
    cie_d60_n(80.440600) , cie_d60_n(82.739580) , cie_d60_n(82.915027) , cie_d60_n(79.009168) , cie_d60_n(77.676264) ,
    cie_d60_n(85.163609) , cie_d60_n(95.681274) , cie_d60_n(103.267764), cie_d60_n(107.954821), cie_d60_n(109.777964),
    cie_d60_n(109.559187), cie_d60_n(108.418402), cie_d60_n(107.758141), cie_d60_n(109.071548), cie_d60_n(109.671404),
    cie_d60_n(106.734741), cie_d60_n(103.707873), cie_d60_n(103.981942), cie_d60_n(105.232199), cie_d60_n(105.235867),
    cie_d60_n(104.427667), cie_d60_n(103.052881), cie_d60_n(102.522934), cie_d60_n(104.371416), cie_d60_n(106.052671),
    cie_d60_n(104.948900), cie_d60_n(103.315154), cie_d60_n(103.416286), cie_d60_n(103.538599), cie_d60_n(102.099304),
    cie_d60_n(100.000000), cie_d60_n(97.992725) , cie_d60_n(96.751421) , cie_d60_n(97.102402) , cie_d60_n(96.712823) ,
    cie_d60_n(93.174457) , cie_d60_n(89.921479) , cie_d60_n(90.351933) , cie_d60_n(91.999793) , cie_d60_n(92.384009) ,
    cie_d60_n(92.098710) , cie_d60_n(91.722859) , cie_d60_n(90.646003) , cie_d60_n(88.327552) , cie_d60_n(86.526483) ,
    cie_d60_n(87.034239) , cie_d60_n(87.579186) , cie_d60_n(85.884584) , cie_d60_n(83.976140) , cie_d60_n(83.743140) ,
    cie_d60_n(84.724074) , cie_d60_n(86.450818) , cie_d60_n(87.493491) , cie_d60_n(86.546330) , cie_d60_n(83.483070) ,
    cie_d60_n(78.268785) , cie_d60_n(74.172451) , cie_d60_n(74.275184) , cie_d60_n(76.620385) , cie_d60_n(79.423856) ,
    cie_d60_n(79.051849) , cie_d60_n(71.763360) , cie_d60_n(65.471371) , cie_d60_n(67.984085) , cie_d60_n(74.106079) ,
    cie_d60_n(78.556612) , cie_d60_n(79.527120) , cie_d60_n(75.584935) , cie_d60_n(67.307163) , cie_d60_n(55.275106) ,
    cie_d60_n(49.273538) , cie_d60_n(59.008629) , cie_d60_n(70.892412) , cie_d60_n(70.950115) , cie_d60_n(67.163996) ,
    cie_d60_n(67.445480) , cie_d60_n(68.171371) , cie_d60_n(66.466636) , cie_d60_n(62.989809) , cie_d60_n(58.067786) ,
    cie_d60_n(54.990892) , cie_d60_n(56.915942) , cie_d60_n(60.825601) , cie_d60_n(62.987850) , cie_d60_n(0.0)       ,
];

#[rustfmt::skip]
const XYZ_TO_SRGB: [[f64; 3]; 3] = [[ 3.240479, -1.537150, -0.498535],
                                    [-0.969256,  1.875991,  0.041556],
                                    [ 0.055648, -0.204043,  1.057311]];

#[rustfmt::skip]
const SRGB_TO_XYZ: [[f64; 3]; 3] = [[0.412453, 0.357580, 0.180423],
                                    [0.212671, 0.715160, 0.072169],
                                    [0.019334, 0.119193, 0.950227]];

#[rustfmt::skip]
const XYZ_TO_XYZ: [[f64; 3]; 3] = [[1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0]];

#[rustfmt::skip]
const XYZ_TO_ERGB: [[f64; 3]; 3] = [[ 2.689989, -1.276020, -0.413844],
                                    [-1.022095,  1.978261,  0.043821],
                                    [ 0.061203, -0.224411,  1.162859]];

#[rustfmt::skip]
const ERGB_TO_XYZ: [[f64; 3]; 3] = [[0.496859, 0.339094, 0.164047],
                                    [0.256193, 0.678188, 0.065619],
                                    [0.023290, 0.113031, 0.863978]];

#[rustfmt::skip]
const XYZ_TO_PRO_PHOTO_RGB: [[f64; 3]; 3] = [[ 1.3459433, -0.2556075, -0.0511118],
                                             [-0.5445989,  1.5081673,  0.0205351],
                                             [ 0.0000000,  0.0000000,  1.2118128]];

#[rustfmt::skip]
const PRO_PHOTO_RGB_TO_XYZ: [[f64; 3]; 3] = [[0.7976749, 0.1351917, 0.0313534],
                                             [0.2880402, 0.7118741, 0.0000857],
                                             [0.0000000, 0.0000000, 0.8252100]];

#[rustfmt::skip]
const XYZ_TO_ACES2065_1: [[f64; 3]; 3] = [[ 1.0498110175, 0.0000000000, -0.0000974845],
                                          [-0.4959030231, 1.3733130458,  0.0982400361],
                                          [ 0.0000000000, 0.0000000000,  0.9912520182]];

#[rustfmt::skip]
const ACES2065_1_TO_XYZ: [[f64; 3]; 3] = [[0.9525523959, 0.0000000000,  0.0000936786],
                                          [0.3439664498, 0.7281660966, -0.0721325464],
                                          [0.0000000000, 0.0000000000,  1.0088251844]];

#[rustfmt::skip]
const XYZ_TO_REC2020: [[f64; 3]; 3] = [[ 1.7166511880, -0.3556707838, -0.2533662814],
                                       [-0.6666843518,  1.6164812366,  0.0157685458],
                                       [ 0.0176398574, -0.0427706133,  0.9421031212]];

#[rustfmt::skip]
const REC2020_TO_XYZ: [[f64; 3]; 3] = [[0.6369580483, 0.1446169036, 0.1688809752],
                                       [0.2627002120, 0.6779980715, 0.0593017165],
                                       [0.0000000000, 0.0280726930, 1.0609850577]];

#[rustfmt::skip]
const XYZ_TO_DCIP3: [[f64; 3]; 3] = [[ 2.493174800, -0.93126315, -0.402658820],
                                     [-0.829504250,  1.76269650,  0.023625137],
                                     [ 0.035853732, -0.07618918,  0.957095200]];

#[rustfmt::skip]
const DCIP3_TO_XYZ: [[f64; 3]; 3] = [[0.48663378, 0.26566276, 0.198173660],
                                     [0.22900413, 0.69172573, 0.079269454],
                                     [0.00000000, 0.04511256, 1.043714500]];

fn cie_interp(data: &[f64; CIE_SAMPLES], mut x: f64) -> f64 {
    x -= CIE_LAMBDA_MIN;
    x *= ((CIE_SAMPLES - 1) as f64) / (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN);

    let offset = if x < 0.0 {
        0
    } else if x as usize > CIE_SAMPLES - 2 {
        CIE_SAMPLES - 2
    } else {
        x as usize
    };

    let weight = x - offset as f64;

    (1.0 - weight) * data[offset] + weight * data[offset + 1]
}

fn lup_decompose<const N: usize>(a: &mut [[f64; N]; N], p: &mut [usize], tol: f64) -> Option<()> {
    assert_eq!(p.len(), N + 1);

    for i in 0..=N {
        p[i] = i;
    }

    for i in 0..N {
        let mut max_a = 0.0;
        let mut max_i = i;

        for k in i..N {
            let abs_a = a[k][i].abs();

            if abs_a > max_a {
                max_a = abs_a;
                max_i = k;
            }
        }

        if max_a < tol {
            return None;
        }

        if max_i != i {
            p.swap(i, max_i);
            a.swap(i, max_i);
            p[N] += 1;
        }

        for j in (i + 1)..N {
            a[j][i] /= a[i][i];

            for k in (i + 1)..N {
                a[j][k] -= a[j][i] * a[i][k];
            }
        }
    }

    Some(())
}

fn lup_solve<const N: usize>(a: &[[f64; N]; N], p: &[usize], b: &[f64; N]) -> [f64; N] {
    assert_eq!(p.len(), N + 1);

    let mut x = [0.0; N];

    for i in 0..N {
        x[i] = b[p[i]];

        for k in 0..i {
            x[i] -= a[i][k] * x[k];
        }
    }

    for i in (0..N).rev() {
        for k in (i + 1)..N {
            x[i] -= a[i][k] * x[k];
        }

        x[i] /= a[i][i];
    }

    x
}

const CIE_FINE_SAMPLES: usize = (CIE_SAMPLES - 1) * 3 + 1;
const RGB_TO_SPEC_EPSILON: f64 = 1.0e-4;

pub struct RgbToSpecTables {
    pub lambda_tbl: [f64; CIE_FINE_SAMPLES],
    pub rgb_tbl: [[f64; CIE_FINE_SAMPLES]; 3],
    pub rgb_to_xyz: [[f64; 3]; 3],
    pub xyz_to_rgb: [[f64; 3]; 3],
    pub xyz_whitepoint: [f64; 3],
}

#[derive(Copy, Clone, PartialEq)]
pub enum Gamut {
    Srgb,
    ProPhotoRgb,
    Aces2065_1,
    Rec2020,
    Ergb,
    Xyz,
    DciP3,
}

#[inline(always)]
fn sigmoid(x: f64) -> f64 {
    0.5 * x / (1.0 + x * x).sqrt() + 0.5
}

#[inline(always)]
pub fn smooth_step(x: f64) -> f64 {
    x * x * (3.0 - 2.0 * x)
}

#[inline(always)]
pub fn sqr(x: f64) -> f64 {
    x * x
}

fn cie_lab(tables: &RgbToSpecTables, p: &mut [f64; 3]) {
    let [xw, yw, zw] = tables.xyz_whitepoint;

    let mut x = 0.0;
    let mut y = 0.0;
    let mut z = 0.0;

    for i in 0..3 {
        x += p[i] * tables.rgb_to_xyz[0][i];
        y += p[i] * tables.rgb_to_xyz[1][i];
        z += p[i] * tables.rgb_to_xyz[2][i];
    }

    let f: fn(f64) -> f64 = |t| {
        const DELTA: f64 = 6.0 / 29.0;

        if t > DELTA * DELTA * DELTA {
            t.cbrt()
        } else {
            t / (DELTA * DELTA * 3.0) + 4.0 / 29.0
        }
    };

    p[0] = 116.0 * f(y / yw) - 16.0;
    p[1] = 500.0 * (f(x / xw) - f(y / yw));
    p[2] = 200.0 * (f(y / yw) - f(z / zw));
}

pub fn init_tables(gamut: Gamut) -> RgbToSpecTables {
    let h = (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN) / (CIE_FINE_SAMPLES - 1) as f64;

    let mut tables = RgbToSpecTables {
        lambda_tbl: [0.0; CIE_FINE_SAMPLES],
        rgb_tbl: [[0.0; CIE_FINE_SAMPLES]; 3],
        rgb_to_xyz: match gamut {
            Gamut::Srgb => SRGB_TO_XYZ,
            Gamut::Ergb => ERGB_TO_XYZ,
            Gamut::Xyz => XYZ_TO_XYZ,
            Gamut::ProPhotoRgb => PRO_PHOTO_RGB_TO_XYZ,
            Gamut::Aces2065_1 => ACES2065_1_TO_XYZ,
            Gamut::Rec2020 => REC2020_TO_XYZ,
            Gamut::DciP3 => DCIP3_TO_XYZ,
        },
        xyz_to_rgb: match gamut {
            Gamut::Srgb => XYZ_TO_SRGB,
            Gamut::Ergb => XYZ_TO_ERGB,
            Gamut::Xyz => XYZ_TO_XYZ,
            Gamut::ProPhotoRgb => XYZ_TO_PRO_PHOTO_RGB,
            Gamut::Aces2065_1 => XYZ_TO_ACES2065_1,
            Gamut::Rec2020 => XYZ_TO_REC2020,
            Gamut::DciP3 => XYZ_TO_DCIP3,
        },
        xyz_whitepoint: [0.0; 3],
    };

    let illuminant = match gamut {
        Gamut::Srgb | Gamut::Rec2020 | Gamut::DciP3 => &CIE_D65,
        Gamut::ProPhotoRgb => &CIE_D50,
        Gamut::Aces2065_1 => &CIE_D60,
        Gamut::Ergb | Gamut::Xyz => &CIE_E,
    };

    for i in 0..CIE_FINE_SAMPLES {
        let lambda = CIE_LAMBDA_MIN + (i as f64) * h;

        let x = cie_interp(&CIE_X, lambda);
        let y = cie_interp(&CIE_Y, lambda);
        let z = cie_interp(&CIE_Z, lambda);
        let illuminant_interp = cie_interp(illuminant, lambda);

        const CIE_FINE_SAMPLES_MINUS_1: usize = CIE_FINE_SAMPLES - 1;

        let weight = match i {
            0 | CIE_FINE_SAMPLES_MINUS_1 => 0.375 * h,
            _ if (i - 1) % 3 == 2 => 0.75 * h,
            _ => 1.125 * h,
        };

        tables.lambda_tbl[i] = lambda;

        for k in 0..3 {
            for j in 0..3 {
                tables.rgb_tbl[k][i] += tables.xyz_to_rgb[k][j] * [x, y, z][j] * illuminant_interp * weight;
            }
        }

        for i in 0..3 {
            tables.xyz_whitepoint[i] += [x, y, z][i] * illuminant_interp * weight;
        }
    }

    tables
}

fn eval_residual(tables: &RgbToSpecTables, coefficients: &[f64; 3], rgb: &[f64; 3], residual: &mut [f64; 3]) {
    let mut out = [0.0; 3];

    for i in 0..CIE_FINE_SAMPLES {
        let lambda = (tables.lambda_tbl[i] - CIE_LAMBDA_MIN) / (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN);

        let mut x = 0.0;
        for j in 0..3 {
            x = x * lambda + coefficients[j];
        }
        let s = sigmoid(x);

        for j in 0..3 {
            out[j] += tables.rgb_tbl[j][i] * s;
        }
    }

    let mut lab_out = out;
    cie_lab(tables, &mut lab_out);

    let mut lab_rgb = *rgb;
    cie_lab(tables, &mut lab_rgb);

    *residual = [lab_rgb[0] - lab_out[0], lab_rgb[1] - lab_out[1], lab_rgb[2] - lab_out[2]];
}

fn eval_jacobian(tables: &RgbToSpecTables, coefficients: &[f64; 3], rgb: &[f64; 3]) -> [[f64; 3]; 3] {
    let mut r0 = [0.0; 3];
    let mut r1 = [0.0; 3];
    let mut jacobian = [[0.0; 3]; 3];

    for i in 0..3 {
        let mut tmp_coe = *coefficients;
        tmp_coe[i] -= RGB_TO_SPEC_EPSILON;
        eval_residual(tables, &tmp_coe, rgb, &mut r0);

        let mut tmp_coe = *coefficients;
        tmp_coe[i] += RGB_TO_SPEC_EPSILON;
        eval_residual(tables, &tmp_coe, rgb, &mut r1);

        for j in 0..3 {
            jacobian[j][i] = (r1[j] - r0[j]) / (2.0 * RGB_TO_SPEC_EPSILON);
        }
    }

    jacobian
}

pub fn gauss_newton(tables: &RgbToSpecTables, rgb: [f64; 3], coefficients: &mut [f64; 3], max_iter: usize) -> Option<()> {
    for _ in 0..max_iter {
        let mut residual = [0.0; 3];
        eval_residual(tables, coefficients, &rgb, &mut residual);
        let mut jacobian = eval_jacobian(tables, coefficients, &rgb);

        let mut p = [0; 4];
        lup_decompose(&mut jacobian, &mut p, 1.0e-15)?;

        let x = lup_solve(&mut jacobian, &mut p, &residual);

        let mut r = 0.0;
        for j in 0..3 {
            coefficients[j] -= x[j];
            r += sqr(residual[j]);
        }

        let max_coefficients = coefficients[0].max(coefficients[1]).max(coefficients[2]);

        if max_coefficients > 200.0 {
            let scale = 200.0 / max_coefficients;

            for j in 0..3 {
                coefficients[j] *= scale;
            }
        }

        if r < 1.0e-6 {
            break;
        }
    }

    Some(())
}

pub fn generate_spectrum_tables(gamut: Gamut, res: usize) -> (Vec<f32>, Vec<f32>) {
    let tables = init_tables(gamut);

    let mut scale = Vec::with_capacity(res);
    for k in 0..res {
        scale.push(smooth_step(smooth_step(k as f64 / (res - 1) as f64)) as f32);
    }

    let tables_ref = Arc::new(tables);
    let scale_ref = Arc::new(scale);

    let out = Arc::new(Mutex::new(vec![0.0f32; 9 * res * res * res]));

    for l in 0..3 {
        let mut handles = vec![];

        for j in 0..res {
            let out_clone = out.clone();
            let tables_clone = tables_ref.clone();
            let scale_clone = scale_ref.clone();

            let handle = thread::spawn(move || {
                let y = j as f64 / (res - 1) as f64;

                for i in 0..res {
                    let x = i as f64 / (res - 1) as f64;
                    let mut coes = [0.0; 3];
                    let start = res / 5;

                    for k in start..res {
                        let b = scale_clone[k] as f64;
                        let mut rgb = [0.0; 3];
                        rgb[l] = b;
                        rgb[(l + 1) % 3] = x * b;
                        rgb[(l + 2) % 3] = y * b;

                        gauss_newton(&tables_clone, rgb, &mut coes, 15).expect("Gauss-Newton optimization failed");

                        let c0 = 360.0;
                        let c1 = 1.0 / (830.0 - 360.0);
                        let a_in = coes[0];
                        let b_in = coes[1];
                        let c_in = coes[2];
                        let idx = ((l * res + k) * res + j) * res + i;

                        let mut out_guard = out_clone.lock().unwrap();
                        out_guard[3 * idx] = (a_in * crate::sqr(c1)) as f32;
                        out_guard[3 * idx + 1] = (b_in * c1 - 2.0 * a_in * c0 * crate::sqr(c1)) as f32;
                        out_guard[3 * idx + 2] = (c_in - b_in * c0 * c1 + a_in * crate::sqr(c0) * crate::sqr(c1)) as f32;
                    }

                    coes = [0.0; 3];
                    for k in (0..start).rev() {
                        let b = scale_clone[k] as f64;
                        let mut rgb = [0.0; 3];
                        rgb[l] = b;
                        rgb[(l + 1) % 3] = x * b;
                        rgb[(l + 2) % 3] = y * b;

                        gauss_newton(&tables_clone, rgb, &mut coes, 15).expect("Gauss-Newton optimization failed");

                        let c0 = 360.0;
                        let c1 = 1.0 / (830.0 - 360.0);
                        let a_in = coes[0];
                        let b_in = coes[1];
                        let c_in = coes[2];
                        let idx = ((l * res + k) * res + j) * res + i;

                        let mut out_guard = out_clone.lock().unwrap();
                        out_guard[3 * idx] = (a_in * crate::sqr(c1)) as f32;
                        out_guard[3 * idx + 1] = (b_in * c1 - 2.0 * a_in * c0 * crate::sqr(c1)) as f32;
                        out_guard[3 * idx + 2] = (c_in - b_in * c0 * c1 + a_in * crate::sqr(c0) * crate::sqr(c1)) as f32;
                    }
                }

                println!("Finish loop: {} - {}", l, j);
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    let scale = Arc::try_unwrap(scale_ref).unwrap();
    let out = Arc::try_unwrap(out).unwrap().into_inner().unwrap();

    (scale, out)
}
