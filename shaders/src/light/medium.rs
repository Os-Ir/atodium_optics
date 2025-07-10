use crate::light::ray::Ray;
use crate::spectrum::{SampledSpectrum, SampledWavelengths};
use core::ops::Deref;
use spirv_std::glam::{Vec2, Vec3};

#[repr(C)]
#[derive(Copy, Clone)]
pub struct PhaseFunctionSample {
    val: f32,
    pdf: f32,
    input_direction: Vec3,
}

pub trait IPhaseFunction {
    fn func_value(&self, output_direction: Vec3, input_direction: Vec3) -> f32;

    fn sample(&self, output_direction: Vec3, u: Vec2) -> Option<PhaseFunctionSample>;

    fn pdf(&self, output_direction: Vec3, input_direction: Vec3) -> f32;
}

#[derive(Clone, Copy)]
pub enum PhaseFunction {
    Hg,
}

impl Deref for PhaseFunction {
    type Target = dyn IPhaseFunction;

    fn deref(&self) -> &Self::Target {
        todo!()
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RayMajorantSegment {
    t_min: f32,
    t_max: f32,
    sigma_majorant: SampledSpectrum,
}

pub trait IRayMajorantIterator {
    fn next(&mut self) -> Option<RayMajorantSegment>;
}

#[derive(Clone, Copy)]
pub enum RayMajorantIterator {}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MediumProperties {
    sigma_a: SampledSpectrum,
    sigma_s: SampledSpectrum,
    phase: PhaseFunction,
    emitted: SampledSpectrum,
}

pub trait IMedium {
    fn is_emissive(&self) -> bool;

    fn sample_point(&self, point: Vec3, lambda: SampledWavelengths) -> f32;

    fn sample_ray(&self, ray: Ray, t_max: f32, lambda: SampledWavelengths) -> RayMajorantIterator;
}

#[derive(Clone, Copy)]
pub enum Medium {
    Homogeneous,
    Grid,
    RgbGrid,
    Cloud,
    NanoVdb,
}

impl Deref for Medium {
    type Target = dyn IMedium;

    fn deref(&self) -> &Self::Target {
        todo!()
    }
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct MediumInterface {
    pub inside: Option<Medium>,
    pub outside: Option<Medium>,
}

impl MediumInterface {
    pub fn new(inside: Option<Medium>, outside: Option<Medium>) -> Self {
        Self { inside, outside }
    }

    pub fn is_medium_transition(&self) -> bool {
        todo!()
    }
}
