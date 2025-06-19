use core::ops::Deref;
use spirv_std::glam::Vec2;

pub struct FilterSample {
    pub point: Vec2,
    pub weight: f32,
}

pub trait IFilmFilter {
    fn radius(&self) -> Vec2;

    fn evaluate(&self, point: Vec2) -> f32;

    fn integral(&self) -> f32;

    fn sample(&self, u: Vec2) -> FilterSample;
}

#[derive(Copy, Clone)]
pub enum FilmFilter {
    BoxFilter,
    GaussianFilter,
    MitchellFilter,
    LanczosSincFilter,
    TriangleFilter,
}

impl Deref for FilmFilter {
    type Target = dyn IFilmFilter;

    fn deref(&self) -> &Self::Target {
        todo!()
    }
}
