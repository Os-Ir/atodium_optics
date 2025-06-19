use spirv_std::glam::{Mat4, Quat, Vec3};

#[derive(Copy, Clone)]
#[repr(C)]
pub struct AnimatedTransform {
    start_transform: Mat4,
    end_transform: Mat4,
    start_time: f32,
    end_time: f32,
    actually_animated: bool,

    translate: [Vec3; 2],
    rotate: [Quat; 2],
    scale: [Mat4; 2],
}

impl AnimatedTransform {
    pub fn new(start_transform: Mat4, end_transform: Mat4, start_time: f32, end_time: f32) -> Self {
        let actually_animated = start_transform != end_transform;

        let (scale_start, rotation_start, translation_start) = start_transform.to_scale_rotation_translation();
        let (scale_end, rotation_end, translation_end) = end_transform.to_scale_rotation_translation();

        Self {
            start_transform,
            end_transform,
            start_time,
            end_time,
            actually_animated,
            translate: [translation_start, translation_end],
            rotate: [rotation_start, rotation_end],
            scale: [Mat4::from_scale(scale_start), Mat4::from_scale(scale_end)],
        }
    }

    pub fn interpolate(&self, time: f32) -> Mat4 {
        if !self.actually_animated || time <= self.start_time {
            return self.start_transform;
        }

        if time >= self.end_time {
            return self.end_transform;
        }

        let dt = (time - self.start_time) / (self.end_time - self.start_time);

        let scale = self.scale[0] + (self.scale[1] - self.scale[0]) * dt;
        let rotation = self.rotate[0].slerp(self.rotate[1], dt);
        let translation = self.translate[0] + (self.translate[1] - self.translate[0]) * dt;

        Mat4::from_scale_rotation_translation(Vec3::new(scale.x_axis.x, scale.y_axis.y, scale.z_axis.z), rotation, translation)
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct CameraTransform {
    render_from_camera: Mat4,
    world_from_render: AnimatedTransform,
}
