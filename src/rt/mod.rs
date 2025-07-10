use crate::memory::render_buffer::{RenderBuffer, RenderBufferAllocator};
use crate::render::device::WrappedDevice;
use anyhow::{anyhow, bail, Result};
use ash::vk::{
    AccelerationStructureBuildGeometryInfoKHR, AccelerationStructureBuildRangeInfoKHR, AccelerationStructureBuildSizesInfoKHR, AccelerationStructureBuildTypeKHR, AccelerationStructureCreateInfoKHR,
    AccelerationStructureGeometryKHR, AccelerationStructureKHR, AccelerationStructureTypeKHR, BufferUsageFlags, BuildAccelerationStructureFlagsKHR, BuildAccelerationStructureModeKHR,
};
use gpu_allocator::MemoryLocation;

pub mod blas;
pub mod tlas;

pub fn allocate_acceleration_structure(
    device: &WrappedDevice,
    allocator: &RenderBufferAllocator,
    ty: AccelerationStructureTypeKHR,
    build_sizes: AccelerationStructureBuildSizesInfoKHR,
) -> Result<(AccelerationStructureKHR, RenderBuffer)> {
    let buffer = allocator.allocate(
        build_sizes.acceleration_structure_size,
        BufferUsageFlags::SHADER_DEVICE_ADDRESS | BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        MemoryLocation::GpuOnly,
    )?;

    let create_info = AccelerationStructureCreateInfoKHR::default().ty(ty).buffer(buffer.buffer).size(build_sizes.acceleration_structure_size);

    let acceleration_structure = unsafe { device.acceleration_device.create_acceleration_structure(&create_info, None)? };

    Ok((acceleration_structure, buffer))
}

pub struct AccelerationStructureBuildData<'a> {
    ty: AccelerationStructureTypeKHR,
    geometries: Vec<AccelerationStructureGeometryKHR<'a>>,
    build_range_infos: Vec<AccelerationStructureBuildRangeInfoKHR>,
    build_size: Option<AccelerationStructureBuildSizesInfoKHR<'a>>,
}

impl<'a> AccelerationStructureBuildData<'a> {
    pub fn add_geometry(&mut self, as_geometry: AccelerationStructureGeometryKHR<'a>, as_build_range_info: AccelerationStructureBuildRangeInfoKHR) -> &mut Self {
        self.geometries.push(as_geometry);
        self.build_range_infos.push(as_build_range_info);
        self
    }

    pub fn finalize_geometry(&mut self, device: &WrappedDevice, flags: BuildAccelerationStructureFlagsKHR) -> Result<AccelerationStructureBuildSizesInfoKHR> {
        if self.geometries.is_empty() {
            bail!("No geometry added to build acceleration structure")
        }

        let build_geometry_info = AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(self.ty)
            .flags(flags)
            .mode(BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&self.geometries);

        let mut build_sizes = AccelerationStructureBuildSizesInfoKHR::default();

        let primitive_counts = self.build_range_infos.iter().map(|build_range_info| build_range_info.primitive_count).collect::<Vec<_>>();

        unsafe {
            device
                .acceleration_device
                .get_acceleration_structure_build_sizes(AccelerationStructureBuildTypeKHR::DEVICE, &build_geometry_info, &primitive_counts, &mut build_sizes)
        };

        self.build_size = Some(build_sizes);

        Ok(build_sizes)
    }

    pub fn make_create_info(&self) -> Result<AccelerationStructureCreateInfoKHR> {
        if self.geometries.is_empty() {
            bail!("No geometry added to build acceleration structure")
        }

        let create_info = AccelerationStructureCreateInfoKHR::default().ty(self.ty).size(
            self.build_size
                .ok_or_else(|| anyhow!("Build size for making create info is not finalized"))?
                .acceleration_structure_size,
        );

        Ok(create_info)
    }
}
