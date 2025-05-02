use crate::render_resource::render_buffer::{RenderBuffer, RenderBufferAllocator};
use crate::vk_context::device::WrappedDevice;
use anyhow::Result;
use ash::vk::{
    AccelerationStructureBuildGeometryInfoKHR, AccelerationStructureBuildRangeInfoKHR, AccelerationStructureBuildSizesInfoKHR, AccelerationStructureBuildTypeKHR, AccelerationStructureCreateInfoKHR,
    AccelerationStructureGeometryDataKHR, AccelerationStructureGeometryKHR, AccelerationStructureGeometryTrianglesDataKHR, AccelerationStructureKHR, AccelerationStructureTypeKHR, BufferUsageFlags,
    BuildAccelerationStructureFlagsKHR, BuildAccelerationStructureModeKHR, DeviceOrHostAddressConstKHR, DeviceOrHostAddressKHR, DeviceSize, Format, GeometryFlagsKHR, GeometryTypeKHR, IndexType,
};
use gpu_allocator::MemoryLocation;
use std::slice;
use crate::model::mesh::MeshBuffer;
use crate::model::vertex::Vertex;

pub struct Blas {
    pub handle: AccelerationStructureKHR,
    pub buffer: RenderBuffer,
}

pub fn create_blas(device: &WrappedDevice, allocator: &RenderBufferAllocator, mesh_buffer: &MeshBuffer) -> Result<Blas> {
    let vertex_device_addr = DeviceOrHostAddressConstKHR {
        device_address: mesh_buffer.vertex_buffer.device_addr(),
    };

    let index_device_addr = DeviceOrHostAddressConstKHR {
        device_address: mesh_buffer.index_buffer.device_addr(),
    };

    let triangles_data = AccelerationStructureGeometryTrianglesDataKHR::default()
        .vertex_data(vertex_device_addr)
        .vertex_format(Format::R32G32B32_SFLOAT)
        .vertex_stride(size_of::<Vertex>() as DeviceSize)
        .max_vertex(mesh_buffer.vertices.len() as u32)
        .index_type(IndexType::UINT32)
        .index_data(index_device_addr);

    let geometry_data = AccelerationStructureGeometryDataKHR { triangles: triangles_data };

    let geometry = AccelerationStructureGeometryKHR::default()
        .flags(GeometryFlagsKHR::OPAQUE)
        .geometry_type(GeometryTypeKHR::TRIANGLES)
        .geometry(geometry_data);

    let build_geometry_info = AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .mode(BuildAccelerationStructureModeKHR::BUILD)
        .flags(BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .geometries(slice::from_ref(&geometry));

    let triangle_count = (mesh_buffer.indices.len() / 3) as u32;

    let mut build_sizes = AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        device
            .acceleration_device
            .get_acceleration_structure_build_sizes(AccelerationStructureBuildTypeKHR::DEVICE, &build_geometry_info, &[triangle_count], &mut build_sizes)
    };

    let blas_buffer = allocator.allocate(
        build_sizes.acceleration_structure_size,
        BufferUsageFlags::SHADER_DEVICE_ADDRESS | BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        MemoryLocation::GpuOnly,
    )?;

    let blas_info = AccelerationStructureCreateInfoKHR::default()
        .ty(AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .buffer(blas_buffer.buffer)
        .size(build_sizes.acceleration_structure_size);

    let blas = unsafe { device.acceleration_device.create_acceleration_structure(&blas_info, None)? };

    let scratch_buffer = allocator.allocate(
        build_sizes.build_scratch_size,
        BufferUsageFlags::SHADER_DEVICE_ADDRESS | BufferUsageFlags::STORAGE_BUFFER,
        MemoryLocation::GpuOnly,
    )?;

    let build_geometry_info = AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .flags(BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .geometries(slice::from_ref(&geometry))
        .mode(BuildAccelerationStructureModeKHR::BUILD)
        .dst_acceleration_structure(blas)
        .scratch_data(DeviceOrHostAddressKHR {
            device_address: scratch_buffer.device_addr(),
        });

    let build_range_info = vec![AccelerationStructureBuildRangeInfoKHR::default().primitive_count(triangle_count)];

    device.single_time_command(|device, command_buffer| unsafe {
        device
            .acceleration_device
            .cmd_build_acceleration_structures(command_buffer, slice::from_ref(&build_geometry_info), slice::from_ref(&build_range_info.as_slice()));
    })?;

    Ok(Blas {
        handle: blas,
        buffer: blas_buffer,
    })
}
