use crate::model::RenderModel;
use crate::memory::render_buffer::{RenderBuffer, RenderBufferAllocator};
use crate::rt;
use crate::rt::blas::Blas;
use crate::render::device::{WrappedDevice, WrappedDeviceRef};
use anyhow::Result;
use ash::vk::{
    AccelerationStructureBuildGeometryInfoKHR, AccelerationStructureBuildRangeInfoKHR, AccelerationStructureBuildSizesInfoKHR, AccelerationStructureBuildTypeKHR,
    AccelerationStructureDeviceAddressInfoKHR, AccelerationStructureGeometryDataKHR, AccelerationStructureGeometryInstancesDataKHR, AccelerationStructureGeometryKHR, AccelerationStructureInstanceKHR,
    AccelerationStructureKHR, AccelerationStructureReferenceKHR, AccelerationStructureTypeKHR, BufferUsageFlags, BuildAccelerationStructureFlagsKHR, BuildAccelerationStructureModeKHR,
    DeviceOrHostAddressConstKHR, DeviceOrHostAddressKHR, DeviceSize, GeometryFlagsKHR, GeometryInstanceFlagsKHR, GeometryTypeKHR, Packed24_8, TransformMatrixKHR,
};
use glam::Affine3A;
use gpu_allocator::MemoryLocation;
use std::{mem, slice};

pub struct Tlas {
    device: WrappedDeviceRef,

    pub handle: AccelerationStructureKHR,
    pub tlas_buffer: RenderBuffer,
    pub instance_buffer: RenderBuffer,
}

impl Drop for Tlas {
    fn drop(&mut self) {
        unsafe {
            self.device.acceleration_device.destroy_acceleration_structure(self.handle, None);
        }
    }
}

pub fn create_acceleration_instance(device: &WrappedDevice, blas: &[Blas], models: &[RenderModel]) -> Result<Vec<AccelerationStructureInstanceKHR>> {
    let mut acceleration_instances: Vec<AccelerationStructureInstanceKHR> = Vec::with_capacity(blas.len());
    let mut blas_idx = 0;

    for model in models {
        for &(_, mesh_transform) in model.meshes.iter() {
            let affine_transform = Affine3A::from_mat4(mesh_transform).to_cols_array_2d();

            let transform = TransformMatrixKHR {
                matrix: [
                    affine_transform[0][0],
                    affine_transform[1][0],
                    affine_transform[2][0],
                    affine_transform[3][0],
                    affine_transform[0][1],
                    affine_transform[1][1],
                    affine_transform[2][1],
                    affine_transform[3][1],
                    affine_transform[0][2],
                    affine_transform[1][2],
                    affine_transform[2][2],
                    affine_transform[3][2],
                ],
            };

            let acceleration_address_info = AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(blas[blas_idx].handle);
            let acceleration_device_handle = unsafe { device.acceleration_device.get_acceleration_structure_device_address(&acceleration_address_info) };
            let acceleration_reference = AccelerationStructureReferenceKHR {
                device_handle: acceleration_device_handle,
            };

            let as_instance = AccelerationStructureInstanceKHR {
                transform,
                acceleration_structure_reference: acceleration_reference,
                instance_custom_index_and_mask: Packed24_8::new(0, 0xff),
                instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(0, GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8),
            };

            acceleration_instances.push(as_instance);

            blas_idx += 1;
        }
    }

    Ok(acceleration_instances)
}

pub fn create_tlas(device: WrappedDeviceRef, allocator: &RenderBufferAllocator, blas: &[Blas], models: &[RenderModel]) -> Result<Tlas> {
    let acceleration_instances = create_acceleration_instance(&device, blas, models)?;

    let instance_buffer = allocator.allocate(
        (acceleration_instances.len() * mem::size_of::<AccelerationStructureInstanceKHR>()) as DeviceSize,
        BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::SHADER_DEVICE_ADDRESS | BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        MemoryLocation::GpuOnly,
    )?;

    allocator.upload_data::<AccelerationStructureInstanceKHR>(&instance_buffer, &acceleration_instances)?;

    let geometry = AccelerationStructureGeometryKHR::default()
        .flags(GeometryFlagsKHR::OPAQUE | GeometryFlagsKHR::NO_DUPLICATE_ANY_HIT_INVOCATION)
        .geometry_type(GeometryTypeKHR::INSTANCES)
        .geometry(AccelerationStructureGeometryDataKHR {
            instances: AccelerationStructureGeometryInstancesDataKHR::default().array_of_pointers(false).data(DeviceOrHostAddressConstKHR {
                device_address: instance_buffer.device_addr().unwrap(),
            }),
        });

    let mut build_geometry_info = AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(AccelerationStructureTypeKHR::TOP_LEVEL)
        .mode(BuildAccelerationStructureModeKHR::BUILD)
        .flags(BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE | BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION)
        .geometries(slice::from_ref(&geometry));

    let acceleration_instances_len = acceleration_instances.len() as u32;

    let acceleration_build_sizes = unsafe {
        let mut acceleration_build_sizes = AccelerationStructureBuildSizesInfoKHR::default();

        device.acceleration_device.get_acceleration_structure_build_sizes(
            AccelerationStructureBuildTypeKHR::DEVICE,
            &build_geometry_info,
            &[acceleration_instances_len],
            &mut acceleration_build_sizes,
        );

        acceleration_build_sizes
    };

    let scratch_buffer = allocator.allocate(
        acceleration_build_sizes.build_scratch_size,
        BufferUsageFlags::SHADER_DEVICE_ADDRESS | BufferUsageFlags::STORAGE_BUFFER,
        MemoryLocation::GpuOnly,
    )?;

    let (tlas, tlas_buffer) = rt::allocate_acceleration_structure(&device, &allocator, AccelerationStructureTypeKHR::TOP_LEVEL, acceleration_build_sizes)?;

    build_geometry_info = build_geometry_info.dst_acceleration_structure(tlas).scratch_data(DeviceOrHostAddressKHR {
        device_address: scratch_buffer.device_addr().unwrap(),
    });

    let build_range_info = vec![AccelerationStructureBuildRangeInfoKHR::default().primitive_count(acceleration_instances_len)];

    device.single_time_command(|cmd_buf| unsafe {
        device
            .acceleration_device
            .cmd_build_acceleration_structures(cmd_buf, slice::from_ref(&build_geometry_info), slice::from_ref(&build_range_info.as_slice()));
    })?;

    Ok(Tlas {
        device,
        handle: tlas,
        tlas_buffer,
        instance_buffer,
    })
}
