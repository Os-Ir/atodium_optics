use crate::vulkan_context::device::WrappedDeviceRef;
use anyhow::Result;
use ash::vk::{Buffer, BufferCopy, BufferCreateInfo, BufferDeviceAddressInfo, BufferUsageFlags, CommandBuffer, DeviceAddress, DeviceSize, IndexType, SharingMode};
use core::slice;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator as GpuAllocator, AllocatorCreateDesc};
use gpu_allocator::{AllocatorDebugSettings, MemoryLocation};
use std::ops::Deref;
use std::sync::{Arc, Mutex};
use std::{cmp, ptr};

pub struct RenderBuffer {
    pub device: WrappedDeviceRef,
    pub gpu_allocator: GpuAllocatorRef,
    pub memory_location: MemoryLocation,
    pub size: DeviceSize,
    pub buffer: Buffer,
    pub allocation: Option<Allocation>,
}

impl RenderBuffer {
    pub fn new(device: WrappedDeviceRef, gpu_allocator: GpuAllocatorRef, memory_location: MemoryLocation, size: DeviceSize, buffer: Buffer, allocation: Allocation) -> Self {
        Self {
            device,
            gpu_allocator,
            memory_location,
            size,
            buffer,
            allocation: Some(allocation),
        }
    }

    pub fn bind_as_vertex_buffer(&self, command_buffer: CommandBuffer, binding_index: u32, offset: DeviceSize) {
        unsafe {
            self.device
                .cmd_bind_vertex_buffers(command_buffer, binding_index, slice::from_ref(&self.buffer), slice::from_ref(&offset))
        };
    }

    pub fn bind_as_index_buffer(&self, command_buffer: CommandBuffer, offset: DeviceSize, index_type: IndexType) {
        unsafe { self.device.cmd_bind_index_buffer(command_buffer, self.buffer, offset, index_type) };
    }

    pub fn copy_from(&self, source: &RenderBuffer) -> Result<()> {
        self.device.single_time_command(|device, command_buffer| unsafe {
            let region = BufferCopy::default().size(cmp::min(self.size, source.size));

            device.cmd_copy_buffer(command_buffer, source.buffer, self.buffer, slice::from_ref(&region))
        })?;

        Ok(())
    }

    pub fn device_addr(&self) -> DeviceAddress {
        let info = BufferDeviceAddressInfo::default().buffer(self.buffer);

        unsafe { self.device.get_buffer_device_address(&info) }
    }
}

impl Drop for RenderBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);

            if let Some(allocation) = self.allocation.take() {
                self.gpu_allocator.lock().unwrap().free(allocation).unwrap();
            }
        }
    }
}

pub type GpuAllocatorRef = Arc<Mutex<GpuAllocator>>;

#[derive(Clone)]
pub struct RenderBufferAllocatorRef(Arc<RenderBufferAllocator>);

impl Deref for RenderBufferAllocatorRef {
    type Target = Arc<RenderBufferAllocator>;

    fn deref(&self) -> &Arc<RenderBufferAllocator> {
        &self.0
    }
}

impl From<RenderBufferAllocator> for RenderBufferAllocatorRef {
    fn from(allocator: RenderBufferAllocator) -> Self {
        Self(Arc::new(allocator))
    }
}

pub struct RenderBufferAllocator {
    device: WrappedDeviceRef,
    gpu_allocator: GpuAllocatorRef,
}

impl RenderBufferAllocator {
    pub fn new(device: WrappedDeviceRef) -> Result<Self> {
        let debug_settings = AllocatorDebugSettings {
            log_leaks_on_shutdown: false,
            log_memory_information: true,
            log_allocations: true,
            log_stack_traces: true,
            ..Default::default()
        };

        let gpu_allocator_desc = AllocatorCreateDesc {
            instance: device.instance.clone(),
            device: device.handle.clone(),
            physical_device: device.physical_device,
            debug_settings,
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        };

        let gpu_allocator = GpuAllocator::new(&gpu_allocator_desc)?;

        Ok(Self {
            device,
            gpu_allocator: Arc::new(Mutex::new(gpu_allocator)),
        })
    }

    pub fn allocate(&self, size: DeviceSize, usage: BufferUsageFlags, location: MemoryLocation) -> Result<RenderBuffer> {
        unsafe {
            let buffer_info = BufferCreateInfo::default().size(size).usage(usage).sharing_mode(SharingMode::EXCLUSIVE);

            let buffer = self.device.create_buffer(&buffer_info, None)?;
            let requirements = self.device.get_buffer_memory_requirements(buffer);

            let allocate_create_desc = AllocationCreateDesc {
                name: "buffer allocation",
                requirements,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            };

            let allocation = self.gpu_allocator.lock().unwrap().allocate(&allocate_create_desc)?;
            let memory = allocation.memory();

            self.device.bind_buffer_memory(buffer, memory, allocation.offset())?;

            let _addr = {
                let info = BufferDeviceAddressInfo::default().buffer(buffer);
                self.device.get_buffer_device_address(&info)
            };

            Ok(RenderBuffer::new(self.device.clone(), self.gpu_allocator.clone(), location, size, buffer, allocation))
        }
    }

    pub fn upload_data<T: Copy>(&self, buffer: &RenderBuffer, data: &[T]) -> Result<()> {
        unsafe {
            let data_ptr = data.as_ptr() as *const u8;
            let data_size = size_of_val(data);

            if buffer.memory_location != MemoryLocation::GpuOnly {
                let allocation = buffer.allocation.as_ref().unwrap();

                let dst = allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;
                let dst_size = allocation.size() as usize;

                ptr::copy_nonoverlapping(data_ptr, dst, cmp::min(data_size, dst_size));
            } else {
                let staging_buffer = self.allocate(buffer.size, BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu)?;
                let staging_allocation = staging_buffer.allocation.as_ref().unwrap();

                let staging_ptr = staging_allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;
                let staging_size = staging_allocation.size() as usize;
                ptr::copy_nonoverlapping(data_ptr, staging_ptr, cmp::min(data_size, staging_size));

                self.device.single_time_command(|device, command_buffer| {
                    let regions = BufferCopy::default().size(buffer.size).src_offset(0).dst_offset(0);

                    device.handle.cmd_copy_buffer(command_buffer, staging_buffer.buffer, buffer.buffer, slice::from_ref(&regions));
                })?;
            }

            Ok(())
        }
    }
}
