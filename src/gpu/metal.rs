//! The Metal backend.
//!
//! Two things here are worth more than the rest of the file put together.
//!
//! **Compilation is from source, at runtime.** `newLibraryWithSource` uses the compiler
//! that ships with the OS, so this needs neither Xcode nor an accepted Xcode licence --
//! `xcrun metal` does not exist on a Command Line Tools install, which is what settled
//! the question. Measured at 139 ms for one kernel and 221 ms for two SHA-512 variants on
//! an M1 Pro (2026-08-13), against a sweep measured in days.
//!
//! **The filter is not copied.** `newBufferWithBytesNoCopy` wraps `bloom`'s own
//! allocation, so on unified memory one set of pages serves the GPU and every CPU worker
//! at once. On a 16 GB machine holding a 7.6 GB filter that is the difference between
//! working and swapping. It is also why `bloom::Words` exists: the call requires the
//! pointer *and* the length to be page-aligned, and `Vec<u64>` guarantees neither.
//!
//! Measured on an M1 Pro: uniformly random 8-byte probes over a 7.6 GB no-copy buffer run
//! at 266 M/s, against 383 M/s over 0.25 GB. A 30x working set costs 1.4x, so the TLB
//! behaviour that could have sunk this design does not.

use anyhow::{Result, anyhow, bail};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;

use super::{Arg, Backend, BufferId};

type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

pub struct Metal {
    device: Device,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Option<Retained<ProtocolObject<dyn MTLLibrary>>>,
    /// Pipelines are built on first use and kept: compiling one is not free and the
    /// launch loop asks for the same handful thousands of times.
    pipelines: HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    buffers: Vec<Buffer>,
    /// Wall time and call count per kernel, kept only when MILKSAD_GPU_PROFILE is set.
    ///
    /// Every dispatch already waits for its own completion, so this is just a clock around
    /// the wait -- accurate, and free when the variable is unset. It exists because
    /// reasoning about which kernel is slow was repeatedly wrong: the arithmetic
    /// benchmarks said the curve code ran near this device's ceiling while the pipeline
    /// ran 18x below it, and only a per-kernel measurement settled where the time went.
    profile: Option<HashMap<String, (std::time::Duration, u64)>>,
}

impl Metal {
    pub fn open() -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice().ok_or_else(|| {
            anyhow::Error::new(super::NoDevice("no Metal device on this machine".into()))
        })?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| anyhow!("could not create a Metal command queue"))?;
        Ok(Self {
            device,
            queue,
            library: None,
            pipelines: HashMap::new(),
            buffers: Vec::new(),
            profile: std::env::var_os("MILKSAD_GPU_PROFILE").map(|_| HashMap::new()),
        })
    }

    /// The largest single buffer this device will allocate.
    ///
    /// The filter is checked against this on open: it is 9.53 GB on an M1 Pro, so a 7.6 GB
    /// filter fits whole, but a smaller Mac reports proportionally less.
    pub fn max_buffer_length(&self) -> usize {
        self.device.maxBufferLength()
    }

    fn pipeline(
        &mut self,
        name: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        if let Some(pso) = self.pipelines.get(name) {
            return Ok(pso.clone());
        }
        let library = self
            .library
            .as_ref()
            .ok_or_else(|| anyhow!("dispatch of `{name}` before the kernels were compiled"))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(name))
            .ok_or_else(|| anyhow!("the compiled library has no kernel named `{name}`"))?;
        let pso = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| anyhow!("building a pipeline for `{name}`: {e}"))?;
        self.pipelines.insert(name.to_string(), pso.clone());
        Ok(pso)
    }

    fn get(&self, id: BufferId) -> Result<&Buffer> {
        self.buffers
            .get(id.0)
            .ok_or_else(|| anyhow!("buffer {} does not exist", id.0))
    }
}

impl Backend for Metal {
    fn name(&self) -> String {
        self.device.name().to_string()
    }

    fn compile(&mut self, source: &str) -> Result<()> {
        let library = self
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(source), None)
            // The compiler's own diagnostics are the only thing that makes a
            // runtime-compiled kernel debuggable, so they are the error, not a footnote.
            .map_err(|e| anyhow!("compiling the Metal kernels:\n{e}"))?;
        self.library = Some(library);
        self.pipelines.clear();
        Ok(())
    }

    fn buffer(&mut self, bytes: usize) -> Result<BufferId> {
        let buf = self
            .device
            .newBufferWithLength_options(bytes.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow!("allocating a {bytes}-byte device buffer"))?;
        self.buffers.push(buf);
        Ok(BufferId(self.buffers.len() - 1))
    }

    fn buffer_from(&mut self, data: &[u8]) -> Result<BufferId> {
        let id = self.buffer(data.len())?;
        let buf = self.get(id)?;
        // SAFETY: `contents` points at `data.len()` bytes we just allocated, and the
        // regions cannot overlap -- one is a fresh device allocation.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                buf.contents().as_ptr() as *mut u8,
                data.len(),
            );
        }
        Ok(id)
    }

    unsafe fn buffer_no_copy(&mut self, ptr: *const u8, bytes: usize) -> Result<BufferId> {
        if bytes > self.max_buffer_length() {
            bail!(
                "the filter is {bytes} bytes but this device's largest buffer is {}. \
                 Rebuild the filter smaller with `keyscan bf-gen` at a higher \
                 false-positive rate, or scan on a device with more memory.",
                self.max_buffer_length()
            );
        }
        // SAFETY: the caller guarantees page alignment of both pointer and length, and
        // that the allocation outlives every dispatch reading it. A nil deallocator is
        // what tells Metal it does not own these pages -- `bloom::Words` still frees them.
        let buf = unsafe {
            self.device.newBufferWithBytesNoCopy_length_options_deallocator(
                NonNull::new(ptr as *mut c_void)
                    .ok_or_else(|| anyhow!("a null pointer cannot be wrapped no-copy"))?,
                bytes,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        }
        .ok_or_else(|| {
            anyhow!(
                "Metal refused to wrap {bytes} bytes at {ptr:p} without copying. \
                 This is what an unaligned pointer or length looks like."
            )
        })?;
        self.buffers.push(buf);
        Ok(BufferId(self.buffers.len() - 1))
    }

    fn dispatch(&mut self, kernel: &str, threads: usize, args: &[Arg<'_>]) -> Result<()> {
        if threads == 0 {
            return Ok(());
        }
        let pso = self.pipeline(kernel)?;
        let cb = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow!("no command buffer available"))?;
        let enc = cb
            .computeCommandEncoder()
            .ok_or_else(|| anyhow!("no compute encoder available"))?;
        enc.setComputePipelineState(&pso);

        for (i, arg) in args.iter().enumerate() {
            match arg {
                Arg::Buffer(id) => {
                    let buf = self.get(*id)?;
                    // SAFETY: index and buffer are both valid; the encoder retains it.
                    unsafe { enc.setBuffer_offset_atIndex(Some(buf), 0, i) };
                }
                Arg::Scalar(bytes) => {
                    // SAFETY: `bytes` outlives the call -- `setBytes` copies immediately
                    // into the command buffer rather than retaining the pointer.
                    unsafe {
                        enc.setBytes_length_atIndex(
                            NonNull::new(bytes.as_ptr() as *mut c_void)
                                .ok_or_else(|| anyhow!("an empty scalar argument"))?,
                            bytes.len(),
                            i,
                        )
                    };
                }
            }
        }

        // A partial final threadgroup is fine -- every kernel bounds-checks `gid` against
        // the count it was given, because the alternative is a wrong answer at the tail of
        // any launch whose size is not a multiple of the threadgroup width.
        let width = pso.maxTotalThreadsPerThreadgroup().min(256);
        let groups = threads.div_ceil(width);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: groups, height: 1, depth: 1 },
            MTLSize { width, height: 1, depth: 1 },
        );
        enc.endEncoding();
        let began = self.profile.is_some().then(std::time::Instant::now);
        cb.commit();
        // Serialised for now: the launch loop syncs after every stage anyway, because each
        // kernel consumes the previous one's output. Overlapping launches is a Stage 7
        // question, and doing it here before there is anything to overlap would be guesswork.
        cb.waitUntilCompleted();

        // Checked, not assumed. A command buffer that fails -- the device is wedged, the
        // GPU watchdog fired, memory ran out under concurrent load -- leaves its output
        // buffer holding whatever was there before, and the only symptom downstream is
        // wrong hashes. Silence here would look exactly like an arithmetic bug.
        if let Some(error) = cb.error() {
            bail!("dispatching `{kernel}` over {threads} threads: {error}");
        }
        if let (Some(profile), Some(began)) = (self.profile.as_mut(), began) {
            let entry = profile.entry(kernel.to_string()).or_default();
            entry.0 += began.elapsed();
            entry.1 += 1;
        }
        Ok(())
    }

    fn report(&mut self) -> Option<String> {
        let profile = self.profile.as_ref()?;
        let mut rows: Vec<_> = profile.iter().collect();
        rows.sort_by_key(|(_, (d, _))| std::cmp::Reverse(*d));
        let total: f64 = rows.iter().map(|(_, (d, _))| d.as_secs_f64()).sum();
        let mut out = format!("gpu kernel profile ({total:.2}s total)\n");
        for (name, (d, calls)) in rows {
            out.push_str(&format!(
                "  {name:<16} {:>7.2}s  {:>5.1}%  {calls} calls\n",
                d.as_secs_f64(),
                100.0 * d.as_secs_f64() / total.max(1e-9)
            ));
        }
        Some(out)
    }

    fn sync(&mut self) -> Result<()> {
        // `dispatch` already waits; this exists so the trait does not promise something
        // the launch loop cannot rely on when a backend batches differently.
        Ok(())
    }

    fn write(&mut self, buffer: BufferId, data: &[u8]) -> Result<()> {
        let buf = self.get(buffer)?;
        if buf.length() < data.len() {
            bail!("writing {} bytes into a {}-byte buffer", data.len(), buf.length());
        }
        // SAFETY: shared storage is host-visible, the length check bounds the copy, and
        // every prior dispatch has completed -- see `dispatch`.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                buf.contents().as_ptr() as *mut u8,
                data.len(),
            );
        }
        Ok(())
    }

    fn has_unified_memory(&self) -> bool {
        self.device.hasUnifiedMemory()
    }

    /// What Apple recommends keeping the working set under, which on unified memory is a
    /// share of system RAM rather than a separate pool -- so the filter's own pages count
    /// against it exactly as the launch buffers do.
    fn device_memory(&self) -> Option<u64> {
        match self.device.recommendedMaxWorkingSetSize() {
            0 => None,
            n => Some(n),
        }
    }

    fn allocated_bytes(&self) -> usize {
        self.buffers.iter().map(|b| b.length()).sum()
    }

    fn read(&mut self, buffer: BufferId, out: &mut [u8]) -> Result<()> {
        let buf = self.get(buffer)?;
        if buf.length() < out.len() {
            bail!(
                "reading {} bytes from a {}-byte buffer",
                out.len(),
                buf.length()
            );
        }
        // SAFETY: shared storage means `contents` is host-visible, and the length check
        // above bounds the copy. Every dispatch has completed -- see `dispatch`.
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf.contents().as_ptr() as *const u8,
                out.as_mut_ptr(),
                out.len(),
            );
        }
        Ok(())
    }
}

/// Metal objects are not `Send` by default under objc2's rules, but a `Metal` is only
/// ever moved into the GPU worker thread and used from there.
///
/// SAFETY: nothing here is shared across threads. `Gpu` owns exactly one `Metal`, hands it
/// to one worker, and every method takes `&mut self`, so there is no concurrent access to
/// synchronise. The device and queue are documented as thread-safe regardless; the
/// pipeline cache and buffer table are plain Rust behind `&mut`.
unsafe impl Send for Metal {}
