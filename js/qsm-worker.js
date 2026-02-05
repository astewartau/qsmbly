/**
 * QSM Processing Web Worker
 *
 * Runs Pyodide and the QSM pipeline in a separate thread
 * to keep the main UI responsive.
 */

let pyodide = null;
let wasmModule = null;

// Post progress updates to main thread
function postProgress(value, text) {
  self.postMessage({ type: 'progress', value, text });
}

// Post log messages to main thread
function postLog(message) {
  self.postMessage({ type: 'log', message });
}

// Send intermediate stage data for live display
async function sendStageData(stage, dataName, description) {
  try {
    await pyodide.runPython(`
import nibabel as nib
import tempfile
import os

# Get the data
_stage_data = ${dataName}
print(f"Sending {${JSON.stringify(description)}}: shape {_stage_data.shape}")

# Create NIfTI file
_nii_img = nib.Nifti1Image(_stage_data, affine_matrix, header_info)

# Save to bytes
_temp_path = '/tmp/stage_output.nii'
_nii_img.to_filename(_temp_path)

# Read the file as bytes
with open(_temp_path, 'rb') as f:
    _stage_bytes = f.read()

# Clean up
os.remove(_temp_path)
`);
    const stageBytes = pyodide.globals.get('_stage_bytes').toJs();
    self.postMessage({ type: 'stageData', stage, data: stageBytes, description });
  } catch (error) {
    postLog(`Warning: Could not send ${stage} data: ${error.message}`);
  }
}

// Post error to main thread
function postError(message) {
  self.postMessage({ type: 'error', message });
}

// Post completion to main thread
function postComplete(results) {
  self.postMessage({ type: 'complete', results });
}

async function initializePyodide() {
  postLog("Initializing Pyodide...");

  importScripts('https://cdn.jsdelivr.net/pyodide/v0.27.1/full/pyodide.js');
  pyodide = await loadPyodide();

  postLog("Installing Python packages...");
  await pyodide.loadPackage(["numpy", "scipy", "micropip"]);

  postLog("Installing nibabel...");
  await pyodide.runPythonAsync(`
    import micropip
    await micropip.install("nibabel")
  `);

  // Try to load WASM acceleration module
  try {
    postLog("Loading WASM acceleration...");
    // Construct URLs relative to worker location
    const baseUrl = self.location.href.replace(/\/js\/.*$/, '');
    const jsUrl = `${baseUrl}/wasm/qsm_wasm.js`;
    const wasmBinaryUrl = `${baseUrl}/wasm/qsm_wasm_bg.wasm`;

    const module = await import(jsUrl);
    // Pass explicit WASM binary URL since import.meta.url doesn't work in workers
    await module.default(wasmBinaryUrl);
    wasmModule = module;
    if (wasmModule.wasm_health_check()) {
      postLog(`WASM acceleration loaded (v${wasmModule.get_version()})`);
    }
  } catch (e) {
    postLog("WASM not available, using Python fallback");
    console.warn("WASM load failed:", e);
    wasmModule = null;
  }

  postLog("Pyodide ready");
  return pyodide;
}

/**
 * WASM-accelerated region growing phase unwrapping
 * Called from Python via JS bridge
 *
 * Takes flat arrays and dimensions directly to avoid Pyodide proxy issues
 */
function wasmGrowRegionUnwrap(phaseFlatProxy, weightsFlatProxy, maskFlatProxy, nx, ny, nz, seedI, seedJ, seedK) {
  if (!wasmModule) {
    return null; // Signal to use Python fallback
  }

  try {
    // Convert Pyodide proxies to TypedArrays
    // The arrays are already flattened by Python in C order
    const phaseFlat = new Float64Array(phaseFlatProxy.toJs());
    const weightsFlat = new Uint8Array(weightsFlatProxy.toJs());
    const maskFlat = new Uint8Array(maskFlatProxy.toJs());

    console.log(`WASM unwrap: ${nx}x${ny}x${nz}, seed=(${seedI},${seedJ},${seedK})`);
    console.log(`  phase length: ${phaseFlat.length}, weights length: ${weightsFlat.length}, mask length: ${maskFlat.length}`);

    // Run WASM unwrapping (modifies phaseFlat in-place)
    const processed = wasmModule.grow_region_unwrap_wasm(
      phaseFlat, weightsFlat, maskFlat,
      nx, ny, nz, seedI, seedJ, seedK
    );

    console.log(`WASM processed ${processed} voxels`);

    // Return the modified phase as a regular array
    // Python will reshape it back to 3D
    return Array.from(phaseFlat);

  } catch (e) {
    console.error("WASM unwrapping failed:", e);
    return null; // Signal to use Python fallback
  }
}

/**
 * WASM-accelerated TKD dipole inversion
 */
function wasmTKD(fieldProxy, maskProxy, nx, ny, nz, vsx, vsy, vsz, threshold) {
  if (!wasmModule) return null;
  try {
    const field = new Float64Array(fieldProxy.toJs());
    const mask = new Uint8Array(maskProxy.toJs());
    console.log(`WASM TKD: ${nx}x${ny}x${nz}, threshold=${threshold}`);
    const result = wasmModule.tkd_wasm(field, mask, nx, ny, nz, vsx, vsy, vsz, 0, 0, 1, threshold);
    return Array.from(result);
  } catch (e) {
    console.error("WASM TKD failed:", e);
    return null;
  }
}

/**
 * WASM-accelerated Tikhonov dipole inversion
 */
function wasmTikhonov(fieldProxy, maskProxy, nx, ny, nz, vsx, vsy, vsz, lambda, regType) {
  if (!wasmModule) return null;
  try {
    const field = new Float64Array(fieldProxy.toJs());
    const mask = new Uint8Array(maskProxy.toJs());
    console.log(`WASM Tikhonov: ${nx}x${ny}x${nz}, lambda=${lambda}`);
    const result = wasmModule.tikhonov_wasm(field, mask, nx, ny, nz, vsx, vsy, vsz, 0, 0, 1, lambda, regType);
    return Array.from(result);
  } catch (e) {
    console.error("WASM Tikhonov failed:", e);
    return null;
  }
}

/**
 * WASM-accelerated TV-ADMM dipole inversion
 */
function wasmTVADMM(fieldProxy, maskProxy, nx, ny, nz, vsx, vsy, vsz, lambda, rho, tol, maxIter) {
  if (!wasmModule) return null;
  try {
    const field = new Float64Array(fieldProxy.toJs());
    const mask = new Uint8Array(maskProxy.toJs());
    console.log(`WASM TV-ADMM: ${nx}x${ny}x${nz}, lambda=${lambda}, maxIter=${maxIter}`);
    const result = wasmModule.tv_admm_wasm(field, mask, nx, ny, nz, vsx, vsy, vsz, 0, 0, 1, lambda, rho, tol, maxIter);
    return Array.from(result);
  } catch (e) {
    console.error("WASM TV-ADMM failed:", e);
    return null;
  }
}

/**
 * WASM-accelerated RTS dipole inversion
 */
function wasmRTS(fieldProxy, maskProxy, nx, ny, nz, vsx, vsy, vsz, delta, mu, rho, tol, maxIter, lsmrIter) {
  if (!wasmModule) return null;
  try {
    const field = new Float64Array(fieldProxy.toJs());
    const mask = new Uint8Array(maskProxy.toJs());
    console.log(`WASM RTS: ${nx}x${ny}x${nz}, delta=${delta}, mu=${mu}`);
    const result = wasmModule.rts_wasm(field, mask, nx, ny, nz, vsx, vsy, vsz, 0, 0, 1, delta, mu, rho, tol, maxIter, lsmrIter);
    return Array.from(result);
  } catch (e) {
    console.error("WASM RTS failed:", e);
    return null;
  }
}

/**
 * WASM-accelerated V-SHARP background removal
 */
function wasmVSHARP(fieldProxy, maskProxy, nx, ny, nz, vsx, vsy, vsz, radiiProxy, threshold) {
  if (!wasmModule) return null;
  try {
    const field = new Float64Array(fieldProxy.toJs());
    const mask = new Uint8Array(maskProxy.toJs());
    const radii = new Float64Array(radiiProxy.toJs());
    console.log(`WASM V-SHARP: ${nx}x${ny}x${nz}, ${radii.length} radii`);
    const result = wasmModule.vsharp_wasm(field, mask, nx, ny, nz, vsx, vsy, vsz, radii, threshold);
    // Result contains local_field followed by eroded_mask
    return Array.from(result);
  } catch (e) {
    console.error("WASM V-SHARP failed:", e);
    return null;
  }
}

/**
 * WASM-accelerated SHARP background removal
 */
function wasmSHARP(fieldProxy, maskProxy, nx, ny, nz, vsx, vsy, vsz, radius, threshold) {
  if (!wasmModule) return null;
  try {
    const field = new Float64Array(fieldProxy.toJs());
    const mask = new Uint8Array(maskProxy.toJs());
    console.log(`WASM SHARP: ${nx}x${ny}x${nz}, radius=${radius}`);
    const result = wasmModule.sharp_wasm(field, mask, nx, ny, nz, vsx, vsy, vsz, radius, threshold);
    return Array.from(result);
  } catch (e) {
    console.error("WASM SHARP failed:", e);
    return null;
  }
}

/**
 * WASM-accelerated PDF background removal
 */
function wasmPDF(fieldProxy, maskProxy, nx, ny, nz, vsx, vsy, vsz, tol, maxIter) {
  if (!wasmModule) return null;
  try {
    const field = new Float64Array(fieldProxy.toJs());
    const mask = new Uint8Array(maskProxy.toJs());
    console.log(`WASM PDF: ${nx}x${ny}x${nz}`);
    const result = wasmModule.pdf_wasm(field, mask, nx, ny, nz, vsx, vsy, vsz, 0, 0, 1, tol, maxIter);
    return Array.from(result);
  } catch (e) {
    console.error("WASM PDF failed:", e);
    return null;
  }
}

/**
 * WASM-accelerated iSMV background removal
 */
function wasmISMV(fieldProxy, maskProxy, nx, ny, nz, vsx, vsy, vsz, radius, tol, maxIter) {
  if (!wasmModule) return null;
  try {
    const field = new Float64Array(fieldProxy.toJs());
    const mask = new Uint8Array(maskProxy.toJs());
    console.log(`WASM iSMV: ${nx}x${ny}x${nz}, radius=${radius}`);
    const result = wasmModule.ismv_wasm(field, mask, nx, ny, nz, vsx, vsy, vsz, radius, tol, maxIter);
    return Array.from(result);
  } catch (e) {
    console.error("WASM iSMV failed:", e);
    return null;
  }
}

/**
 * WASM-accelerated ROMEO weight calculation
 */
function wasmRomeoWeights(phaseProxy, magProxy, phase2Proxy, te1, te2, maskProxy, nx, ny, nz) {
  if (!wasmModule) return null;
  try {
    const phase = new Float64Array(phaseProxy.toJs());
    const mag = magProxy ? new Float64Array(magProxy.toJs()) : new Float64Array(0);
    const phase2 = phase2Proxy ? new Float64Array(phase2Proxy.toJs()) : new Float64Array(0);
    const mask = new Uint8Array(maskProxy.toJs());
    console.log(`WASM ROMEO weights: ${nx}x${ny}x${nz}`);
    const result = wasmModule.calculate_weights_romeo_wasm(phase, mag, phase2, te1, te2, mask, nx, ny, nz);
    return Array.from(result);
  } catch (e) {
    console.error("WASM ROMEO weights failed:", e);
    return null;
  }
}

/**
 * WASM-accelerated MEDI L1 dipole inversion
 */
function wasmMEDI(fieldProxy, maskProxy, magProxy, nx, ny, nz, vsx, vsy, vsz, lambda, maxIter, cgMaxIter, cgTol) {
  if (!wasmModule) return null;
  try {
    const field = new Float64Array(fieldProxy.toJs());
    const mask = new Uint8Array(maskProxy.toJs());
    const mag = new Float64Array(magProxy.toJs());
    console.log(`WASM MEDI: ${nx}x${ny}x${nz}, lambda=${lambda}, maxIter=${maxIter}`);
    const result = wasmModule.medi_l1_wasm(field, mask, mag, nx, ny, nz, vsx, vsy, vsz, 0, 0, 1, lambda, maxIter, cgMaxIter, cgTol);
    return Array.from(result);
  } catch (e) {
    console.error("WASM MEDI failed:", e);
    return null;
  }
}

async function loadPythonAlgorithms(code) {
  postLog("Loading algorithms...");
  await pyodide.runPython(code);

  // Set up WASM bridge for Python
  if (wasmModule) {
    pyodide.globals.set('js_wasm_grow_region_unwrap', wasmGrowRegionUnwrap);
    pyodide.globals.set('js_wasm_tkd', wasmTKD);
    pyodide.globals.set('js_wasm_tikhonov', wasmTikhonov);
    pyodide.globals.set('js_wasm_tv_admm', wasmTVADMM);
    pyodide.globals.set('js_wasm_rts', wasmRTS);
    pyodide.globals.set('js_wasm_vsharp', wasmVSHARP);
    pyodide.globals.set('js_wasm_sharp', wasmSHARP);
    pyodide.globals.set('js_wasm_pdf', wasmPDF);
    pyodide.globals.set('js_wasm_ismv', wasmISMV);
    pyodide.globals.set('js_wasm_romeo_weights', wasmRomeoWeights);
    pyodide.globals.set('js_wasm_medi', wasmMEDI);
    postLog("WASM bridge enabled for all algorithms");
  } else {
    pyodide.globals.set('js_wasm_grow_region_unwrap', null);
    pyodide.globals.set('js_wasm_tkd', null);
    pyodide.globals.set('js_wasm_tikhonov', null);
    pyodide.globals.set('js_wasm_tv_admm', null);
    pyodide.globals.set('js_wasm_rts', null);
    pyodide.globals.set('js_wasm_vsharp', null);
    pyodide.globals.set('js_wasm_sharp', null);
    pyodide.globals.set('js_wasm_pdf', null);
    pyodide.globals.set('js_wasm_ismv', null);
    pyodide.globals.set('js_wasm_romeo_weights', null);
    pyodide.globals.set('js_wasm_medi', null);
  }
}

async function runPipeline(data) {
  const { magnitudeBuffers, phaseBuffers, echoTimes, magField, unwrapMode, maskThreshold, customMaskBuffer, pipelineSettings, skipStages } = data;
  const thresholdFraction = (maskThreshold || 15) / 100;  // Default to 15% if not provided
  const hasCustomMask = customMaskBuffer !== null && customMaskBuffer !== undefined;

  // Intelligent caching - determine which stages to skip
  const canSkipUnwrap = skipStages?.skipUnwrap || false;
  const canSkipBgRemoval = skipStages?.skipBgRemoval || false;

  // Extract pipeline settings with defaults
  const unwrapMethod = pipelineSettings?.unwrapMethod || 'romeo';
  const romeoSettings = pipelineSettings?.romeo || { weighting: 'phase_snr' };
  const backgroundMethod = pipelineSettings?.backgroundRemoval || 'vsharp';
  // Use nullish coalescing for individual values (allows dynamic defaults from app)
  const vsharpSettings = {
    maxRadius: pipelineSettings?.vsharp?.maxRadius ?? 18,
    minRadius: pipelineSettings?.vsharp?.minRadius ?? 2,
    threshold: pipelineSettings?.vsharp?.threshold ?? 0.05
  };
  const smvSettings = {
    radius: pipelineSettings?.smv?.radius ?? 18
  };
  const ismvSettings = {
    radius: pipelineSettings?.ismv?.radius ?? 2,
    tol: pipelineSettings?.ismv?.tol ?? 0.001,
    maxit: pipelineSettings?.ismv?.maxit ?? 500
  };
  const pdfSettings = {
    tol: pipelineSettings?.pdf?.tol ?? 0.00001,
    maxit: pipelineSettings?.pdf?.maxit ?? 100
  };
  const dipoleMethod = pipelineSettings?.dipoleInversion || 'rts';
  const rtsSettings = pipelineSettings?.rts || { delta: 0.15, mu: 100000, rho: 10, maxIter: 20 };
  const mediSettings = pipelineSettings?.medi || { lambda: 7.5e-5, maxIter: 30, cgMaxIter: 10, cgTol: 0.01, edgePercent: 0.3, merit: false };
  const tkdSettings = pipelineSettings?.tkd || { threshold: 0.15 };
  const tikhonovSettings = pipelineSettings?.tikhonov || { lambda: 0.01, reg: 'identity' };
  const tvSettings = pipelineSettings?.tv || { lambda: 0.001, maxIter: 250, tol: 0.001 };

  try {
    postProgress(0.1, 'Loading data...');
    postLog("Loading multi-echo data...");

    // Transfer data to Python
    for (let i = 0; i < echoTimes.length; i++) {
      pyodide.globals.set(`mag_data_${i}`, new Uint8Array(magnitudeBuffers[i]));
      pyodide.globals.set(`phase_data_${i}`, new Uint8Array(phaseBuffers[i]));
    }

    // Transfer custom mask if provided
    if (hasCustomMask) {
      pyodide.globals.set('custom_mask_data', new Uint8Array(customMaskBuffer));
      postLog("Using custom edited mask");
    }
    pyodide.globals.set('has_custom_mask', hasCustomMask);

    pyodide.globals.set("echo_times", new Float64Array(echoTimes));
    pyodide.globals.set("num_echoes", echoTimes.length);

    // Load data in Python
    await pyodide.runPython(`
import numpy as np
import nibabel as nib
from io import BytesIO

# Convert JsProxy objects to Python objects
echo_times_py = echo_times.to_py()

print(f"Loading {num_echoes} echoes...")
print(f"Echo times: {list(echo_times_py)} ms")

# Load magnitude and phase data
magnitude_4d = []
phase_4d = []

for i in range(num_echoes):
    # Load magnitude
    mag_bytes = globals()[f'mag_data_{i}'].to_py()
    mag_fh = nib.FileHolder(BytesIO(mag_bytes))
    mag_img = nib.Nifti1Image.from_file_map({'image': mag_fh, 'header': mag_fh})
    mag_data = mag_img.get_fdata()
    magnitude_4d.append(mag_data)

    # Load phase
    phase_bytes = globals()[f'phase_data_{i}'].to_py()
    phase_fh = nib.FileHolder(BytesIO(phase_bytes))
    phase_img = nib.Nifti1Image.from_file_map({'image': phase_fh, 'header': phase_fh})
    phase_data = phase_img.get_fdata()

    # Scale phase to [-π, +π] range
    # This must be done BEFORE any phase processing
    phase_min = np.min(phase_data)
    phase_max = np.max(phase_data)
    phase_range = phase_max - phase_min

    print(f"  Echo {i+1} phase input range: [{phase_min:.2f}, {phase_max:.2f}]")

    # Check if phase needs scaling (not already in approximately -π to +π)
    if phase_range > 2 * np.pi * 1.1 or phase_max > np.pi * 1.5 or phase_min < -np.pi * 1.5:
        # Linear scale from [min, max] to [-π, +π]
        phase_data = (phase_data - phase_min) / phase_range * 2 * np.pi - np.pi
        print(f"  Echo {i+1} phase scaled to [-π, +π]")

    # Always wrap to ensure exactly [-π, +π] using complex exponential
    phase_data = np.angle(np.exp(1j * phase_data))

    phase_4d.append(phase_data)

# Stack into 4D arrays
magnitude_4d = np.stack(magnitude_4d, axis=3)
phase_4d = np.stack(phase_4d, axis=3)

print(f"Data shape: {magnitude_4d.shape}")
print(f"Phase range: [{np.min(phase_4d):.3f}, {np.max(phase_4d):.3f}]")
print(f"Magnitude range: [{np.min(magnitude_4d):.1f}, {np.max(magnitude_4d):.1f}]")

# Store header info from first echo
header_info = mag_img.header
affine_matrix = mag_img.affine
`);

    // Check if we can skip unwrapping (cached data exists)
    let actuallySkipUnwrap = false;
    if (canSkipUnwrap) {
      const hasCachedUnwrap = await pyodide.runPython(`
'B0_fieldmap' in dir() and B0_fieldmap is not None and 'processing_mask' in dir() and processing_mask is not None
`);
      actuallySkipUnwrap = hasCachedUnwrap;
      if (actuallySkipUnwrap) {
        postLog("Using cached unwrapped phase data");
        postProgress(0.76, 'Skipped unwrapping (cached)');
      } else {
        postLog("Cannot skip unwrapping - cached data not available");
      }
    }

    if (!actuallySkipUnwrap) {
      postProgress(0.1, 'Unwrapping phase...');

      // Choose unwrapping method
      const useRomeo = unwrapMethod === 'romeo';
      const individual = unwrapMode === 'individual';
      pyodide.globals.set('use_romeo_unwrap', useRomeo);

      if (useRomeo) {
        postLog("Running ROMEO phase unwrapping...");

      // Create progress callback for Python to call
      // ROMEO takes ~75% of total time, maps to 1-76% of progress bar
      const unwrapProgressCallback = (stage, progress) => {
        const mappedProgress = 0.01 + (progress / 100) * 0.75;
        postProgress(mappedProgress, `Unwrapping: ${progress}%`);
      };
      pyodide.globals.set('js_progress_callback', unwrapProgressCallback);

      const romeoWeighting = romeoSettings.weighting;
      pyodide.globals.set('romeo_weighting', romeoWeighting);
      await pyodide.runPython(`
print("Starting ROMEO unwrapping...")
print(f"Mode: {'Individual' if ${individual ? 'True' : 'False'} else 'Temporal'}")
print(f"Weighting: {romeo_weighting}")

# Set up progress callback
set_progress_callback(js_progress_callback)

results = romeo_multi_echo_unwrap(
    phase_4d, magnitude_4d, echo_times_py,
    individual=${individual ? 'True' : 'False'},
    B0_calculation=True,
    weighting=romeo_weighting
)

report_progress("complete", 100)

unwrapped_phase = results['unwrapped']
B0_fieldmap = results['B0']
processing_mask = results['mask']

print(f"ROMEO unwrapping completed!")
print(f"B0 range: [{np.min(B0_fieldmap):.1f}, {np.max(B0_fieldmap):.1f}] Hz")

# Use first echo magnitude and B0 fieldmap for subsequent processing
magnitude_combined = magnitude_4d[:, :, :, 0]  # First echo magnitude
fieldmap = B0_fieldmap  # B0 fieldmap from ROMEO

print(f"Using first echo magnitude: {magnitude_combined.shape}")
print(f"Using B0 fieldmap: {fieldmap.shape}")
`);
    } else {
      // Laplacian unwrapping
      postLog("Running Laplacian phase unwrapping...");

      await pyodide.runPython(`
print("Starting Laplacian unwrapping...")

def laplacian_unwrap(wrapped_phase, voxel_size=(1.0, 1.0, 1.0)):
    """
    Laplacian phase unwrapping using Fourier-domain Poisson solver.

    The method exploits that the Laplacian of the true phase equals
    the Laplacian of the wrapped phase (computed via complex derivatives).

    Based on: Schofield & Zhu, Optics Letters 2003
    """
    shape = wrapped_phase.shape

    # Compute cosine and sine of wrapped phase
    cos_phi = np.cos(wrapped_phase)
    sin_phi = np.sin(wrapped_phase)

    # Compute Laplacian using finite differences (7-point stencil)
    def laplacian_3d(f, voxel_size):
        dx, dy, dz = voxel_size
        lap = np.zeros_like(f)
        # x direction
        lap += (np.roll(f, -1, axis=0) - 2*f + np.roll(f, 1, axis=0)) / (dx*dx)
        # y direction
        lap += (np.roll(f, -1, axis=1) - 2*f + np.roll(f, 1, axis=1)) / (dy*dy)
        # z direction
        lap += (np.roll(f, -1, axis=2) - 2*f + np.roll(f, 1, axis=2)) / (dz*dz)
        return lap

    # Compute Laplacian of cosine and sine
    lap_cos = laplacian_3d(cos_phi, voxel_size)
    lap_sin = laplacian_3d(sin_phi, voxel_size)

    # Compute Laplacian of unwrapped phase
    # Derivation: sin(φ)∇²cos(φ) - cos(φ)∇²sin(φ) = -∇²φ
    # So: ∇²φ = cos(φ)∇²sin(φ) - sin(φ)∇²cos(φ)
    lap_phi = cos_phi * lap_sin - sin_phi * lap_cos

    # Create Laplacian kernel in Fourier space
    nx, ny, nz = shape
    dx, dy, dz = voxel_size

    kx = np.fft.fftfreq(nx, dx)
    ky = np.fft.fftfreq(ny, dy)
    kz = np.fft.fftfreq(nz, dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

    # Discrete Laplacian in frequency domain (matches finite difference stencil)
    # Note: eigenvalues are negative (or zero at DC)
    L = (2 * (np.cos(2 * np.pi * KX * dx) - 1) / (dx * dx) +
         2 * (np.cos(2 * np.pi * KY * dy) - 1) / (dy * dy) +
         2 * (np.cos(2 * np.pi * KZ * dz) - 1) / (dz * dz))

    # Regularize to avoid division by very small values
    # Add small positive value to make all entries non-zero
    eps = 1e-6
    L_reg = np.where(np.abs(L) < eps, -eps, L)

    # Solve Poisson equation: ∇²φ = lap_phi  =>  φ = F⁻¹[F(lap_phi) / L]
    F_lap_phi = np.fft.fftn(lap_phi)
    F_phi = F_lap_phi / L_reg
    F_phi[0, 0, 0] = 0  # Set DC to zero (removes global offset)

    unwrapped = np.real(np.fft.ifftn(F_phi))

    print(f"  Debug - wrapped range: [{np.min(wrapped_phase):.4f}, {np.max(wrapped_phase):.4f}]")
    print(f"  Debug - lap_phi range: [{np.min(lap_phi):.4f}, {np.max(lap_phi):.4f}]")
    print(f"  Debug - L range: [{np.min(L):.4f}, {np.max(L):.4f}]")
    print(f"  Debug - unwrapped range: [{np.min(unwrapped):.4f}, {np.max(unwrapped):.4f}]")

    return unwrapped

def compute_b0_from_unwrapped(unwrapped_4d, echo_times_ms, mask):
    """Compute B0 field map from unwrapped phase.

    For single echo: B0 = phase / (2π * TE)
    For multi-echo: weighted linear fit of phase vs TE
    """
    n_echoes = unwrapped_4d.shape[3]
    echo_times_s = np.array(echo_times_ms) / 1000.0  # Convert to seconds

    if n_echoes == 1:
        # Single echo: direct conversion
        # phase (rad) = 2π * B0 (Hz) * TE (s)
        # B0 (Hz) = phase / (2π * TE)
        TE = echo_times_s[0]
        B0_Hz = unwrapped_4d[:,:,:,0] / (2 * np.pi * TE)
        print(f"  Single echo B0 computation: TE={TE*1000:.1f}ms")
    else:
        # Multi-echo: weighted linear fit
        # phase = B0 * TE + phase0
        weights = np.arange(1, n_echoes + 1, dtype=np.float64)
        weights = weights / np.sum(weights)

        te_mean = np.sum(weights * echo_times_s)

        numerator = np.zeros(unwrapped_4d.shape[:3])
        denominator = 0.0

        for i in range(n_echoes):
            te_diff = echo_times_s[i] - te_mean
            numerator += weights[i] * te_diff * unwrapped_4d[:,:,:,i]
            denominator += weights[i] * te_diff * te_diff

        B0_rad_per_s = numerator / (denominator + 1e-10)
        B0_Hz = B0_rad_per_s / (2 * np.pi)
        print(f"  Multi-echo B0 computation: {n_echoes} echoes")

    B0_Hz = B0_Hz * mask
    return B0_Hz

# Get voxel size
try:
    voxel_size = tuple(header_info.get_zooms()[:3])
except:
    voxel_size = (1.0, 1.0, 1.0)

print(f"Voxel size: {voxel_size}")

# Unwrap phase data
n_echoes = phase_4d.shape[3]
unwrapped_4d = np.zeros_like(phase_4d)
use_temporal = not ${individual ? 'True' : 'False'}

if use_temporal and n_echoes > 1:
    # Temporal mode: spatially unwrap first echo, temporally unwrap rest
    print(f"Using temporal unwrapping mode ({n_echoes} echoes)")
    template_idx = 0
    print(f"  Spatially unwrapping template echo {template_idx + 1} with Laplacian...")
    unwrapped_template = laplacian_unwrap(phase_4d[:,:,:,template_idx], voxel_size)

    # Create mask for temporal unwrapping - use custom mask if provided
    magnitude_combined = magnitude_4d[:, :, :, 0]
    if has_custom_mask:
        print("  Using custom mask for temporal unwrapping...")
        mask_bytes = custom_mask_data.to_py()
        mask_fh = nib.FileHolder(BytesIO(mask_bytes))
        mask_img = nib.Nifti1Image.from_file_map({'image': mask_fh, 'header': mask_fh})
        mask_data = mask_img.get_fdata()
        temp_mask = mask_data > 0.5
    else:
        mag_threshold = np.max(magnitude_combined) * ${thresholdFraction}
        temp_mask = magnitude_combined > mag_threshold

    # Use temporal_unwrap from romeo_python.py
    unwrapped_4d = temporal_unwrap(unwrapped_template, phase_4d, echo_times_py, template_idx, temp_mask)
else:
    # Individual mode: unwrap each echo separately
    print(f"Using individual unwrapping mode ({n_echoes} echoes)")
    for i in range(n_echoes):
        print(f"  Unwrapping echo {i+1}/{n_echoes}...")
        unwrapped_4d[:,:,:,i] = laplacian_unwrap(phase_4d[:,:,:,i], voxel_size)

print("Laplacian unwrapping completed!")

# Create processing mask - use custom mask if provided, otherwise threshold
magnitude_combined = magnitude_4d[:, :, :, 0]  # First echo
if has_custom_mask:
    print("Loading custom mask for B0 computation...")
    mask_bytes = custom_mask_data.to_py()
    mask_fh = nib.FileHolder(BytesIO(mask_bytes))
    mask_img = nib.Nifti1Image.from_file_map({'image': mask_fh, 'header': mask_fh})
    mask_data = mask_img.get_fdata()
    processing_mask = mask_data > 0.5
    print(f"Custom mask loaded: {processing_mask.shape}")
else:
    mag_threshold = np.max(magnitude_combined) * ${thresholdFraction}
    processing_mask = magnitude_combined > mag_threshold
    print(f"Using threshold mask: {${thresholdFraction} * 100:.0f}% of max magnitude")

print(f"Mask coverage: {np.sum(processing_mask)}/{processing_mask.size} voxels ({100*np.sum(processing_mask)/processing_mask.size:.1f}%)")

# Compute B0 fieldmap from unwrapped phase
print("Computing B0 field map...")
B0_fieldmap = compute_b0_from_unwrapped(unwrapped_4d, echo_times_py, processing_mask)
fieldmap = B0_fieldmap

print(f"B0 range: [{np.min(B0_fieldmap[processing_mask]):.1f}, {np.max(B0_fieldmap[processing_mask]):.1f}] Hz")
print(f"Using first echo magnitude: {magnitude_combined.shape}")
`);
    }

    // Send B0 fieldmap for live display
    await sendStageData('B0', 'B0_fieldmap', 'B0 Field Map (Hz)');

    } // End of if (!actuallySkipUnwrap)

    // Check if we can skip background removal (cached data exists)
    let actuallySkipBgRemoval = false;
    if (canSkipBgRemoval) {
      const hasCachedBgRemoval = await pyodide.runPython(`
'local_fieldmap' in dir() and local_fieldmap is not None
`);
      actuallySkipBgRemoval = hasCachedBgRemoval;
      if (actuallySkipBgRemoval) {
        postLog("Using cached background-removed data");
        postProgress(0.8, 'Skipped background removal (cached)');
      } else {
        postLog("Cannot skip background removal - cached data not available");
      }
    }

    if (!actuallySkipBgRemoval) {
      postProgress(0.76, 'Removing background...');
      postLog(`Removing background field using ${backgroundMethod.toUpperCase()}...`);

    // Create progress callback for background removal
    const bgProgressCallback = (current, total) => {
      const progress = 0.76 + (current / total) * 0.08;
      postProgress(progress, `${backgroundMethod.toUpperCase()}: ${current}/${total}`);
    };
    pyodide.globals.set('js_bg_progress', bgProgressCallback);
    pyodide.globals.set('background_method', backgroundMethod);

    await pyodide.runPython(`
# Background removal - V-SHARP or SMV
import numpy as np

def create_smv_kernel_kspace(shape, voxel_size, radius):
    """Create spherical mean value kernel in k-space (VECTORIZED)"""
    nx, ny, nz = shape
    dx, dy, dz = voxel_size
    r2 = radius * radius

    i = np.arange(nx)
    j = np.arange(ny)
    k = np.arange(nz)

    x = np.where(i <= nx//2, i, i - nx) * dx
    y = np.where(j <= ny//2, j, j - ny) * dy
    z = np.where(k <= nz//2, k, k - nz) * dz

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    sphere = (X*X + Y*Y + Z*Z <= r2).astype(np.float64)

    sphere_sum = np.sum(sphere)
    if sphere_sum > 0:
        sphere /= sphere_sum

    S = np.real(np.fft.fftn(sphere))
    return S

def vsharp_background_removal(fieldmap, mask, voxel_size=(1.0, 1.0, 1.0),
                               radii=None, threshold=0.05):
    """V-SHARP background field removal via WASM"""
    if radii is None:
        min_vox = min(voxel_size)
        max_vox = max(voxel_size)
        radii = list(np.arange(18*min_vox, 2*max_vox - 0.001, -2*max_vox))
        if len(radii) == 0:
            radii = [6.0, 4.0, 2.0]

    radii = sorted(radii, reverse=True)
    print(f"V-SHARP (WASM) radii (mm): {[f'{r:.1f}' for r in radii]}")

    shape = fieldmap.shape
    nx, ny, nz = shape
    vsx, vsy, vsz = voxel_size
    mask_u8 = mask.astype(np.uint8)
    radii_arr = np.array(radii, dtype=np.float64)
    result = js_wasm_vsharp(
        fieldmap.flatten().astype(np.float64),
        mask_u8.flatten(),
        nx, ny, nz, vsx, vsy, vsz, radii_arr, threshold
    )
    result_arr = np.array(result)
    local_field = result_arr[:nx*ny*nz].reshape(shape)
    eroded_mask = result_arr[nx*ny*nz:].reshape(shape) > 0.5
    print("V-SHARP completed")
    return local_field, eroded_mask

def smv_background_removal(fieldmap, mask, voxel_size, radius=5.0):
    """SMV background field removal with single radius"""
    print(f"SMV background removal with radius={radius:.1f}mm")
    shape = fieldmap.shape

    try:
        js_bg_progress(1, 3)
    except:
        pass

    S = create_smv_kernel_kspace(shape, voxel_size, radius)

    try:
        js_bg_progress(2, 3)
    except:
        pass

    HP = 1.0 - S
    F = np.fft.fftn(fieldmap)
    local_field = np.real(np.fft.ifftn(HP * F))

    M = np.fft.fftn(mask.astype(np.float64))
    eroded = np.real(np.fft.ifftn(S * M))
    eroded_mask = eroded > 0.999

    local_field = local_field * eroded_mask

    try:
        js_bg_progress(3, 3)
    except:
        pass

    return local_field, eroded_mask

def ismv_background_removal(fieldmap, mask, voxel_size, radius=5.0, tol=1e-3, maxit=500):
    """iSMV background field removal via WASM"""
    print(f"iSMV (WASM): radius={radius:.1f}mm, tol={tol}, maxit={maxit}")
    shape = fieldmap.shape
    nx, ny, nz = shape
    vsx, vsy, vsz = voxel_size
    mask_u8 = mask.astype(np.uint8)
    result = js_wasm_ismv(
        fieldmap.flatten().astype(np.float64),
        mask_u8.flatten(),
        nx, ny, nz, vsx, vsy, vsz, radius, tol, maxit
    )
    result_arr = np.array(result)
    local_field = result_arr[:nx*ny*nz].reshape(shape)
    eroded_mask = result_arr[nx*ny*nz:].reshape(shape) > 0.5
    print("iSMV completed")
    return local_field, eroded_mask

def pdf_background_removal(fieldmap, mask, voxel_size, bdir=(0, 0, 1), tol=1e-5, maxit=None):
    """PDF background field removal via WASM"""
    shape = fieldmap.shape
    nx, ny, nz = shape
    n_voxels = nx * ny * nz
    if maxit is None:
        maxit = int(np.ceil(np.sqrt(n_voxels)))
    print(f"PDF (WASM): tol={tol}, maxit={maxit}")
    vsx, vsy, vsz = voxel_size
    mask_u8 = mask.astype(np.uint8)
    result = js_wasm_pdf(
        fieldmap.flatten().astype(np.float64),
        mask_u8.flatten(),
        nx, ny, nz, vsx, vsy, vsz, tol, maxit
    )
    local_field = np.array(result).reshape(shape)
    print("PDF completed")
    return local_field, mask

# Create processing mask - use custom mask if provided, otherwise threshold
if has_custom_mask:
    print("Loading custom edited mask...")
    mask_bytes = custom_mask_data.to_py()
    mask_fh = nib.FileHolder(BytesIO(mask_bytes))
    mask_img = nib.Nifti1Image.from_file_map({'image': mask_fh, 'header': mask_fh})
    mask_data = mask_img.get_fdata()
    processing_mask = mask_data > 0.5
    print(f"Custom mask loaded: {processing_mask.shape}")
else:
    mask_threshold_fraction = ${thresholdFraction}
    print(f"Using mask threshold: {mask_threshold_fraction * 100:.0f}% of max magnitude")
    processing_mask = magnitude_combined > mask_threshold_fraction * np.max(magnitude_combined)

print(f"Processing mask coverage: {np.sum(processing_mask)}/{processing_mask.size} voxels ({100*np.sum(processing_mask)/processing_mask.size:.1f}%)")

# Get voxel size from header
try:
    voxel_size = tuple(header_info.get_zooms()[:3])
    print(f"Voxel size: {voxel_size} mm")
except:
    voxel_size = (1.0, 1.0, 1.0)
    print(f"Using default voxel size: {voxel_size} mm")

# Run background removal based on selected method
bg_method = background_method
if bg_method == 'vsharp':
    print("Running V-SHARP background removal...")
    vsharp_max_radius = ${vsharpSettings.maxRadius}
    vsharp_min_radius = ${vsharpSettings.minRadius}
    vsharp_threshold = ${vsharpSettings.threshold}
    print(f"V-SHARP settings: max_radius={vsharp_max_radius}mm, min_radius={vsharp_min_radius}mm, threshold={vsharp_threshold}")
    radii = list(np.arange(vsharp_max_radius, vsharp_min_radius - 0.001, -2.0))
    local_fieldmap, eroded_mask = vsharp_background_removal(
        fieldmap, processing_mask, voxel_size=voxel_size,
        radii=radii, threshold=vsharp_threshold
    )
elif bg_method == 'ismv':
    print("Running iSMV background removal...")
    ismv_radius = ${ismvSettings.radius}
    ismv_tol = ${ismvSettings.tol}
    ismv_maxit = ${ismvSettings.maxit}
    print(f"iSMV settings: radius={ismv_radius}mm, tol={ismv_tol}, maxit={ismv_maxit}")
    local_fieldmap, eroded_mask = ismv_background_removal(
        fieldmap, processing_mask, voxel_size=voxel_size,
        radius=ismv_radius, tol=ismv_tol, maxit=ismv_maxit
    )
elif bg_method == 'pdf':
    print("Running PDF background removal...")
    pdf_tol = ${pdfSettings.tol}
    pdf_maxit = ${pdfSettings.maxit}
    print(f"PDF settings: tol={pdf_tol}, maxit={pdf_maxit}")
    local_fieldmap, eroded_mask = pdf_background_removal(
        fieldmap, processing_mask, voxel_size=voxel_size,
        tol=pdf_tol, maxit=pdf_maxit
    )
else:  # smv
    print("Running SMV background removal...")
    smv_radius = ${smvSettings.radius}
    print(f"SMV settings: radius={smv_radius}mm")
    local_fieldmap, eroded_mask = smv_background_removal(
        fieldmap, processing_mask, voxel_size=voxel_size,
        radius=smv_radius
    )

# Update processing mask to eroded version
processing_mask = eroded_mask

print(f"Eroded mask coverage: {np.sum(processing_mask)}/{processing_mask.size} voxels ({100*np.sum(processing_mask)/processing_mask.size:.1f}%)")
print(f"Local field range: [{np.min(local_fieldmap[processing_mask]):.1f}, {np.max(local_fieldmap[processing_mask]):.1f}] Hz")
print("Background removal completed!")
`);
    }

    // Send local fieldmap for live display
    await sendStageData('bgRemoved', 'local_fieldmap', 'Local Field Map (Hz)');

    // End of if (!actuallySkipBgRemoval)

    postProgress(0.8, 'Dipole inversion...');
    postLog(`Running ${dipoleMethod.toUpperCase()} dipole inversion...`);

    // Create progress callback for QSM
    const qsmProgressCallback = (iteration, maxiter) => {
      const progress = 0.84 + (iteration / maxiter) * 0.16;
      postProgress(progress, `${dipoleMethod.toUpperCase()}: iteration ${iteration}/${maxiter}`);
    };
    pyodide.globals.set('js_qsm_progress', qsmProgressCallback);
    pyodide.globals.set('dipole_method', dipoleMethod);

    await pyodide.runPython(`
# Dipole inversion - RTS or MEDI

def create_dipole_kernel(shape, voxel_size, bdir=(0, 0, 1)):
    """Create dipole kernel in k-space (full FFT version)"""
    nx, ny, nz = shape
    dx, dy, dz = voxel_size
    kx = np.fft.fftfreq(nx, dx)
    ky = np.fft.fftfreq(ny, dy)
    kz = np.fft.fftfreq(nz, dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    bdir = np.array(bdir, dtype=np.float64)
    bdir = bdir / np.linalg.norm(bdir)
    k_dot_b = KX * bdir[0] + KY * bdir[1] + KZ * bdir[2]
    k2 = KX**2 + KY**2 + KZ**2
    k2[0, 0, 0] = 1e-12
    D = 1/3 - (k_dot_b**2) / k2
    D[0, 0, 0] = 0
    return D

def create_dipole_kernel_rfft(shape, voxel_size, bdir=(0, 0, 1)):
    """Create dipole kernel for rfft (half-spectrum, ~2x faster)"""
    nx, ny, nz = shape
    dx, dy, dz = voxel_size
    # Full frequencies for x and y, half for z (rfft)
    kx = np.fft.fftfreq(nx, dx)
    ky = np.fft.fftfreq(ny, dy)
    kz = np.fft.rfftfreq(nz, dz)  # Only positive frequencies
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    bdir = np.array(bdir, dtype=np.float64)
    bdir = bdir / np.linalg.norm(bdir)
    k_dot_b = KX * bdir[0] + KY * bdir[1] + KZ * bdir[2]
    k2 = KX**2 + KY**2 + KZ**2
    k2[0, 0, 0] = 1e-12
    D = 1/3 - (k_dot_b**2) / k2
    D[0, 0, 0] = 0
    return D

def create_laplacian_kernel(shape, voxel_size):
    """Create negative Laplacian kernel in k-space"""
    nx, ny, nz = shape
    dx, dy, dz = voxel_size
    kx = np.fft.fftfreq(nx, dx)
    ky = np.fft.fftfreq(ny, dy)
    kz = np.fft.fftfreq(nz, dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    Lx = 2 * (1 - np.cos(2 * np.pi * KX * dx)) / (dx * dx)
    Ly = 2 * (1 - np.cos(2 * np.pi * KY * dy)) / (dy * dy)
    Lz = 2 * (1 - np.cos(2 * np.pi * KZ * dz)) / (dz * dz)
    return Lx + Ly + Lz

# --- Helper functions for TV-ADMM (from QSM.jl fd.jl) ---
def gradient_periodic(x, voxel_size):
    """Forward gradient with periodic boundaries"""
    dx, dy, dz = voxel_size
    gx = (np.roll(x, -1, axis=0) - x) / dx
    gy = (np.roll(x, -1, axis=1) - x) / dy
    gz = (np.roll(x, -1, axis=2) - x) / dz
    return gx, gy, gz

def divergence_periodic(gx, gy, gz, voxel_size):
    """Negative divergence with periodic boundaries (adjoint of gradient)"""
    dx, dy, dz = voxel_size
    div_x = (gx - np.roll(gx, 1, axis=0)) / dx
    div_y = (gy - np.roll(gy, 1, axis=1)) / dy
    div_z = (gz - np.roll(gz, 1, axis=2)) / dz
    return -(div_x + div_y + div_z)

def _shrink_update(u, d, threshold):
    """Combined shrink and dual update for TV-ADMM

    From QSM.jl tv.jl lines 328-334:
    v = u + grad_x           # intermediate
    z = shrink(v, λ/ρ)       # z-subproblem
    new_u = v - z            # dual update
    new_d = 2*z - v          # precompute z - new_u for next x-subproblem
    """
    v = u + d  # intermediate: v = old_u + grad_x
    z = np.sign(v) * np.maximum(np.abs(v) - threshold, 0)  # shrink
    new_u = v - z  # dual update
    new_d = 2*z - v  # precompute for next iteration (equals z - new_u)
    return new_u, new_d

# --- TKD (Truncated K-space Division) - WASM only ---
def tkd_qsm(local_field, mask, voxel_size, bdir=(0, 0, 1), thr=0.15):
    """TKD dipole inversion via WASM"""
    print(f"TKD (WASM): threshold={thr}")
    shape = local_field.shape
    nx, ny, nz = shape
    vsx, vsy, vsz = voxel_size
    mask_u8 = mask.astype(np.uint8)
    result = js_wasm_tkd(
        local_field.flatten().astype(np.float64),
        mask_u8.flatten(),
        nx, ny, nz, vsx, vsy, vsz, thr
    )
    chi = np.array(result).reshape(shape)
    print("TKD completed")
    return chi

# --- Tikhonov Regularization from QSM.jl direct.jl ---
def tikh_qsm(local_field, mask, voxel_size, bdir=(0, 0, 1),
             lambda_=0.01, reg='identity'):
    """Tikhonov dipole inversion via WASM"""
    print(f"Tikhonov (WASM): lambda={lambda_}, reg={reg}")
    shape = local_field.shape
    nx, ny, nz = shape
    vsx, vsy, vsz = voxel_size
    reg_type = {'identity': 0, 'gradient': 1, 'laplacian': 2}.get(reg, 0)
    mask_u8 = mask.astype(np.uint8)
    result = js_wasm_tikhonov(
        local_field.flatten().astype(np.float64),
        mask_u8.flatten(),
        nx, ny, nz, vsx, vsy, vsz, lambda_, reg_type
    )
    chi = np.array(result).reshape(shape)
    print("Tikhonov completed")
    return chi

# --- TV-ADMM from QSM.jl tv.jl ---
def tv_admm_qsm(local_field, mask, voxel_size, bdir=(0, 0, 1),
                lambda_=0.001, rho=None, tol=0.001, maxit=250):
    """TV-ADMM dipole inversion via WASM"""
    if rho is None:
        rho = 100 * lambda_
    print(f"TV-ADMM (WASM): lambda={lambda_}, rho={rho}, tol={tol}, maxit={maxit}")
    shape = local_field.shape
    nx, ny, nz = shape
    vsx, vsy, vsz = voxel_size
    mask_u8 = mask.astype(np.uint8)
    result = js_wasm_tv_admm(
        local_field.flatten().astype(np.float64),
        mask_u8.flatten(),
        nx, ny, nz, vsx, vsy, vsz, lambda_, rho, tol, maxit
    )
    chi = np.array(result).reshape(shape)
    print("TV-ADMM completed")
    return chi

def rts_qsm(local_field, mask, voxel_size, bdir=(0, 0, 1),
            delta=0.15, mu=1e5, rho=10.0, tol=1e-2, maxit=20):
    """RTS dipole inversion via WASM"""
    print(f"RTS (WASM): delta={delta}, mu={mu}, rho={rho}, maxit={maxit}")
    shape = local_field.shape
    nx, ny, nz = shape
    vsx, vsy, vsz = voxel_size
    mask_u8 = mask.astype(np.uint8)
    result = js_wasm_rts(
        local_field.flatten().astype(np.float64),
        mask_u8.flatten(),
        nx, ny, nz, vsx, vsy, vsz, delta, mu, rho, tol, maxit, 4
    )
    chi = np.array(result).reshape(shape)
    print("RTS completed")
    return chi

# Keep the Python helper functions for any remaining code that might use them
def _rts_python_fallback(local_field, mask, voxel_size, bdir, delta, mu, rho, tol, maxit):
    """Python RTS fallback - kept for reference but not used"""
    D = create_dipole_kernel(local_field.shape, voxel_size, bdir)
    L = create_laplacian_kernel(local_field.shape, voxel_size)
    well_conditioned = np.abs(D) > delta
    M = np.where(well_conditioned, mu, 0.0)
    f = local_field * mask
    F = np.fft.fftn(f)

    print("Step 1: Direct inversion...")
    D_sq = D * D
    D_sq_safe = np.where(D_sq > delta**2, D_sq, delta**2)
    X = D * F / D_sq_safe
    x = np.real(np.fft.ifftn(X)) * mask

    print("Step 2: ADMM with TV...")
    denominator = M + rho * L
    # Handle near-zero denominators properly
    eps = 1e-10
    small_denom = np.abs(denominator) < eps
    denom_safe = np.where(small_denom, 1.0, denominator)
    iA = np.where(small_denom, 0.0, rho / denom_safe)
    X = np.fft.fftn(x)
    F_const = np.where(small_denom, 0.0, M * X / denom_safe)
    px, py, pz = np.zeros(shape), np.zeros(shape), np.zeros(shape)

    def gradient(x):
        return np.roll(x,-1,0)-x, np.roll(x,-1,1)-x, np.roll(x,-1,2)-x
    def divergence(px, py, pz):
        return (px-np.roll(px,1,0)) + (py-np.roll(py,1,1)) + (pz-np.roll(pz,1,2))
    def shrink(x, t):
        return np.sign(x) * np.maximum(np.abs(x) - t, 0)

    inv_rho = 1.0 / rho
    x_prev = x.copy()
    for iteration in range(maxit):
        try: js_qsm_progress(iteration + 1, maxit)
        except: pass
        gx, gy, gz = gradient(x)
        yx, yy, yz = shrink(gx+px, inv_rho), shrink(gy+py, inv_rho), shrink(gz+pz, inv_rho)
        px, py, pz = px+gx-yx, py+gy-yy, pz+gz-yz
        div_v = divergence(yx-px, yy-py, yz-pz)
        X = iA * np.fft.fftn(div_v) + F_const
        x = np.real(np.fft.ifftn(X))
        diff = np.linalg.norm(x - x_prev) / (np.linalg.norm(x) + 1e-12)
        if diff < tol:
            print(f"  Converged at iteration {iteration + 1}")
            break
        x_prev = x.copy()
        if (iteration + 1) % 5 == 0:
            print(f"  Iteration {iteration + 1}/{maxit}, diff = {diff:.2e}")
    return x * mask

# MEDI helper functions
def fgrad(chi, voxel_size):
    dx, dy, dz = voxel_size
    grad = np.zeros((*chi.shape, 3), dtype=chi.dtype)
    grad[:-1,:,:,0] = (chi[1:,:,:] - chi[:-1,:,:]) / dx
    grad[:,:-1,:,1] = (chi[:,1:,:] - chi[:,:-1,:]) / dy
    grad[:,:,:-1,2] = (chi[:,:,1:] - chi[:,:,:-1]) / dz
    return grad

def bdiv(grad_field, voxel_size):
    dx, dy, dz = voxel_size
    gx, gy, gz = grad_field[:,:,:,0], grad_field[:,:,:,1], grad_field[:,:,:,2]
    div = np.zeros_like(gx)
    div[0,:,:] = gx[0,:,:]/dx; div[1:-1,:,:] += (gx[1:-1,:,:]-gx[:-2,:,:])/dx; div[-1,:,:] += -gx[-2,:,:]/dx
    div[:,0,:] += gy[:,0,:]/dy; div[:,1:-1,:] += (gy[:,1:-1,:]-gy[:,:-2,:])/dy; div[:,-1,:] += -gy[:,-2,:]/dy
    div[:,:,0] += gz[:,:,0]/dz; div[:,:,1:-1] += (gz[:,:,1:-1]-gz[:,:,:-2])/dz; div[:,:,-1] += -gz[:,:,-2]/dz
    return div

def gradient_mask(magnitude, mask, percentage=0.3):
    gy, gx, gz = np.gradient(magnitude)
    grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    masked_grad = grad_mag[mask > 0]
    if len(masked_grad) == 0: return np.ones_like(magnitude)
    threshold = np.percentile(masked_grad, percentage * 100)
    wG = np.ones_like(magnitude)
    wG[grad_mag > threshold] = 0
    return wG * mask

def cg_solve(A_op, b, tol=0.01, max_iter=100, precond=None):
    """Conjugate gradient solver with optional preconditioner"""
    x = np.zeros_like(b)
    r = b.copy()
    b_norm = np.sqrt(np.sum(b * b))
    if b_norm < 1e-12: return x

    if precond is not None:
        # Preconditioned CG
        z = precond(r)
        p = z.copy()
        rz_old = np.sum(r * z)
        for i in range(max_iter):
            Ap = A_op(p)
            pAp = np.sum(p * Ap)
            if np.abs(pAp) < 1e-12: break
            alpha = rz_old / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            if np.sqrt(np.sum(r * r)) / b_norm < tol: break
            z = precond(r)
            rz_new = np.sum(r * z)
            p = z + (rz_new / rz_old) * p
            rz_old = rz_new
        return x
    else:
        # Standard CG (no preconditioner)
        p = r.copy()
        rsold = np.sum(r * r)
        for i in range(max_iter):
            Ap = A_op(p)
            pAp = np.sum(p * Ap)
            if np.abs(pAp) < 1e-12: break
            alpha = rsold / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = np.sum(r * r)
            if np.sqrt(rsnew) / b_norm < tol: break
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        return x

def medi_l1(local_field, mask, magnitude, voxel_size, lambda_=7.5e-5, max_iter=30,
            cg_max_iter=10, cg_tol=0.01, edge_percent=0.3, merit=False):
    """MEDI L1 dipole inversion via WASM"""
    print(f"MEDI (WASM): lambda={lambda_}, max_iter={max_iter}, cg_max_iter={cg_max_iter}")
    shape = local_field.shape
    nx, ny, nz = shape
    vsx, vsy, vsz = voxel_size
    mask_u8 = mask.astype(np.uint8)
    result = js_wasm_medi(
        local_field.flatten().astype(np.float64),
        mask_u8.flatten(),
        magnitude.flatten().astype(np.float64),
        nx, ny, nz, vsx, vsy, vsz, lambda_, max_iter, cg_max_iter, cg_tol
    )
    chi = np.array(result).reshape(shape)
    print("MEDI completed")
    return chi

# Get voxel size
try:
    voxel_size = tuple(header_info.get_zooms()[:3])
except:
    voxel_size = (1.0, 1.0, 1.0)

print(f"Voxel size: {voxel_size} mm")
print(f"B0 field strength: ${magField} T")

# Run dipole inversion based on selected method
inv_method = dipole_method
if inv_method == 'tkd':
    print("Running TKD QSM dipole inversion (instant)...")
    tkd_thr = ${tkdSettings.threshold}
    print(f"TKD settings: threshold={tkd_thr}")
    qsm_result = tkd_qsm(
        local_fieldmap, processing_mask, voxel_size,
        bdir=(0, 0, 1), thr=tkd_thr
    )
elif inv_method == 'tikhonov':
    print("Running Tikhonov QSM dipole inversion (instant)...")
    tikh_lambda = ${tikhonovSettings.lambda}
    tikh_reg = '${tikhonovSettings.reg}'
    print(f"Tikhonov settings: lambda={tikh_lambda}, reg={tikh_reg}")
    qsm_result = tikh_qsm(
        local_fieldmap, processing_mask, voxel_size,
        bdir=(0, 0, 1), lambda_=tikh_lambda, reg=tikh_reg
    )
elif inv_method == 'tv':
    print("Running TV-ADMM QSM dipole inversion...")
    tv_lambda = ${tvSettings.lambda}
    tv_maxiter = ${tvSettings.maxIter}
    tv_tol = ${tvSettings.tol}
    print(f"TV-ADMM settings: lambda={tv_lambda}, maxiter={tv_maxiter}, tol={tv_tol}")
    qsm_result = tv_admm_qsm(
        local_fieldmap, processing_mask, voxel_size,
        bdir=(0, 0, 1), lambda_=tv_lambda, maxit=tv_maxiter, tol=tv_tol
    )
else:  # rts (default)
    print("Running RTS QSM dipole inversion...")
    rts_delta = ${rtsSettings.delta}
    rts_mu = ${rtsSettings.mu}
    rts_rho = ${rtsSettings.rho}
    rts_maxiter = ${rtsSettings.maxIter}
    print(f"RTS settings: delta={rts_delta}, mu={rts_mu}, rho={rts_rho}, maxiter={rts_maxiter}")
    qsm_result = rts_qsm(
        local_fieldmap, processing_mask, voxel_size,
        delta=rts_delta, mu=rts_mu, rho=rts_rho, maxit=rts_maxiter
    )

print(f"QSM result range (raw): [{np.min(qsm_result[processing_mask]):.4f}, {np.max(qsm_result[processing_mask]):.4f}]")

# Scale to ppm: χ (ppm) = χ_raw / (γ × B0) × 1e6
# The dipole inversion solves D*χ = f where f is in Hz
# Physical relationship: f = γ × B0 × D × χ, so χ = f / (γ × B0 × D)
# γ (proton gyromagnetic ratio) = 42.576 MHz/T
gamma = 42.576e6  # Hz/T
B0_tesla = ${magField}
qsm_result = qsm_result / (gamma * B0_tesla) * 1e6  # Convert to ppm

print(f"QSM result range (ppm): [{np.min(qsm_result[processing_mask]):.4f}, {np.max(qsm_result[processing_mask]):.4f}] ppm")
print("Dipole inversion completed!")

# Final cleanup
qsm_result[~processing_mask] = 0
`);

    // Send QSM result for live display
    await sendStageData('final', 'qsm_result', 'QSM Result (ppm)');

    postProgress(1.0, 'Complete');
    postLog("Pipeline completed successfully!");
    postComplete({ success: true });

  } catch (error) {
    postError(error.message);
    throw error;
  }
}

// Post BET-specific progress
function postBETProgress(value, text) {
  self.postMessage({ type: 'betProgress', value, text });
}

function postBETLog(message) {
  self.postMessage({ type: 'betLog', message });
}

function postBETComplete(maskData, coverage) {
  self.postMessage({ type: 'betComplete', maskData, coverage });
}

function postBETError(message) {
  self.postMessage({ type: 'betError', message });
}

async function runBET(data) {
  const { magnitudeBuffer, voxelSize, betCode, fractionalIntensity, iterations, subdivisions } = data;
  const betIterations = iterations || 1000;
  const betSubdivisions = subdivisions || 4;

  try {
    // Initialize Pyodide if not already done
    if (!pyodide) {
      postBETLog("Initializing Pyodide for BET...");
      postBETProgress(0.05, 'Loading Pyodide...');

      importScripts('https://cdn.jsdelivr.net/pyodide/v0.27.1/full/pyodide.js');
      pyodide = await loadPyodide();

      postBETLog("Installing Python packages...");
      postBETProgress(0.1, 'Installing packages...');
      await pyodide.loadPackage(["numpy", "scipy", "micropip"]);

      postBETLog("Installing nibabel...");
      await pyodide.runPythonAsync(`
        import micropip
        await micropip.install("nibabel")
      `);
    }

    postBETProgress(0.15, 'Loading BET algorithm...');
    postBETLog("Loading BET algorithm...");

    // Load BET code
    await pyodide.runPython(betCode);

    postBETProgress(0.2, 'Loading magnitude data...');
    postBETLog("Loading magnitude data...");

    // Transfer data to Python
    pyodide.globals.set('mag_buffer', new Uint8Array(magnitudeBuffer));
    pyodide.globals.set('voxel_size', voxelSize);
    pyodide.globals.set('fractional_intensity', fractionalIntensity || 0.5);
    pyodide.globals.set('bet_iterations', betIterations);
    pyodide.globals.set('bet_subdivisions', betSubdivisions);

    // Set up progress callback for BET
    const betProgressCallback = (iteration, total) => {
      // BET runs from 20% to 90% of the progress bar
      const progress = 0.2 + (iteration / total) * 0.7;
      postBETProgress(progress, `BET: iteration ${iteration}/${total}`);
    };
    pyodide.globals.set('js_bet_progress', betProgressCallback);

    // Load and run BET
    await pyodide.runPython(`
import numpy as np
import nibabel as nib
from io import BytesIO

print("Loading magnitude image...")
mag_bytes = mag_buffer.to_py()
mag_fh = nib.FileHolder(BytesIO(mag_bytes))
mag_img = nib.Nifti1Image.from_file_map({'image': mag_fh, 'header': mag_fh})
mag_data = mag_img.get_fdata()

print(f"Image shape: {mag_data.shape}")
print(f"Voxel size from JS: {list(voxel_size.to_py())}")

# Get voxel size from header if available
try:
    header_voxel_size = tuple(mag_img.header.get_zooms()[:3])
    # Use header voxel size (it's in x, y, z order)
    vs = (header_voxel_size[2], header_voxel_size[1], header_voxel_size[0])  # Convert to z, y, x for Python
    print(f"Using header voxel size: {vs} mm (z, y, x)")
except:
    vs = tuple(voxel_size.to_py())
    print(f"Using provided voxel size: {vs} mm")

# Define progress callback wrapper
def progress_wrapper(iteration, total):
    try:
        js_bet_progress(iteration, total)
    except:
        pass

print(f"Running BET brain extraction (iterations={int(bet_iterations)}, subdivisions={int(bet_subdivisions)})...")
bet_mask = run_bet(
    mag_data,
    voxel_size=vs,
    fractional_intensity=float(fractional_intensity),
    iterations=int(bet_iterations),
    subdivisions=int(bet_subdivisions),
    progress_callback=progress_wrapper
)

print(f"BET mask shape: {bet_mask.shape}")
mask_count = np.sum(bet_mask > 0)
total_voxels = bet_mask.size
coverage_pct = (mask_count / total_voxels) * 100
print(f"Mask coverage: {mask_count}/{total_voxels} voxels ({coverage_pct:.1f}%)")

# Flatten mask for transfer - use Fortran order to match NIfTI convention
bet_mask_flat = bet_mask.flatten(order='F').astype(np.float32)
`);

    postBETProgress(0.95, 'Transferring mask...');
    postBETLog("Transferring mask data...");

    // Get the mask data
    const maskArray = pyodide.globals.get('bet_mask_flat').toJs();
    const coverage = pyodide.globals.get('coverage_pct');

    postBETProgress(1.0, 'Complete');
    postBETComplete(maskArray, `${coverage.toFixed(1)}%`);

  } catch (error) {
    postBETError(error.message);
    console.error('BET error:', error);
  }
}

async function getStageData(stage) {
  let dataName, description;

  switch (stage) {
    case 'magnitude':
      dataName = 'magnitude_combined';
      description = 'Magnitude (First Echo)';
      break;
    case 'phase':
      dataName = 'phase_4d[:,:,:,0]';
      description = 'Phase (First Echo)';
      break;
    case 'mask':
      dataName = 'processing_mask.astype(np.float32)';
      description = 'Processing Mask';
      break;
    case 'B0':
      dataName = 'B0_fieldmap';
      description = 'B0 Field Map';
      break;
    case 'bgRemoved':
      dataName = 'local_fieldmap';
      description = 'Local Field Map';
      break;
    case 'final':
      dataName = 'qsm_result';
      description = 'QSM Result';
      break;
    default:
      throw new Error(`Unknown stage: ${stage}`);
  }

  await pyodide.runPython(`
import nibabel as nib

# Get the data
display_data = ${dataName}
print(f"Exporting {${JSON.stringify(description)}}: shape {display_data.shape}")

# Create NIfTI file
nii_img = nib.Nifti1Image(display_data, affine_matrix, header_info)

# Save to bytes
import tempfile
import os
temp_path = '/tmp/temp_output.nii'
nii_img.to_filename(temp_path)

# Read the file as bytes
with open(temp_path, 'rb') as f:
    output_bytes = f.read()

# Clean up
os.remove(temp_path)
`);

  const outputBytes = pyodide.globals.get('output_bytes').toJs();
  return { stage, data: outputBytes, description };
}

// Handle messages from main thread
self.onmessage = async function (e) {
  const { type, data } = e.data;

  try {
    switch (type) {
      case 'init':
        await initializePyodide();
        await loadPythonAlgorithms(data.romeoCode);
        self.postMessage({ type: 'initialized' });
        break;

      case 'run':
        await runPipeline(data);
        break;

      case 'getStage':
        const result = await getStageData(data.stage);
        self.postMessage({ type: 'stageData', ...result });
        break;

      case 'runBET':
        await runBET(data);
        break;

      default:
        postError(`Unknown message type: ${type}`);
    }
  } catch (error) {
    postError(error.message);
    console.error(error);
  }
};
