/**
 * QSM Processing Web Worker - Pure JavaScript/WASM
 *
 * Runs QSM pipeline entirely in WASM without Pyodide.
 * All computation is done in Rust/WASM, no Python dependencies.
 */

let wasmModule = null;

// Post progress updates to main thread
function postProgress(value, text) {
  self.postMessage({ type: 'progress', value, text });
}

// Post log messages to main thread
function postLog(message) {
  self.postMessage({ type: 'log', message });
}

// Post error to main thread
function postError(message) {
  self.postMessage({ type: 'error', message });
}

// Post completion to main thread
function postComplete(results) {
  self.postMessage({ type: 'complete', results });
}

// Send intermediate stage data for live display
function sendStageData(stage, data, dims, voxelSize, affine, description) {
  // Save as NIfTI and send to main thread
  const niftiBytes = wasmModule.save_nifti_wasm(
    data,
    dims[0], dims[1], dims[2],
    voxelSize[0], voxelSize[1], voxelSize[2],
    affine
  );
  self.postMessage({ type: 'stageData', stage, data: niftiBytes, description });
}

async function initializeWasm() {
  postLog("Loading WASM module...");

  try {
    // Construct URLs relative to worker location
    const baseUrl = self.location.href.replace(/\/js\/.*$/, '');
    const jsUrl = `${baseUrl}/wasm/qsm_wasm.js`;
    const wasmBinaryUrl = `${baseUrl}/wasm/qsm_wasm_bg.wasm`;

    const module = await import(jsUrl);
    await module.default(wasmBinaryUrl);
    wasmModule = module;

    if (wasmModule.wasm_health_check()) {
      postLog(`WASM loaded (v${wasmModule.get_version()}) - Pure JS mode`);
    }
  } catch (e) {
    postError(`WASM load failed: ${e.message}`);
    throw e;
  }
}

// Scale phase to [-π, +π] range
function scalePhase(phase) {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < phase.length; i++) {
    if (phase[i] < min) min = phase[i];
    if (phase[i] > max) max = phase[i];
  }

  const range = max - min;
  const pi = Math.PI;

  // Check if phase needs scaling
  if (range > 2 * pi * 1.1 || max > pi * 1.5 || min < -pi * 1.5) {
    // Linear scale from [min, max] to [-π, +π]
    const scaled = new Float64Array(phase.length);
    for (let i = 0; i < phase.length; i++) {
      scaled[i] = (phase[i] - min) / range * 2 * pi - pi;
    }
    return scaled;
  }

  // Wrap to ensure exactly [-π, +π]
  const wrapped = new Float64Array(phase.length);
  for (let i = 0; i < phase.length; i++) {
    wrapped[i] = Math.atan2(Math.sin(phase[i]), Math.cos(phase[i]));
  }
  return wrapped;
}

// Create threshold-based mask from magnitude
function createThresholdMask(magnitude, thresholdFraction) {
  let maxVal = 0;
  for (let i = 0; i < magnitude.length; i++) {
    if (magnitude[i] > maxVal) maxVal = magnitude[i];
  }

  const threshold = maxVal * thresholdFraction;
  const mask = new Uint8Array(magnitude.length);
  for (let i = 0; i < magnitude.length; i++) {
    mask[i] = magnitude[i] > threshold ? 1 : 0;
  }
  return mask;
}

// Find seed point (center of mass of mask)
function findSeedPoint(mask, nx, ny, nz) {
  let sumX = 0, sumY = 0, sumZ = 0, count = 0;

  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      for (let k = 0; k < nz; k++) {
        const idx = i * ny * nz + j * nz + k;
        if (mask[idx]) {
          sumX += i;
          sumY += j;
          sumZ += k;
          count++;
        }
      }
    }
  }

  if (count === 0) {
    return [Math.floor(nx / 2), Math.floor(ny / 2), Math.floor(nz / 2)];
  }

  return [
    Math.floor(sumX / count),
    Math.floor(sumY / count),
    Math.floor(sumZ / count)
  ];
}

// Compute B0 fieldmap from unwrapped phase
function computeB0FromUnwrapped(unwrappedPhase, echoTimes, nx, ny, nz) {
  const nEchoes = echoTimes.length;
  const voxelCount = nx * ny * nz;

  // Convert echo times from ms to seconds
  const teSec = echoTimes.map(t => t / 1000);

  if (nEchoes === 1) {
    // Single echo: B0 = phase / (2π * TE)
    const b0 = new Float64Array(voxelCount);
    const factor = 1 / (2 * Math.PI * teSec[0]);
    for (let i = 0; i < voxelCount; i++) {
      b0[i] = unwrappedPhase[i] * factor;
    }
    return b0;
  }

  // Multi-echo: weighted linear fit
  const weights = new Float64Array(nEchoes);
  let weightSum = 0;
  for (let i = 0; i < nEchoes; i++) {
    weights[i] = i + 1;
    weightSum += weights[i];
  }
  for (let i = 0; i < nEchoes; i++) {
    weights[i] /= weightSum;
  }

  let teMean = 0;
  for (let i = 0; i < nEchoes; i++) {
    teMean += weights[i] * teSec[i];
  }

  const b0 = new Float64Array(voxelCount);

  for (let v = 0; v < voxelCount; v++) {
    let numerator = 0;
    let denominator = 0;

    for (let e = 0; e < nEchoes; e++) {
      const teDiff = teSec[e] - teMean;
      const phaseIdx = e * voxelCount + v;
      numerator += weights[e] * teDiff * unwrappedPhase[phaseIdx];
      denominator += weights[e] * teDiff * teDiff;
    }

    const b0RadPerSec = numerator / (denominator + 1e-10);
    b0[v] = b0RadPerSec / (2 * Math.PI);
  }

  return b0;
}

async function runPipeline(data) {
  const {
    magnitudeBuffers, phaseBuffers, echoTimes, magField,
    maskThreshold, customMaskBuffer, pipelineSettings
  } = data;

  const thresholdFraction = (maskThreshold || 15) / 100;
  const hasCustomMask = customMaskBuffer !== null && customMaskBuffer !== undefined;

  // Extract pipeline settings
  const unwrapMethod = pipelineSettings?.unwrapMethod || 'romeo';
  const backgroundMethod = pipelineSettings?.backgroundRemoval || 'smv';
  const dipoleMethod = pipelineSettings?.dipoleInversion || 'rts';

  // Validate methods upfront - never silently fall back to a different algorithm
  const validUnwrapMethods = ['romeo', 'laplacian'];
  const validBgMethods = ['vsharp', 'sharp', 'smv', 'ismv', 'pdf', 'lbv'];
  const validInversionMethods = ['tkd', 'tsvd', 'tikhonov', 'tv', 'rts', 'nltv', 'medi'];

  if (!validUnwrapMethods.includes(unwrapMethod)) {
    throw new Error(`Unknown unwrapping method: '${unwrapMethod}'. Valid options are: ${validUnwrapMethods.join(', ')}`);
  }
  if (!validBgMethods.includes(backgroundMethod)) {
    throw new Error(`Unknown background removal method: '${backgroundMethod}'. Valid options are: ${validBgMethods.join(', ')}`);
  }
  if (!validInversionMethods.includes(dipoleMethod)) {
    throw new Error(`Unknown dipole inversion method: '${dipoleMethod}'. Valid options are: ${validInversionMethods.join(', ')}`);
  }
  const vsharpSettings = {
    maxRadius: pipelineSettings?.vsharp?.maxRadius ?? 18,
    minRadius: pipelineSettings?.vsharp?.minRadius ?? 2,
    threshold: pipelineSettings?.vsharp?.threshold ?? 0.05
  };
  const lbvSettings = pipelineSettings?.lbv || { tol: 0.001, maxit: 500 };
  const rtsSettings = pipelineSettings?.rts || { delta: 0.15, mu: 100000, rho: 10, maxIter: 20 };
  const tkdSettings = pipelineSettings?.tkd || { threshold: 0.15 };
  const tsvdSettings = pipelineSettings?.tsvd || { threshold: 0.15 };
  const tikhonovSettings = pipelineSettings?.tikhonov || { lambda: 0.01, reg: 'identity' };
  const tvSettings = pipelineSettings?.tv || { lambda: 0.001, maxIter: 250, tol: 0.001 };
  const nltvSettings = pipelineSettings?.nltv || { lambda: 0.001, mu: 1, maxIter: 250, tol: 0.001, newtonMaxIter: 10 };
  const mediSettings = pipelineSettings?.medi || {
    lambda: 1000, percentage: 0.9, maxIter: 10, cgMaxIter: 100, cgTol: 0.01, tol: 0.1,
    smv: false, smvRadius: 5, merit: false, dataWeighting: 1
  };

  try {
    // =========================================================================
    // Step 1: Load NIfTI data (0% - 10%)
    // =========================================================================
    postProgress(0.02, 'Loading NIfTI data...');
    postLog("Loading multi-echo data via WASM...");

    const nEchoes = echoTimes.length;
    let magnitude4d = [];
    let phase4d = [];
    let dims, voxelSize, affine;

    for (let e = 0; e < nEchoes; e++) {
      postProgress(0.02 + (e / nEchoes) * 0.08, `Loading echo ${e + 1}/${nEchoes}...`);

      // Load magnitude
      const magResult = wasmModule.load_nifti_wasm(new Uint8Array(magnitudeBuffers[e]));
      const magData = Array.from(magResult.data);
      dims = Array.from(magResult.dims);
      voxelSize = Array.from(magResult.voxelSize);
      affine = Array.from(magResult.affine);
      magnitude4d.push(magData);

      // Load phase
      const phaseResult = wasmModule.load_nifti_wasm(new Uint8Array(phaseBuffers[e]));
      let phaseData = Array.from(phaseResult.data);

      // Scale phase to [-π, +π]
      phaseData = scalePhase(new Float64Array(phaseData));
      phase4d.push(Array.from(phaseData));

      postLog(`  Echo ${e + 1}: shape ${dims[0]}x${dims[1]}x${dims[2]}`);
    }

    const [nx, ny, nz] = dims;
    const [vsx, vsy, vsz] = voxelSize;
    const voxelCount = nx * ny * nz;

    postLog(`Data shape: ${nx}x${ny}x${nz}, voxel: ${vsx.toFixed(2)}x${vsy.toFixed(2)}x${vsz.toFixed(2)}mm`);

    // =========================================================================
    // Step 2: Create or load mask (10% - 15%)
    // =========================================================================
    postProgress(0.10, 'Creating mask...');
    let mask;

    if (hasCustomMask) {
      postLog("Loading custom mask...");
      const maskResult = wasmModule.load_nifti_wasm(new Uint8Array(customMaskBuffer));
      const maskData = Array.from(maskResult.data);
      mask = new Uint8Array(voxelCount);
      for (let i = 0; i < voxelCount; i++) {
        mask[i] = maskData[i] > 0.5 ? 1 : 0;
      }
    } else {
      postLog(`Creating threshold mask (${thresholdFraction * 100}%)...`);
      mask = createThresholdMask(new Float64Array(magnitude4d[0]), thresholdFraction);
    }

    const maskCount = mask.reduce((a, b) => a + b, 0);
    postLog(`Mask coverage: ${maskCount}/${voxelCount} voxels (${(100 * maskCount / voxelCount).toFixed(1)}%)`);

    // =========================================================================
    // Step 3: Phase unwrapping (15% - 40%)
    // =========================================================================
    let unwrappedPhase;
    const phase1 = new Float64Array(phase4d[0]);

    if (unwrapMethod === 'laplacian') {
      postProgress(0.15, 'Laplacian: Unwrapping phase...');
      postLog("Running Laplacian phase unwrapping...");

      unwrappedPhase = new Float64Array(wasmModule.laplacian_unwrap_wasm(
        phase1, mask, nx, ny, nz, vsx, vsy, vsz
      ));

      postProgress(0.35, 'Laplacian: Complete');
    } else if (unwrapMethod === 'romeo') {
      postProgress(0.15, 'ROMEO: Calculating weights...');
      postLog("Running ROMEO phase unwrapping...");

      // Calculate ROMEO weights
      const mag1 = new Float64Array(magnitude4d[0]);
      const phase2 = nEchoes > 1 ? new Float64Array(phase4d[1]) : new Float64Array(0);
      const te1 = echoTimes[0];
      const te2 = nEchoes > 1 ? echoTimes[1] : 0;

      postLog("  Calculating ROMEO weights...");
      const weights = wasmModule.calculate_weights_romeo_wasm(
        phase1, mag1, phase2, te1, te2, mask, nx, ny, nz
      );

      // Unwrap first echo
      postProgress(0.25, 'ROMEO: Unwrapping echo 1...');
      postLog("  Unwrapping first echo...");
      const [seedI, seedJ, seedK] = findSeedPoint(mask, nx, ny, nz);
      postLog(`  Seed point: (${seedI}, ${seedJ}, ${seedK})`);

      unwrappedPhase = new Float64Array(phase1);
      const workMask = new Uint8Array(mask);
      wasmModule.grow_region_unwrap_wasm(
        unwrappedPhase, weights, workMask,
        nx, ny, nz, seedI, seedJ, seedK
      );
    } else {
      throw new Error(`Unknown unwrapping method: '${unwrapMethod}'. Valid options are: romeo, laplacian`);
    }

    postProgress(0.40, 'Computing B0 field...');

    // For multi-echo, do temporal unwrapping
    let allUnwrapped;
    if (nEchoes > 1) {
      postLog("  Temporal unwrapping remaining echoes...");
      allUnwrapped = new Float64Array(voxelCount * nEchoes);
      allUnwrapped.set(unwrappedPhase, 0);

      for (let e = 1; e < nEchoes; e++) {
        const currentPhase = new Float64Array(phase4d[e]);

        // Temporal unwrap: adjust to minimize difference from scaled template
        const teRatio = echoTimes[e] / echoTimes[0];
        for (let i = 0; i < voxelCount; i++) {
          if (mask[i]) {
            const expected = unwrappedPhase[i] * teRatio;
            let diff = currentPhase[i] - (expected % (2 * Math.PI));
            // Wrap diff to [-π, π]
            while (diff > Math.PI) diff -= 2 * Math.PI;
            while (diff < -Math.PI) diff += 2 * Math.PI;
            allUnwrapped[e * voxelCount + i] = expected + diff;
          }
        }
      }
    } else {
      allUnwrapped = unwrappedPhase;
    }

    // Compute B0 fieldmap
    postLog("Computing B0 field map...");
    const b0Fieldmap = computeB0FromUnwrapped(allUnwrapped, echoTimes, nx, ny, nz);

    // Apply mask
    for (let i = 0; i < voxelCount; i++) {
      if (!mask[i]) b0Fieldmap[i] = 0;
    }

    // Calculate B0 range efficiently (avoid spread operator on large arrays)
    let b0Min = Infinity, b0Max = -Infinity;
    for (let i = 0; i < voxelCount; i++) {
      if (mask[i]) {
        if (b0Fieldmap[i] < b0Min) b0Min = b0Fieldmap[i];
        if (b0Fieldmap[i] > b0Max) b0Max = b0Fieldmap[i];
      }
    }
    postLog(`B0 range: [${b0Min.toFixed(1)}, ${b0Max.toFixed(1)}] Hz`);

    // Send B0 for display
    sendStageData('B0', b0Fieldmap, dims, voxelSize, affine, 'B0 Field Map (Hz)');

    // =========================================================================
    // Step 4: Background field removal (40% - 65%)
    // =========================================================================
    postProgress(0.42, `Preparing ${backgroundMethod.toUpperCase()} background removal...`);
    postLog(`Removing background field using ${backgroundMethod.toUpperCase()}...`);

    let localField, erodedMask;

    if (backgroundMethod === 'vsharp') {
      // Create radii array
      const radii = [];
      for (let r = vsharpSettings.maxRadius; r >= vsharpSettings.minRadius; r -= 2) {
        radii.push(r);
      }
      const numRadii = radii.length;
      postLog(`  V-SHARP radii: ${radii.map(r => r.toFixed(1)).join(', ')}`);

      // Progress callback for V-SHARP radii
      const vsharpProgress = (current, total) => {
        const progress = 0.42 + (current / total) * 0.20;
        postProgress(progress, `V-SHARP: Radius ${current}/${total}`);
      };

      const result = wasmModule.vsharp_wasm_with_progress(
        b0Fieldmap, mask, nx, ny, nz, vsx, vsy, vsz,
        new Float64Array(radii), vsharpSettings.threshold,
        vsharpProgress
      );

      postProgress(0.63, 'V-SHARP: Extracting results...');
      localField = new Float64Array(result.slice(0, voxelCount));
      erodedMask = new Uint8Array(voxelCount);
      for (let i = 0; i < voxelCount; i++) {
        erodedMask[i] = result[voxelCount + i] > 0.5 ? 1 : 0;
      }
    } else if (backgroundMethod === 'pdf') {
      const pdfSettings = pipelineSettings?.pdf || { tol: 0.00001, maxit: 100 };

      // Progress callback for PDF iterations
      const pdfProgress = (current, total) => {
        const progress = 0.42 + (current / total) * 0.20;
        postProgress(progress, `PDF: Iteration ${current}/${total}`);
      };

      localField = new Float64Array(wasmModule.pdf_wasm_with_progress(
        b0Fieldmap, mask, nx, ny, nz, vsx, vsy, vsz,
        0, 0, 1, pdfSettings.tol, pdfSettings.maxit,
        pdfProgress
      ));
      erodedMask = mask;
    } else if (backgroundMethod === 'ismv') {
      const ismvSettings = pipelineSettings?.ismv || { radius: 5, tol: 0.001, maxit: 500 };

      // Progress callback for iSMV iterations
      const ismvProgress = (current, total) => {
        const progress = 0.42 + (current / total) * 0.20;
        postProgress(progress, `iSMV: Iteration ${current}/${total}`);
      };

      const result = wasmModule.ismv_wasm_with_progress(
        b0Fieldmap, mask, nx, ny, nz, vsx, vsy, vsz,
        ismvSettings.radius, ismvSettings.tol, ismvSettings.maxit,
        ismvProgress
      );
      postProgress(0.63, 'iSMV: Extracting results...');
      localField = new Float64Array(result.slice(0, voxelCount));
      erodedMask = new Uint8Array(voxelCount);
      for (let i = 0; i < voxelCount; i++) {
        erodedMask[i] = result[voxelCount + i] > 0.5 ? 1 : 0;
      }
    } else if (backgroundMethod === 'smv') {
      // Simple SMV (single radius, no deconvolution)
      const smvSettings = pipelineSettings?.smv || { radius: 5 };
      postProgress(0.45, `SMV: Processing radius ${smvSettings.radius}mm...`);
      const result = wasmModule.smv_wasm(
        b0Fieldmap, mask, nx, ny, nz, vsx, vsy, vsz,
        smvSettings.radius
      );
      postProgress(0.60, 'SMV: Extracting results...');
      localField = new Float64Array(result.slice(0, voxelCount));
      erodedMask = new Uint8Array(voxelCount);
      for (let i = 0; i < voxelCount; i++) {
        erodedMask[i] = result[voxelCount + i] > 0.5 ? 1 : 0;
      }
    } else if (backgroundMethod === 'sharp') {
      // SHARP (high-pass filter + deconvolution)
      const sharpSettings = pipelineSettings?.sharp || { radius: 6, threshold: 0.05 };
      postProgress(0.45, `SHARP: Processing radius ${sharpSettings.radius}mm...`);
      const result = wasmModule.sharp_wasm(
        b0Fieldmap, mask, nx, ny, nz, vsx, vsy, vsz,
        sharpSettings.radius, sharpSettings.threshold
      );
      postProgress(0.60, 'SHARP: Extracting results...');
      localField = new Float64Array(result.slice(0, voxelCount));
      erodedMask = new Uint8Array(voxelCount);
      for (let i = 0; i < voxelCount; i++) {
        erodedMask[i] = result[voxelCount + i] > 0.5 ? 1 : 0;
      }
    } else if (backgroundMethod === 'lbv') {
      // LBV (Laplacian Boundary Value)
      const lbvProgress = (current, total) => {
        const progress = 0.42 + (current / total) * 0.20;
        postProgress(progress, `LBV: Iteration ${current}/${total}`);
      };

      postProgress(0.45, 'LBV: Solving Poisson equation...');
      const result = wasmModule.lbv_wasm_with_progress(
        b0Fieldmap, mask, nx, ny, nz, vsx, vsy, vsz,
        lbvSettings.tol, lbvSettings.maxit,
        lbvProgress
      );
      postProgress(0.63, 'LBV: Extracting results...');
      localField = new Float64Array(result.slice(0, voxelCount));
      erodedMask = new Uint8Array(voxelCount);
      for (let i = 0; i < voxelCount; i++) {
        erodedMask[i] = result[voxelCount + i] > 0.5 ? 1 : 0;
      }
    } else {
      throw new Error(`Unknown background removal method: '${backgroundMethod}'. Valid options are: vsharp, sharp, smv, ismv, pdf, lbv`);
    }

    const erodedCount = erodedMask.reduce((a, b) => a + b, 0);
    postLog(`Eroded mask: ${erodedCount} voxels (${(100 * erodedCount / voxelCount).toFixed(1)}%)`);
    postProgress(0.65, 'Sending local field for display...');

    // Send local field for display
    sendStageData('bgRemoved', localField, dims, voxelSize, affine, 'Local Field Map (Hz)');

    // =========================================================================
    // Step 5: Dipole inversion (65% - 95%)
    // =========================================================================
    postProgress(0.67, `Preparing ${dipoleMethod.toUpperCase()} dipole inversion...`);
    postLog(`Running ${dipoleMethod.toUpperCase()} dipole inversion...`);

    let qsmResult;

    if (dipoleMethod === 'tkd') {
      // TKD is non-iterative (single FFT-based inversion)
      postProgress(0.70, 'TKD: Computing thresholded k-space division...');
      qsmResult = new Float64Array(wasmModule.tkd_wasm(
        localField, erodedMask, nx, ny, nz, vsx, vsy, vsz,
        0, 0, 1, tkdSettings.threshold
      ));
      postProgress(0.90, 'TKD: Complete');
    } else if (dipoleMethod === 'tsvd') {
      // TSVD is non-iterative (zeros small values instead of truncating)
      postProgress(0.70, 'TSVD: Computing truncated SVD inversion...');
      qsmResult = new Float64Array(wasmModule.tsvd_wasm(
        localField, erodedMask, nx, ny, nz, vsx, vsy, vsz,
        0, 0, 1, tsvdSettings.threshold
      ));
      postProgress(0.90, 'TSVD: Complete');
    } else if (dipoleMethod === 'tikhonov') {
      // Tikhonov is non-iterative (single FFT-based inversion)
      const regType = { 'identity': 0, 'gradient': 1, 'laplacian': 2 }[tikhonovSettings.reg] || 0;
      postProgress(0.70, `Tikhonov: Solving (λ=${tikhonovSettings.lambda})...`);
      qsmResult = new Float64Array(wasmModule.tikhonov_wasm(
        localField, erodedMask, nx, ny, nz, vsx, vsy, vsz,
        0, 0, 1, tikhonovSettings.lambda, regType
      ));
      postProgress(0.90, 'Tikhonov: Complete');
    } else if (dipoleMethod === 'tv') {
      const rho = tvSettings.rho || 100 * tvSettings.lambda;

      // Progress callback for TV-ADMM iterations
      const tvProgress = (current, total) => {
        const progress = 0.67 + (current / total) * 0.25;
        postProgress(progress, `TV-ADMM: Iteration ${current}/${total}`);
      };

      qsmResult = new Float64Array(wasmModule.tv_admm_wasm_with_progress(
        localField, erodedMask, nx, ny, nz, vsx, vsy, vsz,
        0, 0, 1, tvSettings.lambda, rho, tvSettings.tol, tvSettings.maxIter,
        tvProgress
      ));
    } else if (dipoleMethod === 'rts') {
      // RTS (Rapid Two-Step)
      // Progress callback for RTS iterations
      const rtsProgress = (current, total) => {
        const progress = 0.67 + (current / total) * 0.25;
        postProgress(progress, `RTS: Iteration ${current}/${total}`);
      };

      qsmResult = new Float64Array(wasmModule.rts_wasm_with_progress(
        localField, erodedMask, nx, ny, nz, vsx, vsy, vsz,
        0, 0, 1,
        rtsSettings.delta, rtsSettings.mu, rtsSettings.rho,
        0.01, rtsSettings.maxIter, 4,
        rtsProgress
      ));
    } else if (dipoleMethod === 'nltv') {
      // NLTV (Nonlinear Total Variation)
      const nltvProgress = (current, total) => {
        const progress = 0.67 + (current / total) * 0.25;
        postProgress(progress, `NLTV: Iteration ${current}/${total}`);
      };

      postProgress(0.70, 'NLTV: Running nonlinear TV inversion...');
      qsmResult = new Float64Array(wasmModule.nltv_wasm_with_progress(
        localField, erodedMask, nx, ny, nz, vsx, vsy, vsz,
        0, 0, 1,
        nltvSettings.lambda, nltvSettings.mu,
        nltvSettings.tol, nltvSettings.maxIter, nltvSettings.newtonMaxIter,
        nltvProgress
      ));
    } else if (dipoleMethod === 'medi') {
      // MEDI (Morphology Enabled Dipole Inversion)
      const mediProgress = (current, total) => {
        const progress = 0.67 + (current / total) * 0.25;
        postProgress(progress, `MEDI: Iteration ${current}/${total}`);
      };

      // MEDI requires magnitude image for gradient weighting
      // Use the first echo magnitude data
      const magnitudeData = new Float64Array(magnitude4d[0]);

      // Create noise std map (uniform for now - could be computed from data)
      const nStd = new Float64Array(voxelCount).fill(1.0);

      qsmResult = new Float64Array(wasmModule.medi_l1_wasm_with_progress(
        localField,       // local field
        nStd,             // noise standard deviation
        magnitudeData,    // magnitude for edge weighting
        erodedMask,       // mask
        nx, ny, nz,
        vsx, vsy, vsz,
        0, 0, 1,          // B0 direction
        mediSettings.lambda,
        mediSettings.merit,
        mediSettings.smv,
        mediSettings.smvRadius,
        mediSettings.dataWeighting,
        mediSettings.percentage,
        mediSettings.cgTol,
        mediSettings.cgMaxIter,
        mediSettings.maxIter,
        mediSettings.tol,
        mediProgress
      ));
    } else {
      throw new Error(`Unknown dipole inversion method: '${dipoleMethod}'. Valid options are: tkd, tsvd, tikhonov, tv, rts, nltv, medi`);
    }

    // Scale to ppm: χ (ppm) = χ_raw / (γ × B0) × 1e6
    postProgress(0.92, 'Scaling to ppm...');
    const gamma = 42.576e6; // Hz/T
    const b0Tesla = magField || 3.0;
    const scaleFactor = 1e6 / (gamma * b0Tesla);

    for (let i = 0; i < voxelCount; i++) {
      qsmResult[i] *= scaleFactor;
      if (!erodedMask[i]) qsmResult[i] = 0;
    }

    // Calculate QSM range efficiently
    let qsmMin = Infinity, qsmMax = -Infinity;
    for (let i = 0; i < voxelCount; i++) {
      if (erodedMask[i]) {
        if (qsmResult[i] < qsmMin) qsmMin = qsmResult[i];
        if (qsmResult[i] > qsmMax) qsmMax = qsmResult[i];
      }
    }
    postLog(`QSM range: [${qsmMin.toFixed(4)}, ${qsmMax.toFixed(4)}] ppm`);

    // Send QSM result for display
    postProgress(0.95, 'Sending QSM result...');
    sendStageData('final', qsmResult, dims, voxelSize, affine, 'QSM Result (ppm)');

    postProgress(1.0, 'Pipeline complete!');
    postLog("Pipeline completed successfully!");
    postComplete({ success: true });

  } catch (error) {
    postError(error.message);
    throw error;
  }
}

// BET-specific handlers
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
  const { magnitudeBuffer, fractionalIntensity, iterations, subdivisions } = data;
  const betIterations = iterations || 1000;
  const betSubdivisions = subdivisions || 4;

  try {
    // Load magnitude data
    postBETProgress(0.1, 'Loading data...');
    postBETLog("Loading magnitude image via WASM...");

    const magResult = wasmModule.load_nifti_wasm(new Uint8Array(magnitudeBuffer));
    const magData = new Float64Array(magResult.data);
    const dims = Array.from(magResult.dims);
    const voxelSize = Array.from(magResult.voxelSize);

    const [nx, ny, nz] = dims;
    const [vsx, vsy, vsz] = voxelSize;

    postBETLog(`Image: ${nx}x${ny}x${nz}, voxel: ${vsx.toFixed(2)}x${vsy.toFixed(2)}x${vsz.toFixed(2)}mm`);

    // TEST: Create a simple sphere mask to verify data transfer works
    const TEST_SPHERE = false;  // Set to true to test with sphere instead of BET

    let mask;
    if (TEST_SPHERE) {
      postBETProgress(0.2, 'Creating test sphere...');
      postBETLog(`Creating test sphere mask at center (${nx/2}, ${ny/2}, ${nz/2}), radius=${Math.min(nx,ny,nz)/3}`);
      mask = wasmModule.create_sphere_mask(
        nx, ny, nz,
        nx / 2, ny / 2, nz / 2,
        Math.min(nx, ny, nz) / 3
      );
    } else {
      // Run BET with progress callback
      postBETProgress(0.15, 'Running BET...');
      postBETLog(`Running BET (iterations=${betIterations}, subdivisions=${betSubdivisions})...`);

      // Progress callback that updates the progress bar during iteration
      const progressCallback = (current, total) => {
        // Map iterations to 0.15 - 0.9 range (leave room for mask conversion)
        const progress = 0.15 + (current / total) * 0.75;
        const pct = Math.round((current / total) * 100);
        postBETProgress(progress, `BET iteration ${current}/${total} (${pct}%)`);
      };

      mask = wasmModule.bet_wasm_with_progress(
        magData, nx, ny, nz, vsx, vsy, vsz,
        fractionalIntensity || 0.5, betIterations, betSubdivisions,
        progressCallback
      );

      postBETProgress(0.95, 'Converting mask...');
    }

    const maskCount = mask.reduce((a, b) => a + b, 0);
    const totalVoxels = mask.length;
    const coveragePct = (maskCount / totalVoxels) * 100;

    postBETLog(`Mask coverage: ${maskCount}/${totalVoxels} voxels (${coveragePct.toFixed(1)}%)`);

    // Convert mask to Float32 for transfer
    const maskFloat = new Float32Array(totalVoxels);
    for (let i = 0; i < totalVoxels; i++) {
      maskFloat[i] = mask[i];
    }

    postBETProgress(1.0, 'Complete');
    postBETComplete(maskFloat, `${coveragePct.toFixed(1)}%`);

  } catch (error) {
    postBETError(error.message);
    console.error('BET error:', error);
  }
}

// Handle messages from main thread
self.onmessage = async function (e) {
  const { type, data } = e.data;

  try {
    switch (type) {
      case 'init':
        await initializeWasm();
        self.postMessage({ type: 'initialized' });
        break;

      case 'run':
        await runPipeline(data);
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
