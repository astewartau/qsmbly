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
// Set displayNow=false to cache without displaying (e.g., for auxiliary outputs)
function sendStageData(stage, data, dims, voxelSize, affine, description, displayNow = true) {
  // Save as NIfTI and send to main thread
  const niftiBytes = wasmModule.save_nifti_wasm(
    data,
    dims[0], dims[1], dims[2],
    voxelSize[0], voxelSize[1], voxelSize[2],
    affine
  );
  self.postMessage({ type: 'stageData', stage, data: niftiBytes, description, displayNow });
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

// 3D box filter (mean filter) for smoothing
// Used for R_0 reliability map computation
function boxFilter3D(data, nx, ny, nz, radius) {
  const voxelCount = nx * ny * nz;
  const result = new Float64Array(voxelCount);

  const idx = (i, j, k) => i + j * nx + k * nx * ny;

  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        let sum = 0;
        let count = 0;

        // Box neighborhood
        for (let dk = -radius; dk <= radius; dk++) {
          const kk = k + dk;
          if (kk < 0 || kk >= nz) continue;

          for (let dj = -radius; dj <= radius; dj++) {
            const jj = j + dj;
            if (jj < 0 || jj >= ny) continue;

            for (let di = -radius; di <= radius; di++) {
              const ii = i + di;
              if (ii < 0 || ii >= nx) continue;

              sum += data[idx(ii, jj, kk)];
              count++;
            }
          }
        }

        result[idx(i, j, k)] = count > 0 ? sum / count : 0;
      }
    }
  }

  return result;
}

// Compute B0 fieldmap from unwrapped phase
// method: 'ols' (simple least squares through origin) or 'ols_offset' (estimates phase offset)
function computeB0FromUnwrapped(unwrappedPhase, echoTimes, nx, ny, nz, method = 'ols') {
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

  const b0 = new Float64Array(voxelCount);

  if (method === 'ols_offset') {
    // OLS with phase offset estimation (matching QSM.jl _multi_echo_linear_fit! with α)
    // Model: phase = α + β * TE
    // Solve using centered data to avoid numerical issues

    // Compute mean TE
    let teMean = 0;
    for (let e = 0; e < nEchoes; e++) {
      teMean += teSec[e];
    }
    teMean /= nEchoes;

    // Compute centered TE and sum of squared centered TEs
    const teCentered = teSec.map(t => t - teMean);
    let sumTeCenteredSq = 0;
    for (let e = 0; e < nEchoes; e++) {
      sumTeCenteredSq += teCentered[e] * teCentered[e];
    }

    for (let v = 0; v < voxelCount; v++) {
      // Compute mean phase for this voxel
      let phaseMean = 0;
      for (let e = 0; e < nEchoes; e++) {
        phaseMean += unwrappedPhase[e * voxelCount + v];
      }
      phaseMean /= nEchoes;

      // Compute slope β = Σ((TE - TE_mean) * (phase - phase_mean)) / Σ((TE - TE_mean)²)
      let sumTeCenteredPhase = 0;
      for (let e = 0; e < nEchoes; e++) {
        const phaseIdx = e * voxelCount + v;
        sumTeCenteredPhase += teCentered[e] * (unwrappedPhase[phaseIdx] - phaseMean);
      }

      const b0RadPerSec = sumTeCenteredPhase / (sumTeCenteredSq + 1e-10);
      b0[v] = b0RadPerSec / (2 * Math.PI);
    }
  } else {
    // Simple OLS through origin (default, matching QSM.jl _multi_echo_linear_fit! without α)
    // Model: phase = β * TE (assumes zero phase at TE=0)
    // Slope: β = Σ(TE * phase) / Σ(TE²)

    // Precompute sum of TE² (same for all voxels)
    let sumTeSq = 0;
    for (let e = 0; e < nEchoes; e++) {
      sumTeSq += teSec[e] * teSec[e];
    }

    for (let v = 0; v < voxelCount; v++) {
      let sumTePhase = 0;

      for (let e = 0; e < nEchoes; e++) {
        const phaseIdx = e * voxelCount + v;
        sumTePhase += teSec[e] * unwrappedPhase[phaseIdx];
      }

      const b0RadPerSec = sumTePhase / (sumTeSq + 1e-10);
      b0[v] = b0RadPerSec / (2 * Math.PI);
    }
  }

  return b0;
}

async function runPipeline(data) {
  const {
    magnitudeBuffers, phaseBuffers, echoTimes, magField,
    maskThreshold, customMaskBuffer, preparedMagnitude, pipelineSettings
  } = data;

  const thresholdFraction = (maskThreshold || 15) / 100;
  const hasCustomMask = customMaskBuffer !== null && customMaskBuffer !== undefined;
  const hasPreparedMagnitude = preparedMagnitude !== null && preparedMagnitude !== undefined;

  // Check for combined method (TGV)
  const combinedMethod = pipelineSettings?.combinedMethod || 'none';

  if (combinedMethod === 'tgv') {
    // Use TGV single-step reconstruction
    return await runTgvPipeline(data);
  } else if (combinedMethod === 'qsmart') {
    // Use QSMART two-stage reconstruction
    return await runQsmartPipeline(data);
  }

  // Extract pipeline settings for standard pipeline
  const unwrapMethod = pipelineSettings?.unwrapMethod || 'romeo';
  const multiEchoMethod = pipelineSettings?.multiEchoMethod || 'ols_offset'; // 'ols', 'ols_offset', 'mcpc3ds'
  const mcpc3dsSettings = pipelineSettings?.mcpc3ds || {
    sigma: [10, 10, 5],  // Smoothing sigma in voxels [x, y, z]
    weightType: 'phase_snr'  // 'phase_snr', 'phase_var', 'average', 'tes', 'mag'
  };
  const backgroundMethod = pipelineSettings?.backgroundRemoval || 'smv';
  const dipoleMethod = pipelineSettings?.dipoleInversion || 'rts';

  // Validate methods upfront - never silently fall back to a different algorithm
  const validUnwrapMethods = ['romeo', 'laplacian'];
  const validBgMethods = ['vsharp', 'sharp', 'smv', 'ismv', 'pdf', 'lbv'];
  const validInversionMethods = ['tkd', 'tsvd', 'tikhonov', 'tv', 'rts', 'nltv', 'medi', 'ilsqr'];

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
  const ilsqrSettings = pipelineSettings?.ilsqr || { tol: 0.01, maxIter: 50 };

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
      // Use prepared magnitude if available (combined/bias-corrected), otherwise first echo
      const maskMagnitude = hasPreparedMagnitude
        ? new Float64Array(preparedMagnitude)
        : new Float64Array(magnitude4d[0]);
      const magSource = hasPreparedMagnitude ? 'prepared' : 'first echo';
      postLog(`Creating threshold mask (${thresholdFraction * 100}%) from ${magSource} magnitude...`);
      mask = createThresholdMask(maskMagnitude, thresholdFraction);
    }

    const maskCount = mask.reduce((a, b) => a + b, 0);
    postLog(`Mask coverage: ${maskCount}/${voxelCount} voxels (${(100 * maskCount / voxelCount).toFixed(1)}%)`);

    // Send mask as intermediate stage
    const maskFloat = new Float64Array(voxelCount);
    for (let i = 0; i < voxelCount; i++) {
      maskFloat[i] = mask[i] ? 1.0 : 0.0;
    }
    sendStageData('mask', maskFloat, dims, voxelSize, affine, 'Brain Mask');

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

    // Handle multi-echo phase combination
    let b0Fieldmap;

    if (nEchoes > 1 && multiEchoMethod === 'mcpc3ds') {
      // MCPC-3D-S workflow from MriResearchTools.jl
      postLog(`Using MCPC-3D-S multi-echo combination (sigma=[${mcpc3dsSettings.sigma.join(',')}], weight=${mcpc3dsSettings.weightType})...`);
      postProgress(0.41, 'MCPC-3D-S: Preparing data...');

      // Flatten phase and magnitude data for WASM
      const phasesFlat = new Float64Array(nEchoes * voxelCount);
      const magsFlat = new Float64Array(nEchoes * voxelCount);
      for (let e = 0; e < nEchoes; e++) {
        phasesFlat.set(phase4d[e], e * voxelCount);
        magsFlat.set(magnitude4d[e], e * voxelCount);
      }

      postProgress(0.42, 'MCPC-3D-S: Running pipeline...');

      // Call the full MCPC-3D-S + B0 pipeline
      const result = wasmModule.mcpc3ds_b0_pipeline_wasm(
        phasesFlat, magsFlat,
        new Float64Array(echoTimes),
        mask,
        nx, ny, nz,
        mcpc3dsSettings.sigma[0], mcpc3dsSettings.sigma[1], mcpc3dsSettings.sigma[2],
        mcpc3dsSettings.weightType
      );

      postProgress(0.44, 'MCPC-3D-S: Extracting B0...');

      // Extract B0 from result (first n_total elements)
      b0Fieldmap = new Float64Array(result.slice(0, voxelCount));

      // Could also extract phase_offset (next n_total) and corrected_phases (remaining)
      // for debugging/visualization if needed

      postLog(`  MCPC-3D-S B0 calculation complete`);

    } else {
      // Standard workflow: unwrap each echo independently, then fit B0
      let allUnwrapped;
      if (nEchoes > 1) {
        allUnwrapped = new Float64Array(voxelCount * nEchoes);
        allUnwrapped.set(unwrappedPhase, 0);

        postLog(`  Unwrapping ${nEchoes} echoes...`);
        const [seedI, seedJ, seedK] = findSeedPoint(mask, nx, ny, nz);

        for (let e = 1; e < nEchoes; e++) {
          postProgress(0.40 + (e / nEchoes) * 0.05, `Unwrapping echo ${e + 1}/${nEchoes}...`);

          if (unwrapMethod === 'laplacian') {
            const phaseE = new Float64Array(phase4d[e]);
            const unwrappedE = new Float64Array(wasmModule.laplacian_unwrap_wasm(
              phaseE, mask, nx, ny, nz, vsx, vsy, vsz
            ));
            allUnwrapped.set(unwrappedE, e * voxelCount);
          } else {
            // ROMEO for each echo
            const magE = new Float64Array(magnitude4d[e]);
            const phaseE = new Float64Array(phase4d[e]);
            const phaseNext = (e + 1 < nEchoes) ? new Float64Array(phase4d[e + 1]) : new Float64Array(0);
            const teE = echoTimes[e];
            const teNext = (e + 1 < nEchoes) ? echoTimes[e + 1] : 0;

            const weightsE = wasmModule.calculate_weights_romeo_wasm(
              phaseE, magE, phaseNext, teE, teNext, mask, nx, ny, nz
            );

            const unwrappedE = new Float64Array(phaseE);
            const workMaskE = new Uint8Array(mask);
            wasmModule.grow_region_unwrap_wasm(
              unwrappedE, weightsE, workMaskE,
              nx, ny, nz, seedI, seedJ, seedK
            );
            allUnwrapped.set(unwrappedE, e * voxelCount);
          }
        }

        // Align echoes globally to remove 2π ambiguities between independent unwrappings
        postLog("  Aligning echoes to remove 2π ambiguities...");
        for (let e = 1; e < nEchoes; e++) {
          const teRatio = echoTimes[e] / echoTimes[0];

          // Calculate mean difference from expected (based on first echo)
          let sumDiff = 0;
          let count = 0;
          for (let i = 0; i < voxelCount; i++) {
            if (mask[i]) {
              const expected = unwrappedPhase[i] * teRatio;
              const actual = allUnwrapped[e * voxelCount + i];
              sumDiff += (actual - expected);
              count++;
            }
          }
          const meanDiff = sumDiff / count;

          // Round to nearest multiple of 2π and correct
          const correction = Math.round(meanDiff / (2 * Math.PI)) * (2 * Math.PI);
          if (Math.abs(correction) > 0.1) {
            postLog(`    Echo ${e + 1}: correcting by ${(correction / Math.PI).toFixed(2)}π`);
            for (let i = 0; i < voxelCount; i++) {
              if (mask[i]) {
                allUnwrapped[e * voxelCount + i] -= correction;
              }
            }
          }
        }
      } else {
        allUnwrapped = unwrappedPhase;
      }

      // Compute B0 fieldmap using OLS
      const useOffset = multiEchoMethod === 'ols_offset';
      const b0FitMethod = useOffset ? 'ols_offset' : 'ols';
      postLog(`Computing B0 field map (${useOffset ? 'estimating phase offset' : 'assuming zero offset'})...`);
      b0Fieldmap = computeB0FromUnwrapped(allUnwrapped, echoTimes, nx, ny, nz, b0FitMethod);
    }

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
      // Use prepared magnitude if available (combined/bias-corrected), otherwise first echo
      const magnitudeData = hasPreparedMagnitude
        ? new Float64Array(preparedMagnitude)
        : new Float64Array(magnitude4d[0]);
      const magSource = hasPreparedMagnitude ? 'prepared' : 'first echo';
      postLog(`MEDI using ${magSource} magnitude for gradient weighting`);

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
    } else if (dipoleMethod === 'ilsqr') {
      // iLSQR (iterative LSQR with streaking artifact removal)
      const ilsqrProgress = (current, total) => {
        const progress = 0.67 + (current / total) * 0.25;
        postProgress(progress, `iLSQR: Step ${current}/${total}`);
      };

      postProgress(0.70, 'iLSQR: Running 4-step artifact removal...');
      postLog(`iLSQR parameters: tol=${ilsqrSettings.tol}, maxIter=${ilsqrSettings.maxIter}`);

      qsmResult = new Float64Array(wasmModule.ilsqr_wasm_with_progress(
        localField, erodedMask, nx, ny, nz, vsx, vsy, vsz,
        0, 0, 1,  // B0 direction
        ilsqrSettings.tol, ilsqrSettings.maxIter,
        ilsqrProgress
      ));
    } else {
      throw new Error(`Unknown dipole inversion method: '${dipoleMethod}'. Valid options are: tkd, tsvd, tikhonov, tv, rts, nltv, medi, ilsqr`);
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

// TGV single-step pipeline
async function runTgvPipeline(data) {
  const {
    magnitudeBuffers, phaseBuffers, echoTimes, magField,
    maskThreshold, customMaskBuffer, preparedMagnitude, pipelineSettings
  } = data;

  const thresholdFraction = (maskThreshold || 15) / 100;
  const hasCustomMask = customMaskBuffer !== null && customMaskBuffer !== undefined;
  const hasPreparedMagnitude = preparedMagnitude !== null && preparedMagnitude !== undefined;

  // TGV settings
  const tgvSettings = pipelineSettings?.tgv || { regularization: 2, iterations: 1000, erosions: 3 };
  const alphas = wasmModule.tgv_get_default_alpha(tgvSettings.regularization);
  const alpha0 = alphas[0];
  const alpha1 = alphas[1];

  try {
    // =========================================================================
    // Step 1: Load NIfTI data (0% - 15%)
    // =========================================================================
    postProgress(0.02, 'Loading NIfTI data...');
    postLog("TGV: Loading data via WASM...");

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
    // Step 2: Create or load mask (15% - 20%)
    // =========================================================================
    postProgress(0.15, 'Creating mask...');
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
      // Use prepared magnitude if available (combined/bias-corrected), otherwise first echo
      const maskMagnitude = hasPreparedMagnitude
        ? new Float64Array(preparedMagnitude)
        : new Float64Array(magnitude4d[0]);
      const magSource = hasPreparedMagnitude ? 'prepared' : 'first echo';
      postLog(`Creating threshold mask (${thresholdFraction * 100}%) from ${magSource} magnitude...`);
      mask = createThresholdMask(maskMagnitude, thresholdFraction);
    }

    const maskCount = mask.reduce((a, b) => a + b, 0);
    postLog(`Mask coverage: ${maskCount}/${voxelCount} voxels (${(100 * maskCount / voxelCount).toFixed(1)}%)`);

    // Note: magnitude, phase, and mask are available from the app's uploaded files
    // and mask editing UI. We don't need to send them back - only send computed results.

    // =========================================================================
    // Step 3: TGV reconstruction (20% - 95%)
    // =========================================================================
    postProgress(0.20, 'Starting TGV reconstruction...');
    postLog(`TGV parameters: alpha0=${alpha0.toFixed(4)}, alpha1=${alpha1.toFixed(4)}, iterations=${tgvSettings.iterations}, erosions=${tgvSettings.erosions}`);

    // Use first echo phase for TGV
    const phase1 = new Float64Array(phase4d[0]);
    const te = echoTimes[0] / 1000;  // Convert ms to seconds
    const fieldstrength = magField || 3.0;

    postLog(`Using TE=${(te * 1000).toFixed(2)}ms, B0=${fieldstrength}T`);

    // Progress callback
    const tgvProgress = (current, total) => {
      const progress = 0.20 + (current / total) * 0.70;
      postProgress(progress, `TGV: Iteration ${current}/${total}`);
    };

    const qsmResult = new Float64Array(wasmModule.tgv_qsm_wasm_with_progress(
      phase1, mask, nx, ny, nz, vsx, vsy, vsz,
      0, 0, 1,  // B0 direction
      alpha0, alpha1,
      tgvSettings.iterations, tgvSettings.erosions,
      te, fieldstrength,
      tgvProgress
    ));

    // Calculate QSM range
    let qsmMin = Infinity, qsmMax = -Infinity;
    for (let i = 0; i < voxelCount; i++) {
      if (mask[i] && qsmResult[i] !== 0) {
        if (qsmResult[i] < qsmMin) qsmMin = qsmResult[i];
        if (qsmResult[i] > qsmMax) qsmMax = qsmResult[i];
      }
    }
    postLog(`QSM range: [${qsmMin.toFixed(4)}, ${qsmMax.toFixed(4)}] ppm`);

    // Send QSM result for display
    postProgress(0.95, 'Sending QSM result...');
    sendStageData('final', qsmResult, dims, voxelSize, affine, 'QSM Result (ppm) - TGV');

    postProgress(1.0, 'TGV pipeline complete!');
    postLog("TGV pipeline completed successfully!");
    postComplete({ success: true });

  } catch (error) {
    postError(error.message);
    throw error;
  }
}

// QSMART two-stage pipeline
async function runQsmartPipeline(data) {
  const {
    magnitudeBuffers, phaseBuffers, echoTimes, magField,
    maskThreshold, customMaskBuffer, preparedMagnitude, pipelineSettings
  } = data;

  const thresholdFraction = (maskThreshold || 15) / 100;
  const hasCustomMask = customMaskBuffer !== null && customMaskBuffer !== undefined;
  const hasPreparedMagnitude = preparedMagnitude !== null && preparedMagnitude !== undefined;

  // QSMART settings with defaults from Demo_QSMART.m
  const qsmartSettings = pipelineSettings?.qsmart || {};
  const sdfSigma1Stage1 = qsmartSettings.sdfSigma1Stage1 ?? 10;
  const sdfSigma2Stage1 = qsmartSettings.sdfSigma2Stage1 ?? 0;
  const sdfSigma1Stage2 = qsmartSettings.sdfSigma1Stage2 ?? 8;
  const sdfSigma2Stage2 = qsmartSettings.sdfSigma2Stage2 ?? 2;
  const sdfSpatialRadius = qsmartSettings.sdfSpatialRadius ?? 8;
  const sdfLowerLim = qsmartSettings.sdfLowerLim ?? 0.6;
  const sdfCurvConstant = qsmartSettings.sdfCurvConstant ?? 500;
  // Curvature disabled by default for now - our implementation differs from reference
  // which uses Delaunay triangulation + mesh-based curvature
  const useCurvature = qsmartSettings.useCurvature === true;
  // Vasculature sphere radius - can be specified in mm or voxels
  // Default 8mm - the reference uses 8 voxels tuned for ~1mm isotropic data
  const vasculatureSphereRadiusMm = qsmartSettings.vascSphereRadiusMm ?? 8.0;  // mm
  // Legacy support: direct voxel-based radius (will be overridden by mm-based if not set)
  const vasculatureSphereRadiusOverride = qsmartSettings.vascSphereRadius ?? qsmartSettings.vasculatureSphereRadius;
  // Frangi scale parameters - can be specified in mm (physical) or voxels
  // Default reference values are in mm, tuned for typical brain vessel sizes
  const frangiScaleMinMm = qsmartSettings.frangiScaleMinMm ?? 1.0;  // mm - minimum vessel radius (QSMART default: 1)
  const frangiScaleMaxMm = qsmartSettings.frangiScaleMaxMm ?? 10.0;  // mm - maximum vessel radius (QSMART default: 10)
  const frangiScaleRatioMm = qsmartSettings.frangiScaleRatioMm ?? 2.0;  // mm - step between scales (QSMART default: 2)
  // Legacy support: direct voxel-based scales (will be overridden by mm-based if not set)
  const frangiScaleMinVoxelOverride = qsmartSettings.frangiScaleRange?.[0] ?? qsmartSettings.frangiScaleMin;
  const frangiScaleMaxVoxelOverride = qsmartSettings.frangiScaleRange?.[1] ?? qsmartSettings.frangiScaleMax;
  const frangiScaleRatioOverride = qsmartSettings.frangiScaleRatio;
  const frangiC = qsmartSettings.frangiC ?? 500;
  const ilsqrTol = qsmartSettings.ilsqrTol ?? 0.01;
  const ilsqrMaxIter = qsmartSettings.ilsqrMaxIter ?? 50;
  const enableVasculature = qsmartSettings.enableVasculature !== false;

  // Compute ppm conversion factor (gyro * B0 / 1e6)
  const gyro = 2.675e8;  // Proton gyromagnetic ratio rad/s/T
  const b0Tesla = magField || 7.0;  // QSMART optimized for 7T
  const ppmFactor = gyro * b0Tesla / 1e6;

  try {
    // =========================================================================
    // Step 1: Load NIfTI data (0% - 10%)
    // =========================================================================
    postProgress(0.02, 'Loading NIfTI data...');
    postLog("QSMART: Loading multi-echo data via WASM...");

    const nEchoes = echoTimes.length;
    let magnitude4d = [];
    let phase4d = [];
    let dims, voxelSize, affine;

    for (let e = 0; e < nEchoes; e++) {
      postProgress(0.02 + (e / nEchoes) * 0.06, `Loading echo ${e + 1}/${nEchoes}...`);

      const magResult = wasmModule.load_nifti_wasm(new Uint8Array(magnitudeBuffers[e]));
      magnitude4d.push(Array.from(magResult.data));
      dims = Array.from(magResult.dims);
      voxelSize = Array.from(magResult.voxelSize);
      affine = Array.from(magResult.affine);

      const phaseResult = wasmModule.load_nifti_wasm(new Uint8Array(phaseBuffers[e]));
      let phaseData = scalePhase(new Float64Array(phaseResult.data));
      phase4d.push(Array.from(phaseData));

      postLog(`  Echo ${e + 1}: shape ${dims[0]}x${dims[1]}x${dims[2]}`);
    }

    const [nx, ny, nz] = dims;
    const [vsx, vsy, vsz] = voxelSize;
    const voxelCount = nx * ny * nz;

    postLog(`Data: ${nx}x${ny}x${nz}, voxel: ${vsx.toFixed(2)}x${vsy.toFixed(2)}x${vsz.toFixed(2)}mm, B0=${b0Tesla}T`);

    // =========================================================================
    // Step 2: Create or load mask (10% - 12%)
    // =========================================================================
    postProgress(0.10, 'Creating mask...');
    let mask;

    if (hasCustomMask) {
      postLog("Loading custom mask...");
      const maskResult = wasmModule.load_nifti_wasm(new Uint8Array(customMaskBuffer));
      mask = new Uint8Array(voxelCount);
      for (let i = 0; i < voxelCount; i++) {
        mask[i] = maskResult.data[i] > 0.5 ? 1 : 0;
      }
    } else {
      const maskMagnitude = hasPreparedMagnitude
        ? new Float64Array(preparedMagnitude)
        : new Float64Array(magnitude4d[0]);
      mask = createThresholdMask(maskMagnitude, thresholdFraction);
    }

    const maskCount = mask.reduce((a, b) => a + b, 0);
    postLog(`Mask: ${maskCount}/${voxelCount} voxels (${(100 * maskCount / voxelCount).toFixed(1)}%)`);

    // Send mask as intermediate stage
    const maskFloat = new Float64Array(voxelCount);
    for (let i = 0; i < voxelCount; i++) {
      maskFloat[i] = mask[i] ? 1.0 : 0.0;
    }
    sendStageData('mask', maskFloat, dims, voxelSize, affine, 'Brain Mask');

    // =========================================================================
    // Step 3: Phase unwrapping and B0 estimation (12% - 20%)
    // =========================================================================
    postProgress(0.12, 'Phase unwrapping (Laplacian)...');
    postLog("Running Laplacian phase unwrapping...");

    // Unwrap first echo
    const phase1 = new Float64Array(phase4d[0]);
    let unwrappedPhase = new Float64Array(wasmModule.laplacian_unwrap_wasm(
      phase1, mask, nx, ny, nz, vsx, vsy, vsz
    ));

    // Multi-echo B0 estimation
    postProgress(0.16, 'Computing total field shift...');
    let tfs;
    if (nEchoes > 1) {
      // Unwrap all echoes and fit
      const allUnwrapped = new Float64Array(voxelCount * nEchoes);
      allUnwrapped.set(unwrappedPhase, 0);

      for (let e = 1; e < nEchoes; e++) {
        const phaseE = new Float64Array(phase4d[e]);
        const unwrappedE = new Float64Array(wasmModule.laplacian_unwrap_wasm(
          phaseE, mask, nx, ny, nz, vsx, vsy, vsz
        ));
        allUnwrapped.set(unwrappedE, e * voxelCount);
      }

      tfs = computeB0FromUnwrapped(allUnwrapped, echoTimes, nx, ny, nz, 'ols');
    } else {
      // Single echo
      const te = echoTimes[0] / 1000;
      tfs = new Float64Array(voxelCount);
      for (let i = 0; i < voxelCount; i++) {
        tfs[i] = unwrappedPhase[i] / (2 * Math.PI * te);
      }
    }

    // Apply mask
    for (let i = 0; i < voxelCount; i++) {
      if (!mask[i]) tfs[i] = 0;
    }

    // Compute TFS range without spread operator (avoid stack overflow for large arrays)
    let tfsMin = Infinity, tfsMax = -Infinity;
    for (let i = 0; i < voxelCount; i++) {
      if (mask[i]) {
        if (tfs[i] < tfsMin) tfsMin = tfs[i];
        if (tfs[i] > tfsMax) tfsMax = tfs[i];
      }
    }
    postLog(`TFS range: [${tfsMin.toFixed(1)}, ${tfsMax.toFixed(1)}] Hz`);

    // Send TFS as intermediate stage
    sendStageData('tfs', tfs, dims, voxelSize, affine, 'Total Field Shift (Hz)');

    // =========================================================================
    // Step 4: Generate vasculature mask (20% - 30%)
    // =========================================================================
    let vascOnly;
    if (enableVasculature) {
      postProgress(0.20, 'Detecting vasculature (Frangi filter)...');

      // Compute parameters in voxels from mm-based parameters
      // Use average voxel size for isotropic-equivalent scaling
      const avgVoxelSize = (vsx + vsy + vsz) / 3.0;

      // Morphological sphere radius: use override if set, otherwise compute from mm
      const sphereRadiusVoxels = vasculatureSphereRadiusOverride ?? Math.round(vasculatureSphereRadiusMm / avgVoxelSize);
      // Ensure minimum radius of 2 voxels for meaningful morphological filtering
      const effectiveSphereRadius = Math.max(sphereRadiusVoxels, 2);

      // Frangi scales: use override values if explicitly set, otherwise compute from mm
      const frangiScaleMin = frangiScaleMinVoxelOverride ?? (frangiScaleMinMm / avgVoxelSize);
      const frangiScaleMax = frangiScaleMaxVoxelOverride ?? (frangiScaleMaxMm / avgVoxelSize);
      const frangiScaleRatio = frangiScaleRatioOverride ?? (frangiScaleRatioMm / avgVoxelSize);

      // Ensure minimum scale ratio to avoid too many iterations
      const effectiveScaleRatio = Math.max(frangiScaleRatio, 0.1);

      postLog(`Generating vasculature mask:`);
      postLog(`  Voxel size: ${vsx.toFixed(2)}x${vsy.toFixed(2)}x${vsz.toFixed(2)}mm (avg=${avgVoxelSize.toFixed(2)}mm)`);
      postLog(`  Sphere radius: ${effectiveSphereRadius} voxels (${vasculatureSphereRadiusMm.toFixed(1)}mm)`);
      postLog(`  Frangi scales: [${frangiScaleMin.toFixed(2)}, ${frangiScaleMax.toFixed(2)}] voxels (step=${effectiveScaleRatio.toFixed(2)})`);
      postLog(`  (Physical: [${frangiScaleMinMm.toFixed(1)}, ${frangiScaleMaxMm.toFixed(1)}]mm, Frangi C: ${frangiC})`);

      const avgMag = hasPreparedMagnitude
        ? new Float64Array(preparedMagnitude)
        : new Float64Array(magnitude4d[0]);

      const vascProgress = (current, total) => {
        postProgress(0.20 + (current / total) * 0.10, `Vasculature: Step ${current}/${total}`);
      };

      vascOnly = new Float64Array(wasmModule.vasculature_mask_wasm_with_progress(
        avgMag, mask, nx, ny, nz,
        effectiveSphereRadius,
        frangiScaleMin, frangiScaleMax, effectiveScaleRatio,
        frangiC,
        vascProgress
      ));

      const vascCount = vascOnly.filter(v => v === 0).length;
      postLog(`Vessel voxels: ${vascCount} (${(100 * vascCount / maskCount).toFixed(1)}% of brain)`);

      // Send vasculature detection result (inverted: 1 = vessel, 0 = tissue)
      const vascDisplay = new Float64Array(voxelCount);
      for (let i = 0; i < voxelCount; i++) {
        vascDisplay[i] = mask[i] ? (1 - vascOnly[i]) : 0;
      }
      sendStageData('vascDetect', vascDisplay, dims, voxelSize, affine, 'Vessel Detection (Frangi)');
    } else {
      postLog("Vasculature detection disabled - using full mask for both stages");
      vascOnly = new Float64Array(voxelCount).fill(1.0);
    }

    // =========================================================================
    // Step 5: Create weighted mask (simplified - use mask directly)
    // =========================================================================
    // Note: Full R_0 reliability computation from echo fitting residuals
    // requires storing all unwrapped phases. For now, use mask directly.
    // The reference QSMART uses R_0 to exclude voxels with poor echo fits,
    // but for most data this doesn't significantly affect results.

    const weightedMask = new Float64Array(voxelCount);
    for (let i = 0; i < voxelCount; i++) {
      weightedMask[i] = mask[i] ? 1.0 : 0.0;
    }

    postLog(`Using full mask (${maskCount} voxels) for SDF`);

    // =========================================================================
    // Step 6: Stage 1 - SDF + iLSQR on whole ROI (30% - 50%)
    // =========================================================================
    postProgress(0.30, 'Stage 1: SDF background removal...');
    postLog(`Stage 1 SDF: sigma1=${sdfSigma1Stage1}, sigma2=${sdfSigma2Stage1}, curvature=${useCurvature}`);

    // All-ones for vasc_only in stage 1
    const onesArray = new Float64Array(voxelCount).fill(1.0);

    const sdfProgress1 = (current, total) => {
      postProgress(0.30 + (current / total) * 0.10, `Stage 1 SDF: ${current}/${total} alphas`);
    };

    const lfsStage1 = new Float64Array(wasmModule.sdf_wasm_with_progress(
      tfs, weightedMask, onesArray,
      nx, ny, nz,
      sdfSigma1Stage1, sdfSigma2Stage1,
      sdfSpatialRadius,
      sdfLowerLim, sdfCurvConstant,
      useCurvature,
      sdfProgress1
    ));

    // Compute LFS range for diagnostics
    let lfs1Min = Infinity, lfs1Max = -Infinity;
    for (let i = 0; i < voxelCount; i++) {
      if (weightedMask[i] > 0) {
        if (lfsStage1[i] < lfs1Min) lfs1Min = lfsStage1[i];
        if (lfsStage1[i] > lfs1Max) lfs1Max = lfsStage1[i];
      }
    }
    postLog(`Stage 1 LFS range: [${lfs1Min.toFixed(2)}, ${lfs1Max.toFixed(2)}] Hz`);

    // Scale to ppm for storage (will scale back for offset adjustment)
    const lfsStage1Ppm = new Float64Array(voxelCount);
    for (let i = 0; i < voxelCount; i++) {
      lfsStage1Ppm[i] = lfsStage1[i] * ppmFactor;
    }

    // Send stage 1 local field for display
    sendStageData('lfsStage1', lfsStage1, dims, voxelSize, affine, 'Stage 1 Local Field (Hz)');

    postProgress(0.42, 'Stage 1: iLSQR inversion...');
    postLog(`Stage 1 iLSQR: tol=${ilsqrTol}, maxIter=${ilsqrMaxIter}`);

    // Create binary mask for iLSQR
    const maskStage1 = new Uint8Array(voxelCount);
    for (let i = 0; i < voxelCount; i++) {
      maskStage1[i] = weightedMask[i] > 0.1 ? 1 : 0;
    }

    const ilsqrProgress1 = (current, total) => {
      postProgress(0.42 + (current / total) * 0.08, `Stage 1 iLSQR: Step ${current}/${total}`);
    };

    const chiStage1 = new Float64Array(wasmModule.ilsqr_wasm_with_progress(
      lfsStage1, maskStage1, nx, ny, nz, vsx, vsy, vsz,
      0, 0, 1,  // B0 direction
      ilsqrTol, ilsqrMaxIter,
      ilsqrProgress1
    ));

    // Compute chi1 range for diagnostics
    let chi1Min = Infinity, chi1Max = -Infinity;
    for (let i = 0; i < voxelCount; i++) {
      if (maskStage1[i]) {
        if (chiStage1[i] < chi1Min) chi1Min = chiStage1[i];
        if (chiStage1[i] > chi1Max) chi1Max = chiStage1[i];
      }
    }
    postLog(`Stage 1 Chi range: [${chi1Min.toFixed(4)}, ${chi1Max.toFixed(4)}]`);

    sendStageData('chiStage1', chiStage1, dims, voxelSize, affine, 'Stage 1 QSM (arb)');

    // =========================================================================
    // Step 7: Stage 2 - SDF + iLSQR on tissue only (50% - 75%)
    // =========================================================================
    postProgress(0.50, 'Stage 2: SDF on tissue region...');
    postLog(`Stage 2 SDF: sigma1=${sdfSigma1Stage2}, sigma2=${sdfSigma2Stage2}`);

    // Weighted TFS for stage 2: tfs * mask * R_0
    const tfsWeighted = new Float64Array(voxelCount);
    for (let i = 0; i < voxelCount; i++) {
      tfsWeighted[i] = tfs[i] * weightedMask[i];
    }

    const sdfProgress2 = (current, total) => {
      postProgress(0.50 + (current / total) * 0.12, `Stage 2 SDF: ${current}/${total} alphas`);
    };

    const lfsStage2 = new Float64Array(wasmModule.sdf_wasm_with_progress(
      tfsWeighted, weightedMask, vascOnly,
      nx, ny, nz,
      sdfSigma1Stage2, sdfSigma2Stage2,
      sdfSpatialRadius,
      sdfLowerLim, sdfCurvConstant,
      useCurvature,
      sdfProgress2
    ));

    const lfsStage2Ppm = new Float64Array(voxelCount);
    for (let i = 0; i < voxelCount; i++) {
      lfsStage2Ppm[i] = lfsStage2[i] * ppmFactor;
    }

    sendStageData('lfsStage2', lfsStage2, dims, voxelSize, affine, 'Stage 2 Local Field (Hz)');

    postProgress(0.64, 'Stage 2: iLSQR inversion...');

    // Mask for stage 2: mask * vasc_only * R_0
    const maskStage2 = new Uint8Array(voxelCount);
    for (let i = 0; i < voxelCount; i++) {
      maskStage2[i] = (weightedMask[i] > 0.1 && vascOnly[i] > 0.5) ? 1 : 0;
    }

    const ilsqrProgress2 = (current, total) => {
      postProgress(0.64 + (current / total) * 0.10, `Stage 2 iLSQR: Step ${current}/${total}`);
    };

    const chiStage2 = new Float64Array(wasmModule.ilsqr_wasm_with_progress(
      lfsStage2, maskStage2, nx, ny, nz, vsx, vsy, vsz,
      0, 0, 1,
      ilsqrTol, ilsqrMaxIter,
      ilsqrProgress2
    ));

    sendStageData('chiStage2', chiStage2, dims, voxelSize, affine, 'Stage 2 QSM (arb)');

    // =========================================================================
    // Step 8: Combine stages with offset adjustment (75% - 90%)
    // =========================================================================
    postProgress(0.75, 'Combining stages with offset adjustment...');
    postLog("Computing offset adjustment in Fourier space...");

    // Removed voxels = mask * R_0 - vasc_only
    const removedVoxels = new Float64Array(voxelCount);
    for (let i = 0; i < voxelCount; i++) {
      removedVoxels[i] = weightedMask[i] - vascOnly[i];
    }

    const chiQsmart = new Float64Array(wasmModule.qsmart_adjust_offset_wasm(
      removedVoxels, lfsStage1Ppm, chiStage1, chiStage2,
      nx, ny, nz, vsx, vsy, vsz,
      0, 0, 1,  // B0 direction
      ppmFactor
    ));

    // =========================================================================
    // Step 9: Scale to ppm and finalize (90% - 100%)
    // =========================================================================
    postProgress(0.90, 'Scaling to ppm...');

    // Scale to ppm: χ (ppm) = χ_raw / (γ × B0) × 1e6
    const gamma = 42.576e6; // Hz/T
    const scaleFactor = 1e6 / (gamma * b0Tesla);

    const qsmResult = new Float64Array(voxelCount);
    for (let i = 0; i < voxelCount; i++) {
      qsmResult[i] = chiQsmart[i] * scaleFactor;
      if (!mask[i]) qsmResult[i] = 0;
    }

    // Calculate QSM range
    let qsmMin = Infinity, qsmMax = -Infinity;
    for (let i = 0; i < voxelCount; i++) {
      if (mask[i] && qsmResult[i] !== 0) {
        if (qsmResult[i] < qsmMin) qsmMin = qsmResult[i];
        if (qsmResult[i] > qsmMax) qsmMax = qsmResult[i];
      }
    }
    postLog(`QSMART QSM range: [${qsmMin.toFixed(4)}, ${qsmMax.toFixed(4)}] ppm`);

    // Send final result
    postProgress(0.95, 'Sending QSMART result...');
    sendStageData('final', qsmResult, dims, voxelSize, affine, 'QSMART QSM (ppm)');

    postProgress(1.0, 'QSMART pipeline complete!');
    postLog("QSMART two-stage pipeline completed successfully!");
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

/**
 * Run bias field correction on magnitude data
 * @param {Object} data - Contains magnitude, dimensions, voxel sizes, and parameters
 */
async function runBiasCorrection(data) {
  const { magnitude, nx, ny, nz, vx, vy, vz, sigma_mm, nbox } = data;

  try {
    // Ensure WASM is initialized
    if (!wasmModule) {
      await initializeWasm();
    }

    console.log(`[Worker] Bias correction: ${nx}x${ny}x${nz}, voxel size=${vx.toFixed(2)}x${vy.toFixed(2)}x${vz.toFixed(2)}mm, sigma=${sigma_mm}mm, nbox=${nbox}`);

    const inputArray = new Float64Array(magnitude);
    const inputSum = inputArray.reduce((a, b) => a + b, 0);
    console.log(`[Worker] Input data length: ${inputArray.length}, sum: ${inputSum.toExponential(3)}`);

    // Call WASM bias correction
    const result = wasmModule.makehomogeneous_wasm(
      inputArray,
      nx, ny, nz,
      vx, vy, vz,
      sigma_mm,
      nbox
    );

    const resultSum = result.reduce((a, b) => a + b, 0);
    console.log(`[Worker] Bias correction complete, result sum: ${resultSum.toExponential(3)}`);

    // Send result back
    self.postMessage({
      type: 'biasCorrection',
      result: Array.from(result)
    });

  } catch (error) {
    console.error('[Worker] Bias correction error:', error);
    self.postMessage({
      type: 'biasCorrection',
      error: error.message
    });
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

      case 'biasCorrection':
        await runBiasCorrection(data);
        break;


      default:
        postError(`Unknown message type: ${type}`);
    }
  } catch (error) {
    postError(error.message);
    console.error(error);
  }
};
