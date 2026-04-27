/**
 * Generate a qsmxt.rs CLI command from qsmbly pipeline settings.
 * Only emits flags that differ from defaults (imported from config.js).
 */

import {
  PIPELINE_DEFAULTS,
  RTS_DEFAULTS, TV_DEFAULTS, TKD_DEFAULTS, TSVD_DEFAULTS,
  TIKHONOV_DEFAULTS, NLTV_DEFAULTS, MEDI_DEFAULTS,
  VSHARP_DEFAULTS, PDF_DEFAULTS, LBV_DEFAULTS, ISMV_DEFAULTS, SHARP_DEFAULTS,
  TGV_DEFAULTS, QSMART_DEFAULTS,
  ROMEO_DEFAULTS, MCPC3DS_DEFAULTS,
  SWI_DEFAULTS,
} from '../app/config.js';

// qsmxt.rs default mask ops (robust threshold)
const DEFAULT_MASK_OPS = ['threshold:otsu', 'dilate:2', 'fill-holes:0', 'erode:2'];

/**
 * @param {Object} settings - Pipeline settings from PipelineSettingsController.save()
 * @param {string[]} maskOps - Mask operations history
 * @param {Object} options - Additional options (doSwi, doT2star, doR2star)
 * @returns {string} The qsmxt.rs CLI command
 */
export function generateQsmxtCommand(settings, maskOps = [], options = {}) {
  const parts = ['qsmxt', 'run', '<bids_dir>', '<output_dir>'];
  const d = PIPELINE_DEFAULTS;

  if (!settings) return parts.join(' ');

  // --- QSM Algorithm ---
  const isTgv = settings.combinedMethod === 'tgv';
  const isQsmart = settings.combinedMethod === 'qsmart';

  if (isTgv) {
    emit(parts, '--qsm-algorithm', 'tgv', d.dipoleInversion);
  } else if (isQsmart) {
    emit(parts, '--qsm-algorithm', 'qsmart', d.dipoleInversion);
  } else {
    emit(parts, '--qsm-algorithm', settings.dipoleInversion, d.dipoleInversion);
  }

  // --- Unwrapping ---
  if (!isTgv && !isQsmart) {
    emit(parts, '--unwrapping-algorithm', settings.unwrapMethod, d.unwrapMethod);

    // ROMEO weight params
    if (settings.romeo?.phaseGradientCoherence === false) {
      parts.push('--no-romeo-phase-gradient-coherence');
    }
    if (settings.romeo?.magCoherence === false) {
      parts.push('--no-romeo-mag-coherence');
    }
    if (settings.romeo?.magWeight === false) {
      parts.push('--no-romeo-mag-weight');
    }
  }

  // --- MCPC-3D-S sigma ---
  if (settings.mcpc3ds?.sigma) {
    const s = settings.mcpc3ds.sigma;
    const ds = MCPC3DS_DEFAULTS.sigma;
    if (s[0] !== ds[0] || s[1] !== ds[1] || s[2] !== ds[2]) {
      parts.push(`--mcpc3ds-sigma ${s[0]} ${s[1]} ${s[2]}`);
    }
  }

  // --- Phase combination ---
  if (settings.fieldCalculationMethod === 'linear_fit' && d.fieldCalculationMethod !== 'linear_fit') {
    parts.push('--combine-phase false');
  }

  // --- Background removal ---
  const isMediSmv = !isTgv && !isQsmart && settings.dipoleInversion === 'medi' && settings.medi?.smv;
  if (!isTgv && !isQsmart && !isMediSmv) {
    emit(parts, '--bf-algorithm', settings.backgroundRemoval, d.backgroundRemoval);

    switch (settings.backgroundRemoval) {
      case 'vsharp':
        emitNum(parts, '--vsharp-threshold', settings.vsharp?.threshold, VSHARP_DEFAULTS.threshold);
        emitNum(parts, '--vsharp-max-radius-factor', settings.vsharp?.maxRadius, VSHARP_DEFAULTS.maxRadiusFactor);
        emitNum(parts, '--vsharp-min-radius-factor', settings.vsharp?.minRadius, VSHARP_DEFAULTS.minRadiusFactor);
        break;
      case 'pdf':
        emitNum(parts, '--pdf-tol', settings.pdf?.tol, PDF_DEFAULTS.tol);
        break;
      case 'lbv':
        emitNum(parts, '--lbv-tol', settings.lbv?.tol, LBV_DEFAULTS.tol);
        break;
      case 'ismv':
        emitNum(parts, '--ismv-tol', settings.ismv?.tol, ISMV_DEFAULTS.tol);
        emitNum(parts, '--ismv-max-iter', settings.ismv?.maxit, ISMV_DEFAULTS.maxit);
        emitNum(parts, '--ismv-radius-factor', settings.ismv?.radius, ISMV_DEFAULTS.radiusFactor);
        break;
      case 'sharp':
        emitNum(parts, '--sharp-threshold', settings.sharp?.threshold, SHARP_DEFAULTS.threshold);
        emitNum(parts, '--sharp-radius-factor', settings.sharp?.radius, SHARP_DEFAULTS.radiusFactor);
        break;
    }
  }

  // --- QSM Reference ---
  if (settings.referenceMean === false) {
    parts.push('--qsm-reference none');
  }

  // --- Algorithm-specific params ---
  if (isTgv) {
    emitNum(parts, '--tgv-iterations', settings.tgv?.iterations, TGV_DEFAULTS.iterations);
    emitNum(parts, '--tgv-erosions', settings.tgv?.erosions, TGV_DEFAULTS.erosions);
    emitNum(parts, '--tgv-alpha0', settings.tgv?.alpha0, TGV_DEFAULTS.alpha0);
    emitNum(parts, '--tgv-alpha1', settings.tgv?.alpha1, TGV_DEFAULTS.alpha1);
    emitNum(parts, '--tgv-step-size', settings.tgv?.stepSize, TGV_DEFAULTS.stepSize);
    emitNum(parts, '--tgv-tol', settings.tgv?.tol, TGV_DEFAULTS.tol);
  } else if (isQsmart) {
    emitNum(parts, '--qsmart-ilsqr-tol', settings.qsmart?.ilsqrTol, QSMART_DEFAULTS.ilsqrTol);
    emitNum(parts, '--qsmart-ilsqr-max-iter', settings.qsmart?.ilsqrMaxIter, QSMART_DEFAULTS.ilsqrMaxIter);
    emitNum(parts, '--qsmart-vasc-sphere-radius', settings.qsmart?.vascSphereRadiusMm, QSMART_DEFAULTS.vascSphereRadiusMm);
    emitNum(parts, '--qsmart-sdf-spatial-radius', settings.qsmart?.sdfSpatialRadius, QSMART_DEFAULTS.sdfSpatialRadius);
  } else {
    const algo = settings.dipoleInversion || d.dipoleInversion;
    switch (algo) {
      case 'rts':
        emitNum(parts, '--rts-delta', settings.rts?.delta, RTS_DEFAULTS.delta);
        emitNum(parts, '--rts-mu', settings.rts?.mu, RTS_DEFAULTS.mu);
        emitNum(parts, '--rts-rho', settings.rts?.rho, RTS_DEFAULTS.rho);
        emitNum(parts, '--rts-tol', settings.rts?.tol, RTS_DEFAULTS.tol);
        emitNum(parts, '--rts-max-iter', settings.rts?.maxIter, RTS_DEFAULTS.maxIter);
        emitNum(parts, '--rts-lsmr-iter', settings.rts?.lsmrIter, RTS_DEFAULTS.lsmrIter);
        break;
      case 'tv':
        emitNum(parts, '--tv-lambda', settings.tv?.lambda, TV_DEFAULTS.lambda);
        emitNum(parts, '--tv-rho', settings.tv?.rho, TV_DEFAULTS.rho);
        emitNum(parts, '--tv-tol', settings.tv?.tol, TV_DEFAULTS.tol);
        emitNum(parts, '--tv-max-iter', settings.tv?.maxIter, TV_DEFAULTS.maxIter);
        break;
      case 'tkd':
        emitNum(parts, '--tkd-threshold', settings.tkd?.threshold, TKD_DEFAULTS.threshold);
        break;
      case 'tsvd':
        emitNum(parts, '--tsvd-threshold', settings.tsvd?.threshold, TSVD_DEFAULTS.threshold);
        break;
      case 'tikhonov':
        emitNum(parts, '--tikhonov-lambda', settings.tikhonov?.lambda, TIKHONOV_DEFAULTS.lambda);
        break;
      case 'ilsqr':
        emitNum(parts, '--ilsqr-tol', settings.ilsqr?.tol, QSMART_DEFAULTS.ilsqrTol);
        emitNum(parts, '--ilsqr-max-iter', settings.ilsqr?.maxIter, QSMART_DEFAULTS.ilsqrMaxIter);
        break;
      case 'nltv':
        emitNum(parts, '--nltv-lambda', settings.nltv?.lambda, NLTV_DEFAULTS.lambda);
        emitNum(parts, '--nltv-mu', settings.nltv?.mu, NLTV_DEFAULTS.mu);
        emitNum(parts, '--nltv-tol', settings.nltv?.tol, NLTV_DEFAULTS.tol);
        emitNum(parts, '--nltv-max-iter', settings.nltv?.maxIter, NLTV_DEFAULTS.maxIter);
        emitNum(parts, '--nltv-newton-iter', settings.nltv?.newtonMaxIter, NLTV_DEFAULTS.newtonMaxIter);
        break;
      case 'medi':
        emitNum(parts, '--medi-lambda', settings.medi?.lambda, MEDI_DEFAULTS.lambda);
        emitNum(parts, '--medi-percentage', settings.medi?.percentage, MEDI_DEFAULTS.percentage);
        emitNum(parts, '--medi-max-iter', settings.medi?.maxIter, MEDI_DEFAULTS.maxIter);
        emitNum(parts, '--medi-cg-max-iter', settings.medi?.cgMaxIter, MEDI_DEFAULTS.cgMaxIter);
        emitNum(parts, '--medi-cg-tol', settings.medi?.cgTol, MEDI_DEFAULTS.cgTol);
        emitNum(parts, '--medi-tol', settings.medi?.tol, MEDI_DEFAULTS.tol);
        emitNum(parts, '--medi-smv-radius', settings.medi?.smvRadius, MEDI_DEFAULTS.smvRadius);
        if (settings.medi?.smv !== MEDI_DEFAULTS.smv) {
          if (settings.medi?.smv) {
            parts.push('--medi-smv');
          }
        }
        break;
    }
  }

  // --- Mask sections ---
  // qsmbly tracks mask ops as a flat list; emit as a single --mask
  if (maskOps && maskOps.length > 0) {
    const defaultSection = `phase-quality,${DEFAULT_MASK_OPS.join(',')}`;
    const currentSection = `phase-quality,${maskOps.join(',')}`;

    if (currentSection !== defaultSection) {
      parts.push(`--mask ${currentSection}`);
    }
  }

  // --- SWI params (only if SWI computed) ---
  if (options.doSwi) {
    parts.push('--do-swi');
    if (settings.swi) {
      const s = settings.swi;
      // hp_sigma
      if (s.hpSigma && SWI_DEFAULTS.hpSigma) {
        const ds = SWI_DEFAULTS.hpSigma;
        if (s.hpSigma[0] !== ds[0] || s.hpSigma[1] !== ds[1] || s.hpSigma[2] !== ds[2]) {
          parts.push(`--swi-hp-sigma ${s.hpSigma[0]} ${s.hpSigma[1]} ${s.hpSigma[2]}`);
        }
      }
      emit(parts, '--swi-scaling', s.scaling, SWI_DEFAULTS.scaling);
      emitNum(parts, '--swi-strength', s.strength, SWI_DEFAULTS.strength);
      emitNum(parts, '--swi-mip-window', s.mipWindow, SWI_DEFAULTS.mipWindow);
    }
  }
  if (options.doT2star) parts.push('--do-t2starmap');
  if (options.doR2star) parts.push('--do-r2starmap');

  return parts.join(' \\\n  ');
}

function emit(parts, flag, value, defaultValue) {
  if (value != null && value !== defaultValue) {
    parts.push(`${flag} ${value}`);
  }
}

function emitNum(parts, flag, value, defaultValue) {
  if (value != null && Number(value) !== Number(defaultValue)) {
    parts.push(`${flag} ${value}`);
  }
}
