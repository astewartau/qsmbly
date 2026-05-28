/**
 * Config Bridge — converts qsmbly pipeline settings to TOML
 * and calls qsmxt-config WASM (via worker) for command/methods generation.
 *
 * Replaces CommandGenerator.js with a single source of truth
 * shared between qsmxt.rs and qsmbly.
 */

/**
 * Convert qsmbly pipeline settings to TOML string.
 * @param {Object} settings - Pipeline settings from PipelineSettingsController.save()
 * @param {string[]} maskOps - Mask operations history
 * @param {string} maskSource - Mask input source
 * @param {Object} options - { doSwi, doT2star, doR2star }
 * @returns {string} TOML config string
 */
export function settingsToToml(settings, maskOps = [], maskSource = 'phase_quality', options = {}) {
  if (!settings) return '';

  const isTgv = settings.combinedMethod === 'tgv';
  const isQsmart = settings.combinedMethod === 'qsmart';

  const config = {
    pipeline: {
      do_qsm: true,
      do_swi: !!options.doSwi,
      do_t2starmap: !!options.doT2star,
      do_r2starmap: !!options.doR2star,
    },
    field_mapping: {
      phase_offset_removal: settings.phaseOffsetMethod !== 'none',
      phase_offset_sigma: settings.mcpc3ds?.sigma || [4, 4, 4],
      bipolar_correction: !!settings.bipolarCorrection,
      unwrapping_algorithm: settings.unwrapMethod || 'romeo',
      b0_estimation: (settings.fieldCalculationMethod || 'weighted_avg').replace(/_/g, '-'),
      b0_weight_type: (settings.b0WeightType || 'phase_snr').replace(/_/g, '-'),
      romeo: {
        individual: settings.romeo?.individual ?? true,
        correct_global: settings.romeo?.correctGlobal ?? true,
        template: settings.romeo?.template ?? 0,
        phase_gradient_coherence: settings.romeo?.phaseGradientCoherence ?? true,
        mag_coherence: settings.romeo?.magCoherence ?? true,
        mag_weight: settings.romeo?.magWeight ?? false,
      },
    },
    masking: { inhomogeneity_correction: true },
    bg_removal: { algorithm: settings.backgroundRemoval || 'vsharp' },
    inversion: {},
    qsm: { reference: settings.referenceMean === false ? 'none' : 'mean' },
  };

  if (isTgv) config.inversion.algorithm = 'tgv';
  else if (isQsmart) config.inversion.algorithm = 'qsmart';
  else config.inversion.algorithm = settings.dipoleInversion || 'rts';

  // Algorithm params
  if (settings.rts) config.inversion.rts = settings.rts;
  if (settings.tv) config.inversion.tv = settings.tv;
  if (settings.tkd) config.inversion.tkd = settings.tkd;
  if (settings.tsvd) config.inversion.tsvd = settings.tsvd;
  if (settings.tikhonov) config.inversion.tikhonov = settings.tikhonov;
  if (settings.nltv) config.inversion.nltv = settings.nltv;
  if (settings.medi) config.inversion.medi = settings.medi;
  if (settings.ilsqr) config.inversion.ilsqr = settings.ilsqr;
  if (settings.tgv) config.inversion.tgv = {
    iterations: settings.tgv.iterations, erosions: settings.tgv.erosions,
    alpha0: settings.tgv.alpha0, alpha1: settings.tgv.alpha1,
    step_size: settings.tgv.stepSize, tol: settings.tgv.tol,
  };
  if (settings.qsmart) config.inversion.qsmart = {
    ilsqr_tol: settings.qsmart.ilsqrTol, ilsqr_max_iter: settings.qsmart.ilsqrMaxIter,
    vasc_sphere_radius: settings.qsmart.vascSphereRadiusMm, sdf_spatial_radius: settings.qsmart.sdfSpatialRadius,
  };

  // BG removal params
  if (settings.vsharp) config.bg_removal.vsharp = settings.vsharp;
  if (settings.pdf) config.bg_removal.pdf = settings.pdf;
  if (settings.lbv) config.bg_removal.lbv = settings.lbv;
  if (settings.ismv) config.bg_removal.ismv = { tol: settings.ismv.tol, max_iter: settings.ismv.maxit, radius_factor: settings.ismv.radius };
  if (settings.sharp) config.bg_removal.sharp = { threshold: settings.sharp.threshold, radius_factor: settings.sharp.radius };
  if (settings.resharp) config.bg_removal.resharp = settings.resharp;
  if (settings.harperella) config.bg_removal.harperella = settings.harperella;
  if (settings.iharperella) config.bg_removal.iharperella = settings.iharperella;

  if (settings.swi) config.swi = {
    hp_sigma: settings.swi.hpSigma, scaling: settings.swi.scaling,
    strength: settings.swi.strength, mip_window: settings.swi.mipWindow,
  };

  return toTomlString(config);
}

/**
 * Append mask CLI flags to a command string.
 * (Mask sections use complex serde tagged enums that can't easily go through TOML.)
 */
export function appendMaskFlags(cmd, maskOps, maskSource) {
  const DEFAULT_MASK_OPS = ['threshold:otsu', 'dilate:1', 'fill-holes:0', 'erode:1'];
  if (maskOps && maskOps.length > 0) {
    const inputMap = {
      'phase_quality': 'phase-quality', 'combined': 'magnitude',
      'first_echo': 'magnitude-first', 'last_echo': 'magnitude-last',
    };
    const input = inputMap[maskSource] || 'phase-quality';
    const defaultSection = `phase-quality,${DEFAULT_MASK_OPS.join(',')}`;
    const currentSection = `${input},${maskOps.join(',')}`;
    if (currentSection !== defaultSection) {
      cmd += ` --mask ${currentSection}`;
    }
  }
  return cmd;
}

/**
 * Minimal TOML serializer for nested config objects.
 */
function toTomlString(obj, prefix = '') {
  let lines = [];
  const tables = [];

  for (const [key, value] of Object.entries(obj)) {
    if (value === null || value === undefined) continue;
    const fullKey = prefix ? `${prefix}.${key}` : key;

    if (Array.isArray(value)) {
      if (value.length > 0 && typeof value[0] === 'object') {
        for (const item of value) {
          lines.push(`\n[[${fullKey}]]`);
          lines.push(toTomlString(item, '').trim());
        }
      } else {
        const formatted = value.map(v => typeof v === 'string' ? `"${v}"` : v);
        lines.push(`${key} = [${formatted.join(', ')}]`);
      }
    } else if (typeof value === 'object') {
      tables.push({ key: fullKey, value });
    } else if (typeof value === 'string') {
      lines.push(`${key} = "${value}"`);
    } else if (typeof value === 'boolean') {
      lines.push(`${key} = ${value}`);
    } else if (typeof value === 'number') {
      lines.push(`${key} = ${value}`);
    }
  }

  for (const { key, value } of tables) {
    lines.push(`\n[${key}]`);
    lines.push(toTomlString(value, key).trim());
  }

  return lines.join('\n') + '\n';
}
