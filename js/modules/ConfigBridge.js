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

  const isTgv = settings.combined_method === 'tgv';
  const isQsmart = settings.combined_method === 'qsmart';

  const config = {
    pipeline: {
      do_qsm: true,
      do_swi: !!options.doSwi,
      do_t2starmap: !!options.doT2star,
      do_r2starmap: !!options.doR2star,
    },
    field_mapping: {
      phase_offset_removal: settings.phase_offset_method !== 'none',
      phase_offset_sigma: settings.mcpc3ds?.sigma || [4, 4, 4],
      bipolar_correction: !!settings.bipolar_correction,
      unwrapping_algorithm: settings.unwrapping_algorithm || 'romeo',
      b0_estimation: (settings.b0_estimation || 'weighted_avg').replace(/_/g, '-'),
      b0_weight_type: (settings.b0_weight_type || 'phase_snr').replace(/_/g, '-'),
      romeo: {
        individual: settings.romeo?.individual ?? true,
        correct_global: settings.romeo?.correct_global ?? true,
        template: settings.romeo?.template ?? 0,
        phase_gradient_coherence: settings.romeo?.phase_gradient_coherence ?? true,
        mag_coherence: settings.romeo?.mag_coherence ?? true,
        mag_weight: settings.romeo?.mag_weight ?? false,
      },
    },
    masking: { inhomogeneity_correction: true },
    bg_removal: { algorithm: settings.bf_algorithm || 'vsharp' },
    inversion: {},
    qsm: { reference: settings.reference_mean === false ? 'none' : 'mean' },
  };

  if (isTgv) config.inversion.algorithm = 'tgv';
  else if (isQsmart) config.inversion.algorithm = 'qsmart';
  else config.inversion.algorithm = settings.dipole_inversion || 'rts';

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
    step_size: settings.tgv.step_size, tol: settings.tgv.tol,
  };
  if (settings.qsmart) config.inversion.qsmart = {
    ilsqr_tol: settings.qsmart.ilsqr_tol, ilsqr_max_iter: settings.qsmart.ilsqr_max_iter,
    vasc_sphere_radius: settings.qsmart.vasc_sphere_radius, sdf_spatial_radius: settings.qsmart.sdf_spatial_radius,
    inversion: settings.qsmart.inversion_algorithm || 'ilsqr',
    sdf_sigma1_stage1: settings.qsmart.sdf_sigma1_stage1, sdf_sigma2_stage1: settings.qsmart.sdf_sigma2_stage1,
    sdf_sigma1_stage2: settings.qsmart.sdf_sigma1_stage2, sdf_sigma2_stage2: settings.qsmart.sdf_sigma2_stage2,
    sdf_lower_lim: settings.qsmart.sdf_lower_lim, sdf_curv_constant: settings.qsmart.sdf_curv_constant,
    frangi_scale_min: settings.qsmart.frangi_scale_min, frangi_scale_max: settings.qsmart.frangi_scale_max,
    frangi_scale_ratio: settings.qsmart.frangi_scale_ratio, frangi_c: settings.qsmart.frangi_c,
  };

  // BG removal params
  if (settings.vsharp) config.bg_removal.vsharp = settings.vsharp;
  if (settings.pdf) config.bg_removal.pdf = settings.pdf;
  if (settings.lbv) config.bg_removal.lbv = settings.lbv;
  if (settings.ismv) config.bg_removal.ismv = { tol: settings.ismv.tol, max_iter: settings.ismv.max_iter, radius_factor: settings.ismv.radius };
  if (settings.sharp) config.bg_removal.sharp = { threshold: settings.sharp.threshold, radius_factor: settings.sharp.radius_factor };
  if (settings.resharp) config.bg_removal.resharp = settings.resharp;
  if (settings.harperella) config.bg_removal.harperella = settings.harperella;
  if (settings.iharperella) config.bg_removal.iharperella = settings.iharperella;

  if (settings.swi) config.swi = {
    // scaling uses snake_case in the UI (e.g. negative_tanh) but qsmxt-config expects
    // kebab-case (negative-tanh), same as b0_estimation/b0_weight_type above.
    hp_sigma: settings.swi.hp_sigma, scaling: (settings.swi.scaling || 'tanh').replace(/_/g, '-'),
    strength: settings.swi.strength, mip_window: settings.swi.mip_window,
  };

  return toTomlString(config);
}

/**
 * Append mask CLI flags to a command string.
 * (Mask sections use complex serde tagged enums that can't easily go through TOML.)
 */
/**
 * Default mask ops from qsmxt-config (robust threshold preset).
 * Set dynamically from WASM default config; falls back to hardcoded.
 */
let DEFAULT_MASK_OPS = ['threshold:otsu', 'dilate:1', 'fill-holes:0', 'erode:1'];

/**
 * Update default mask ops from a parsed qsmxt-config default config JSON.
 */
export function setDefaultsFromConfig(config) {
  if (config?.masking?.sections?.[0]) {
    const s = config.masking.sections[0];
    const ops = [];
    if (s.generator) ops.push(maskOpToString(s.generator));
    if (s.refinements) s.refinements.forEach(r => ops.push(maskOpToString(r)));
    if (ops.length > 0) DEFAULT_MASK_OPS = ops;
  }
}

function maskOpToString(op) {
  switch (op.op) {
    case 'threshold': return `threshold:${op.method || 'otsu'}`;
    case 'bet': return `bet:${op.fractional_intensity ?? 0.5}`;
    case 'erode': return `erode:${op.iterations ?? 1}`;
    case 'dilate': return `dilate:${op.iterations ?? 1}`;
    case 'close': return `close:${op.radius ?? 1}`;
    case 'fill-holes': return `fill-holes:${op.max_size ?? 0}`;
    case 'gaussian-smooth': return `gaussian:${op.sigma_mm ?? 4.0}`;
    default: return op.op;
  }
}

export function appendMaskFlags(cmd, maskOps, maskSource) {
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
