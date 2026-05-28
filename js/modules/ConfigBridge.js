/**
 * Config Bridge — converts qsmbly pipeline settings to TOML
 * and calls qsmxt-config WASM for command/methods generation.
 *
 * Replaces CommandGenerator.js with a single source of truth
 * shared between qsmxt.rs and qsmbly.
 */

let wasmModule = null;

/**
 * Initialize the WASM module for config operations.
 * Loads the QSM WASM module in the main thread (lightweight — only used for
 * string-based config/command/methods generation, not data processing).
 */
export async function initConfigBridge() {
  if (wasmModule) return;
  try {
    const wasm = await import('../../wasm/qsm_wasm.js');
    await wasm.default();
    wasmModule = wasm;
  } catch (e) {
    console.warn('ConfigBridge: WASM load failed, command generation unavailable', e);
  }
}

/**
 * Set the WASM module reference directly (alternative to initConfigBridge).
 */
export function setWasmModule(wasm) {
  wasmModule = wasm;
}

/**
 * Generate a qsmxt CLI command from pipeline settings.
 * @param {Object} settings - Pipeline settings from PipelineSettingsController.save()
 * @param {string[]} maskOps - Mask operations history
 * @param {string} maskSource - Mask input source (e.g. 'phase_quality')
 * @param {Object} options - { doSwi, doT2star, doR2star }
 * @returns {string} CLI command string
 */
export function generateCommand(settings, maskOps = [], maskSource = 'phase_quality', options = {}) {
  if (!wasmModule) return 'ERROR: WASM not loaded';
  const toml = settingsToToml(settings, maskOps, maskSource, options);
  let cmd = wasmModule.generate_command_wasm(toml);

  // Append --mask flag if mask differs from default
  // (Mask sections use complex serde tagged enums that are hard to serialize
  // from JS to TOML, so we handle them directly here)
  const DEFAULT_MASK_OPS = ['threshold:otsu', 'dilate:1', 'fill-holes:0', 'erode:1'];
  if (maskOps && maskOps.length > 0) {
    const inputMap = {
      'phase_quality': 'phase-quality',
      'combined': 'magnitude',
      'first_echo': 'magnitude-first',
      'last_echo': 'magnitude-last',
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
 * Generate methods text with citations from pipeline settings.
 * @returns {string} Markdown methods text
 */
export function generateMethods(settings, maskOps = [], maskSource = 'phase_quality', options = {}) {
  if (!wasmModule) return 'ERROR: WASM not loaded';
  const toml = settingsToToml(settings, maskOps, maskSource, options);
  return wasmModule.generate_methods_wasm(toml);
}

/**
 * Get default config as TOML string.
 */
export function getDefaultConfigToml() {
  if (!wasmModule) return '';
  return wasmModule.get_default_config_toml_wasm();
}

/**
 * Convert qsmbly pipeline settings to TOML string.
 * Maps JS camelCase settings to the nested PipelineConfig TOML format.
 */
export function settingsToToml(settings, maskOps = [], maskSource = 'phase_quality', options = {}) {
  if (!settings) return '';

  const isTgv = settings.combinedMethod === 'tgv';
  const isQsmart = settings.combinedMethod === 'qsmart';

  // Build nested config object matching PipelineConfig structure
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
    masking: {
      inhomogeneity_correction: true,
    },
    bg_removal: {
      algorithm: settings.backgroundRemoval || 'vsharp',
    },
    inversion: {},
    qsm: {
      reference: settings.referenceMean === false ? 'none' : 'mean',
    },
  };

  // Inversion algorithm
  if (isTgv) {
    config.inversion.algorithm = 'tgv';
  } else if (isQsmart) {
    config.inversion.algorithm = 'qsmart';
  } else {
    config.inversion.algorithm = settings.dipoleInversion || 'rts';
  }

  // Algorithm-specific params (only emit non-default)
  if (settings.rts) config.inversion.rts = settings.rts;
  if (settings.tv) config.inversion.tv = settings.tv;
  if (settings.tkd) config.inversion.tkd = settings.tkd;
  if (settings.tsvd) config.inversion.tsvd = settings.tsvd;
  if (settings.tikhonov) config.inversion.tikhonov = settings.tikhonov;
  if (settings.nltv) config.inversion.nltv = settings.nltv;
  if (settings.medi) config.inversion.medi = settings.medi;
  if (settings.ilsqr) config.inversion.ilsqr = settings.ilsqr;
  if (settings.tgv) config.inversion.tgv = {
    iterations: settings.tgv.iterations,
    erosions: settings.tgv.erosions,
    alpha0: settings.tgv.alpha0,
    alpha1: settings.tgv.alpha1,
    step_size: settings.tgv.stepSize,
    tol: settings.tgv.tol,
  };
  if (settings.qsmart) config.inversion.qsmart = {
    ilsqr_tol: settings.qsmart.ilsqrTol,
    ilsqr_max_iter: settings.qsmart.ilsqrMaxIter,
    vasc_sphere_radius: settings.qsmart.vascSphereRadiusMm,
    sdf_spatial_radius: settings.qsmart.sdfSpatialRadius,
  };

  // BG removal params
  if (settings.vsharp) config.bg_removal.vsharp = settings.vsharp;
  if (settings.pdf) config.bg_removal.pdf = settings.pdf;
  if (settings.lbv) config.bg_removal.lbv = settings.lbv;
  if (settings.ismv) config.bg_removal.ismv = {
    tol: settings.ismv.tol,
    max_iter: settings.ismv.maxit,
    radius_factor: settings.ismv.radius,
  };
  if (settings.sharp) config.bg_removal.sharp = {
    threshold: settings.sharp.threshold,
    radius_factor: settings.sharp.radius,
  };
  if (settings.resharp) config.bg_removal.resharp = settings.resharp;
  if (settings.harperella) config.bg_removal.harperella = settings.harperella;
  if (settings.iharperella) config.bg_removal.iharperella = settings.iharperella;

  // SWI params
  if (settings.swi) config.swi = {
    hp_sigma: settings.swi.hpSigma,
    scaling: settings.swi.scaling,
    strength: settings.swi.strength,
    mip_window: settings.swi.mipWindow,
  };

  // Mask sections — don't emit in TOML (complex tagged enum structure).
  // The generate_command() in Rust uses defaults. The command preview
  // won't show --mask flags unless we add mask CLI flag generation
  // directly in JS (like the old CommandGenerator did).
  // TODO: serialize mask sections properly or handle via separate mechanism.

  return toTomlString(config);
}

/**
 * Parse a mask op string like "threshold:otsu" into a TOML-compatible object.
 */
function parseMaskOp(opStr) {
  const parts = opStr.split(':');
  switch (parts[0]) {
    case 'threshold':
      return { op: 'threshold', method: parts[1] || 'otsu', value: parts[2] ? parseFloat(parts[2]) : null };
    case 'bet':
      return { op: 'bet', fractional_intensity: parseFloat(parts[1] || '0.5') };
    case 'erode':
      return { op: 'erode', iterations: parseInt(parts[1] || '1') };
    case 'dilate':
      return { op: 'dilate', iterations: parseInt(parts[1] || '1') };
    case 'close':
      return { op: 'close', radius: parseInt(parts[1] || '1') };
    case 'fill-holes':
      return { op: 'fill-holes', max_size: parseInt(parts[1] || '1000') };
    case 'gaussian':
      return { op: 'gaussian-smooth', sigma_mm: parseFloat(parts[1] || '4.0') };
    default:
      return { op: parts[0] };
  }
}

/**
 * Minimal TOML serializer for nested config objects.
 * Handles strings, numbers, booleans, arrays, and nested objects.
 */
function toTomlString(obj, prefix = '') {
  let lines = [];
  const tables = [];

  for (const [key, value] of Object.entries(obj)) {
    if (value === null || value === undefined) continue;
    const fullKey = prefix ? `${prefix}.${key}` : key;

    if (Array.isArray(value)) {
      if (value.length > 0 && typeof value[0] === 'object') {
        // Array of tables
        for (const item of value) {
          lines.push(`\n[[${fullKey}]]`);
          lines.push(toTomlString(item, '').trim());
        }
      } else {
        // Inline array
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
      lines.push(`${key} = ${Number.isInteger(value) ? value : value}`);
    }
  }

  // Emit nested tables after simple keys
  for (const { key, value } of tables) {
    lines.push(`\n[${key}]`);
    lines.push(toTomlString(value, key).trim());
  }

  return lines.join('\n') + '\n';
}
