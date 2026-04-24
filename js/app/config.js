/**
 * QSM-WASM Configuration Module
 *
 * Centralized configuration for all magic numbers, default values,
 * and constants used across the application.
 *
 * This module works as both an ES module (for modern scripts) and
 * can be loaded via importScripts in web workers.
 */

// Detect environment and set up exports appropriately
const isWorker = typeof WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope;
const isModule = typeof exports !== 'undefined' || (typeof window !== 'undefined' && window.QSMConfig === undefined);

// Application version (keep in sync with package.json, Cargo.toml, and git tags)
export const VERSION = '0.10.1';

// QSM.rs core library version (keep in sync with qsm-core dependency in rust-wasm/Cargo.toml)
export const QSM_RS_VERSION = '0.7.0';

// Physics constants
export const PHYSICS = {
  GYROMAGNETIC_RATIO: 42.576e6  // Hz/T (gamma for protons)
};

// NiiVue viewer configuration
export const VIEWER_CONFIG = {
  loadingText: "",
  dragToMeasure: false,
  isColorbar: false,
  textHeight: 0.03,
  show3Dcrosshair: false,
  crosshairColor: [0.23, 0.51, 0.96, 1.0],  // #3b82f6 - matches site primary color
  crosshairWidth: 0.75
};

// Input mode configuration
export const INPUT_MODES = {
  RAW: 'raw',              // Raw magnitude + phase images (default)
  TOTAL_FIELD: 'totalField', // Pre-computed total field map (B0)
  LOCAL_FIELD: 'localField'  // Pre-computed local field map
};

export const FIELD_MAP_UNITS = {
  HZ: 'hz',       // Hertz
  RAD_S: 'rad_s',  // Radians per second
  PPM: 'ppm'       // Parts per million (already normalized)
};

export const INPUT_DEFAULTS = {
  inputMode: 'dicom',
  fieldMapUnits: 'hz'
};

// Algorithm defaults — sourced from QSM.rs via auto-generated qsm-defaults.js
// These re-exports adapt the QSM.rs field names to the JS conventions used throughout qsmbly.
import {
  RTS_DEFAULTS as _RTS,
  TV_DEFAULTS as _TV,
  TKD_DEFAULTS as _TKD,
  TGV_DEFAULTS as _TGV,
  BET_DEFAULTS as _BET,
  VSHARP_DEFAULTS as _VSHARP,
  PDF_DEFAULTS as _PDF,
  LBV_DEFAULTS as _LBV,
  ISMV_DEFAULTS as _ISMV,
  SWI_DEFAULTS as _SWI,
  SHARP_DEFAULTS as _SHARP,
  TIKHONOV_DEFAULTS as _TIKHONOV,
  NLTV_DEFAULTS as _NLTV,
  MEDI_DEFAULTS as _MEDI,
  QSMART_DEFAULTS as _QSMART,
} from './qsm-defaults.js';

// Re-export with JS-convention field names (camelCase, matching existing usage)
export const RTS_DEFAULTS = {
  delta: _RTS.delta,
  mu: _RTS.mu,
  rho: _RTS.rho,
  tol: _RTS.tol,
  maxIter: _RTS.max_iter,
  lsmrIter: _RTS.lsmr_iter,
};

export const TV_DEFAULTS = {
  lambda: _TV.lambda,
  rho: _TV.rho,
  tol: _TV.tol,
  maxIter: _TV.max_iter,
};

export const TKD_DEFAULTS = {
  threshold: _TKD.threshold,
};

export const VSHARP_DEFAULTS = {
  threshold: _VSHARP.threshold,
};

export const PDF_DEFAULTS = {
  tol: _PDF.tol,
};

export const LBV_DEFAULTS = {
  tol: _LBV.tol,
};

export const ISMV_DEFAULTS = {
  tol: _ISMV.tol,
  maxit: _ISMV.max_iter,
};

// Adapted re-exports (field name mapping from snake_case to camelCase)
export const BET_DEFAULTS = {
  fractionalIntensity: _BET.fractional_intensity,
  smoothness: _BET.smoothness,
  gradientThreshold: _BET.gradient_threshold,
  iterations: _BET.iterations,
  subdivisions: _BET.subdivisions,
  erosions: 2  // mask erosion count (qsmbly-specific, not a BET param)
};

export const TGV_DEFAULTS = {
  regularization: 2,  // UI preset level (qsmbly-specific)
  alpha0: _TGV.alpha0,
  alpha1: _TGV.alpha1,
  iterations: _TGV.iterations,
  erosions: _TGV.erosions,
  stepSize: _TGV.step_size,
  tol: _TGV.tol,
};

export const SWI_DEFAULTS = {
  hpSigma: _SWI.hp_sigma,
  scaling: 'tanh',  // PhaseScaling enum (JS-side choice)
  strength: _SWI.strength,
  mipWindow: _SWI.mip_window,
};

// Mask configuration defaults
export const MASK_CONFIG = {
  defaultThreshold: 15,         // Percentage of max magnitude
  drawingOpacity: 0.5,
  defaultBrushSize: 2
};

// Mask preparation settings
export const MASK_PREP_DEFAULTS = {
  source: 'phase_quality',      // 'first_echo', 'combined', or 'phase_quality'
  biasCorrection: true
};

// Progress animation settings
export const PROGRESS_CONFIG = {
  animationSpeed: 0.5           // 50% per second - catches up quickly
};

// (TGV_DEFAULTS and SWI_DEFAULTS are now sourced from QSM.rs — see imports above)

// QSMART pipeline defaults (from QSM.rs QsmartParams)
export const QSMART_DEFAULTS = {
  sdfSigma1Stage1: _QSMART.sdf_sigma1_stage1,
  sdfSigma2Stage1: _QSMART.sdf_sigma2_stage1,
  sdfSigma1Stage2: _QSMART.sdf_sigma1_stage2,
  sdfSigma2Stage2: _QSMART.sdf_sigma2_stage2,
  sdfSpatialRadius: _QSMART.sdf_spatial_radius,
  sdfLowerLim: _QSMART.sdf_lower_lim,
  sdfCurvConstant: _QSMART.sdf_curv_constant,
  vascSphereRadiusMm: _QSMART.vasc_sphere_radius,
  frangiScaleMinMm: _QSMART.frangi_scale_range[0],
  frangiScaleMaxMm: _QSMART.frangi_scale_range[1],
  frangiScaleRatioMm: _QSMART.frangi_scale_ratio,
  frangiC: _QSMART.frangi_c,
  ilsqrTol: _QSMART.ilsqr_tol,
  ilsqrMaxIter: _QSMART.ilsqr_max_iter,
};

// ROMEO unwrapping defaults
export const ROMEO_DEFAULTS = {
  weighting: 'phase_snr',
  phaseGradientCoherence: true,
  magCoherence: true,
  magWeight: true
};

// MCPC-3D-S phase offset correction defaults
export const MCPC3DS_DEFAULTS = {
  sigma: [10, 10, 5]
};

// Linear fit B0 calculation defaults
export const LINEAR_FIT_DEFAULTS = {
  estimateOffset: true
};

export const SHARP_DEFAULTS = {
  threshold: _SHARP.threshold,
};

// TSVD (Truncated SVD) defaults (shares TKD threshold)
export const TSVD_DEFAULTS = {
  threshold: _TKD.threshold,
};

// (VSHARP_DEFAULTS, ISMV_DEFAULTS, PDF_DEFAULTS, LBV_DEFAULTS,
//  TKD_DEFAULTS, RTS_DEFAULTS, TV_DEFAULTS are now sourced from QSM.rs
//  — see imports above)

// Tikhonov regularization defaults
export const TIKHONOV_DEFAULTS = {
  lambda: _TIKHONOV.lambda,
  reg: _TIKHONOV.reg,
};

// (TV_DEFAULTS, RTS_DEFAULTS sourced from QSM.rs — see imports above)

export const NLTV_DEFAULTS = {
  lambda: _NLTV.lambda,
  mu: _NLTV.mu,
  maxIter: _NLTV.max_iter,
  tol: _NLTV.tol,
  newtonMaxIter: _NLTV.newton_iter,
};

export const MEDI_DEFAULTS = {
  lambda: _MEDI.lambda,
  percentage: _MEDI.percentage,
  maxIter: _MEDI.max_iter,
  cgMaxIter: _MEDI.cg_max_iter,
  cgTol: _MEDI.cg_tol,
  tol: _MEDI.tol,
  smv: _MEDI.smv,
  smvRadius: _MEDI.smv_radius,
  merit: _MEDI.merit,
  dataWeighting: _MEDI.data_weighting,
};

// Stage display names for UI
export const STAGE_DISPLAY_NAMES = {
  'magnitude': 'Magnitude',
  'phase': 'Phase',
  'mask': 'Mask',
  'B0': 'B0 Field',
  'bgRemoved': 'Local Field',
  'final': 'QSM',
  'tfs': 'TFS',
  'lfsStage1': 'LFS 1',
  'lfsStage2': 'LFS 2',
  'chiStage1': 'χ1',
  'chiStage2': 'χ2',
  'vasculature': 'Vessels',
  'vascDetect': 'Vessels',
  'frangi': 'Frangi',
  'vascMask': 'Vasc Mask',
  'bottomHat': 'Bottom Hat',
  'unwrapped': 'Unwrapped',
  'swi': 'SWI',
  'mip': 'mIP',
  'r2star': 'R2*',
  't2star': 'T2*'
};

// Pipeline method options
export const PIPELINE_METHODS = {
  combined: ['none', 'tgv', 'qsmart'],
  unwrap: ['romeo', 'laplacian'],
  phaseOffset: ['mcpc3ds', 'none'],
  fieldCalculation: ['weighted_avg', 'linear_fit'],
  b0WeightType: ['phase_snr', 'phase_var', 'average', 'tes', 'mag'],
  backgroundRemoval: ['vsharp', 'sharp', 'ismv', 'pdf', 'lbv'],
  dipoleInversion: ['tkd', 'tsvd', 'tikhonov', 'tv', 'rts', 'nltv', 'medi']
};

// Default pipeline settings (assembled from individual defaults)
export const PIPELINE_DEFAULTS = {
  combinedMethod: 'none',
  swi: { ...SWI_DEFAULTS },
  tgv: { ...TGV_DEFAULTS },
  qsmart: { ...QSMART_DEFAULTS },
  unwrapMethod: 'romeo',
  phaseOffsetMethod: 'mcpc3ds',
  fieldCalculationMethod: 'weighted_avg',
  mcpc3ds: { ...MCPC3DS_DEFAULTS },
  b0WeightType: 'phase_snr',
  linearFit: { ...LINEAR_FIT_DEFAULTS },
  romeo: { ...ROMEO_DEFAULTS },
  backgroundRemoval: 'vsharp',
  vsharp: { ...VSHARP_DEFAULTS, maxRadius: null, minRadius: null },
  sharp: { radius: 6, ...SHARP_DEFAULTS },
  ismv: { ...ISMV_DEFAULTS, radius: null },
  pdf: { ...PDF_DEFAULTS, maxit: null },
  lbv: { ...LBV_DEFAULTS },
  dipoleInversion: 'rts',
  tkd: { ...TKD_DEFAULTS },
  tsvd: { ...TSVD_DEFAULTS },
  tikhonov: { ...TIKHONOV_DEFAULTS },
  tv: { ...TV_DEFAULTS },
  rts: { ...RTS_DEFAULTS },
  nltv: { ...NLTV_DEFAULTS },
  medi: { ...MEDI_DEFAULTS }
};

/**
 * Calculate dynamic defaults based on voxel size
 * Matches QSM.jl behavior for radius calculations
 *
 * @param {number[]} voxelSize - [dx, dy, dz] in mm
 * @param {number[]} maskDims - [nx, ny, nz] dimensions (optional, for PDF maxit)
 * @returns {Object} Calculated defaults
 */
export function getVoxelBasedDefaults(voxelSize = [1, 1, 1], maskDims = null) {
  const minVsz = Math.min(...voxelSize);
  const maxVsz = Math.max(...voxelSize);

  // Calculate mask size for PDF maxit (if available)
  const maskSize = maskDims ? maskDims[0] * maskDims[1] * maskDims[2] : 100000;

  // Max dimension for LBV adaptive maxit
  const maxDim = maskDims ? Math.max(maskDims[0], maskDims[1], maskDims[2]) : 256;

  return {
    // V-SHARP: maxRadius = 18 * min(vsz), minRadius = 2 * max(vsz)
    vsharpMaxRadius: Math.round(18 * minVsz),
    vsharpMinRadius: Math.round(Math.max(1, 2 * minVsz)),
    // SHARP: radius = 18 * min(vsz) - matches QSM.jl sharp.jl line 22
    sharpRadius: Math.round(18 * minVsz),
    // iSMV: radius = 2 * max(vsz) - matches QSM.jl ismv.jl line 20
    ismvRadius: Math.round(Math.max(2, 2 * maxVsz)),
    // PDF: maxit = ceil(sqrt(numel(mask))) - matches QSM.jl pdf.jl
    pdfMaxit: Math.ceil(Math.sqrt(maskSize)),
    // LBV: maxit = max(dims) - matches QSM.jl lbv.jl
    lbvMaxit: maxDim
  };
}

/**
 * Phase scaling constants
 */
export const PHASE_SCALING = {
  PI_THRESHOLD_MULTIPLIER: 1.1,   // Range > 2π * 1.1 triggers scaling
  MAX_PI_MULTIPLIER: 1.5          // Values > π * 1.5 trigger scaling
};

/**
 * Box filter default radius for reliability map computation
 */
export const BOX_FILTER_DEFAULTS = {
  reliabilityRadius: 1
};

// Make config available globally for non-module scripts and workers
const QSMConfig = {
  VERSION,
  QSM_RS_VERSION,
  PHYSICS,
  INPUT_MODES,
  FIELD_MAP_UNITS,
  INPUT_DEFAULTS,
  VIEWER_CONFIG,
  MASK_CONFIG,
  BET_DEFAULTS,
  MASK_PREP_DEFAULTS,
  PROGRESS_CONFIG,
  SWI_DEFAULTS,
  TGV_DEFAULTS,
  QSMART_DEFAULTS,
  ROMEO_DEFAULTS,
  MCPC3DS_DEFAULTS,
  LINEAR_FIT_DEFAULTS,
  VSHARP_DEFAULTS,
  SHARP_DEFAULTS,
  ISMV_DEFAULTS,
  PDF_DEFAULTS,
  LBV_DEFAULTS,
  TKD_DEFAULTS,
  TSVD_DEFAULTS,
  TIKHONOV_DEFAULTS,
  TV_DEFAULTS,
  RTS_DEFAULTS,
  NLTV_DEFAULTS,
  MEDI_DEFAULTS,
  STAGE_DISPLAY_NAMES,
  PIPELINE_METHODS,
  PIPELINE_DEFAULTS,
  getVoxelBasedDefaults,
  PHASE_SCALING,
  BOX_FILTER_DEFAULTS
};

// Export for different environments
if (typeof self !== 'undefined' && typeof WorkerGlobalScope !== 'undefined') {
  // Web Worker context
  self.QSMConfig = QSMConfig;
} else if (typeof window !== 'undefined') {
  // Browser context
  window.QSMConfig = QSMConfig;
}
