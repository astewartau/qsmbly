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
export const VERSION = '0.4.0';

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
  inputMode: 'raw',
  fieldMapUnits: 'hz'
};

// Mask configuration defaults
export const MASK_CONFIG = {
  defaultThreshold: 15,         // Percentage of max magnitude
  drawingOpacity: 0.5,
  defaultBrushSize: 2
};

// BET (Brain Extraction Tool) defaults
export const BET_DEFAULTS = {
  fractionalIntensity: 0.5,
  iterations: 1000,
  subdivisions: 4
};

// Mask preparation settings
export const MASK_PREP_DEFAULTS = {
  source: 'combined',           // 'first_echo' or 'combined'
  biasCorrection: true
};

// Progress animation settings
export const PROGRESS_CONFIG = {
  animationSpeed: 0.5           // 50% per second - catches up quickly
};

// TGV (Total Generalized Variation) pipeline defaults
export const TGV_DEFAULTS = {
  regularization: 2,
  iterations: 1000,
  erosions: 3
};

// QSMART pipeline defaults
export const QSMART_DEFAULTS = {
  // SDF (Spatial Domain Filtering) parameters
  sdfSigma1Stage1: 10,
  sdfSigma2Stage1: 0,
  sdfSigma1Stage2: 8,
  sdfSigma2Stage2: 2,
  sdfSpatialRadius: 8,
  sdfLowerLim: 0.6,
  sdfCurvConstant: 500,

  // Vasculature detection parameters (in mm - auto-scaled to voxels)
  vascSphereRadiusMm: 8.0,      // mm - morphological filter radius
  frangiScaleMinMm: 1.0,        // mm - minimum vessel radius to detect (QSMART default: 1)
  frangiScaleMaxMm: 10.0,       // mm - maximum vessel radius to detect (QSMART default: 10)
  frangiScaleRatioMm: 2.0,      // mm - step between scales (QSMART default: 2)
  frangiC: 500,                 // noise sensitivity threshold

  // iLSQR solver parameters
  ilsqrTol: 0.01,
  ilsqrMaxIter: 50
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

// V-SHARP background removal defaults
export const VSHARP_DEFAULTS = {
  threshold: 0.05
  // maxRadius and minRadius are calculated from voxel size
};

// SHARP background removal defaults
export const SHARP_DEFAULTS = {
  threshold: 0.05
  // radius is calculated from voxel size
};

// SMV (Spherical Mean Value) defaults
export const SMV_DEFAULTS = {
  // radius is calculated from voxel size
};

// iSMV (iterative SMV) defaults
export const ISMV_DEFAULTS = {
  tol: 0.001,
  maxit: 500
  // radius is calculated from voxel size
};

// PDF (Projection onto Dipole Fields) defaults
export const PDF_DEFAULTS = {
  tol: 0.00001
  // maxit is calculated from mask size
};

// LBV (Laplacian Boundary Value) defaults
export const LBV_DEFAULTS = {
  tol: 0.001,
  maxit: 500
};

// TKD (Thresholded K-space Division) defaults
export const TKD_DEFAULTS = {
  threshold: 0.15
};

// TSVD (Truncated SVD) defaults
export const TSVD_DEFAULTS = {
  threshold: 0.15
};

// Tikhonov regularization defaults
export const TIKHONOV_DEFAULTS = {
  lambda: 0.01,
  reg: 'identity'
};

// TV (Total Variation) regularization defaults
export const TV_DEFAULTS = {
  lambda: 0.001,
  maxIter: 250,
  tol: 0.001
};

// RTS (Rapid Two-Step) defaults
export const RTS_DEFAULTS = {
  delta: 0.15,
  mu: 100000,
  rho: 10,
  maxIter: 20
};

// NLTV (Nonlinear Total Variation) defaults
export const NLTV_DEFAULTS = {
  lambda: 0.001,
  mu: 1,
  maxIter: 250,
  tol: 0.001,
  newtonMaxIter: 10
};

// MEDI (Morphology Enabled Dipole Inversion) defaults
export const MEDI_DEFAULTS = {
  lambda: 7.5e-5,
  percentage: 0.3,
  maxIter: 30,
  cgMaxIter: 10,
  cgTol: 0.01,
  tol: 0.1,
  smv: false,
  smvRadius: 5,
  merit: false,
  dataWeighting: 1
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
  'unwrapped': 'Unwrapped'
};

// Pipeline method options
export const PIPELINE_METHODS = {
  combined: ['none', 'tgv', 'qsmart'],
  unwrap: ['romeo', 'laplacian'],
  phaseOffset: ['mcpc3ds', 'none'],
  fieldCalculation: ['weighted_avg', 'linear_fit'],
  b0WeightType: ['phase_snr', 'phase_var', 'average', 'tes', 'mag'],
  backgroundRemoval: ['vsharp', 'sharp', 'smv', 'ismv', 'pdf', 'lbv'],
  dipoleInversion: ['tkd', 'tsvd', 'tikhonov', 'tv', 'rts', 'nltv', 'medi']
};

// Default pipeline settings (assembled from individual defaults)
export const PIPELINE_DEFAULTS = {
  combinedMethod: 'none',
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
  smv: { radius: null },
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

  return {
    // V-SHARP: maxRadius = 18 * min(vsz), minRadius = 2 * max(vsz)
    vsharpMaxRadius: Math.round(18 * minVsz),
    vsharpMinRadius: Math.round(Math.max(1, 2 * minVsz)),
    // SHARP: radius = 18 * min(vsz) - matches QSM.jl sharp.jl line 22
    sharpRadius: Math.round(18 * minVsz),
    // Simple SMV: radius = 5 * max(vsz) - smaller than SHARP since no deconvolution
    smvRadius: Math.round(Math.max(4, 5 * maxVsz)),
    // iSMV: radius = 2 * max(vsz) - matches QSM.jl ismv.jl line 20
    ismvRadius: Math.round(Math.max(2, 2 * maxVsz)),
    // PDF: maxit = ceil(sqrt(numel(mask)))
    pdfMaxit: Math.ceil(Math.sqrt(maskSize))
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
  PHYSICS,
  INPUT_MODES,
  FIELD_MAP_UNITS,
  INPUT_DEFAULTS,
  VIEWER_CONFIG,
  MASK_CONFIG,
  BET_DEFAULTS,
  MASK_PREP_DEFAULTS,
  PROGRESS_CONFIG,
  TGV_DEFAULTS,
  QSMART_DEFAULTS,
  ROMEO_DEFAULTS,
  MCPC3DS_DEFAULTS,
  LINEAR_FIT_DEFAULTS,
  VSHARP_DEFAULTS,
  SHARP_DEFAULTS,
  SMV_DEFAULTS,
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
