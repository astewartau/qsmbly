/**
 * Pipeline Settings Controller
 *
 * Manages the pipeline settings modal UI - form population, visibility toggling,
 * reset to defaults, and reading form values.
 */

import {
  PIPELINE_DEFAULTS as D,
  TGV_DEFAULTS, SWI_DEFAULTS, QSMART_DEFAULTS, MCPC3DS_DEFAULTS,
  ROMEO_DEFAULTS, LINEAR_FIT_DEFAULTS,
  VSHARP_DEFAULTS, SHARP_DEFAULTS, RESHARP_DEFAULTS, HARPERELLA_DEFAULTS,
  ISMV_DEFAULTS, PDF_DEFAULTS, LBV_DEFAULTS,
  TKD_DEFAULTS, TSVD_DEFAULTS, TIKHONOV_DEFAULTS,
  TV_DEFAULTS, RTS_DEFAULTS, NLTV_DEFAULTS, MEDI_DEFAULTS,
} from '../app/config.js';

export class PipelineSettingsController {
  constructor(modalElement) {
    this.modal = modalElement;
    this.inputMode = 'dicom'; // 'dicom', 'raw', 'totalField', or 'localField'
    this._setupTabs();
    this._setupEventListeners();
  }

  /**
   * Set the current input mode - controls which pipeline sections are visible
   * @param {string} mode - 'raw', 'totalField', or 'localField'
   */
  setInputMode(mode) {
    this.inputMode = mode;
  }

  /**
   * Open the modal and populate form from settings
   * @param {Object} settings - Current pipeline settings
   * @param {Object} defaults - Voxel-based default values
   * @param {number} nEchoes - Number of echo files loaded
   * @param {boolean} hasMagnitude - Whether magnitude data is available
   */
  open(settings, defaults, nEchoes, hasMagnitude = true) {
    this.hasMagnitude = hasMagnitude;
    this.nEchoes = nEchoes;
    this._populateForm(settings, defaults);
    this.updateVisibility(nEchoes);
    this._switchTab('tabQsmPipeline');
    this.modal.classList.add('active');
  }

  /**
   * Close the modal
   */
  close() {
    this.modal.classList.remove('active');
  }

  /**
   * Reset form to default values
   * @param {Object} defaults - Voxel-based default values
   */
  reset(defaults) {
    // Combined method
    this._setEl('combinedMethod', D.combinedMethod);

    // TGV defaults
    this._setEl('tgvRegularization', 2); // UI preset level (qsmbly-specific)
    this._setEl('tgvIterations', TGV_DEFAULTS.iterations);
    this._setEl('tgvErosions', TGV_DEFAULTS.erosions);

    // SWI defaults
    this._setEl('swiScaling', SWI_DEFAULTS.scaling);
    this._setEl('swiStrength', SWI_DEFAULTS.strength);
    this._setEl('swiHpSigmaX', SWI_DEFAULTS.hpSigma[0]);
    this._setEl('swiHpSigmaY', SWI_DEFAULTS.hpSigma[1]);
    this._setEl('swiHpSigmaZ', SWI_DEFAULTS.hpSigma[2]);
    this._setEl('swiMipWindow', SWI_DEFAULTS.mipWindow);

    // QSMART defaults
    this._setEl('qsmartSdfSigma1Stage1', QSMART_DEFAULTS.sdfSigma1Stage1);
    this._setEl('qsmartSdfSigma2Stage1', QSMART_DEFAULTS.sdfSigma2Stage1);
    this._setEl('qsmartSdfSigma1Stage2', QSMART_DEFAULTS.sdfSigma1Stage2);
    this._setEl('qsmartSdfSigma2Stage2', QSMART_DEFAULTS.sdfSigma2Stage2);
    this._setEl('qsmartSdfSpatialRadius', QSMART_DEFAULTS.sdfSpatialRadius);
    this._setEl('qsmartSdfLowerLim', QSMART_DEFAULTS.sdfLowerLim);
    this._setEl('qsmartSdfCurvConstant', QSMART_DEFAULTS.sdfCurvConstant);
    this._setEl('qsmartVascSphereRadius', QSMART_DEFAULTS.vascSphereRadiusMm);
    this._setEl('qsmartFrangiScaleMin', QSMART_DEFAULTS.frangiScaleMinMm);
    this._setEl('qsmartFrangiScaleMax', QSMART_DEFAULTS.frangiScaleMaxMm);
    this._setEl('qsmartFrangiScaleRatio', QSMART_DEFAULTS.frangiScaleRatioMm);
    this._setEl('qsmartFrangiC', QSMART_DEFAULTS.frangiC);
    this._setEl('qsmartIlsqrTol', QSMART_DEFAULTS.ilsqrTol);
    this._setEl('qsmartIlsqrMaxIter', QSMART_DEFAULTS.ilsqrMaxIter);

    // Phase offset
    this._setChecked('phaseOffsetEnabled', true);
    this._setEl('phaseOffsetMethod', D.phaseOffsetMethod);
    this._setEl('mcpc3dsSigmaX', MCPC3DS_DEFAULTS.sigma[0]);
    this._setEl('mcpc3dsSigmaY', MCPC3DS_DEFAULTS.sigma[1]);
    this._setEl('mcpc3dsSigmaZ', MCPC3DS_DEFAULTS.sigma[2]);

    // Bipolar correction
    this._setChecked('bipolarCorrectionEnabled', false);

    // Unwrap method
    this._setEl('unwrapMethod', D.unwrapMethod);
    const resetHint = document.getElementById('unwrapLockedHint');
    if (resetHint) resetHint.style.display = 'none';
    this._showEl('romeoSettings', true);
    this._showEl('laplacianSettings', false);

    // ROMEO weight checkboxes
    this._setChecked('romeoPhaseGradientCoherence', ROMEO_DEFAULTS.phaseGradientCoherence);
    this._setChecked('romeoMagCoherence', ROMEO_DEFAULTS.magCoherence);
    this._setChecked('romeoMagWeight', ROMEO_DEFAULTS.magWeight);

    // Field calculation method
    this._setEl('fieldCalculationMethod', D.fieldCalculationMethod);
    this._showEl('weightedAvgSettings', true);
    this._showEl('linearFitSettings', false);
    this._setEl('b0WeightType', D.b0WeightType);

    // Linear fit defaults
    this._setChecked('linearFitEstimateOffset', LINEAR_FIT_DEFAULTS.estimateOffset);

    // Background removal
    this._setEl('bgRemovalMethod', D.backgroundRemoval);
    this._showEl('vsharpSettings', true);
    this._showEl('sharpSettings', false);
    this._showEl('resharpSettings', false);
    this._showEl('ismvSettings', false);
    this._showEl('pdfSettings', false);
    this._showEl('lbvSettings', false);
    this._showEl('harperellaSettings', false);
    this._showEl('iharperellaSettings', false);

    this._setEl('vsharpMaxRadius', defaults.vsharpMaxRadius);
    this._setEl('vsharpMinRadius', defaults.vsharpMinRadius);
    this._setEl('vsharpThreshold', VSHARP_DEFAULTS.threshold);
    this._setEl('sharpRadius', defaults.sharpRadius);
    this._setEl('sharpThreshold', SHARP_DEFAULTS.threshold);
    this._setEl('ismvRadius', defaults.ismvRadius);
    this._setEl('ismvTol', ISMV_DEFAULTS.tol);
    this._setEl('ismvMaxit', ISMV_DEFAULTS.maxit);
    this._setEl('pdfTol', PDF_DEFAULTS.tol);
    this._setEl('pdfMaxit', defaults.pdfMaxit);
    this._setEl('lbvTol', LBV_DEFAULTS.tol);
    this._setEl('lbvMaxit', defaults.lbvMaxit);
    this._setEl('resharpRadius', RESHARP_DEFAULTS.radius);
    this._setEl('resharpTikReg', RESHARP_DEFAULTS.tikReg);
    this._setEl('resharpTol', RESHARP_DEFAULTS.tol);
    this._setEl('resharpMaxIter', RESHARP_DEFAULTS.maxIter);
    this._setEl('harperellaRadius', HARPERELLA_DEFAULTS.radius);
    this._setEl('harperellaMaxIter', HARPERELLA_DEFAULTS.maxIter);
    this._setEl('iharperellaRadius', HARPERELLA_DEFAULTS.radius);
    this._setEl('iharperellaMaxIter', HARPERELLA_DEFAULTS.maxIter);

    // Dipole inversion
    this._setEl('dipoleMethod', D.dipoleInversion);
    this._showEl('tkdSettings', false);
    this._showEl('tsvdSettings', false);
    this._showEl('tikhonovSettings', false);
    this._showEl('tvSettings', false);
    this._showEl('rtsSettings', true);
    this._showEl('nltvSettings', false);
    this._showEl('mediSettings', false);
    this._showEl('ilsqrSettings', false);

    this._setEl('tkdThreshold', TKD_DEFAULTS.threshold);
    this._setEl('tsvdThreshold', TSVD_DEFAULTS.threshold);
    this._setEl('tikhLambda', TIKHONOV_DEFAULTS.lambda);
    this._setEl('tikhReg', 'identity');

    this._setEl('tvLambda', TV_DEFAULTS.lambda);
    this._setEl('tvMaxIter', TV_DEFAULTS.maxIter);
    this._setEl('tvTol', TV_DEFAULTS.tol);

    this._setEl('rtsDelta', RTS_DEFAULTS.delta);
    this._setEl('rtsMu', RTS_DEFAULTS.mu);
    this._setEl('rtsRho', RTS_DEFAULTS.rho);
    this._setEl('rtsMaxIter', RTS_DEFAULTS.maxIter);

    this._setEl('nltvLambda', NLTV_DEFAULTS.lambda);
    this._setEl('nltvMu', NLTV_DEFAULTS.mu);
    this._setEl('nltvMaxIter', NLTV_DEFAULTS.maxIter);
    this._setEl('nltvTol', NLTV_DEFAULTS.tol);
    this._setEl('nltvNewtonMaxIter', NLTV_DEFAULTS.newtonMaxIter);

    this._setEl('mediLambda', MEDI_DEFAULTS.lambda);
    this._setEl('mediPercentage', MEDI_DEFAULTS.percentage);
    this._setEl('mediMaxIter', MEDI_DEFAULTS.maxIter);
    this._setEl('mediCgMaxIter', MEDI_DEFAULTS.cgMaxIter);
    this._setChecked('mediSmv', MEDI_DEFAULTS.smv);
    this._setEl('mediSmvRadius', MEDI_DEFAULTS.smvRadius);
    this._showEl('mediSmvRadiusGroup', MEDI_DEFAULTS.smv);
    this._setChecked('mediMerit', MEDI_DEFAULTS.merit);

    this._setEl('ilsqrTol', QSMART_DEFAULTS.ilsqrTol);
    this._setEl('ilsqrMaxIter', QSMART_DEFAULTS.ilsqrMaxIter);
  }

  /**
   * Read form values and return settings object
   * @param {number} nEchoes - Number of echo files loaded
   * @returns {Object} Pipeline settings object
   */
  save(nEchoes) {
    const isMultiEcho = nEchoes > 1;

    // Phase offset: use checkbox state (multi-echo) or 'none' (single-echo)
    const phaseOffsetEnabled = isMultiEcho && (this._getChecked('phaseOffsetEnabled') ?? true);
    const phaseOffsetMethod = phaseOffsetEnabled ? (this._getEl('phaseOffsetMethod') || 'mcpc3ds') : 'none';

    const unwrapMethod = isMultiEcho
      ? this._getEl('unwrapMethod')
      : this._getEl('singleEchoUnwrapMethod');

    // ROMEO weight settings
    const romeoPhaseGradientCoherence = isMultiEcho
      ? this._getChecked('romeoPhaseGradientCoherence') ?? true
      : true;
    const romeoMagCoherence = this._getChecked('romeoMagCoherence') ?? true;
    const romeoMagWeight = this._getChecked('romeoMagWeight') ?? true;

    return {
      combinedMethod: this._getEl('combinedMethod'),
      referenceMean: this._getChecked('qsmReferenceMean') ?? true,
      swi: {
        hpSigma: [
          parseFloat(this._getEl('swiHpSigmaX')),
          parseFloat(this._getEl('swiHpSigmaY')),
          parseFloat(this._getEl('swiHpSigmaZ'))
        ],
        scaling: this._getEl('swiScaling') || 'tanh',
        strength: parseFloat(this._getEl('swiStrength')),
        mipWindow: parseInt(this._getEl('swiMipWindow'))
      },
      tgv: {
        regularization: parseInt(this._getEl('tgvRegularization')),
        iterations: parseInt(this._getEl('tgvIterations')),
        erosions: parseInt(this._getEl('tgvErosions'))
      },
      qsmart: {
        sdfSigma1Stage1: parseFloat(this._getEl('qsmartSdfSigma1Stage1')),
        sdfSigma2Stage1: parseFloat(this._getEl('qsmartSdfSigma2Stage1')),
        sdfSigma1Stage2: parseFloat(this._getEl('qsmartSdfSigma1Stage2')),
        sdfSigma2Stage2: parseFloat(this._getEl('qsmartSdfSigma2Stage2')),
        sdfSpatialRadius: parseInt(this._getEl('qsmartSdfSpatialRadius')),
        sdfLowerLim: parseFloat(this._getEl('qsmartSdfLowerLim')),
        sdfCurvConstant: parseFloat(this._getEl('qsmartSdfCurvConstant')),
        vascSphereRadiusMm: parseFloat(this._getEl('qsmartVascSphereRadius')),
        frangiScaleMinMm: parseFloat(this._getEl('qsmartFrangiScaleMin')),
        frangiScaleMaxMm: parseFloat(this._getEl('qsmartFrangiScaleMax')),
        frangiScaleRatioMm: parseFloat(this._getEl('qsmartFrangiScaleRatio')),
        frangiC: parseFloat(this._getEl('qsmartFrangiC')),
        ilsqrTol: parseFloat(this._getEl('qsmartIlsqrTol')),
        ilsqrMaxIter: parseInt(this._getEl('qsmartIlsqrMaxIter'))
      },
      unwrapMethod: unwrapMethod,
      phaseOffsetMethod: phaseOffsetMethod,
      bipolarCorrection: isMultiEcho && nEchoes >= 3 && (this._getChecked('bipolarCorrectionEnabled') ?? false),
      fieldCalculationMethod: this._getEl('fieldCalculationMethod') || 'weighted_avg',
      mcpc3ds: {
        sigma: [
          parseInt(this._getEl('mcpc3dsSigmaX')),
          parseInt(this._getEl('mcpc3dsSigmaY')),
          parseInt(this._getEl('mcpc3dsSigmaZ'))
        ]
      },
      b0WeightType: this._getEl('b0WeightType') || 'phase_snr',
      linearFit: {
        estimateOffset: this._getChecked('linearFitEstimateOffset') ?? true
      },
      romeo: {
        phaseGradientCoherence: romeoPhaseGradientCoherence,
        magCoherence: romeoMagCoherence,
        magWeight: romeoMagWeight
      },
      backgroundRemoval: this._getEl('bgRemovalMethod'),
      vsharp: {
        maxRadius: parseFloat(this._getEl('vsharpMaxRadius')),
        minRadius: parseFloat(this._getEl('vsharpMinRadius')),
        threshold: parseFloat(this._getEl('vsharpThreshold'))
      },
      sharp: {
        radius: parseFloat(this._getEl('sharpRadius')),
        threshold: parseFloat(this._getEl('sharpThreshold'))
      },
      ismv: {
        radius: parseFloat(this._getEl('ismvRadius')),
        tol: parseFloat(this._getEl('ismvTol')),
        maxit: parseInt(this._getEl('ismvMaxit'))
      },
      pdf: {
        tol: parseFloat(this._getEl('pdfTol')),
        maxit: parseInt(this._getEl('pdfMaxit'))
      },
      resharp: {
        radius: parseFloat(this._getEl('resharpRadius')),
        tikReg: parseFloat(this._getEl('resharpTikReg')),
        tol: parseFloat(this._getEl('resharpTol')),
        maxIter: parseInt(this._getEl('resharpMaxIter'))
      },
      harperella: {
        radius: parseFloat(this._getEl('harperellaRadius')),
        maxIter: parseInt(this._getEl('harperellaMaxIter')),
        tol: 1e-6
      },
      iharperella: {
        radius: parseFloat(this._getEl('iharperellaRadius')),
        maxIter: parseInt(this._getEl('iharperellaMaxIter')),
        tol: 1e-6
      },
      lbv: {
        tol: parseFloat(this._getEl('lbvTol')),
        maxit: parseInt(this._getEl('lbvMaxit'))
      },
      dipoleInversion: this._getEl('dipoleMethod'),
      tkd: {
        threshold: parseFloat(this._getEl('tkdThreshold'))
      },
      tsvd: {
        threshold: parseFloat(this._getEl('tsvdThreshold'))
      },
      tikhonov: {
        lambda: parseFloat(this._getEl('tikhLambda')),
        reg: this._getEl('tikhReg')
      },
      tv: {
        lambda: parseFloat(this._getEl('tvLambda')),
        maxIter: parseInt(this._getEl('tvMaxIter')),
        tol: parseFloat(this._getEl('tvTol'))
      },
      rts: {
        delta: parseFloat(this._getEl('rtsDelta')),
        mu: parseFloat(this._getEl('rtsMu')),
        rho: parseFloat(this._getEl('rtsRho')),
        maxIter: parseInt(this._getEl('rtsMaxIter'))
      },
      nltv: {
        lambda: parseFloat(this._getEl('nltvLambda')),
        mu: parseFloat(this._getEl('nltvMu')),
        maxIter: parseInt(this._getEl('nltvMaxIter')),
        tol: parseFloat(this._getEl('nltvTol')),
        newtonMaxIter: parseInt(this._getEl('nltvNewtonMaxIter'))
      },
      medi: {
        lambda: parseFloat(this._getEl('mediLambda')),
        percentage: parseFloat(this._getEl('mediPercentage')),
        maxIter: parseInt(this._getEl('mediMaxIter')),
        cgMaxIter: parseInt(this._getEl('mediCgMaxIter')),
        cgTol: 0.01,
        tol: 0.1,
        smv: this._getChecked('mediSmv'),
        smvRadius: parseFloat(this._getEl('mediSmvRadius')),
        merit: this._getChecked('mediMerit'),
        dataWeighting: 1
      },
      ilsqr: {
        tol: parseFloat(this._getEl('ilsqrTol')),
        maxIter: parseInt(this._getEl('ilsqrMaxIter'))
      }
    };
  }

  /**
   * Update visibility of sections based on method selections and echo count
   * @param {number} nEchoes - Number of echo files loaded
   */
  updateVisibility(nEchoes) {
    const isRawMode = this.inputMode === 'raw' || this.inputMode === 'dicom';
    const isTotalFieldMode = this.inputMode === 'totalField';
    const isLocalFieldMode = this.inputMode === 'localField';
    const isFieldMapMode = isTotalFieldMode || isLocalFieldMode;

    const combinedMethod = this._getEl('combinedMethod');
    const phaseOffsetMethod = this._getEl('phaseOffsetMethod') || 'mcpc3ds';
    const isTgv = combinedMethod === 'tgv';
    const isQsmart = combinedMethod === 'qsmart';
    const isCombined = isTgv || isQsmart;
    const isMcpc3ds = phaseOffsetMethod === 'mcpc3ds';
    const isMultiEcho = nEchoes > 1;

    // Combined method selector - available in all modes
    const combinedMethodGroup = document.getElementById('combinedMethod')?.closest('.param-group');
    if (combinedMethodGroup) combinedMethodGroup.style.display = '';

    // TGV settings - show when TGV selected in any mode
    this._showEl('tgvSettings', isTgv);

    // QSMART settings - show when QSMART selected in any mode
    this._showEl('qsmartSettings', isQsmart);

    // Phase offset removal
    const phaseOffsetEnabled = this._getChecked('phaseOffsetEnabled') ?? true;
    this._disableEl('phaseOffsetContent', !phaseOffsetEnabled);
    this._showWarning('phaseOffsetEnabled', 'phaseOffsetWarning',
      phaseOffsetEnabled && nEchoes === 1,
      'Requires multi-echo data', 'error');

    // Hide the old locked hint (no longer forcing ROMEO)
    const hint = document.getElementById('unwrapLockedHint');
    if (hint) hint.style.display = 'none';

    // Bipolar correction
    const bipolarEnabled = this._getChecked('bipolarCorrectionEnabled');
    this._showWarning('bipolarCorrectionEnabled', 'bipolarWarning',
      bipolarEnabled && nEchoes >= 1 && nEchoes < 3,
      nEchoes === 1 ? 'Requires multi-echo data (3+ echoes)' : 'Requires 3+ echoes',
      'error');

    // Phase unwrapping
    const currentUnwrapMethod = this._getEl('unwrapMethod') || 'romeo';
    this._showEl('romeoSettings', currentUnwrapMethod === 'romeo');
    this._showEl('laplacianSettings', currentUnwrapMethod === 'laplacian');
    this._showWarning('unwrapMethod', 'laplacianMultiEchoWarning',
      currentUnwrapMethod === 'laplacian' && isMultiEcho,
      'No temporal coherence — ROMEO recommended for multi-echo',
      'warning');

    // Phase Gradient Coherence only meaningful for multi-echo
    const pgcLabel = document.getElementById('romeoPgcLabel');
    if (pgcLabel) pgcLabel.style.display = isMultiEcho ? '' : 'none';

    // Multi-echo fitting
    const fieldCalcMethod = this._getEl('fieldCalculationMethod') || 'weighted_avg';
    this._showEl('weightedAvgSettings', fieldCalcMethod === 'weighted_avg');
    this._showEl('linearFitSettings', fieldCalcMethod === 'linear_fit');
    this._showWarning('fieldCalculationMethod', 'multiEchoFittingWarning',
      nEchoes === 1, 'Requires multi-echo data', 'error');

    // Background removal - show for:
    // - Raw mode standard pipeline (not TGV/QSMART)
    // - Total field standard pipeline
    // - Total field + QSMART (QSMART uses SDF for total field)
    // NOT shown for: TGV (handles BG removal internally), local field, raw+QSMART (handled internally)
    const showBgRemoval = (!isCombined && isRawMode) || (isTotalFieldMode && !isTgv && !isQsmart);
    this._showEl('bgRemovalSection', showBgRemoval);

    // Dipole inversion - show for standard pipeline only (TGV/QSMART handle inversion internally)
    const showDipoleInversion = (!isCombined && isRawMode) || (!isCombined && isFieldMapMode);
    this._showEl('dipoleInversionSection', showDipoleInversion);

    // Check if MEDI with SMV is enabled - show error on background removal
    const dipoleMethod = this._getEl('dipoleMethod');
    const mediSmvEnabled = this._getChecked('mediSmv');
    const bgDisabledByMediSmv = dipoleMethod === 'medi' && mediSmvEnabled && showBgRemoval;

    const bgHint = document.getElementById('bgRemovalDisabledHint');
    if (bgHint) bgHint.style.display = bgDisabledByMediSmv ? '' : 'none';

    // Enable/disable tabs based on pipeline state
    this._setTabEnabled('tabPhaseProcessing', isRawMode);
    this._setTabEnabled('tabBgRemoval', showBgRemoval);
    this._setTabEnabled('tabDipoleInversion', showDipoleInversion);

    // Show errors for magnitude-dependent features when no magnitude is available
    const noMag = this.hasMagnitude === false;

    // QSMART option in combined method dropdown
    const combinedSelect = document.getElementById('combinedMethod');
    if (combinedSelect) {
      this._showWarning('combinedMethod', 'combinedMethodWarning',
        noMag && combinedSelect.value === 'qsmart',
        'Requires magnitude', 'error');
    }

    // MEDI option in dipole inversion dropdown
    const dipoleSelect = document.getElementById('dipoleMethod');
    if (dipoleSelect) {
      this._showWarning('dipoleMethod', 'dipoleMethodWarning',
        noMag && dipoleSelect.value === 'medi',
        'Requires magnitude', 'error');
    }

    // ROMEO magnitude weight checkboxes
    const romeoMagCoh = document.getElementById('romeoMagCoherence');
    const romeoMagWt = document.getElementById('romeoMagWeight');
    if (romeoMagCoh) {
      this._showWarning('romeoMagCoherence', 'romeoMagCohWarning',
        noMag && romeoMagCoh.checked, 'Requires magnitude', 'error');
    }
    if (romeoMagWt) {
      this._showWarning('romeoMagWeight', 'romeoMagWtWarning',
        noMag && romeoMagWt.checked, 'Requires magnitude', 'error');
    }

    // B0 weight type — phase_snr, phase_var, and mag all require magnitude
    const b0WeightSelect = document.getElementById('b0WeightType');
    if (b0WeightSelect) {
      const magWeights = ['phase_snr', 'phase_var', 'mag'];
      this._showWarning('b0WeightType', 'b0WeightWarning',
        noMag && magWeights.includes(b0WeightSelect.value),
        'Requires magnitude', 'error');
    }
  }

  // ---- Tab management ----

  _setupTabs() {
    const tabBar = this.modal.querySelector('.pipeline-tabs');
    if (!tabBar) return;
    tabBar.addEventListener('click', (e) => {
      const btn = e.target.closest('.pipeline-tab');
      if (!btn || btn.disabled) return;
      this._switchTab(btn.dataset.tab);
    });
  }

  _switchTab(tabId) {
    const tabBar = this.modal.querySelector('.pipeline-tabs');
    if (!tabBar) return;
    for (const btn of tabBar.querySelectorAll('.pipeline-tab')) {
      btn.classList.toggle('active', btn.dataset.tab === tabId);
    }
    const body = this.modal.querySelector('.modal-body');
    if (!body) return;
    for (const panel of body.querySelectorAll('.tab-panel')) {
      panel.classList.toggle('active', panel.id === tabId);
    }
  }

  _setTabEnabled(tabId, enabled) {
    const btn = this.modal.querySelector(`.pipeline-tab[data-tab="${tabId}"]`);
    if (!btn) return;
    btn.disabled = !enabled;
    // If the disabled tab was active, switch to first enabled tab
    if (!enabled && btn.classList.contains('active')) {
      const first = this.modal.querySelector('.pipeline-tab:not(:disabled)');
      if (first) this._switchTab(first.dataset.tab);
    }
  }

  // ---- Private helpers ----

  _populateForm(settings, defaults) {
    // Combined method
    this._setEl('combinedMethod', settings.combinedMethod || 'none');

    // SWI settings
    const swiSettings = settings.swi || {};
    this._setEl('swiScaling', swiSettings.scaling || 'tanh');
    this._setEl('swiStrength', swiSettings.strength ?? 4);
    this._setEl('swiHpSigmaX', swiSettings.hpSigma?.[0] ?? 4);
    this._setEl('swiHpSigmaY', swiSettings.hpSigma?.[1] ?? 4);
    this._setEl('swiHpSigmaZ', swiSettings.hpSigma?.[2] ?? 0);
    this._setEl('swiMipWindow', swiSettings.mipWindow ?? 7);

    // TGV settings
    this._setEl('tgvRegularization', settings.tgv.regularization);
    this._setEl('tgvIterations', settings.tgv.iterations);
    this._setEl('tgvErosions', settings.tgv.erosions);

    // Phase offset
    const phaseOffsetMethod = settings.phaseOffsetMethod || 'mcpc3ds';
    this._setChecked('phaseOffsetEnabled', phaseOffsetMethod !== 'none');
    this._setEl('phaseOffsetMethod', phaseOffsetMethod === 'none' ? 'mcpc3ds' : phaseOffsetMethod);

    // MCPC-3D-S settings
    this._setEl('mcpc3dsSigmaX', settings.mcpc3ds?.sigma?.[0] ?? 10);
    this._setEl('mcpc3dsSigmaY', settings.mcpc3ds?.sigma?.[1] ?? 10);
    this._setEl('mcpc3dsSigmaZ', settings.mcpc3ds?.sigma?.[2] ?? 5);

    // Phase unwrap method
    const unwrapMethod = settings.unwrapMethod || 'romeo';
    this._setEl('unwrapMethod', unwrapMethod);
    this._showEl('romeoSettings', unwrapMethod === 'romeo');
    this._showEl('laplacianSettings', unwrapMethod === 'laplacian');

    // ROMEO weight component checkboxes
    const romeoSettings = settings.romeo || {};
    this._setChecked('romeoPhaseGradientCoherence', romeoSettings.phaseGradientCoherence !== false);
    this._setChecked('romeoMagCoherence', romeoSettings.magCoherence !== false);
    this._setChecked('romeoMagWeight', romeoSettings.magWeight !== false);

    // Field calculation method
    const fieldCalcMethod = settings.fieldCalculationMethod || 'weighted_avg';
    this._setEl('fieldCalculationMethod', fieldCalcMethod);
    this._showEl('weightedAvgSettings', fieldCalcMethod === 'weighted_avg');
    this._showEl('linearFitSettings', fieldCalcMethod === 'linear_fit');

    // B0 weight type
    this._setEl('b0WeightType', settings.b0WeightType ?? 'phase_snr');

    // Linear fit settings
    this._setChecked('linearFitEstimateOffset', settings.linearFit?.estimateOffset ?? true);

    // Background removal method
    const bgMethod = settings.backgroundRemoval;
    this._setEl('bgRemovalMethod', bgMethod);
    this._showEl('vsharpSettings', bgMethod === 'vsharp');
    this._showEl('sharpSettings', bgMethod === 'sharp');
    this._showEl('resharpSettings', bgMethod === 'resharp');
    this._showEl('ismvSettings', bgMethod === 'ismv');
    this._showEl('pdfSettings', bgMethod === 'pdf');
    this._showEl('lbvSettings', bgMethod === 'lbv');
    this._showEl('harperellaSettings', bgMethod === 'harperella');
    this._showEl('iharperellaSettings', bgMethod === 'iharperella');

    // V-SHARP settings
    this._setEl('vsharpMaxRadius', settings.vsharp.maxRadius ?? defaults.vsharpMaxRadius);
    this._setEl('vsharpMinRadius', settings.vsharp.minRadius ?? defaults.vsharpMinRadius);
    this._setEl('vsharpThreshold', settings.vsharp.threshold);

    // RESHARP settings
    if (settings.resharp) {
      this._setEl('resharpRadius', settings.resharp.radius);
      this._setEl('resharpTikReg', settings.resharp.tikReg);
      this._setEl('resharpTol', settings.resharp.tol);
      this._setEl('resharpMaxIter', settings.resharp.maxIter);
    }

    // HARPERELLA settings
    if (settings.harperella) {
      this._setEl('harperellaRadius', settings.harperella.radius);
      this._setEl('harperellaMaxIter', settings.harperella.maxIter);
    }

    // iHARPERELLA settings
    if (settings.iharperella) {
      this._setEl('iharperellaRadius', settings.iharperella.radius);
      this._setEl('iharperellaMaxIter', settings.iharperella.maxIter);
    }

    // iSMV settings
    this._setEl('ismvRadius', settings.ismv.radius ?? defaults.ismvRadius);
    this._setEl('ismvTol', settings.ismv.tol);
    this._setEl('ismvMaxit', settings.ismv.maxit);

    // PDF settings
    this._setEl('pdfTol', settings.pdf.tol);
    this._setEl('pdfMaxit', settings.pdf.maxit ?? defaults.pdfMaxit);

    // LBV settings
    this._setEl('lbvTol', settings.lbv.tol);
    this._setEl('lbvMaxit', settings.lbv.maxit ?? defaults.lbvMaxit);

    // Dipole inversion method
    const dipoleMethod = settings.dipoleInversion;
    this._setEl('dipoleMethod', dipoleMethod);
    this._showEl('tkdSettings', dipoleMethod === 'tkd');
    this._showEl('tsvdSettings', dipoleMethod === 'tsvd');
    this._showEl('tikhonovSettings', dipoleMethod === 'tikhonov');
    this._showEl('tvSettings', dipoleMethod === 'tv');
    this._showEl('rtsSettings', dipoleMethod === 'rts');
    this._showEl('nltvSettings', dipoleMethod === 'nltv');
    this._showEl('mediSettings', dipoleMethod === 'medi');
    this._showEl('ilsqrSettings', dipoleMethod === 'ilsqr');

    // TKD settings
    this._setEl('tkdThreshold', settings.tkd.threshold);

    // TSVD settings
    this._setEl('tsvdThreshold', settings.tsvd.threshold);

    // Tikhonov settings
    this._setEl('tikhLambda', settings.tikhonov.lambda);
    this._setEl('tikhReg', settings.tikhonov.reg);

    // TV-ADMM settings
    this._setEl('tvLambda', settings.tv.lambda);
    this._setEl('tvMaxIter', settings.tv.maxIter);
    this._setEl('tvTol', settings.tv.tol);

    // RTS settings
    this._setEl('rtsDelta', settings.rts.delta);
    this._setEl('rtsMu', settings.rts.mu);
    this._setEl('rtsRho', settings.rts.rho);
    this._setEl('rtsMaxIter', settings.rts.maxIter);

    // NLTV settings
    this._setEl('nltvLambda', settings.nltv.lambda);
    this._setEl('nltvMu', settings.nltv.mu);
    this._setEl('nltvMaxIter', settings.nltv.maxIter);
    this._setEl('nltvTol', settings.nltv.tol);
    this._setEl('nltvNewtonMaxIter', settings.nltv.newtonMaxIter);

    // MEDI settings
    this._setEl('mediLambda', settings.medi.lambda);
    this._setEl('mediPercentage', settings.medi.percentage);
    this._setEl('mediMaxIter', settings.medi.maxIter);
    this._setEl('mediCgMaxIter', settings.medi.cgMaxIter);
    this._setChecked('mediSmv', settings.medi.smv);
    this._setEl('mediSmvRadius', settings.medi.smvRadius);
    this._showEl('mediSmvRadiusGroup', settings.medi.smv);
    this._setChecked('mediMerit', settings.medi.merit);

    // iLSQR settings
    this._setEl('ilsqrTol', settings.ilsqr?.tol || 0.01);
    this._setEl('ilsqrMaxIter', settings.ilsqr?.maxIter || 50);
  }

  _setupEventListeners() {
    // Combined method dropdown - show/hide TGV/QSMART settings
    this._on('combinedMethod', 'change', () => this._onCombinedMethodChange());

    // Phase offset enabled checkbox
    this._on('phaseOffsetEnabled', 'change', () => this._onCombinedMethodChange());

    // Bipolar correction checkbox
    this._on('bipolarCorrectionEnabled', 'change', () => this._onCombinedMethodChange());

    // Unwrap method dropdown
    this._on('unwrapMethod', 'change', () => this._onCombinedMethodChange());

    // Background removal method dropdown
    this._on('bgRemovalMethod', 'change', (e) => {
      const method = e.target.value;
      this._showEl('vsharpSettings', method === 'vsharp');
      this._showEl('sharpSettings', method === 'sharp');
      this._showEl('resharpSettings', method === 'resharp');
      this._showEl('ismvSettings', method === 'ismv');
      this._showEl('pdfSettings', method === 'pdf');
      this._showEl('lbvSettings', method === 'lbv');
      this._showEl('harperellaSettings', method === 'harperella');
      this._showEl('iharperellaSettings', method === 'iharperella');
    });

    // MEDI SMV checkbox toggle
    this._on('mediSmv', 'change', (e) => {
      this._showEl('mediSmvRadiusGroup', e.target.checked);
      this._onCombinedMethodChange(); // Re-check visibility
    });

    // Dipole method change
    this._on('dipoleMethod', 'change', (e) => {
      const method = e.target.value;
      this._showEl('tkdSettings', method === 'tkd');
      this._showEl('tsvdSettings', method === 'tsvd');
      this._showEl('tikhonovSettings', method === 'tikhonov');
      this._showEl('tvSettings', method === 'tv');
      this._showEl('rtsSettings', method === 'rts');
      this._showEl('nltvSettings', method === 'nltv');
      this._showEl('mediSettings', method === 'medi');
      this._showEl('ilsqrSettings', method === 'ilsqr');
      this._onCombinedMethodChange(); // Re-check visibility for MEDI SMV
    });

    // Phase offset method dropdown
    this._on('phaseOffsetMethod', 'change', () => this._onCombinedMethodChange());

    // Field calculation method dropdown
    this._on('fieldCalculationMethod', 'change', () => this._onCombinedMethodChange());

    // ROMEO magnitude weight checkboxes - re-evaluate warnings on change
    ['romeoMagCoherence', 'romeoMagWeight'].forEach(id => {
      this._on(id, 'change', () => this._onCombinedMethodChange());
    });

    // BET fractional intensity slider value display
    this._on('betFractionalIntensity', 'input', (e) => {
      const valueEl = document.getElementById('betFractionalIntensityValue');
      if (valueEl) valueEl.textContent = e.target.value;
    });
  }

  _onCombinedMethodChange() {
    this.updateVisibility(this.nEchoes || 0);
  }

  // DOM helper methods
  _getEl(id) {
    const el = document.getElementById(id);
    return el ? el.value : null;
  }

  _setEl(id, value) {
    const el = document.getElementById(id);
    if (el) el.value = value;
  }

  _getChecked(id) {
    const el = document.getElementById(id);
    return el ? el.checked : null;
  }

  _setChecked(id, checked) {
    const el = document.getElementById(id);
    if (el) el.checked = checked;
  }

  _showEl(id, show) {
    const el = document.getElementById(id);
    if (el) el.style.display = show ? 'block' : 'none';
  }

  _disableEl(id, disabled) {
    const el = document.getElementById(id);
    if (!el) return;
    el.style.opacity = disabled ? '0.4' : '';
    el.style.pointerEvents = disabled ? 'none' : '';
    for (const input of el.querySelectorAll('input, select, button')) {
      input.disabled = disabled;
    }
  }


  /**
   * Show/hide an inline warning message near a control
   * @param {string} anchorId - ID of the element to attach warning after
   * @param {string} warningId - Unique ID for the warning element
   * @param {boolean} show - Whether to show the warning
   * @param {string} message - Warning text
   * @param {string} [type='warning'] - 'warning' or 'info'
   */
  _showWarning(anchorId, warningId, show, message, type = 'warning') {
    let warning = document.getElementById(warningId);
    const anchor = document.getElementById(anchorId);

    if (show) {
      if (!warning && anchor) {
        warning = document.createElement('span');
        warning.id = warningId;
        warning.className = `validation-message ${type} inline-warning`;
        warning.innerHTML = '<span></span>';
        // If anchor is inside a <label>, insert after the label instead
        const insertAfter = anchor.parentNode.tagName === 'LABEL' ? anchor.parentNode : anchor;
        insertAfter.parentNode.insertBefore(warning, insertAfter.nextSibling);
      }
      if (warning) {
        warning.querySelector('span').textContent = message;
        warning.style.display = '';
      }
    } else if (warning) {
      warning.style.display = 'none';
    }
  }

  /**
   * Show/hide a warning banner at the top of a section (replaces _disableSection)
   * All inputs remain interactive.
   * @param {string} id - Section element ID
   * @param {boolean} hasWarning - Whether to show the warning
   * @param {string} [warningText] - Warning text
   */
  _showSectionWarning(id, hasWarning, warningText) {
    const section = document.getElementById(id);
    if (!section) return;

    const warningId = id + 'Warning';
    let warning = document.getElementById(warningId);

    if (hasWarning && warningText) {
      if (!warning) {
        warning = document.createElement('div');
        warning.id = warningId;
        warning.className = 'validation-message error inline-warning';
        warning.innerHTML = '<span></span>';
        const heading = section.querySelector('h4');
        if (heading) {
          heading.after(warning);
        } else {
          section.prepend(warning);
        }
      }
      warning.querySelector('span').textContent = warningText;
      warning.style.display = 'flex';
    } else if (warning) {
      warning.style.display = 'none';
    }
  }

  _on(id, event, handler) {
    const el = document.getElementById(id);
    if (el) el.addEventListener(event, handler);
  }
}
