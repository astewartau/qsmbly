/**
 * Pipeline Settings Controller
 *
 * Manages the pipeline settings modal UI - form population, visibility toggling,
 * reset to defaults, and reading form values.
 */

export class PipelineSettingsController {
  constructor(modalElement) {
    this.modal = modalElement;
    this.inputMode = 'raw'; // 'raw', 'totalField', or 'localField'
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
    this._populateForm(settings, defaults);
    this.updateVisibility(nEchoes);
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
    // Combined method - reset to standard pipeline
    this._setEl('combinedMethod', 'none');

    // TGV defaults
    this._setEl('tgvRegularization', 2);
    this._setEl('tgvIterations', 1000);
    this._setEl('tgvErosions', 3);

    // QSMART defaults
    this._setEl('qsmartSdfSigma1Stage1', 10);
    this._setEl('qsmartSdfSigma2Stage1', 0);
    this._setEl('qsmartSdfSigma1Stage2', 8);
    this._setEl('qsmartSdfSigma2Stage2', 2);
    this._setEl('qsmartSdfSpatialRadius', 8);
    this._setEl('qsmartSdfLowerLim', 0.6);
    this._setEl('qsmartSdfCurvConstant', 500);
    this._setEl('qsmartVascSphereRadius', 8);
    this._setEl('qsmartFrangiScaleMin', 1.0);
    this._setEl('qsmartFrangiScaleMax', 10.0);
    this._setEl('qsmartFrangiScaleRatio', 2.0);
    this._setEl('qsmartFrangiC', 500);
    this._setEl('qsmartIlsqrTol', 0.01);
    this._setEl('qsmartIlsqrMaxIter', 50);

    // Phase offset method - default to MCPC-3D-S
    this._setEl('phaseOffsetMethod', 'mcpc3ds');
    this._showEl('mcpc3dsSettings', true);
    this._setEl('mcpc3dsSigmaX', 10);
    this._setEl('mcpc3dsSigmaY', 10);
    this._setEl('mcpc3dsSigmaZ', 5);

    // Unwrap method (auto-set to ROMEO when MCPC-3D-S — no error since they match)
    this._setEl('unwrapMethod', 'romeo');
    const resetHint = document.getElementById('unwrapLockedHint');
    if (resetHint) resetHint.style.display = 'none';
    this._showEl('romeoSettings', true);
    this._showEl('laplacianSettings', false);

    // ROMEO weight checkboxes - all enabled by default
    this._setChecked('romeoPhaseGradientCoherence', true);
    this._setChecked('romeoMagCoherence', true);
    this._setChecked('romeoMagWeight', true);

    // Field calculation method - default to weighted averaging
    this._setEl('fieldCalculationMethod', 'weighted_avg');
    this._showEl('weightedAvgSettings', true);
    this._showEl('linearFitSettings', false);
    this._setEl('b0WeightType', 'phase_snr');

    // Linear fit defaults
    this._setChecked('linearFitEstimateOffset', true);

    // Single-echo unwrap (sync with multi-echo)
    this._setEl('singleEchoUnwrapMethod', 'romeo');
    this._showEl('singleEchoRomeoSettings', true);
    this._setChecked('singleEchoRomeoMagCoherence', true);
    this._setChecked('singleEchoRomeoMagWeight', true);

    // Background removal - default to V-SHARP
    this._setEl('bgRemovalMethod', 'vsharp');
    this._showEl('vsharpSettings', true);
    this._showEl('sharpSettings', false);
    this._showEl('smvSettings', false);
    this._showEl('ismvSettings', false);
    this._showEl('pdfSettings', false);
    this._showEl('lbvSettings', false);

    this._setEl('vsharpMaxRadius', defaults.vsharpMaxRadius);
    this._setEl('vsharpMinRadius', defaults.vsharpMinRadius);
    this._setEl('vsharpThreshold', 0.05);
    this._setEl('sharpRadius', 6);
    this._setEl('sharpThreshold', 0.05);
    this._setEl('smvRadius', defaults.smvRadius);

    // iSMV defaults
    this._setEl('ismvRadius', defaults.ismvRadius);
    this._setEl('ismvTol', 0.001);
    this._setEl('ismvMaxit', 500);

    // PDF defaults
    this._setEl('pdfTol', 0.00001);
    this._setEl('pdfMaxit', defaults.pdfMaxit);

    // LBV defaults
    this._setEl('lbvTol', 0.001);
    this._setEl('lbvMaxit', 500);

    // Dipole inversion - default to RTS
    this._setEl('dipoleMethod', 'rts');
    this._showEl('tkdSettings', false);
    this._showEl('tsvdSettings', false);
    this._showEl('tikhonovSettings', false);
    this._showEl('tvSettings', false);
    this._showEl('rtsSettings', true);
    this._showEl('nltvSettings', false);
    this._showEl('mediSettings', false);
    this._showEl('ilsqrSettings', false);

    // TKD
    this._setEl('tkdThreshold', 0.15);

    // TSVD
    this._setEl('tsvdThreshold', 0.15);

    // Tikhonov
    this._setEl('tikhLambda', 0.01);
    this._setEl('tikhReg', 'identity');

    // TV-ADMM
    this._setEl('tvLambda', 0.001);
    this._setEl('tvMaxIter', 250);
    this._setEl('tvTol', 0.001);

    // RTS
    this._setEl('rtsDelta', 0.15);
    this._setEl('rtsMu', 100000);
    this._setEl('rtsRho', 10);
    this._setEl('rtsMaxIter', 20);

    // NLTV
    this._setEl('nltvLambda', 0.001);
    this._setEl('nltvMu', 1);
    this._setEl('nltvMaxIter', 250);
    this._setEl('nltvTol', 0.001);
    this._setEl('nltvNewtonMaxIter', 10);

    // MEDI
    this._setEl('mediLambda', 1000);
    this._setEl('mediPercentage', 0.9);
    this._setEl('mediMaxIter', 10);
    this._setEl('mediCgMaxIter', 100);
    this._setChecked('mediSmv', false);
    this._setEl('mediSmvRadius', 5);
    this._showEl('mediSmvRadiusGroup', false);
    this._setChecked('mediMerit', false);

    // iLSQR
    this._setEl('ilsqrTol', 0.01);
    this._setEl('ilsqrMaxIter', 50);
  }

  /**
   * Read form values and return settings object
   * @param {number} nEchoes - Number of echo files loaded
   * @returns {Object} Pipeline settings object
   */
  save(nEchoes) {
    const isMultiEcho = nEchoes > 1;

    // Get unwrap method from appropriate dropdown based on echo count
    // Auto-correct: MCPC-3D-S requires ROMEO regardless of dropdown value
    const phaseOffset = this._getEl('phaseOffsetMethod') || 'mcpc3ds';
    const unwrapMethod = isMultiEcho
      ? (phaseOffset === 'mcpc3ds' ? 'romeo' : this._getEl('unwrapMethod'))
      : this._getEl('singleEchoUnwrapMethod');

    // Get ROMEO weight settings from checkboxes
    const romeoPhaseGradientCoherence = isMultiEcho
      ? this._getChecked('romeoPhaseGradientCoherence') ?? true
      : true;
    const romeoMagCoherence = isMultiEcho
      ? this._getChecked('romeoMagCoherence') ?? true
      : this._getChecked('singleEchoRomeoMagCoherence') ?? true;
    const romeoMagWeight = isMultiEcho
      ? this._getChecked('romeoMagWeight') ?? true
      : this._getChecked('singleEchoRomeoMagWeight') ?? true;

    return {
      combinedMethod: this._getEl('combinedMethod'),
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
      phaseOffsetMethod: this._getEl('phaseOffsetMethod') || 'mcpc3ds',
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
      smv: {
        radius: parseFloat(this._getEl('smvRadius'))
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

    // Multi-echo section - only in raw mode with multi-echo data
    // (single-echo has its own separate unwrap section, so hide this to avoid duplication)
    const multiEchoSectionEl = document.getElementById('multiEchoSection');
    if (multiEchoSectionEl) {
      multiEchoSectionEl.style.display = (isRawMode && isMultiEcho) ? '' : 'none';
    }

    // MCPC-3D-S settings and unwrap (raw mode only)
    this._showEl('mcpc3dsSettings', isMcpc3ds && isRawMode);
    if (isRawMode) {
      if (isMcpc3ds) this._setEl('unwrapMethod', 'romeo');
      // Show error only when MCPC-3D-S is active and unwrap is not ROMEO
      const unwrapVal = this._getEl('unwrapMethod') || 'romeo';
      const hint = document.getElementById('unwrapLockedHint');
      if (hint) hint.style.display = (isMcpc3ds && unwrapVal !== 'romeo') ? 'flex' : 'none';
    }

    // Unwrap settings visibility (raw mode only)
    const currentUnwrapMethod = this._getEl('unwrapMethod') || 'romeo';
    this._showEl('romeoSettings', currentUnwrapMethod === 'romeo' && isRawMode);
    this._showEl('laplacianSettings', currentUnwrapMethod === 'laplacian' && isRawMode);

    // Field calculation settings visibility (raw mode only)
    const fieldCalcMethod = this._getEl('fieldCalculationMethod') || 'weighted_avg';
    this._showEl('weightedAvgSettings', fieldCalcMethod === 'weighted_avg' && isRawMode);
    this._showEl('linearFitSettings', fieldCalcMethod === 'linear_fit' && isRawMode);

    // Single-echo unwrap section - show only for single-echo + standard pipeline + raw mode
    this._showEl('singleEchoUnwrapSection', !isMultiEcho && !isCombined && isRawMode);

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
    if (bgHint) bgHint.style.display = bgDisabledByMediSmv ? 'flex' : 'none';

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

    // ROMEO magnitude weight checkboxes — multi-echo section
    const romeoMagCoh = document.getElementById('romeoMagCoherence');
    const romeoMagWt = document.getElementById('romeoMagWeight');
    this._showWarning('romeoSettings', 'romeoMagWarning',
      noMag && (romeoMagCoh?.checked || romeoMagWt?.checked),
      'Requires magnitude', 'error');

    // ROMEO magnitude weight checkboxes — single-echo section
    const singleRomeoMagCoh = document.getElementById('singleEchoRomeoMagCoherence');
    const singleRomeoMagWt = document.getElementById('singleEchoRomeoMagWeight');
    this._showWarning('singleEchoRomeoSettings', 'singleEchoRomeoMagWarning',
      noMag && (singleRomeoMagCoh?.checked || singleRomeoMagWt?.checked),
      'Requires magnitude', 'error');

    // B0 weight phase_snr option
    const b0WeightSelect = document.getElementById('b0WeightType');
    if (b0WeightSelect) {
      this._showWarning('b0WeightType', 'b0WeightWarning',
        noMag && b0WeightSelect.value === 'phase_snr',
        'Requires magnitude', 'error');
    }
  }

  // ---- Private helpers ----

  _populateForm(settings, defaults) {
    // Combined method
    this._setEl('combinedMethod', settings.combinedMethod || 'none');

    // TGV settings
    this._setEl('tgvRegularization', settings.tgv.regularization);
    this._setEl('tgvIterations', settings.tgv.iterations);
    this._setEl('tgvErosions', settings.tgv.erosions);

    // Phase offset method
    const phaseOffsetMethod = settings.phaseOffsetMethod || 'mcpc3ds';
    this._setEl('phaseOffsetMethod', phaseOffsetMethod);

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

    // Show error only when MCPC-3D-S + non-ROMEO
    const isMcpc3ds = phaseOffsetMethod === 'mcpc3ds';
    const popHint = document.getElementById('unwrapLockedHint');
    if (popHint) popHint.style.display = (isMcpc3ds && unwrapMethod !== 'romeo') ? 'flex' : 'none';

    // Single-echo unwrap method (sync with multi-echo settings)
    this._setEl('singleEchoUnwrapMethod', unwrapMethod);
    this._showEl('singleEchoRomeoSettings', unwrapMethod === 'romeo');
    this._setChecked('singleEchoRomeoMagCoherence', romeoSettings.magCoherence !== false);
    this._setChecked('singleEchoRomeoMagWeight', romeoSettings.magWeight !== false);

    // MCPC-3D-S settings
    this._setEl('mcpc3dsSigmaX', settings.mcpc3ds?.sigma?.[0] ?? 10);
    this._setEl('mcpc3dsSigmaY', settings.mcpc3ds?.sigma?.[1] ?? 10);
    this._setEl('mcpc3dsSigmaZ', settings.mcpc3ds?.sigma?.[2] ?? 5);
    this._showEl('mcpc3dsSettings', isMcpc3ds);

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
    this._showEl('smvSettings', bgMethod === 'smv');
    this._showEl('ismvSettings', bgMethod === 'ismv');
    this._showEl('pdfSettings', bgMethod === 'pdf');
    this._showEl('lbvSettings', bgMethod === 'lbv');

    // V-SHARP settings
    this._setEl('vsharpMaxRadius', settings.vsharp.maxRadius ?? defaults.vsharpMaxRadius);
    this._setEl('vsharpMinRadius', settings.vsharp.minRadius ?? defaults.vsharpMinRadius);
    this._setEl('vsharpThreshold', settings.vsharp.threshold);

    // SMV settings
    this._setEl('smvRadius', settings.smv.radius ?? defaults.smvRadius);

    // iSMV settings
    this._setEl('ismvRadius', settings.ismv.radius ?? defaults.ismvRadius);
    this._setEl('ismvTol', settings.ismv.tol);
    this._setEl('ismvMaxit', settings.ismv.maxit);

    // PDF settings
    this._setEl('pdfTol', settings.pdf.tol);
    this._setEl('pdfMaxit', settings.pdf.maxit ?? defaults.pdfMaxit);

    // LBV settings
    this._setEl('lbvTol', settings.lbv.tol);
    this._setEl('lbvMaxit', settings.lbv.maxit);

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

    // Unwrap method dropdown
    this._on('unwrapMethod', 'change', (e) => {
      const isRomeo = e.target.value === 'romeo';
      this._showEl('romeoSettings', isRomeo);
      this._showEl('laplacianSettings', !isRomeo);
      // Show MCPC-3D-S incompatibility error if needed
      const isMcpc3ds = this._getEl('phaseOffsetMethod') === 'mcpc3ds';
      const hint = document.getElementById('unwrapLockedHint');
      if (hint) hint.style.display = (isMcpc3ds && !isRomeo) ? 'flex' : 'none';
    });

    // Single-echo unwrap method dropdown
    this._on('singleEchoUnwrapMethod', 'change', (e) => {
      const isRomeo = e.target.value === 'romeo';
      this._showEl('singleEchoRomeoSettings', isRomeo);
    });

    // Background removal method dropdown
    this._on('bgRemovalMethod', 'change', (e) => {
      const method = e.target.value;
      this._showEl('vsharpSettings', method === 'vsharp');
      this._showEl('sharpSettings', method === 'sharp');
      this._showEl('smvSettings', method === 'smv');
      this._showEl('ismvSettings', method === 'ismv');
      this._showEl('pdfSettings', method === 'pdf');
      this._showEl('lbvSettings', method === 'lbv');
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
    ['romeoMagCoherence', 'romeoMagWeight', 'singleEchoRomeoMagCoherence', 'singleEchoRomeoMagWeight'].forEach(id => {
      this._on(id, 'change', () => this._onCombinedMethodChange());
    });

    // BET fractional intensity slider value display
    this._on('betFractionalIntensity', 'input', (e) => {
      const valueEl = document.getElementById('betFractionalIntensityValue');
      if (valueEl) valueEl.textContent = e.target.value;
    });
  }

  _onCombinedMethodChange() {
    // Get nEchoes from DOM (count phase file items)
    const phaseList = document.getElementById('phaseList');
    const nEchoes = phaseList ? phaseList.querySelectorAll('.file-item').length : 0;
    this.updateVisibility(nEchoes);
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
        warning = document.createElement('div');
        warning.id = warningId;
        warning.className = `validation-message ${type} inline-warning`;
        warning.innerHTML = '<span></span>';
        anchor.parentNode.insertBefore(warning, anchor.nextSibling);
      }
      if (warning) {
        warning.querySelector('span').textContent = message;
        warning.style.display = 'flex';
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
