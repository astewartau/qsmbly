// Import extracted utility modules
import { createThresholdMask } from './modules/mask/ThresholdUtils.js';
import {
  parseNiftiHeader,
  isGzipped,
  isValidNifti1,
  readNiftiImageData,
  createMaskNifti,
  createNiftiHeaderFromVolume,
  createFloat64Nifti
} from './modules/file-io/NiftiUtils.js';
import { ConsoleOutput } from './modules/ui/ConsoleOutput.js';
import { ModalManager } from './modules/ui/ModalManager.js';
import { ProgressManager } from './modules/ui/ProgressManager.js';
import { EchoNavigator } from './modules/viewer/EchoNavigator.js';
import { FileIOController, PipelineExecutor, PipelineSettingsController, MaskController, ViewerController } from './controllers/index.js';
import { DicomController } from './controllers/DicomController.js';
import { DicompareController } from './controllers/DicompareController.js';
import { DicompareReportRenderer } from './modules/ui/DicompareReportRenderer.js';
import * as QSMConfig from './app/config.js';

// Make config available globally for backward compatibility
window.QSMConfig = QSMConfig;

class QSMApp {
  constructor() {
    // Config is required - no fallbacks
    const cfg = window.QSMConfig;

    this.nv = new window.Niivue({
      ...cfg.VIEWER_CONFIG,
      onLocationChange: (data) => {
        document.getElementById("intensity").innerHTML = data.string;
      }
    });
    this.currentFile = null;
    this.threshold = 75;
    this.progress = 0;

    // Smooth progress animation state
    this.targetProgress = 0;
    this.animatedProgress = 0;
    this.progressAnimationId = null;
    this.lastAnimationTime = 0;
    this.progressAnimationSpeed = cfg.PROGRESS_CONFIG.animationSpeed;

    // Controllers (initialized in init() after DOM ready)
    this.fileIOController = null;
    this.pipelineExecutor = null;

    // Mask threshold (percentage of max magnitude)
    this.maskThreshold = cfg.MASK_CONFIG.defaultThreshold;
    this.magnitudeData = null;
    this.magnitudeMax = 0;

    // Mask editing state
    this.currentMaskData = null;
    this.originalMaskData = null;
    this.maskDims = null;
    this.voxelSize = null;

    // Drawing state
    this.drawingEnabled = false;
    this.brushMode = 'add';
    this.brushSize = cfg.MASK_CONFIG.defaultBrushSize;
    this.savedCrosshairWidth = cfg.VIEWER_CONFIG.crosshairWidth;

    // Pipeline settings from config
    this.pipelineSettings = JSON.parse(JSON.stringify(cfg.PIPELINE_DEFAULTS));

    // BET settings from config
    this.betSettings = { ...cfg.BET_DEFAULTS };

    // Mask preparation settings from config
    this.maskPrepSettings = { ...cfg.MASK_PREP_DEFAULTS, prepared: false };
    this.preparedMagnitudeData = null;
    this.preparedMagnitudeMax = 0;

    // Echo navigation state
    this.currentEchoIndex = 0;
    this.currentViewType = null;

    // Controllers (initialized in init() after DOM ready)
    this.pipelineSettingsController = null;
    this.maskController = null;
    this.viewerController = null;

    // Modal managers (initialized in init() after DOM ready)
    this.betModal = null;
    this.aboutModal = null;
    this.citationsModal = null;
    this.privacyModal = null;

    this.init();
  }

  // Getter for backward compatibility - delegates to FileIOController
  get multiEchoFiles() {
    return this.fileIOController?.getMultiEchoFiles() || {
      magnitude: [], phase: [], json: [], echoTimes: [],
      combinedMagnitude: null, combinedPhase: null
    };
  }

  // Getters for backward compatibility - delegates to PipelineExecutor
  get pipelineRunning() {
    return this.pipelineExecutor?.isRunning() || false;
  }

  get worker() {
    return this.pipelineExecutor?.getWorker() || null;
  }

  get workerReady() {
    return this.pipelineExecutor?.isReady() || false;
  }

  get results() {
    return this.pipelineExecutor?.getResults() || {};
  }

  get stageOrder() {
    return this.pipelineExecutor?.getStageOrder() || [];
  }

  async init() {
    // Display version in header
    const versionEl = document.getElementById('appVersion');
    if (versionEl && window.QSMConfig?.VERSION) {
      versionEl.textContent = `v${window.QSMConfig.VERSION}`;
    }

    // Initialize FileIOController first (other controllers depend on it)
    this.fileIOController = new FileIOController({
      updateOutput: (msg) => this.updateOutput(msg),
      onFilesChanged: () => this.updateEchoInfo(),
      onMagnitudeFilesChanged: (files) => this._onMagnitudeFilesChanged(files),
      onPhaseFilesChanged: (files) => this._onPhaseFilesChanged(files)
    });
    this.fileIOController.setupEchoTagify();

    await this.setupViewer();
    this.setupUIControls();
    this.setupEventListeners();
    this.syncSidebarFromSettings();
    this.updateDownloadButtons();

    // Initialize file lists via controller
    this.fileIOController.updateFileList('magnitude', []);
    this.fileIOController.updateFileList('phase', []);
    this.fileIOController.updateFileList('json', []);

    // Initialize masking controls state (disabled until Prepare is clicked)
    this.updateMaskingControlsState();

    // Sync mask prep settings with actual UI state
    const sourceSelect = document.getElementById('maskInputSource');
    const biasCheckbox = document.getElementById('applyBiasCorrection');
    if (sourceSelect) this.maskPrepSettings.source = sourceSelect.value;
    if (biasCheckbox) this.maskPrepSettings.biasCorrection = biasCheckbox.checked;

    // Initialize controllers
    const pipelineModal = document.getElementById('pipelineSettingsModal');
    if (pipelineModal) {
      this.pipelineSettingsController = new PipelineSettingsController(pipelineModal);
    }

    // Initialize pipeline executor (before mask controller, provides worker)
    this.pipelineExecutor = new PipelineExecutor({
      updateOutput: (msg) => this.updateOutput(msg),
      setProgress: (val, text) => this.setProgress(val, text),
      onStageData: (data) => this._onStageData(data),
      onPipelineComplete: () => this._onPipelineComplete(),
      onPipelineError: () => this._onPipelineError(),
      config: window.QSMConfig
    });

    // Initialize mask controller
    this.maskController = new MaskController({
      nv: this.nv,
      getWorker: () => this.pipelineExecutor?.getWorker(),
      updateOutput: (msg) => this.updateOutput(msg),
      setProgress: (val, text) => this.setProgress(val, text),
      initializeWorker: () => this.pipelineExecutor?.initialize(),
      config: window.QSMConfig
    });

    // Initialize viewer controller
    this.viewerController = new ViewerController({
      nv: this.nv,
      getMultiEchoFiles: () => this.multiEchoFiles,
      updateOutput: (msg) => this.updateOutput(msg),
      showOverlayControl: (show) => this.showOverlayControl(show),
      updateDownloadVolumeButton: () => this.updateDownloadVolumeButton()
    });

    // Initialize dicompare controller
    this.dicompareController = new DicompareController({
      updateOutput: (msg) => this.updateOutput(msg)
    });
    this.dicompareRenderer = new DicompareReportRenderer();

    // Initialize DICOM controller
    this.dicomController = new DicomController({
      updateOutput: (msg) => this.updateOutput(msg),
      onConversionComplete: (classified) => this._onDicomConversionComplete(classified),
      onFilesRetained: (files) => this._onDicomFilesRetained(files)
    });

    // Initialize modal managers
    this.betModal = new ModalManager('betSettingsModal');
    this.aboutModal = new ModalManager('aboutModal');
    this.citationsModal = new ModalManager('citationsModal');
    this.privacyModal = new ModalManager('privacyModal');
    this.dicompareModal = new ModalManager('dicompareModal');

    // Start loading WASM in the background immediately
    this.pipelineExecutor.initialize();
  }

  // Pipeline executor callbacks
  _onStageData(data) {
    // displayNow defaults to true for backward compatibility
    const displayNow = data.displayNow !== false;
    if (displayNow) {
      this.displayLiveStageData(data);
    } else {
      this.cacheStageData(data);
    }
  }

  _onPipelineComplete() {
    this.showStageButtons();
    document.getElementById('cancelPipeline').disabled = true;
    document.getElementById('runSWI').disabled = false;
    this.updateEchoInfo();
  }

  _onPipelineError() {
    document.getElementById('cancelPipeline').disabled = true;
    document.getElementById('runSWI').disabled = false;
    this.updateEchoInfo();
  }

  async setupViewer() {
    await this.nv.attachTo("gl1");
    this.nv.setMultiplanarPadPixels(5);
    this.nv.setSliceType(this.nv.sliceTypeMultiplanar);
    // Clear any default loading text by drawing empty scene
    this.nv.drawScene();
  }

  setupUIControls() {
    // Sidebar toggle
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebar = document.getElementById('sidebar');
    if (sidebarToggle && sidebar) {
      sidebarToggle.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
      });
    }

    // Console toggle
    const consoleHeader = document.querySelector('.console-header');
    const consoleEl = document.getElementById('console');
    if (consoleHeader && consoleEl) {
      consoleHeader.addEventListener('click', () => {
        consoleEl.classList.toggle('collapsed');
      });
    }

    // Stage tab switching
    const stageTabs = document.querySelectorAll('.stage-tab');
    stageTabs.forEach(tab => {
      tab.addEventListener('click', (e) => {
        if (e.target.classList.contains('tab-download')) return;
        stageTabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
      });
    });
  }

  setProgress(value, text = null) {
    this.progress = value;
    this.targetProgress = value;

    const textEl = document.getElementById('progressText');
    if (textEl) textEl.textContent = text || `${Math.round(value * 100)}%`;

    // Update progress bar immediately for accurate feedback
    this.animatedProgress = value;
    this.updateProgressBar();

    // Stop any running animation since we update immediately
    this.stopProgressAnimation();
  }

  animateProgress() {
    const now = performance.now();
    const deltaTime = (now - this.lastAnimationTime) / 1000; // Convert to seconds
    this.lastAnimationTime = now;

    // Move animated progress toward target, but don't exceed it
    if (this.animatedProgress < this.targetProgress) {
      // Calculate how much to move based on time elapsed
      const increment = this.progressAnimationSpeed * deltaTime;
      this.animatedProgress = Math.min(this.animatedProgress + increment, this.targetProgress);
      this.updateProgressBar();
    }
    // If animated progress has caught up to target, we just wait (pause)
    // The bar will resume moving when a new setProgress call increases targetProgress

    // Continue animation loop if not complete
    if (this.targetProgress < 1 && this.targetProgress > 0) {
      this.progressAnimationId = requestAnimationFrame(() => this.animateProgress());
    } else {
      this.progressAnimationId = null;
    }
  }

  updateProgressBar() {
    const fill = document.getElementById('progressFill');
    if (fill) {
      fill.style.width = `${this.animatedProgress * 100}%`;
    }
  }

  stopProgressAnimation() {
    if (this.progressAnimationId) {
      cancelAnimationFrame(this.progressAnimationId);
      this.progressAnimationId = null;
    }
  }

  setupEventListeners() {
    // Input mode tab switching (top-level: dicom vs nifti)
    document.querySelectorAll('.input-mode-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        const topMode = tab.dataset.mode;
        if (topMode === 'dicom') {
          this.switchInputMode('dicom');
        } else {
          // NIfTI selected - use the active sub-type
          const activeSub = document.querySelector('.input-subtype-tab.active');
          this.switchInputMode(activeSub?.dataset.mode || 'raw');
        }
      });
    });

    // NIfTI sub-type tab switching
    document.querySelectorAll('.input-subtype-tab').forEach(tab => {
      tab.addEventListener('click', () => this.switchInputMode(tab.dataset.mode));
    });

    // Field map units dropdown
    document.getElementById('fieldMapUnits')?.addEventListener('change', () => {
      this.updateInputParamsVisibility();
      this.updateEchoInfo();
    });

    // Multi-echo file inputs (raw mode)
    document.getElementById('magnitudeFiles').addEventListener('change', (e) => {
      this.handleMultipleFiles(e, 'magnitude');
    });

    document.getElementById('phaseFiles').addEventListener('change', (e) => {
      this.handleMultipleFiles(e, 'phase');
    });

    document.getElementById('jsonFiles').addEventListener('change', (e) => {
      this.handleMultipleFiles(e, 'json');
    });

    // Total field map mode file inputs
    document.getElementById('totalFieldFiles')?.addEventListener('change', (e) => {
      this.handleMultipleFiles(e, 'totalField');
    });
    document.getElementById('magnitudeTFFiles')?.addEventListener('change', (e) => {
      this.handleMultipleFiles(e, 'magnitudeTF');
    });

    // Local field map mode file inputs
    document.getElementById('localFieldFiles')?.addEventListener('change', (e) => {
      this.handleMultipleFiles(e, 'localField');
    });
    document.getElementById('magnitudeLFFiles')?.addEventListener('change', (e) => {
      this.handleMultipleFiles(e, 'magnitudeLF');
    });

    // Centralized mask file input (in Masking section)
    document.getElementById('maskFiles')?.addEventListener('change', (e) => {
      this.handleMultipleFiles(e, 'mask');
    });

    // DICOM file input (folder picker)
    document.getElementById('dicomFiles')?.addEventListener('change', (e) => {
      this.handleDicomFiles(e);
    });

    // dicompare report
    document.getElementById('dicompareReportBtn')?.addEventListener('click', () => this.runDicompareReport());
    document.getElementById('closeDicompare')?.addEventListener('click', () => this.dicompareModal?.close());
    document.getElementById('closeDicompare2')?.addEventListener('click', () => this.dicompareModal?.close());
    document.getElementById('dicomparePrint')?.addEventListener('click', () => this.printDicompareReport());

    // Preview buttons for field map modes
    document.getElementById('vis_totalField')?.addEventListener('click', () => this.visualizeFieldMap('totalField'));
    document.getElementById('vis_localField')?.addEventListener('click', () => this.visualizeFieldMap('localField'));
    document.getElementById('vis_magnitudeTF')?.addEventListener('click', () => this.visualizeFieldMapMagnitude('magnitudeTF'));
    document.getElementById('vis_magnitudeLF')?.addEventListener('click', () => this.visualizeFieldMapMagnitude('magnitudeLF'));
    document.getElementById('vis_mask')?.addEventListener('click', () => this.visualizeMaskFile());

    // Sidebar pipeline dropdowns
    this.setupSidebarDropdownListeners();

    // Processing buttons
    document.getElementById('openPipelineSettings').addEventListener('click', () => this.openPipelineSettingsModal());
    document.getElementById('cancelPipeline')?.addEventListener('click', () => this.cancelPipeline());
    document.getElementById('vis_magnitude').addEventListener('click', () => this.visualizeMagnitude());
    document.getElementById('vis_phase').addEventListener('click', () => this.visualizePhase());

    // Echo navigation
    document.getElementById('echoPrev')?.addEventListener('click', () => this.navigateEcho(-1));
    document.getElementById('echoNext')?.addEventListener('click', () => this.navigateEcho(1));

    // Note: Stage show/download buttons are now created dynamically in addStageButton()

    // Mask threshold slider with debounce
    const thresholdSlider = document.getElementById('maskThreshold');
    if (thresholdSlider) {
      thresholdSlider.addEventListener('input', (e) => {
        this.maskThreshold = parseInt(e.target.value);
        document.getElementById('thresholdLabel').textContent = `Threshold (${this.maskThreshold}%)`;

        // Debounce the mask preview update
        if (this.maskUpdateTimeout) {
          clearTimeout(this.maskUpdateTimeout);
        }
        this.maskUpdateTimeout = setTimeout(() => {
          if (this.magnitudeData && !this.maskUpdating) {
            this.updateMaskPreview();
          }
        }, 150);
      });
    }

    // Preview mask button - shows Robust/Manual sub-buttons
    const previewMaskBtn = document.getElementById('previewMask');
    if (previewMaskBtn) {
      previewMaskBtn.addEventListener('click', () => {
        document.getElementById('thresholdModeButtons').style.display = '';
      });
    }

    // Threshold Robust button - Otsu + auto-refinement
    document.getElementById('thresholdRobust')?.addEventListener('click', async () => {
      document.getElementById('thresholdModeButtons').style.display = 'none';
      await this.previewMask();
      this.updateOutput("Applying robust refinement (dilate, fill holes, erode x2)...");
      this.dilateMask3D();
      this.fillHoles3D();
      this.erodeMask3D();
      this.erodeMask3D();
      await this.displayCurrentMask();
      this.updateOutput("Robust mask complete");
    });

    // Threshold Manual button - Otsu + slider
    document.getElementById('thresholdManual')?.addEventListener('click', async () => {
      document.getElementById('thresholdModeButtons').style.display = 'none';
      await this.previewMask();
      const sliderGroup = document.getElementById('thresholdSliderGroup');
      if (sliderGroup) sliderGroup.style.display = '';
      this.setThresholdSliderEnabled(true);
    });

    // BET brain extraction button - opens settings modal
    document.getElementById('runBET')?.addEventListener('click', () => this.openBetSettingsModal());

    // Auto threshold button (Otsu)
    document.getElementById('autoThreshold')?.addEventListener('click', () => this.autoDetectThreshold());

    // Mask Input Preparation
    document.getElementById('maskInputSource')?.addEventListener('change', (e) => {
      this.maskPrepSettings.source = e.target.value;
      this.maskPrepSettings.prepared = false;
      this.updatePrepareButtonState();

      // Hide bias correction for phase quality map (not applicable)
      const biasCorrectionGroup = document.getElementById('biasCorrectionGroup');
      if (biasCorrectionGroup) {
        biasCorrectionGroup.style.display = e.target.value === 'phase_quality' ? 'none' : '';
      }
    });

    document.getElementById('applyBiasCorrection')?.addEventListener('change', (e) => {
      this.maskPrepSettings.biasCorrection = e.target.checked;
      this.maskPrepSettings.prepared = false;
      console.log('Bias correction checkbox changed:', e.target.checked);
      this.updatePrepareButtonState();
    });

    document.getElementById('prepareMaskInput')?.addEventListener('click', () => {
      this.prepareMaskInput();
    });

    // Pipeline settings modal
    document.getElementById('closePipelineSettings')?.addEventListener('click', () => this.closePipelineSettingsModal());
    document.getElementById('resetPipelineSettings')?.addEventListener('click', () => this.resetPipelineSettings());
    document.getElementById('savePipelineSettings')?.addEventListener('click', () => this.savePipelineSettings());
    document.getElementById('runPipelineSidebar')?.addEventListener('click', () => this.runPipelineFromSidebar());
    document.getElementById('runSWI')?.addEventListener('click', () => this.runSWI());

    // BET settings modal
    document.getElementById('closeBetSettings')?.addEventListener('click', () => this.betModal?.close());
    document.getElementById('resetBetSettings')?.addEventListener('click', () => this.resetBetSettings());
    document.getElementById('runBetWithSettings')?.addEventListener('click', () => this.runBetWithSettings());

    // About modal
    document.getElementById('openAbout')?.addEventListener('click', () => {
      const cfg = window.QSMConfig;
      const appVer = document.getElementById('aboutAppVersion');
      if (appVer && cfg?.VERSION) appVer.textContent = `v${cfg.VERSION}`;
      const coreVer = document.getElementById('aboutCoreVersion');
      if (coreVer && cfg?.QSM_RS_VERSION) coreVer.textContent = `v${cfg.QSM_RS_VERSION}`;
      this.aboutModal?.open();
    });
    document.getElementById('closeAbout')?.addEventListener('click', () => this.aboutModal?.close());

    // Citations modal
    document.getElementById('openCitations')?.addEventListener('click', () => this.citationsModal?.open());
    document.getElementById('closeCitations')?.addEventListener('click', () => this.citationsModal?.close());

    // Privacy modal
    document.getElementById('openPrivacy')?.addEventListener('click', () => this.privacyModal?.open());
    document.getElementById('closePrivacy')?.addEventListener('click', () => this.privacyModal?.close());

    // BET fractional intensity slider value display
    document.getElementById('betFractionalIntensity')?.addEventListener('input', (e) => {
      document.getElementById('betFractionalIntensityValue').textContent = e.target.value;
    });

    // Overlay opacity slider
    const opacitySlider = document.getElementById('overlayOpacity');
    if (opacitySlider) {
      opacitySlider.addEventListener('input', (e) => {
        const opacity = parseInt(e.target.value) / 100;
        document.getElementById('overlayOpacityValue').textContent = `${e.target.value}%`;
        this.updateOverlayOpacity(opacity);
      });
    }

    // Download current volume button
    document.getElementById('downloadCurrentVolume')?.addEventListener('click', () => {
      this.downloadCurrentVolume();
    });

    // Screenshot button
    document.getElementById('screenshotViewer')?.addEventListener('click', () => {
      this.saveScreenshot();
    });

    // Clear all results button
    document.getElementById('clearAllResults')?.addEventListener('click', () => {
      this.clearAllResults();
    });

    // Close modals on overlay click (BET and Citations handled by ModalManager)
    document.getElementById('pipelineSettingsModal')?.addEventListener('click', (e) => {
      if (e.target.id === 'pipelineSettingsModal') this.closePipelineSettingsModal();
    });

    // Note: Pipeline settings form event listeners are now handled by PipelineSettingsController

    // Morphological operation buttons
    document.getElementById('fillHoles')?.addEventListener('click', async () => {
      this.updateOutput("Filling holes in mask...");
      this.fillHoles3D();
      await this.displayCurrentMask();
      this.updateOutput("Holes filled");
    });

    document.getElementById('erodeMask')?.addEventListener('click', async () => {
      this.updateOutput("Eroding mask...");
      this.erodeMask3D();
      await this.displayCurrentMask();
      this.updateOutput("Mask eroded");
    });

    document.getElementById('dilateMask')?.addEventListener('click', async () => {
      this.updateOutput("Dilating mask...");
      this.dilateMask3D();
      await this.displayCurrentMask();
      this.updateOutput("Mask dilated");
    });

    document.getElementById('resetMask')?.addEventListener('click', async () => {
      this.updateOutput("Clearing mask...");
      await this.clearMask();
      this.updateOutput("Mask cleared. Choose Threshold or BET to create a new mask.");
    });

    // Drawing controls
    document.getElementById('enableDrawing')?.addEventListener('click', async () => {
      await this.toggleDrawingMode();
    });

    document.getElementById('brushAdd')?.addEventListener('click', () => {
      this.setBrushMode('add');
    });

    document.getElementById('brushRemove')?.addEventListener('click', () => {
      this.setBrushMode('remove');
    });

    document.getElementById('brushSize')?.addEventListener('input', (e) => {
      this.brushSize = parseInt(e.target.value);
      document.getElementById('brushSizeValue').textContent = this.brushSize;
      if (this.drawingEnabled) {
        this.nv.setPenValue(this.brushMode === 'add' ? 1 : 0, false);
        this.nv.opts.penSize = this.brushSize;
      }
    });

    document.getElementById('undoDraw')?.addEventListener('click', () => {
      this.nv.drawUndo();
    });

    document.getElementById('applyDrawing')?.addEventListener('click', async () => {
      await this.applyDrawingToMask();
    });
  }

  // Delegate file handling to FileIOController
  async handleMultipleFiles(event, type) {
    await this.fileIOController.handleFileInput(event, type);

    // Handle field map mode file changes
    const mode = this.fileIOController.getInputMode();
    if (type === 'totalField' || type === 'localField') {
      // Update preview button state
      const btnId = type === 'totalField' ? 'vis_totalField' : 'vis_localField';
      const file = type === 'totalField'
        ? this.fileIOController.getTotalFieldFile()
        : this.fileIOController.getLocalFieldFile();
      const btn = document.getElementById(btnId);
      if (btn) btn.disabled = !file;
    }

    // Handle magnitude files in field map modes
    if (type === 'magnitudeTF' || type === 'magnitudeLF') {
      const hasMag = this.fileIOController.hasFieldMapMagnitude();
      const btnId = type === 'magnitudeTF' ? 'vis_magnitudeTF' : 'vis_magnitudeLF';
      const btn = document.getElementById(btnId);
      if (btn) btn.disabled = !hasMag;
      this.updateMagnitudePrepSection();
      this.updateMaskInputSourceOptions();
      this.updateMaskSectionState();
      this.updatePrepareButtonState();
      this.updateSidebarDropdownVisibility(true);
    }

    // Handle centralized mask file
    if (type === 'mask') {
      const hasMask = this.fileIOController.hasMask();
      const btn = document.getElementById('vis_mask');
      if (btn) btn.disabled = !hasMask;
      this.updateMaskSectionState();
      this.updateEchoInfo();
    }
  }

  // Passthrough for backward compatibility (HTML onclick uses app.removeFile)
  removeFile(type, index) {
    this.fileIOController.removeFile(type, index);
  }

  // Callbacks from FileIOController
  _onMagnitudeFilesChanged(files) {
    document.getElementById('vis_magnitude').disabled = files.length === 0;

    // Clear prepared state when magnitude files change
    this.maskPrepSettings.prepared = false;
    this.preparedMagnitudeData = null;
    this.preparedMagnitudeMax = 0;
    this.currentMaskData = null;
    this.originalMaskData = null;

    // Update all dependent sections
    this.updateMagnitudePrepSection();
    this.updateMaskInputSourceOptions();
    this.updatePrepareButtonState();
    this.updateMaskSectionState();
    this.updateSidebarDropdownVisibility(true);

    if (files.length > 0) {
      this.visualizeMagnitude();
    }
  }

  _onPhaseFilesChanged(files) {
    document.getElementById('vis_phase').disabled = files.length === 0;
  }

  // ==================== DICOM Handling ====================

  /**
   * Handle DICOM files from the folder file input.
   */
  async handleDicomFiles(event) {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    this._showDicomStatus('Converting...');
    await this.dicomController.convertFiles(files);
    this._hideDicomStatus();
  }

  /**
   * Handle DICOM files from drag-and-drop (directory traversal).
   */
  async handleDicomDrop(dataTransferItems) {
    this._showDicomStatus('Converting...');
    await this.dicomController.convertDropItems(dataTransferItems);
    this._hideDicomStatus();
  }

  /**
   * Callback when DICOM conversion and classification is complete.
   */
  _onDicomConversionComplete(classified) {
    // Merge into triage state (supports incremental uploads)
    if (!this._triageState) {
      this._triageState = {
        magnitude: [],
        phase: [],
        extras: [],
        jsonFiles: [],
        fieldStrength: null,
        echoTimes: []
      };
    }
    this._triageState.magnitude.push(...classified.magnitude);
    this._triageState.phase.push(...classified.phase);
    this._triageState.extras.push(...(classified.extras || []));
    this._triageState.jsonFiles.push(...classified.jsonFiles);
    if (classified.fieldStrength != null) {
      this._triageState.fieldStrength = classified.fieldStrength;
    }
    // Collect unique echo times
    const teSet = new Set(this._triageState.echoTimes);
    for (const entry of [...classified.magnitude, ...classified.phase, ...(classified.extras || [])]) {
      if (entry.echoTime != null) teSet.add(entry.echoTime);
    }
    this._triageState.echoTimes = [...teSet].sort((a, b) => a - b);

    // Sort magnitude and phase by echo
    const sortByEcho = (a, b) => {
      if (a.echoTime != null && b.echoTime != null) return a.echoTime - b.echoTime;
      if (a.echoNumber != null && b.echoNumber != null) return a.echoNumber - b.echoNumber;
      return 0;
    };
    this._triageState.magnitude.sort(sortByEcho);
    this._triageState.phase.sort(sortByEcho);

    // Push to FileIOController and render triage UI
    this._syncTriageToFileIO();
    this._renderDicomTriage();

    // Update drop zone text for incremental uploads
    const dropLabel = document.querySelector('#dicomDrop .file-drop-label span');
    if (dropLabel) dropLabel.textContent = 'Drop more DICOM files';
    const dropZone = document.getElementById('dicomDrop');
    if (dropZone) dropZone.classList.add('has-files');
  }

  /**
   * Push current triage state (magnitude + phase) to FileIOController.
   * Extras are excluded from the pipeline.
   */
  _syncTriageToFileIO() {
    const ts = this._triageState;
    const magnitudeFileData = ts.magnitude.map(e => ({ file: e.file, name: e.name }));
    const phaseFileData = ts.phase.map(e => ({ file: e.file, name: e.name }));

    // Only pass JSON sidecars for files still in magnitude/phase (not extras)
    const activeBaseNames = new Set();
    for (const e of [...ts.magnitude, ...ts.phase]) {
      activeBaseNames.add(e.name.replace(/\.nii(\.gz)?$/, ''));
    }
    const jsonFiles = ts.jsonFiles.filter(f =>
      activeBaseNames.has(f.name.replace(/\.json$/, ''))
    );

    this.fileIOController.setFilesFromDicom(magnitudeFileData, phaseFileData, jsonFiles);
  }

  /**
   * Render the interactive DICOM triage UI with draggable file cards.
   */
  _renderDicomTriage() {
    const container = document.getElementById('dicomTriage');
    if (!container) return;
    container.style.display = '';

    const ts = this._triageState;
    const eyeSvg = '<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>';
    const gripSvg = '<svg viewBox="0 0 12 24" width="6" height="12" fill="currentColor"><circle cx="3" cy="4" r="1.5"/><circle cx="9" cy="4" r="1.5"/><circle cx="3" cy="12" r="1.5"/><circle cx="9" cy="12" r="1.5"/><circle cx="3" cy="20" r="1.5"/><circle cx="9" cy="20" r="1.5"/></svg>';
    const deleteSvg = '<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
    const buckets = [
      { key: 'magnitude', label: 'Magnitude', items: ts.magnitude },
      { key: 'phase', label: 'Phase', items: ts.phase },
      { key: 'extras', label: 'Extras', items: ts.extras }
    ];

    let html = '<div class="dicom-triage">';

    for (const bucket of buckets) {
      const isExtras = bucket.key === 'extras';
      const emptyClass = bucket.items.length === 0 ? ' empty' : '';

      html += `<div class="dicom-triage-bucket ${bucket.key}">`;
      html += `<div class="dicom-triage-bucket-header">`;
      html += `<span class="dicom-triage-bucket-label">${bucket.label} <span class="dicom-triage-bucket-count">(${bucket.items.length})</span></span>`;
      html += `<div class="dicom-triage-bucket-actions">`;

      if (isExtras && bucket.items.length > 0) {
        html += `<button class="btn-clear-extras" data-action="clearExtras" title="Remove all extras">Clear</button>`;
      }

      if (!isExtras && bucket.items.length > 0) {
        html += `<button class="btn-icon btn-preview dicom-triage-preview-btn" data-category="${bucket.key}" title="Preview ${bucket.label}">`;
        html += `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>`;
        html += `</button>`;
      }

      html += `</div></div>`; // close bucket-actions, bucket-header

      html += `<div class="dicom-triage-drop${emptyClass}" data-bucket="${bucket.key}">`;

      for (let i = 0; i < bucket.items.length; i++) {
        const item = bucket.items[i];
        const teLabel = item.echoTime != null ? `${item.echoTime.toFixed(1)} ms` : '';

        html += `<div class="dicom-triage-card" draggable="true" data-category="${bucket.key}" data-index="${i}">`;
        html += `<span class="dicom-triage-card-grip" aria-hidden="true">${gripSvg}</span>`;
        html += `<span class="dicom-triage-card-name" title="${item.name}">${item.name}</span>`;
        if (teLabel) {
          html += `<span class="dicom-triage-card-te">${teLabel}</span>`;
        }
        html += `<button class="dicom-triage-card-preview" data-category="${bucket.key}" data-index="${i}" title="Preview">${eyeSvg}</button>`;
        html += `<button class="dicom-triage-card-delete" data-category="${bucket.key}" data-index="${i}" title="Remove file">${deleteSvg}</button>`;
        html += `</div>`;
      }

      html += `</div>`; // close dicom-triage-drop
      html += `</div>`; // close dicom-triage-bucket
    }

    html += '</div>';
    container.innerHTML = html;

    this._setupTriageDragDrop(container);
    this._setupTriageClickHandlers(container);
  }

  /**
   * Set up HTML5 drag-and-drop between triage buckets.
   */
  _setupTriageDragDrop(container) {
    let dragData = null;

    container.querySelectorAll('.dicom-triage-card').forEach(card => {
      card.addEventListener('dragstart', (e) => {
        dragData = {
          category: card.dataset.category,
          index: parseInt(card.dataset.index)
        };
        card.classList.add('dragging');
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', '');
      });

      card.addEventListener('dragend', () => {
        card.classList.remove('dragging');
        dragData = null;
        container.querySelectorAll('.dicom-triage-drop.dragover')
          .forEach(el => el.classList.remove('dragover'));
      });
    });

    container.querySelectorAll('.dicom-triage-drop').forEach(dropZone => {
      dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        dropZone.classList.add('dragover');
      });

      dropZone.addEventListener('dragleave', (e) => {
        if (!dropZone.contains(e.relatedTarget)) {
          dropZone.classList.remove('dragover');
        }
      });

      dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (!dragData) return;

        const targetBucket = dropZone.dataset.bucket;
        const sourceBucket = dragData.category;
        if (targetBucket === sourceBucket) return;

        // Move item between arrays
        const ts = this._triageState;
        const [item] = ts[sourceBucket].splice(dragData.index, 1);
        ts[targetBucket].push(item);

        // Re-sort magnitude and phase by echo
        const sortByEcho = (a, b) => {
          if (a.echoTime != null && b.echoTime != null) return a.echoTime - b.echoTime;
          if (a.echoNumber != null && b.echoNumber != null) return a.echoNumber - b.echoNumber;
          return 0;
        };
        ts.magnitude.sort(sortByEcho);
        ts.phase.sort(sortByEcho);

        this._syncTriageToFileIO();
        this._renderDicomTriage();
      });
    });
  }

  /**
   * Set up click handlers for triage card previews and clear button.
   */
  _setupTriageClickHandlers(container) {
    // Individual card preview
    container.querySelectorAll('.dicom-triage-card-preview').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const category = btn.dataset.category;
        const index = parseInt(btn.dataset.index);
        const item = this._triageState[category]?.[index];
        if (item?.file) {
          this.viewerController.loadAndVisualizeFile(item.file, item.name);
        }
      });
    });

    // Bucket-level preview (eye in header)
    container.querySelectorAll('.dicom-triage-preview-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const category = btn.dataset.category;
        if (category === 'magnitude') this.visualizeMagnitude();
        else if (category === 'phase') this.visualizePhase();
      });
    });

    // Clear extras
    container.querySelector('[data-action="clearExtras"]')?.addEventListener('click', () => {
      this._triageState.extras = [];
      this._syncTriageToFileIO();
      this._renderDicomTriage();
    });

    // Per-card delete
    container.querySelectorAll('.dicom-triage-card-delete').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const category = btn.dataset.category;
        const index = parseInt(btn.dataset.index);
        const ts = this._triageState;
        if (ts[category] && index >= 0 && index < ts[category].length) {
          ts[category].splice(index, 1);
          this._syncTriageToFileIO();
          this._renderDicomTriage();
        }
      });
    });
  }

  _showDicomStatus(text) {
    const el = document.getElementById('dicomStatus');
    const textEl = document.getElementById('dicomStatusText');
    if (el) el.style.display = '';
    if (textEl) textEl.textContent = text;
  }

  _hideDicomStatus() {
    const el = document.getElementById('dicomStatus');
    if (el) el.style.display = 'none';
  }

  // ==================== dicompare Integration ====================

  /**
   * Callback when DICOM files are retained for validation.
   */
  async _onDicomFilesRetained(files) {
    await this.dicompareController.retainDicomFiles(files);
    const btn = document.getElementById('dicompareReportBtn');
    if (btn) {
      btn.disabled = files.length === 0;
    }
  }

  /**
   * Run dicompare validation and display results in modal.
   */
  async runDicompareReport() {
    if (!this.dicompareController.hasFiles()) {
      this.updateOutput('No DICOM files available for validation.');
      return;
    }

    const body = document.getElementById('dicompareModalBody');
    const footer = document.getElementById('dicompareModalFooter');

    // If results are already cached, just re-display them
    const cached = this.dicompareController.getCachedResults();
    if (cached) {
      this.dicompareModal.open();
      this.dicompareRenderer.render(body, cached);
      if (footer) footer.style.display = '';
      return;
    }

    // Open modal with loading state
    this.dicompareModal.open();
    if (body) {
      body.innerHTML = `
        <div class="dicompare-loading">
          <div class="dicompare-spinner"></div>
          <p class="dicompare-loading-text" id="dicompareLoadingText">Initializing Python runtime...</p>
          <div class="dicompare-progress-bar">
            <div class="dicompare-progress-fill" id="dicompareProgressFill"></div>
          </div>
        </div>
      `;
    }
    if (footer) footer.style.display = 'none';

    try {
      const result = await this.dicompareController.runValidation((progress) => {
        const textEl = document.getElementById('dicompareLoadingText');
        const fillEl = document.getElementById('dicompareProgressFill');
        if (textEl) textEl.textContent = progress.currentOperation;
        if (fillEl) fillEl.style.width = `${progress.percentage}%`;
      });

      // Render results
      this.dicompareRenderer.render(body, result);
      if (footer) footer.style.display = '';
    } catch (error) {
      if (body) {
        body.innerHTML = `
          <div class="dicompare-error">
            <p>Validation failed: ${error.message}</p>
          </div>
        `;
      }
      console.error('dicompare validation error:', error);
    }
  }

  /**
   * Print the dicompare report in a new window.
   */
  printDicompareReport() {
    if (!this.dicompareController.complianceResults) return;
    const html = this.dicompareRenderer.generatePrintHtml({
      acquisitions: this.dicompareController.acquisitions,
      complianceResults: this.dicompareController.complianceResults,
      schema: JSON.parse(this.dicompareController.schemaContent || '{}')
    });
    const printWindow = window.open('', '_blank');
    if (printWindow) {
      printWindow.document.write(html);
      printWindow.document.close();
      printWindow.focus();
    }
  }

  // ==================== Sidebar Pipeline Dropdowns ====================

  setupSidebarDropdownListeners() {
    const mappings = [
      { id: 'sidebarCombinedMethod', key: 'combinedMethod' },
      { id: 'sidebarUnwrapMethod', key: 'unwrapMethod' },
      { id: 'sidebarBgRemovalMethod', key: 'backgroundRemoval' },
      { id: 'sidebarDipoleMethod', key: 'dipoleInversion' }
    ];

    for (const { id, key } of mappings) {
      document.getElementById(id)?.addEventListener('change', (e) => {
        this.pipelineSettings[key] = e.target.value;
        this.updateSidebarDropdownVisibility();
        this.updateInputParamsVisibility();
        this.updateEchoInfo();
      });
    }

  }

  updateSidebarDropdownVisibility(autoCorrect = false) {
    const mode = this.fileIOController?.getInputMode() || 'raw';
    const isRaw = mode === 'raw' || mode === 'dicom';
    const isTotalField = mode === 'totalField';
    const combined = this.pipelineSettings?.combinedMethod || 'none';
    const isStandard = combined === 'none';

    // Phase unwrapping: raw mode + standard pipeline only
    const unwrapGroup = document.getElementById('sidebarUnwrapGroup');
    if (unwrapGroup) unwrapGroup.style.display = (isRaw && isStandard) ? '' : 'none';

    // Background removal: (raw + standard) or (totalField + standard)
    const bgGroup = document.getElementById('sidebarBgRemovalGroup');
    if (bgGroup) bgGroup.style.display = ((isRaw || isTotalField) && isStandard) ? '' : 'none';

    // Dipole inversion: standard pipeline, any mode
    const dipoleGroup = document.getElementById('sidebarDipoleGroup');
    if (dipoleGroup) dipoleGroup.style.display = isStandard ? '' : 'none';

    // Magnitude gating
    const hasMagnitude = isRaw
      ? (this.multiEchoFiles?.magnitude?.length > 0)
      : (this.fileIOController?.hasFieldMapMagnitude() || this.preparedMagnitudeData !== null);
    const noMag = !hasMagnitude;

    // Auto-correct to safe defaults when data changes
    const combinedSelect = document.getElementById('sidebarCombinedMethod');
    if (combinedSelect) {
      if (autoCorrect && noMag && combinedSelect.value === 'qsmart') {
        combinedSelect.value = 'none';
        this.pipelineSettings.combinedMethod = 'none';
      }
      this._showSidebarWarning('sidebarCombinedMethod', 'sidebarCombinedWarning',
        noMag && combinedSelect.value === 'qsmart',
        'Requires magnitude');
    }

    const dipoleSelect = document.getElementById('sidebarDipoleMethod');
    if (dipoleSelect) {
      if (autoCorrect && noMag && dipoleSelect.value === 'medi') {
        dipoleSelect.value = 'rts';
        this.pipelineSettings.dipoleInversion = 'rts';
      }
      this._showSidebarWarning('sidebarDipoleMethod', 'sidebarDipoleWarning',
        noMag && dipoleSelect.value === 'medi',
        'Requires magnitude');
    }
  }

  _showSidebarWarning(anchorId, warningId, show, message) {
    let warning = document.getElementById(warningId);
    const anchor = document.getElementById(anchorId);

    if (show) {
      if (!warning && anchor) {
        warning = document.createElement('div');
        warning.id = warningId;
        warning.className = 'validation-message error inline-warning';
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

  syncSidebarFromSettings() {
    const s = this.pipelineSettings;
    const set = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.value = val;
    };
    set('sidebarCombinedMethod', s?.combinedMethod || 'none');
    set('sidebarUnwrapMethod', s?.unwrapMethod || 'romeo');
    set('sidebarBgRemovalMethod', s?.backgroundRemoval || 'vsharp');
    set('sidebarDipoleMethod', s?.dipoleInversion || 'tv');
    this.updateSidebarDropdownVisibility();
  }

  // ==================== Input Mode Switching ====================

  switchInputMode(mode) {
    const isDicom = mode === 'dicom';
    const subtypeTabs = document.getElementById('inputSubtypeTabs');

    // Update top-level tab UI (dicom vs nifti)
    document.querySelectorAll('.input-mode-tab').forEach(tab => {
      tab.classList.toggle('active', isDicom ? tab.dataset.mode === 'dicom' : tab.dataset.mode === 'nifti');
    });

    // Show/hide NIfTI sub-type tabs
    subtypeTabs.style.display = isDicom ? 'none' : '';

    // Update sub-type tab active state
    if (!isDicom) {
      document.querySelectorAll('.input-subtype-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.mode === mode);
      });
    }

    // Update content panels
    document.querySelectorAll('.input-mode-content').forEach(content => {
      content.classList.toggle('active', content.dataset.mode === mode);
    });

    // Update controller
    this.fileIOController.setInputMode(mode);

    // Update input parameters visibility
    this.updateInputParamsVisibility();

    // Update magnitude prep and masking sections
    this.updateMagnitudePrepSection();
    this.updateMaskInputSourceOptions();
    this.updateMaskSectionState();

    // Update pipeline settings visibility
    if (this.pipelineSettingsController) {
      this.pipelineSettingsController.setInputMode(mode);
    }

    // Update sidebar pipeline dropdowns visibility (auto-correct for new mode)
    this.updateSidebarDropdownVisibility(true);

    // Update sidebar pipeline section labels
    this.updatePipelineSectionForMode(mode);

    // Update run button state
    this.updateEchoInfo();

    // Update preview button states for field map modes
    if (mode === 'totalField') {
      const hasFile = this.fileIOController.getTotalFieldFile() !== null;
      document.getElementById('vis_totalField').disabled = !hasFile;
      const hasMag = this.fileIOController.hasFieldMapMagnitude();
      document.getElementById('vis_magnitudeTF').disabled = !hasMag;
    } else if (mode === 'localField') {
      const hasFile = this.fileIOController.getLocalFieldFile() !== null;
      document.getElementById('vis_localField').disabled = !hasFile;
      const hasMag = this.fileIOController.hasFieldMapMagnitude();
      document.getElementById('vis_magnitudeLF').disabled = !hasMag;
    }
  }

  updateInputParamsVisibility() {
    const mode = this.fileIOController.getInputMode();
    const isRaw = mode === 'raw' || mode === 'dicom';
    const units = this.fileIOController.getFieldMapUnits();
    const combinedMethod = this.pipelineSettings?.combinedMethod || 'none';
    // Field strength needed for: raw mode, Hz/rad_s units, or TGV/QSMART (internal scaling)
    const needsFieldStrength = isRaw || units !== 'ppm' || combinedMethod !== 'none';

    // Show/hide raw-mode-only parameters
    const isNifti = mode === 'raw';
    document.getElementById('jsonMetadataGroup').style.display = isNifti ? '' : 'none';
    document.getElementById('echoTimesGroup').style.display = isRaw ? '' : 'none';

    // Show/hide field map units (only for field map modes)
    document.getElementById('fieldMapUnitsGroup').style.display = isRaw ? 'none' : '';

    // Show/hide field strength
    document.getElementById('fieldStrengthGroup').style.display =
      needsFieldStrength ? '' : 'none';
  }

  updateMagnitudePrepSection() {
    const section = document.getElementById('magnitudePrepSection');
    if (!section) return;

    const mode = this.fileIOController.getInputMode();
    let hasMag = false;

    if (mode === 'raw' || mode === 'dicom') {
      hasMag = this.multiEchoFiles.magnitude.length > 0;
    } else {
      hasMag = this.fileIOController.hasFieldMapMagnitude();
    }

    // Show/hide inline warning instead of greying out
    let warning = document.getElementById('magnitudePrepWarning');
    if (!hasMag) {
      if (!warning) {
        warning = document.createElement('div');
        warning.id = 'magnitudePrepWarning';
        warning.className = 'validation-message error inline-warning';
        warning.innerHTML = '<span>Requires magnitude</span>';
        const content = section.querySelector('.section-content');
        if (content) content.prepend(warning);
      }
      warning.style.display = 'flex';
    } else if (warning) {
      warning.style.display = 'none';
    }

    // Keep the Prepare button individually disabled when no magnitude (it's an action, not a setting)
    const prepareBtn = document.getElementById('prepareMaskInput');
    if (prepareBtn) prepareBtn.disabled = !hasMag;
  }

  updateMaskSectionState() {
    const section = document.getElementById('maskSection');
    if (!section) return;

    const hasMaskFile = this.fileIOController.hasMask();
    const hasPrepared = this.maskPrepSettings.prepared;

    // Show/hide the "or generate from magnitude" divider
    const divider = document.getElementById('maskDivider');
    if (divider) {
      divider.style.display = hasPrepared ? '' : 'none';
    }

    // When mask file uploaded: disable Threshold/BET generation controls + show info note
    const maskFileNote = document.getElementById('maskFileUploadedNote');
    if (hasMaskFile) {
      const generateButtons = document.getElementById('maskGenerateButtons');
      if (generateButtons) generateButtons.style.opacity = '0.5';
      document.getElementById('previewMask')?.setAttribute('disabled', '');
      document.getElementById('runBET')?.setAttribute('disabled', '');
      document.getElementById('maskThreshold')?.setAttribute('disabled', '');
      const maskOps = document.getElementById('maskOperations');
      if (maskOps) maskOps.style.display = 'none';
      // Show info note
      if (!maskFileNote) {
        const note = document.createElement('div');
        note.id = 'maskFileUploadedNote';
        note.className = 'validation-message info inline-warning';
        note.innerHTML = '<span>Using uploaded mask file. Remove it to use generated masks.</span>';
        const generateBtns = document.getElementById('maskGenerateButtons');
        if (generateBtns) generateBtns.parentNode.insertBefore(note, generateBtns.nextSibling);
      } else {
        maskFileNote.style.display = 'flex';
      }
    } else {
      if (maskFileNote) maskFileNote.style.display = 'none';
      if (hasPrepared) {
        // Magnitude prepared: enable generation controls
        const generateButtons = document.getElementById('maskGenerateButtons');
        if (generateButtons) generateButtons.style.opacity = '1';
        document.getElementById('previewMask')?.removeAttribute('disabled');
        document.getElementById('runBET')?.removeAttribute('disabled');
      }
    }
  }

  updateMaskInputSourceOptions() {
    const mode = this.fileIOController.getInputMode();
    let magCount = 0;

    if (mode === 'raw' || mode === 'dicom') {
      magCount = this.multiEchoFiles.magnitude.length;
    } else {
      magCount = this.fileIOController.getFieldMapMagnitudeCount();
    }

    const select = document.getElementById('maskInputSource');
    if (!select) return;

    const combinedOption = select.querySelector('option[value="combined"]');
    if (combinedOption) {
      combinedOption.disabled = magCount <= 1;
      combinedOption.title = magCount <= 1 ? 'Requires 2+ magnitude files for RSS combination' : '';
      // Force to first_echo if combined was selected but only 1 file
      if (magCount <= 1 && select.value === 'combined') {
        select.value = 'first_echo';
        this.maskPrepSettings.source = 'first_echo';
      }
    }
  }

  updatePipelineSectionForMode(mode) {
    // Update the run button label based on input mode
    const runButton = document.getElementById('runPipelineSidebar');
    if (runButton) {
      const label = runButton.querySelector('span');
      if (label) {
        const labels = {
          raw: 'Start QSM',
          totalField: 'Start QSM',
          localField: 'Start QSM'
        };
        label.textContent = labels[mode] || 'Start QSM';
      }
    }
  }

  async visualizeFieldMap(type) {
    const file = type === 'totalField'
      ? this.fileIOController.getTotalFieldFile()
      : this.fileIOController.getLocalFieldFile();
    if (!file) return;

    const label = type === 'totalField' ? 'Total Field Map' : 'Local Field Map';
    await this.loadAndVisualizeFile(file, label);
    this.hideEchoNavigation();
  }

  async visualizeFieldMapMagnitude(type) {
    const file = this.fileIOController.getFieldMapMagnitudeFile();
    if (!file) return;

    await this.loadAndVisualizeFile(file, 'Magnitude');
    this.hideEchoNavigation();
  }

  async visualizeMaskFile() {
    const file = this.fileIOController.getMaskFile();
    if (!file) return;

    await this.loadAndVisualizeFile(file, 'Mask');
    this.hideEchoNavigation();
  }

  // Update run button state based on file/mask state (mode-aware)
  updateEchoInfo() {
    const mode = this.fileIOController?.getInputMode() || 'raw';
    const isValid = this.fileIOController?.hasValidData() || false;
    const combinedMethod = this.pipelineSettings?.combinedMethod || 'none';
    let canRun = false;

    switch (mode) {
      case 'dicom':
      case 'raw': {
        const hasEchoTimes = this.fileIOController?.hasEchoTimes() || false;
        const hasMask = this.currentMaskData !== null;
        canRun = isValid && hasEchoTimes && hasMask;
        break;
      }
      case 'totalField':
      case 'localField': {
        const units = this.fileIOController.getFieldMapUnits();
        // Field strength needed for Hz/rad_s, and for TGV/QSMART internal scaling
        const needsFieldStrength = units !== 'ppm' || combinedMethod !== 'none';
        const hasFieldStrength = !needsFieldStrength || (this.fileIOController.getFieldStrength() > 0);
        // Mask can come from: UI editing, mask file upload, or magnitude (for threshold generation)
        const hasMaskSource = this.currentMaskData !== null
          || this.fileIOController.hasMask()
          || this.fileIOController.hasFieldMapMagnitude()
          || this.preparedMagnitudeData !== null;
        // QSMART and MEDI require magnitude
        const dipoleMethod = this.pipelineSettings?.dipoleInversion || 'rts';
        const needsMagnitude = combinedMethod === 'qsmart' || dipoleMethod === 'medi';
        const hasMagnitude = this.fileIOController.hasFieldMapMagnitude()
          || this.preparedMagnitudeData !== null;
        const algorithmOk = !needsMagnitude || hasMagnitude;
        canRun = isValid && hasFieldStrength && hasMaskSource && algorithmOk;
        break;
      }
    }

    const runButton = document.getElementById('runPipelineSidebar');
    if (runButton) {
      runButton.disabled = !canRun || this.pipelineRunning;
    }
  }

  // Delegate to FileIOController
  getEchoTimesFromInputs() {
    return this.fileIOController?.getEchoTimesFromInputs() || [];
  }

  // Visualization methods - delegate to ViewerController
  async visualizeMagnitude() {
    await this.viewerController.visualizeMagnitude();
    this.syncViewerState();
  }

  async visualizePhase() {
    await this.viewerController.visualizePhase();
    this.syncViewerState();
  }

  navigateEcho(direction) {
    this.viewerController.navigateEcho(direction);
    this.syncViewerState();
  }

  async visualizeCurrentEcho() {
    await this.viewerController.visualizeCurrentEcho();
    this.syncViewerState();
  }

  updateEchoNavigation() {
    this.viewerController.updateEchoNavigation();
  }

  hideEchoNavigation() {
    this.viewerController.hideEchoNavigation();
    this.currentViewType = null;
  }

  async loadAndVisualizeFile(file, description) {
    await this.viewerController.loadAndVisualizeFile(file, description);
    this.currentFile = this.viewerController.getCurrentFile();
  }

  updateDataUnits(description) {
    this.viewerController.updateDataUnits(description);
  }

  // Sync viewer state from controller to app
  syncViewerState() {
    this.currentEchoIndex = this.viewerController.getCurrentEchoIndex();
    this.currentViewType = this.viewerController.getCurrentViewType();
    this.currentFile = this.viewerController.getCurrentFile();
  }

  /**
   * Update the Prepare button state based on current settings
   */
  updatePrepareButtonState() {
    const btn = document.getElementById('prepareMaskInput');
    const mode = this.fileIOController?.getInputMode() || 'raw';
    let hasMagnitude;
    if (mode === 'raw' || mode === 'dicom') {
      hasMagnitude = this.multiEchoFiles.magnitude.length > 0;
    } else {
      hasMagnitude = this.fileIOController?.hasFieldMapMagnitude() || false;
    }

    if (btn) {
      btn.disabled = !hasMagnitude;
    }

    // Enable/disable masking controls based on prepared state
    this.updateMaskingControlsState();
  }

  /**
   * Enable or disable masking controls based on whether Prepare has been run
   */
  updateMaskingControlsState() {
    const prepared = this.maskPrepSettings.prepared;
    const hasMaskFile = this.fileIOController?.hasMask() || false;

    // Threshold/BET enabled when prepared AND no mask file uploaded directly
    const canGenerate = prepared && !hasMaskFile;

    // Preview Mask (Threshold) button
    const previewBtn = document.getElementById('previewMask');
    if (previewBtn) previewBtn.disabled = !canGenerate;

    // BET button
    const betBtn = document.getElementById('runBET');
    if (betBtn) betBtn.disabled = !canGenerate;

    // Threshold slider and auto-threshold button:
    // Only enabled when Threshold method is active (not BET)
    // This is controlled separately by setThresholdSliderEnabled()
    // On initial prepare, keep them disabled until user clicks Threshold

    // Morphological operations panel - show/hide based on state
    const opsPanel = document.getElementById('maskOperations');
    if (opsPanel) {
      // Only show if prepared AND we have a mask (from either Threshold or BET)
      opsPanel.style.display = (prepared && this.currentMaskData) ? 'block' : 'none';
    }
  }

  /**
   * Enable or disable the threshold slider and auto-threshold button
   */
  setThresholdSliderEnabled(enabled) {
    const thresholdSlider = document.getElementById('maskThreshold');
    if (thresholdSlider) thresholdSlider.disabled = !enabled;

    const autoThresholdBtn = document.getElementById('autoThreshold');
    if (autoThresholdBtn) autoThresholdBtn.disabled = !enabled;
  }

  /**
   * Prepare mask input by combining echoes and/or applying bias correction
   * Delegates to MaskController
   */
  async prepareMaskInput() {
    // Get magnitude files based on current input mode
    const mode = this.fileIOController.getInputMode();
    let magnitudeFiles;
    let phaseFiles;
    if (mode === 'raw' || mode === 'dicom') {
      magnitudeFiles = this.multiEchoFiles.magnitude;
      phaseFiles = this.multiEchoFiles.phase;
    } else {
      // Field map modes: use all magnitude files (may be multiple)
      magnitudeFiles = this.fileIOController.getFieldMapMagnitudeFiles();
      phaseFiles = [];
    }

    await this.maskController.prepareMaskInput({
      magnitudeFiles: magnitudeFiles,
      phaseFiles: phaseFiles,
      echoTimes: this.getEchoTimesFromInputs(),
      maskPrepSettings: this.maskPrepSettings,
      onComplete: () => {
        // Sync state from controller to app
        this.magnitudeData = this.maskController.magnitudeData;
        this.magnitudeMax = this.maskController.magnitudeMax;
        this.preparedMagnitudeData = this.maskController.preparedMagnitudeData;
        this.preparedMagnitudeMax = this.maskController.preparedMagnitudeMax;
        this.magnitudeFileBytes = this.maskController.magnitudeFileBytes;
        this.magnitudeVolume = this.maskController.magnitudeVolume;

        this.maskPrepSettings.prepared = true;
        this.updatePrepareButtonState();
        this.updateMaskSectionState();
        this.hideEchoNavigation();
        this.showStageButtons();
        this.addStageButton('preparedMagnitude', 'Masking input');
        this.autoDetectThreshold();
      }
    });
  }

  /**
   * Display the prepared magnitude data as the base volume
   * Delegates to MaskController
   */
  async displayPreparedMagnitude() {
    await this.maskController.displayPreparedMagnitude();
    this.magnitudeVolume = this.maskController.magnitudeVolume;
    this.updateDownloadVolumeButton();
  }

  // Create NIfTI header from NiiVue volume - delegates to imported module
  createNiftiHeaderFromVolume(vol) {
    return createNiftiHeaderFromVolume(vol);
  }

  /**
   * Read NIfTI header from a file without displaying it
   * Delegates to MaskController
   */
  async readNiftiHeader(file) {
    return this.maskController.readNiftiHeader(file);
  }

  /**
   * Read NIfTI image data from a file without displaying it
   * Delegates to MaskController
   */
  async readNiftiData(file) {
    return this.maskController.readNiftiData(file);
  }

  /**
   * Combine multiple magnitude echoes using Root Sum of Squares (RSS)
   * Delegates to MaskController
   */
  async combineMagnitudeRSS() {
    return this.maskController.combineMagnitudeRSS(this.multiEchoFiles.magnitude);
  }

  /**
   * Apply bias field correction to magnitude data
   * Delegates to MaskController
   */
  async applyBiasCorrection(magnitudeData) {
    return this.maskController.applyBiasCorrection(magnitudeData);
  }

  /**
   * Preview mask based on threshold
   * Delegates to MaskController
   */
  async previewMask() {
    // Sync threshold to controller before previewing
    this.maskController.setMaskThreshold(this.maskThreshold);

    await this.maskController.previewMask(this.maskPrepSettings);

    // Sync state from controller
    this.currentMaskData = this.maskController.currentMaskData;
    this.originalMaskData = this.maskController.originalMaskData;
    this.maskDims = this.maskController.maskDims;
    this.voxelSize = this.maskController.voxelSize;
    this.applyVoxelDefaults();

    // Show morphological operations panel
    const opsPanel = document.getElementById('maskOperations');
    if (opsPanel) opsPanel.style.display = 'block';

    // Add mask to Results section
    this.showStageButtons();
    this.addStageButton('mask', 'Brain Mask');

    // Update run button state
    this.updateEchoInfo();
  }

  /**
   * Update mask preview when threshold changes
   * Delegates to MaskController
   */
  async updateMaskPreview() {
    // Sync threshold to controller
    this.maskController.setMaskThreshold(this.maskThreshold);

    await this.maskController.updateMaskPreview();

    // Sync state from controller
    this.currentMaskData = this.maskController.currentMaskData;
    this.originalMaskData = this.maskController.originalMaskData;
    this.maskDims = this.maskController.maskDims;
    this.voxelSize = this.maskController.voxelSize;

    // Show morphological operations panel
    const opsPanel = document.getElementById('maskOperations');
    if (opsPanel) opsPanel.style.display = 'block';

    // Add mask to Results section
    this.showStageButtons();
    this.addStageButton('mask', 'Brain Mask');

    // Update run button state
    this.updateEchoInfo();
  }

  /**
   * Display the current mask as an overlay
   * Delegates to MaskController
   */
  async displayCurrentMask() {
    // Sync mask data to controller if it was modified locally
    if (this.currentMaskData !== this.maskController.currentMaskData) {
      this.maskController.currentMaskData = this.currentMaskData;
    }
    await this.maskController.displayCurrentMask();
  }

  /**
   * Update the opacity of all overlay volumes
   */
  updateOverlayOpacity(opacity) {
    if (this.nv.volumes.length <= 1) return;

    // Update all overlays (volumes after the first one)
    for (let i = 1; i < this.nv.volumes.length; i++) {
      this.nv.setOpacity(i, opacity);
    }
    this.nv.updateGLVolume();
  }

  /**
   * Show or hide the overlay opacity control
   */
  showOverlayControl(show) {
    const control = document.getElementById('overlayOpacityControl');
    if (control) {
      control.style.display = show ? 'flex' : 'none';
    }
  }

  /**
   * Update the download volume button state
   */
  updateDownloadVolumeButton() {
    const btn = document.getElementById('downloadCurrentVolume');
    if (btn) {
      btn.disabled = !this.nv.volumes || this.nv.volumes.length === 0;
    }
  }

  // 3D morphological erosion - delegates to MaskController
  erodeMask3D() {
    // Sync mask to controller
    this.maskController.currentMaskData = this.currentMaskData;
    this.maskController.maskDims = this.maskDims;

    this.maskController.erodeMask3D();

    // Sync back
    this.currentMaskData = this.maskController.currentMaskData;
  }

  // 3D morphological dilation - delegates to MaskController
  dilateMask3D() {
    // Sync mask to controller
    this.maskController.currentMaskData = this.currentMaskData;
    this.maskController.maskDims = this.maskDims;

    this.maskController.dilateMask3D();

    // Sync back
    this.currentMaskData = this.maskController.currentMaskData;
  }

  // Fill holes in 3D mask - delegates to MaskController
  fillHoles3D() {
    // Sync mask to controller
    this.maskController.currentMaskData = this.currentMaskData;
    this.maskController.maskDims = this.maskDims;

    this.maskController.fillHoles3D();

    // Sync back
    this.currentMaskData = this.maskController.currentMaskData;
  }

  // Clear mask completely - delegates to MaskController
  async clearMask() {
    await this.maskController.clearMask();

    // Sync state
    this.currentMaskData = null;
    this.originalMaskData = null;

    // Hide threshold slider
    const sliderGroup = document.getElementById('thresholdSliderGroup');
    if (sliderGroup) sliderGroup.style.display = 'none';

    // Update run button state (mask no longer available)
    this.updateEchoInfo();
  }

  // Toggle drawing mode on/off - delegates to MaskController
  async toggleDrawingMode() {
    // Sync mask data to controller
    this.maskController.currentMaskData = this.currentMaskData;
    this.maskController.maskDims = this.maskDims;
    this.maskController.brushSize = this.brushSize;

    await this.maskController.toggleDrawingMode();

    // Sync state back
    this.drawingEnabled = this.maskController.drawingEnabled;
    this.brushMode = this.maskController.brushMode;
    this.savedCrosshairWidth = this.maskController.savedCrosshairWidth;
  }

  // Set brush mode (add or remove) - delegates to MaskController
  setBrushMode(mode) {
    this.maskController.setBrushMode(mode);
    this.brushMode = this.maskController.brushMode;
  }

  // Set brush size - delegates to MaskController
  setBrushSize(size) {
    this.brushSize = size;
    this.maskController.setBrushSize(size);
  }

  // Apply the drawing to the current mask - delegates to MaskController
  async applyDrawingToMask() {
    // Sync mask data to controller
    this.maskController.currentMaskData = this.currentMaskData;
    this.maskController.maskDims = this.maskDims;

    await this.maskController.applyDrawingToMask();

    // Sync state back
    this.currentMaskData = this.maskController.currentMaskData;
    this.drawingEnabled = this.maskController.drawingEnabled;
    this.brushMode = this.maskController.brushMode;

    // Update run button state
    this.updateEchoInfo();
  }

  // Create mask NIfTI using source header as template - delegates to imported module
  createMaskNifti(maskData) {
    return createMaskNifti(maskData, this.magnitudeFileBytes);
  }

  async runRomeoQSM() {
    const mode = this.fileIOController.getInputMode();

    if (mode === 'raw' || mode === 'dicom') {
      await this._runRawPipeline();
    } else if (mode === 'totalField') {
      await this._runTotalFieldPipeline();
    } else if (mode === 'localField') {
      await this._runLocalFieldPipeline();
    }
  }

  async _runRawPipeline() {
    // Validation
    const magCount = this.multiEchoFiles.magnitude.length;
    const phaseCount = this.multiEchoFiles.phase.length;
    const echoTimes = this.getEchoTimesFromInputs();
    const echoTimeCount = echoTimes.length;

    if (magCount === 0 || phaseCount === 0) {
      this.updateOutput("Please upload both magnitude and phase files");
      return;
    }

    if (magCount !== phaseCount) {
      this.updateOutput(`File count mismatch: ${magCount} magnitude, ${phaseCount} phase`);
      return;
    }

    if (echoTimeCount === 0) {
      this.updateOutput("Please enter echo times");
      return;
    }

    // Get parameters
    const magField = parseFloat(document.getElementById('magField').value);

    if (!magField || magField <= 0) {
      this.updateOutput("Please enter a valid magnetic field strength");
      return;
    }

    try {
      // Read file buffers
      const magnitudeBuffers = [];
      const phaseBuffers = [];

      for (let i = 0; i < magCount; i++) {
        const magFile = this.multiEchoFiles.magnitude[i]?.file;
        const phaseFile = this.multiEchoFiles.phase[i]?.file;

        if (magFile && phaseFile) {
          magnitudeBuffers.push(await magFile.arrayBuffer());
          phaseBuffers.push(await phaseFile.arrayBuffer());
        }
      }

      // Prepare custom mask if available
      let customMaskBuffer = null;
      if (this.currentMaskData && this.magnitudeFileBytes) {
        const maskNifti = this.createMaskNifti(this.currentMaskData);
        customMaskBuffer = maskNifti;
        this.updateOutput("Using custom edited mask");
      }

      // Determine which stages can be skipped based on settings changes
      const skipStages = this.pipelineExecutor.determineSkipStages(this.pipelineSettings);
      if (skipStages.skipUnwrap) {
        this.updateOutput("Reusing cached unwrapped phase data");
      }
      if (skipStages.skipBgRemoval) {
        this.updateOutput("Reusing cached background-removed data");
      }

      // Show phase image at the start
      await this.visualizePhase();

      // Include prepared magnitude if available (for MEDI gradient weighting and threshold mask)
      const preparedMagnitude = this.preparedMagnitudeData
        ? Array.from(this.preparedMagnitudeData)
        : null;

      // Run pipeline via executor
      const started = await this.pipelineExecutor.run({
        inputMode: 'raw',
        magnitudeBuffers,
        phaseBuffers,
        echoTimes,
        magField,
        maskThreshold: this.maskThreshold,
        customMaskBuffer,
        preparedMagnitude,
        pipelineSettings: this.pipelineSettings,
        skipStages
      });

      if (started) {
        document.getElementById('cancelPipeline').disabled = false;
        document.getElementById('runPipelineSidebar').disabled = true;
      }

    } catch (error) {
      this.updateOutput(`Error: ${error.message}`);
      this.setProgress(0, 'Failed');
      document.getElementById('cancelPipeline').disabled = true;
      this.updateEchoInfo();
      console.error(error);
    }
  }

  async _runTotalFieldPipeline() {
    const totalFieldFile = this.fileIOController.getTotalFieldFile();
    if (!totalFieldFile) {
      this.updateOutput("Please upload a total field map file");
      return;
    }

    const units = this.fileIOController.getFieldMapUnits();
    const combinedMethod = this.pipelineSettings?.combinedMethod || 'none';
    // Field strength needed for Hz/rad_s conversion, and for TGV/QSMART internal scaling
    const needsFieldStrength = units !== 'ppm' || combinedMethod !== 'none';
    const magField = needsFieldStrength ? parseFloat(document.getElementById('magField').value) : null;

    if (needsFieldStrength && (!magField || magField <= 0)) {
      this.updateOutput("Please enter a valid magnetic field strength");
      return;
    }

    // QSMART and MEDI require magnitude
    const dipoleMethod = this.pipelineSettings?.dipoleInversion || 'rts';
    if ((combinedMethod === 'qsmart' || dipoleMethod === 'medi')
        && !this.fileIOController.hasFieldMapMagnitude() && !this.preparedMagnitudeData) {
      const method = combinedMethod === 'qsmart' ? 'QSMART' : 'MEDI';
      this.updateOutput(`${method} requires a magnitude image`);
      return;
    }

    try {
      const totalFieldBuffer = await totalFieldFile.arrayBuffer();

      // Extract voxel size from NIfTI header for pipeline defaults
      if (!isGzipped(new Uint8Array(totalFieldBuffer)) && totalFieldBuffer.byteLength >= 352) {
        const headerInfo = parseNiftiHeader(totalFieldBuffer.slice(0, 352));
        this.voxelSize = headerInfo.voxelSize;
        this.maskDims = [headerInfo.nx, headerInfo.ny, headerInfo.nz];
        this.applyVoxelDefaults();
      }

      // Optional magnitude for masking/MEDI/QSMART vasculature
      const magFile = this.fileIOController.getFieldMapMagnitudeFile();
      const magnitudeBuffer = magFile ? await magFile.arrayBuffer() : null;

      // Mask: from centralized upload, from UI editing, or will be generated from magnitude
      const maskFile = this.fileIOController.getMaskFile();
      let maskBuffer = maskFile ? await maskFile.arrayBuffer() : null;

      // Use custom edited mask if available
      let customMaskBuffer = null;
      if (this.currentMaskData && this.magnitudeFileBytes) {
        const maskNifti = this.createMaskNifti(this.currentMaskData);
        customMaskBuffer = maskNifti;
        this.updateOutput("Using custom edited mask");
      }

      // Preview the field map
      await this.visualizeFieldMap('totalField');

      const started = await this.pipelineExecutor.run({
        inputMode: 'totalField',
        totalFieldBuffer,
        fieldMapUnits: units,
        magnitudeBuffer,
        maskBuffer,
        customMaskBuffer,
        magField,
        maskThreshold: this.maskThreshold,
        preparedMagnitude: this.preparedMagnitudeData ? Array.from(this.preparedMagnitudeData) : null,
        pipelineSettings: this.pipelineSettings
      });

      if (started) {
        document.getElementById('cancelPipeline').disabled = false;
        document.getElementById('runPipelineSidebar').disabled = true;
      }
    } catch (error) {
      this.updateOutput(`Error: ${error.message}`);
      this.setProgress(0, 'Failed');
      document.getElementById('cancelPipeline').disabled = true;
      this.updateEchoInfo();
      console.error(error);
    }
  }

  async _runLocalFieldPipeline() {
    const localFieldFile = this.fileIOController.getLocalFieldFile();
    if (!localFieldFile) {
      this.updateOutput("Please upload a local field map file");
      return;
    }

    const units = this.fileIOController.getFieldMapUnits();
    const combinedMethod = this.pipelineSettings?.combinedMethod || 'none';
    const needsFieldStrength = units !== 'ppm' || combinedMethod !== 'none';
    const magField = needsFieldStrength ? parseFloat(document.getElementById('magField').value) : null;

    if (needsFieldStrength && (!magField || magField <= 0)) {
      this.updateOutput("Please enter a valid magnetic field strength");
      return;
    }

    // QSMART and MEDI require magnitude
    const dipoleMethod = this.pipelineSettings?.dipoleInversion || 'rts';
    if ((combinedMethod === 'qsmart' || dipoleMethod === 'medi')
        && !this.fileIOController.hasFieldMapMagnitude() && !this.preparedMagnitudeData) {
      const method = combinedMethod === 'qsmart' ? 'QSMART' : 'MEDI';
      this.updateOutput(`${method} requires a magnitude image`);
      return;
    }

    try {
      const localFieldBuffer = await localFieldFile.arrayBuffer();

      // Extract voxel size from NIfTI header for pipeline defaults
      if (!isGzipped(new Uint8Array(localFieldBuffer)) && localFieldBuffer.byteLength >= 352) {
        const headerInfo = parseNiftiHeader(localFieldBuffer.slice(0, 352));
        this.voxelSize = headerInfo.voxelSize;
        this.maskDims = [headerInfo.nx, headerInfo.ny, headerInfo.nz];
        this.applyVoxelDefaults();
      }

      // Optional magnitude for MEDI/QSMART vasculature
      const magFile = this.fileIOController.getFieldMapMagnitudeFile();
      const magnitudeBuffer = magFile ? await magFile.arrayBuffer() : null;

      // Mask: from centralized upload or from UI editing
      const maskFile = this.fileIOController.getMaskFile();
      let maskBuffer = maskFile ? await maskFile.arrayBuffer() : null;

      let customMaskBuffer = null;
      if (this.currentMaskData && this.magnitudeFileBytes) {
        const maskNifti = this.createMaskNifti(this.currentMaskData);
        customMaskBuffer = maskNifti;
        this.updateOutput("Using custom edited mask");
      }

      if (!maskBuffer && !customMaskBuffer && combinedMethod === 'none') {
        this.updateOutput("Please provide a mask file or create one from magnitude");
        return;
      }

      // Preview the field map
      await this.visualizeFieldMap('localField');

      const started = await this.pipelineExecutor.run({
        inputMode: 'localField',
        localFieldBuffer,
        fieldMapUnits: units,
        magnitudeBuffer,
        maskBuffer,
        customMaskBuffer,
        magField,
        maskThreshold: this.maskThreshold,
        preparedMagnitude: this.preparedMagnitudeData ? Array.from(this.preparedMagnitudeData) : null,
        pipelineSettings: this.pipelineSettings
      });

      if (started) {
        document.getElementById('cancelPipeline').disabled = false;
        document.getElementById('runPipelineSidebar').disabled = true;
      }
    } catch (error) {
      this.updateOutput(`Error: ${error.message}`);
      this.setProgress(0, 'Failed');
      document.getElementById('cancelPipeline').disabled = true;
      this.updateEchoInfo();
      console.error(error);
    }
  }

  cancelPipeline() {
    this.pipelineExecutor?.cancel();
    document.getElementById('cancelPipeline').disabled = true;
    this.updateEchoInfo();
  }

  showStageButtons() {
    const resultsSection = document.getElementById('stage-buttons');
    resultsSection.classList.remove('hidden');
    // Expand results section if collapsed (don't close if already open)
    if (resultsSection.classList.contains('collapsed')) {
      resultsSection.classList.remove('collapsed');
    }
  }

  // Create or update a stage button dynamically
  addStageButton(stage, description) {
    const container = document.getElementById('dynamicStageButtons');
    if (!container) return;

    // Check if button already exists
    const existingItem = document.getElementById(`stage-item-${stage}`);
    if (existingItem) {
      // Already exists, just make sure it's enabled
      const showBtn = existingItem.querySelector('.stage-tab');
      const downloadBtn = existingItem.querySelector('.download-btn');
      if (showBtn) showBtn.disabled = false;
      if (downloadBtn) downloadBtn.disabled = false;
      return;
    }

    // Track stage order
    if (!this.stageOrder.includes(stage)) {
      this.stageOrder.push(stage);
    }

    // Create display name from stage ID or description
    const displayName = this.getStageDisplayName(stage, description);

    // Create the stage item
    const stageItem = document.createElement('div');
    stageItem.className = 'stage-item';
    stageItem.id = `stage-item-${stage}`;

    // Show button
    const showBtn = document.createElement('button');
    showBtn.className = 'btn btn-secondary btn-sm stage-tab';
    showBtn.textContent = displayName;
    showBtn.title = description || stage;
    showBtn.addEventListener('click', () => this.showStage(stage));

    stageItem.appendChild(showBtn);
    container.appendChild(stageItem);
  }

  // Get a user-friendly display name for a stage
  getStageDisplayName(stage, description) {
    const nameMap = window.QSMConfig.STAGE_DISPLAY_NAMES;

    // Use mapped name, or extract short name from description, or use stage ID
    if (nameMap[stage]) {
      return nameMap[stage];
    }

    // Try to extract a short name from description (first 2 words)
    if (description) {
      const words = description.split(' ').slice(0, 2).join(' ');
      if (words.length <= 15) return words;
    }

    // Fallback to stage ID (camelCase to Title Case)
    return stage.replace(/([A-Z])/g, ' $1').replace(/^./, s => s.toUpperCase()).trim();
  }

  // Clear all dynamic stage buttons (called at pipeline start)
  clearStageButtons() {
    const container = document.getElementById('dynamicStageButtons');
    if (container) {
      container.innerHTML = '';
    }
  }

  // Clear all cached results (internal use)
  clearResults() {
    this.pipelineExecutor?.clearResults();
  }

  // Clear all results including prepared magnitude and mask (user-triggered)
  clearAllResults() {
    // Clear pipeline results
    this.clearResults();
    this.clearStageButtons();

    // Clear prepared magnitude
    this.preparedMagnitudeData = null;
    this.preparedMagnitudeMax = 0;
    this.maskPrepSettings.prepared = false;

    // Clear mask
    this.currentMaskData = null;
    this.originalMaskData = null;

    // Hide morphological operations panel and threshold slider
    const opsPanel = document.getElementById('maskOperations');
    if (opsPanel) opsPanel.style.display = 'none';
    const sliderGroup = document.getElementById('thresholdSliderGroup');
    if (sliderGroup) sliderGroup.style.display = 'none';

    // Hide Results section
    const resultsSection = document.getElementById('stage-buttons');
    if (resultsSection) {
      resultsSection.classList.add('hidden');
      resultsSection.classList.add('collapsed');
    }

    // Update button states
    this.updatePrepareButtonState();
    this.updateEchoInfo();

    this.updateOutput("Results cleared");
  }

  updateDownloadButtons() {
    // Legacy method - now handled by enableStageButtons
  }

  async showStage(stage) {
    try {
      // For magnitude and phase, use the multi-echo viewer with echo navigation
      if (stage === 'magnitude' || stage === 'phase') {
        this.currentViewType = stage;
        this.currentEchoIndex = 0;
        await this.visualizeCurrentEcho();
        this.updateEchoNavigation();
        return;
      }

      // For single 3D volume stages, hide echo navigation
      this.hideEchoNavigation();

      // Handle prepared magnitude (local data, not from pipeline)
      if (stage === 'preparedMagnitude') {
        if (this.preparedMagnitudeData) {
          await this.displayPreparedMagnitude();
          this.updateDataUnits(null);
          this.updateOutput("Displaying: Masking input");
        } else {
          this.updateOutput("Masking input not available - click Prepare first");
        }
        return;
      }

      // Handle mask (local data, not from pipeline)
      if (stage === 'mask') {
        if (this.currentMaskData) {
          await this.displayCurrentMask();
          this.updateDataUnits(null);
          this.updateOutput("Displaying: Brain Mask");
        } else {
          this.updateOutput("Mask not available - generate one first");
        }
        return;
      }

      // Check if we have cached results
      if (this.results[stage]?.file) {
        const description = this.results[stage].description || stage;
        this.updateOutput(`Displaying ${description}...`);
        await this.loadAndVisualizeFile(this.results[stage].file, description);
        return;
      }

      // No cached result available
      this.updateOutput(`${stage} not available - run the pipeline first`);

    } catch (error) {
      this.updateOutput(`Error showing ${stage}: ${error.message}`);
    }
  }

  // Display stage data as it arrives during pipeline processing
  async displayLiveStageData(data) {
    try {
      const { stage, data: stageBytes, description } = data;

      // Show the stage buttons section as soon as first result arrives
      this.showStageButtons();

      // Add/enable the button for this stage (with description for display name)
      this.addStageButton(stage, description);

      // Hide echo navigation - pipeline results are single 3D volumes, not multi-echo
      this.hideEchoNavigation();

      // Create file from bytes
      const blob = new Blob([stageBytes], { type: 'application/octet-stream' });
      const file = new File([blob], `${stage}.nii`, { type: 'application/octet-stream' });

      // Load in viewer
      await this.loadAndVisualizeFile(file, description);

      // Cache the result with description
      this.results[stage] = { file: file, path: `${stage}.nii`, description: description };

      this.updateOutput(`Displaying: ${description}`);
    } catch (error) {
      this.updateOutput(`Error displaying live data: ${error.message}`);
    }
  }

  // Cache stage data without displaying (for auxiliary outputs like vasculature mask)
  cacheStageData(data) {
    try {
      const { stage, data: stageBytes, description } = data;

      // Show the stage buttons section
      this.showStageButtons();

      // Add/enable the button for this stage (with description for display name)
      this.addStageButton(stage, description);

      // Create file from bytes
      const blob = new Blob([stageBytes], { type: 'application/octet-stream' });
      const file = new File([blob], `${stage}.nii`, { type: 'application/octet-stream' });

      // Cache the result (but don't display)
      this.results[stage] = { file: file, path: `${stage}.nii`, description: description };

      this.updateOutput(`Cached: ${description}`);
    } catch (error) {
      this.updateOutput(`Error caching data: ${error.message}`);
    }
  }

  async downloadStage(stage) {
    if (!this.results[stage]?.file) {
      this.updateOutput(`${stage} not available - run the pipeline first`);
      return;
    }

    const file = this.results[stage].file;
    const url = URL.createObjectURL(file);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${stage}_${new Date().toISOString().slice(0,10)}.nii`;
    a.click();
    URL.revokeObjectURL(url);
  }

  /**
   * Download the currently displayed volume as a NIfTI file
   */
  downloadCurrentVolume() {
    if (!this.nv.volumes || this.nv.volumes.length === 0) {
      this.updateOutput("No volume loaded to download");
      return;
    }

    const vol = this.nv.volumes[0];
    const name = vol.name || 'volume';
    const baseName = name.replace(/\.(nii|nii\.gz)$/i, '');

    // Create NIfTI from volume data
    const niftiBuffer = this.createNiftiFromVolume(vol);

    // Download
    const blob = new Blob([niftiBuffer], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${baseName}.nii`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    this.updateOutput(`Downloaded: ${baseName}.nii`);
  }

  /**
   * Save a screenshot of the NiiVue viewer as PNG
   */
  saveScreenshot() {
    if (!this.nv) {
      this.updateOutput("Viewer not initialized");
      return;
    }

    // Generate filename based on current volume or timestamp
    let filename = 'niivue_screenshot.png';
    if (this.nv.volumes && this.nv.volumes.length > 0) {
      const vol = this.nv.volumes[0];
      const name = vol.name || 'volume';
      const baseName = name.replace(/\.(nii|nii\.gz)$/i, '');
      filename = `${baseName}_screenshot.png`;
    }

    this.nv.saveScene(filename);
    this.updateOutput(`Screenshot saved: ${filename}`);
  }

  /**
   * Create a NIfTI buffer from a NiiVue volume
   */
  createNiftiFromVolume(vol) {
    const hdr = vol.hdr;
    const img = vol.img;

    // Determine data type and bytes per voxel
    let datatype = 16;  // FLOAT32 by default
    let bitpix = 32;
    let bytesPerVoxel = 4;

    if (img instanceof Float64Array) {
      datatype = 64;  // FLOAT64
      bitpix = 64;
      bytesPerVoxel = 8;
    } else if (img instanceof Int16Array) {
      datatype = 4;   // INT16
      bitpix = 16;
      bytesPerVoxel = 2;
    } else if (img instanceof Uint8Array) {
      datatype = 2;   // UINT8
      bitpix = 8;
      bytesPerVoxel = 1;
    }

    const headerSize = 352;
    const dataSize = img.length * bytesPerVoxel;
    const buffer = new ArrayBuffer(headerSize + dataSize);
    const view = new DataView(buffer);

    // sizeof_hdr
    view.setInt32(0, 348, true);

    // dim array
    const dims = hdr.dims || [3, vol.dims[1], vol.dims[2], vol.dims[3], 1, 1, 1, 1];
    for (let i = 0; i < 8; i++) {
      view.setInt16(40 + i * 2, dims[i] || 0, true);
    }

    // datatype and bitpix
    view.setInt16(70, datatype, true);
    view.setInt16(72, bitpix, true);

    // pixdim
    const pixdim = hdr.pixDims || [1, 1, 1, 1, 1, 1, 1, 1];
    for (let i = 0; i < 8; i++) {
      view.setFloat32(76 + i * 4, pixdim[i] || 1, true);
    }

    // vox_offset
    view.setFloat32(108, headerSize, true);

    // scl_slope and scl_inter
    view.setFloat32(112, hdr.scl_slope || 1, true);
    view.setFloat32(116, hdr.scl_inter || 0, true);

    // xyzt_units
    view.setUint8(123, 10);  // mm + sec

    // qform_code and sform_code
    view.setInt16(252, hdr.qform_code || 1, true);
    view.setInt16(254, hdr.sform_code || 1, true);

    // Affine matrix
    if (hdr.affine) {
      for (let i = 0; i < 4; i++) {
        view.setFloat32(280 + i * 4, hdr.affine[0][i] || 0, true);
        view.setFloat32(296 + i * 4, hdr.affine[1][i] || 0, true);
        view.setFloat32(312 + i * 4, hdr.affine[2][i] || 0, true);
      }
    }

    // magic
    view.setUint8(344, 0x6E);  // 'n'
    view.setUint8(345, 0x2B);  // '+'
    view.setUint8(346, 0x31);  // '1'
    view.setUint8(347, 0x00);

    // Copy image data
    const dataView = new Uint8Array(buffer, headerSize);
    const imgBytes = new Uint8Array(img.buffer, img.byteOffset, img.byteLength);
    dataView.set(imgBytes);

    return buffer;
  }

  updateOutput(message) {
    const consoleOutput = document.getElementById('consoleOutput');
    if (consoleOutput) {
      const time = new Date().toLocaleTimeString('en-US', { hour12: false });
      const line = document.createElement('div');
      line.className = 'console-line';
      line.innerHTML = `<span class="console-time">[${time}]</span> <span class="console-message">${message}</span>`;
      consoleOutput.appendChild(line);
      // Auto-scroll to bottom
      consoleOutput.scrollTop = consoleOutput.scrollHeight;
    }
    console.log(message);
  }

  /**
   * Auto-detect optimal threshold using Otsu's method and set the slider
   * Delegates computation to imported ThresholdUtils module
   */
  /**
   * Auto-detect optimal threshold using Otsu's method
   * Delegates to MaskController
   */
  autoDetectThreshold() {
    if (!this.preparedMagnitudeData) {
      this.updateOutput("Please click Prepare first");
      return;
    }

    this.updateOutput("Computing optimal threshold (Otsu)...");

    // Sync prepared data to controller
    this.maskController.preparedMagnitudeData = this.preparedMagnitudeData;
    this.maskController.preparedMagnitudeMax = this.preparedMagnitudeMax;

    const result = this.maskController.computeOtsuThreshold();

    if (result.error) {
      this.updateOutput(`Cannot compute threshold: ${result.error}`);
      return;
    }

    const clampedPercent = result.thresholdPercent;

    // Update slider and display
    const slider = document.getElementById('maskThreshold');
    if (slider) {
      slider.value = clampedPercent;
      this.maskThreshold = clampedPercent;
      this.maskController.setMaskThreshold(clampedPercent);
      document.getElementById('thresholdLabel').textContent = `Threshold (${clampedPercent}%)`;
    }

    this.updateOutput(`Otsu threshold: ${clampedPercent}% (${result.thresholdValue.toFixed(1)})`);

    // Only trigger mask preview if threshold slider is enabled (user has clicked Threshold button)
    const thresholdSlider = document.getElementById('maskThreshold');
    if (thresholdSlider && !thresholdSlider.disabled && this.magnitudeData && !this.maskUpdating) {
      this.updateMaskPreview();
    }
  }

  /**
   * Run BET brain extraction
   * Delegates to MaskController
   */
  async runBET() {
    // Disable threshold slider since user chose BET-based masking
    this.setThresholdSliderEnabled(false);

    // Sync state to controller
    this.maskController.magnitudeFileBytes = this.magnitudeFileBytes;
    this.maskController.magnitudeVolume = this.magnitudeVolume;
    this.maskController.magnitudeData = this.magnitudeData;
    this.maskController.magnitudeMax = this.magnitudeMax;
    this.maskController.preparedMagnitudeData = this.preparedMagnitudeData;
    this.maskController.maskDims = this.maskDims;

    // Get magnitude files based on current input mode
    const mode = this.fileIOController.getInputMode();
    let magnitudeFilesForBET;
    if (mode === 'raw' || mode === 'dicom') {
      magnitudeFilesForBET = this.multiEchoFiles.magnitude;
    } else {
      magnitudeFilesForBET = this.fileIOController.getFieldMapMagnitudeFiles();
    }

    await this.maskController.runBET({
      magnitudeFiles: magnitudeFilesForBET,
      betSettings: this.betSettings,
      createNiftiHeaderFromVolume: (vol) => this.createNiftiHeaderFromVolume(vol),
      onComplete: async () => {
        // Sync state from controller
        this.currentMaskData = this.maskController.currentMaskData;
        this.originalMaskData = this.maskController.originalMaskData;
        this.maskDims = this.maskController.maskDims;
        this.magnitudeVolume = this.maskController.magnitudeVolume;
        this.magnitudeData = this.maskController.magnitudeData;
        this.magnitudeMax = this.maskController.magnitudeMax;
        this.magnitudeFileBytes = this.maskController.magnitudeFileBytes;

        // Apply post-BET erosions
        const erosions = this.betSettings.erosions || 0;
        if (erosions > 0) {
          this.updateOutput(`Applying ${erosions} erosion step(s)...`);
          for (let i = 0; i < erosions; i++) {
            this.erodeMask3D();
          }
          await this.displayCurrentMask();
          this.updateOutput(`BET mask complete with ${erosions} erosion(s)`);
        }

        // Show morphological operations panel
        const opsPanel = document.getElementById('maskOperations');
        if (opsPanel) opsPanel.style.display = 'block';

        // Add mask to Results section
        this.showStageButtons();
        this.addStageButton('mask', 'Brain Mask');

        // Update run button state (mask is now available)
        this.updateEchoInfo();
      },
      onError: () => {
        this.updateEchoInfo();
      }
    });
  }

  /**
   * Handle BET completion - delegates to MaskController
   */
  async handleBETComplete(data) {
    await this.maskController.handleBETComplete(data, () => {
      // Sync state from controller
      this.currentMaskData = this.maskController.currentMaskData;
      this.originalMaskData = this.maskController.originalMaskData;

      // Show morphological operations panel
      const opsPanel = document.getElementById('maskOperations');
      if (opsPanel) opsPanel.style.display = 'block';

      // Add mask to Results section
      this.showStageButtons();
      this.addStageButton('mask', 'Brain Mask');

      // Update run button state
      this.updateEchoInfo();
    });
  }

  // Calculate dynamic defaults based on voxel size (matches QSM.jl)
  getVoxelBasedDefaults() {
    return window.QSMConfig.getVoxelBasedDefaults(this.voxelSize || [1, 1, 1], this.maskDims);
  }

  // Apply voxel-based defaults to pipeline settings for any null values.
  // Called when voxel size becomes available (file upload, mask preparation).
  applyVoxelDefaults() {
    if (!this.voxelSize) return;
    const defaults = this.getVoxelBasedDefaults();
    const s = this.pipelineSettings;
    if (s.vsharp.maxRadius == null) s.vsharp.maxRadius = defaults.vsharpMaxRadius;
    if (s.vsharp.minRadius == null) s.vsharp.minRadius = defaults.vsharpMinRadius;
    if (s.ismv.radius == null) s.ismv.radius = defaults.ismvRadius;
    if (s.pdf.maxit == null) s.pdf.maxit = defaults.pdfMaxit;
  }

  // Pipeline Settings Modal - delegates to PipelineSettingsController
  openPipelineSettingsModal() {
    if (!this.pipelineSettingsController) return;
    const defaults = this.getVoxelBasedDefaults();
    const nEchoes = this.multiEchoFiles?.phase?.filter(f => f.file)?.length || 0;
    const inputMode = this.fileIOController?.getInputMode() || 'raw';
    const hasMagnitude = (inputMode === 'raw' || inputMode === 'dicom')
      ? this.multiEchoFiles.magnitude.length > 0
      : (this.fileIOController.hasFieldMapMagnitude() || this.preparedMagnitudeData !== null);
    this.pipelineSettingsController.setInputMode(inputMode);
    this.pipelineSettingsController.open(this.pipelineSettings, defaults, nEchoes, hasMagnitude);
    this.updateEchoInfo();
  }

  closePipelineSettingsModal() {
    if (this.pipelineSettingsController) {
      this.pipelineSettingsController.close();
    }
  }

  resetPipelineSettings() {
    if (!this.pipelineSettingsController) return;
    const defaults = this.getVoxelBasedDefaults();
    this.pipelineSettingsController.reset(defaults);
  }

  savePipelineSettings() {
    if (!this.pipelineSettingsController) return;
    const nEchoes = this.multiEchoFiles?.phase?.filter(f => f.file)?.length || 0;
    this.pipelineSettings = this.pipelineSettingsController.save(nEchoes);
    this.closePipelineSettingsModal();
    // Sync sidebar dropdowns and update visibility
    this.syncSidebarFromSettings();
    this.updateInputParamsVisibility();
    this.updateEchoInfo();
  }

  runPipelineFromSidebar() {
    // Save current settings from modal if it's open
    const modal = document.getElementById('pipelineSettingsModal');
    if (modal && modal.classList.contains('active')) {
      this.savePipelineSettings();
    }
    this.runRomeoQSM();
  }

  async runSWI() {
    const mode = this.fileIOController.getInputMode();
    if (mode !== 'raw' && mode !== 'dicom') {
      this.updateOutput("SWI requires raw magnitude + phase data");
      return;
    }

    const magCount = this.multiEchoFiles.magnitude.length;
    const phaseCount = this.multiEchoFiles.phase.length;

    if (magCount === 0 || phaseCount === 0) {
      this.updateOutput("Please upload both magnitude and phase files");
      return;
    }

    try {
      const magnitudeBuffers = [];
      const phaseBuffers = [];

      // Only need first echo for SWI
      const magFile = this.multiEchoFiles.magnitude[0]?.file;
      const phaseFile = this.multiEchoFiles.phase[0]?.file;

      if (magFile && phaseFile) {
        magnitudeBuffers.push(await magFile.arrayBuffer());
        phaseBuffers.push(await phaseFile.arrayBuffer());
      }

      let customMaskBuffer = null;
      if (this.currentMaskData && this.magnitudeFileBytes) {
        customMaskBuffer = this.createMaskNifti(this.currentMaskData);
      }

      const preparedMagnitude = this.preparedMagnitudeData
        ? Array.from(this.preparedMagnitudeData)
        : null;

      await this.pipelineExecutor.initialize();
      this.pipelineExecutor.pipelineRunning = true;
      this.updateOutput("Starting SWI pipeline...");

      this.pipelineExecutor.getWorker().postMessage({
        type: 'runSWI',
        data: {
          magnitudeBuffers,
          phaseBuffers,
          maskThreshold: this.maskThreshold,
          customMaskBuffer,
          preparedMagnitude,
          pipelineSettings: this.pipelineSettings
        }
      });

      document.getElementById('cancelPipeline').disabled = false;
      document.getElementById('runSWI').disabled = true;

    } catch (error) {
      this.updateOutput(`Error: ${error.message}`);
      this.setProgress(0, 'Failed');
      console.error(error);
    }
  }

  // BET Settings Modal
  openBetSettingsModal() {
    const mode = this.fileIOController.getInputMode();
    const hasMag = (mode === 'raw' || mode === 'dicom')
      ? this.multiEchoFiles.magnitude.length > 0
      : this.fileIOController.hasFieldMapMagnitude();

    if (!hasMag) {
      this.updateOutput("No magnitude files uploaded - please load magnitude data first");
      return;
    }

    // Populate form with current settings
    document.getElementById('betFractionalIntensity').value = this.betSettings.fractionalIntensity;
    document.getElementById('betFractionalIntensityValue').textContent = this.betSettings.fractionalIntensity;
    document.getElementById('betIterations').value = this.betSettings.iterations;
    document.getElementById('betSubdivisions').value = this.betSettings.subdivisions;
    document.getElementById('betErosions').value = this.betSettings.erosions ?? 2;

    this.betModal?.open();
  }

  resetBetSettings() {
    // Reset to defaults
    document.getElementById('betFractionalIntensity').value = 0.5;
    document.getElementById('betFractionalIntensityValue').textContent = '0.5';
    document.getElementById('betIterations').value = 1000;
    document.getElementById('betSubdivisions').value = 4;
    document.getElementById('betErosions').value = 2;
  }

  runBetWithSettings() {
    // Save settings from form
    this.betSettings = {
      fractionalIntensity: parseFloat(document.getElementById('betFractionalIntensity').value),
      iterations: parseInt(document.getElementById('betIterations').value),
      subdivisions: parseInt(document.getElementById('betSubdivisions').value),
      erosions: parseInt(document.getElementById('betErosions').value) || 0
    };

    this.betModal?.close();
    this.runBET();
  }
}

// Export the QSMApp class for ES module usage
export { QSMApp };

// Initialize the app - this will be called after NiiVue is loaded
function initQSMApp() {
  console.log('Initializing QSM App with NiiVue:', window.Niivue);
  window.app = new QSMApp();
}

// Auto-initialize when loaded as a module
// Wait for DOM and NiiVue to be ready
function waitForNiiVue(maxAttempts = 20, attempt = 0) {
  if (window.Niivue) {
    initQSMApp();
  } else if (attempt < maxAttempts) {
    setTimeout(() => waitForNiiVue(maxAttempts, attempt + 1), 100);
  } else {
    document.getElementById("output").textContent = "Error: NiiVue library failed to load. Please refresh the page.";
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => waitForNiiVue());
} else {
  waitForNiiVue();
}