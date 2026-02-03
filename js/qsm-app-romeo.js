class QSMApp {
  constructor() {
    this.worker = null;
    this.workerReady = false;
    this.workerInitializing = false;
    this.nv = new window.Niivue({
      loadingText: "",
      dragToMeasure: false,
      isColorbar: false,
      textHeight: 0.03,
      show3Dcrosshair: false,
      crosshairColor: [0.23, 0.51, 0.96, 1.0],  // #3b82f6 - matches site primary color
      crosshairWidth: 0.75,  // Thinner crosshair
      onLocationChange: (data) => {
        document.getElementById("intensity").innerHTML = data.string;
      }
    });
    this.currentFile = null;
    this.threshold = 75;
    this.progress = 0;

    // Smooth progress animation state
    this.targetProgress = 0;      // Actual progress from pipeline
    this.animatedProgress = 0;    // Displayed (animated) progress
    this.progressAnimationId = null;
    this.lastAnimationTime = 0;
    // Animation speed: move at ~0.67% per second (completes 100% in ~150s typical runtime)
    this.progressAnimationSpeed = 0.5; // 50% per second - catches up quickly

    // Multi-echo file storage
    this.multiEchoFiles = {
      magnitude: [],
      phase: [],
      json: [],
      echoTimes: [],
      combinedMagnitude: null,
      combinedPhase: null
    };

    // Processing results - dynamically populated as stages arrive
    this.results = {};

    // Track stage order for display (in order they arrive)
    this.stageOrder = [];

    // Pending stage requests
    this.pendingStageResolve = null;

    // Mask threshold (percentage of max magnitude)
    this.maskThreshold = 15;
    this.magnitudeData = null;  // Cached magnitude array for mask preview
    this.magnitudeMax = 0;

    // Mask editing state
    this.currentMaskData = null;      // Current edited mask (Float32Array)
    this.originalMaskData = null;     // Original threshold-based mask for reset
    this.maskDims = null;             // [nx, ny, nz] dimensions
    this.voxelSize = null;            // [dx, dy, dz] in mm

    // Drawing state
    this.drawingEnabled = false;
    this.brushMode = 'add';           // 'add' or 'remove'
    this.brushSize = 2;
    this.savedCrosshairWidth = 0.75;  // Store crosshair width when hiding

    // Pipeline settings (null values = calculate from voxel size)
    this.pipelineSettings = {
      combinedMethod: 'none',  // 'none', 'tgv', or 'qsmart'
      tgv: { regularization: 2, iterations: 1000, erosions: 3 },
      qsmart: {
        sdfSigma1Stage1: 10, sdfSigma2Stage1: 0,
        sdfSigma1Stage2: 8, sdfSigma2Stage2: 2,
        sdfSpatialRadius: 8, sdfLowerLim: 0.6, sdfCurvConstant: 500,
        // Vasculature detection parameters (in mm - auto-scaled to voxels)
        vascSphereRadiusMm: 8.0,  // mm - morphological filter radius
        frangiScaleMinMm: 1.0,    // mm - minimum vessel radius to detect (QSMART default: 1)
        frangiScaleMaxMm: 10.0,   // mm - maximum vessel radius to detect (QSMART default: 10)
        frangiScaleRatioMm: 2.0,  // mm - step between scales (QSMART default: 2)
        frangiC: 500,             // noise sensitivity threshold
        ilsqrTol: 0.01, ilsqrMaxIter: 50
      },
      unwrapMethod: 'romeo',  // 'romeo' or 'laplacian'
      multiEchoMethod: 'mcpc3ds',  // 'ols', 'ols_offset', 'mcpc3ds'
      mcpc3ds: { sigma: [10, 10, 5], weightType: 'phase_snr' },
      romeo: { weighting: 'phase_snr' },
      backgroundRemoval: 'vsharp',  // 'vsharp', 'sharp', 'smv', 'ismv', 'pdf', 'lbv'
      vsharp: { maxRadius: null, minRadius: null, threshold: 0.05 },
      sharp: { radius: 6, threshold: 0.05 },
      smv: { radius: null },
      ismv: { radius: null, tol: 0.001, maxit: 500 },
      pdf: { tol: 0.00001, maxit: null },
      lbv: { tol: 0.001, maxit: 500 },
      dipoleInversion: 'rts',  // 'tkd', 'tsvd', 'tikhonov', 'tv', 'rts', 'nltv', 'medi'
      tkd: { threshold: 0.15 },
      tsvd: { threshold: 0.15 },
      tikhonov: { lambda: 0.01, reg: 'identity' },
      tv: { lambda: 0.001, maxIter: 250, tol: 0.001 },
      rts: { delta: 0.15, mu: 100000, rho: 10, maxIter: 20 },
      nltv: { lambda: 0.001, mu: 1, maxIter: 250, tol: 0.001, newtonMaxIter: 10 },
      medi: { lambda: 1000, percentage: 0.9, maxIter: 10, cgMaxIter: 100, cgTol: 0.01, tol: 0.1, smv: false, smvRadius: 5, merit: false, dataWeighting: 1 }
    };

    // BET settings
    this.betSettings = {
      fractionalIntensity: 0.5,
      iterations: 1000,
      subdivisions: 4
    };

    // Mask preparation settings
    this.maskPrepSettings = {
      source: 'combined',        // 'first_echo' or 'combined'
      biasCorrection: true,
      prepared: false
    };
    this.preparedMagnitudeData = null;  // Cached prepared magnitude
    this.preparedMagnitudeMax = 0;

    // Track last run settings for intelligent caching
    this.lastRunSettings = null;
    this.pipelineHasRun = false;

    // Echo navigation state
    this.currentEchoIndex = 0;   // 0-indexed current echo
    this.currentViewType = null; // 'magnitude' or 'phase' when navigating echoes

    // Pipeline running state
    this.pipelineRunning = false;

    this.init();
  }

  async init() {
    await this.setupViewer();
    this.setupUIControls();
    this.setupEventListeners();
    this.updateDownloadButtons();

    // Initialize file lists
    this.updateFileList('magnitude', []);
    this.updateFileList('phase', []);
    this.updateFileList('json', []);

    // Initialize masking controls state (disabled until Prepare is clicked)
    this.updateMaskingControlsState();

    // Sync mask prep settings with actual UI state
    const sourceSelect = document.getElementById('maskInputSource');
    const biasCheckbox = document.getElementById('applyBiasCorrection');
    if (sourceSelect) this.maskPrepSettings.source = sourceSelect.value;
    if (biasCheckbox) this.maskPrepSettings.biasCorrection = biasCheckbox.checked;

    this.updateOutput("Ready. Upload magnitude, phase, and JSON files for each echo.");

    // Start loading WASM in the background immediately
    this.initializeWorker();
  }

  setupWorker() {
    if (this.worker) return;

    this.worker = new Worker('js/qsm-worker-pure.js');

    this.worker.onmessage = (e) => {
      const { type, ...data } = e.data;

      switch (type) {
        case 'progress':
          this.setProgress(data.value, data.text);
          break;

        case 'log':
          this.updateOutput(data.message);
          break;

        case 'error':
          this.updateOutput(`Error: ${data.message}`);
          this.setProgress(0, 'Failed');
          this.pipelineRunning = false;
          document.getElementById('cancelPipeline').disabled = true;
          this.updateEchoInfo(); // Re-enable run button if valid
          break;

        case 'initialized':
          this.workerReady = true;
          this.updateOutput("WASM ready");
          break;

        case 'complete':
          this.updateOutput("Pipeline completed successfully!");
          this.showStageButtons();
          // Add magnitude and phase buttons if we have multi-echo data
          if (this.multiEchoFiles.magnitude.length > 0) {
            this.addStageButton('magnitude', 'Input Magnitude');
          }
          if (this.multiEchoFiles.phase.length > 0) {
            this.addStageButton('phase', 'Input Phase');
          }
          // Save settings for intelligent caching on next run
          this.lastRunSettings = JSON.parse(JSON.stringify(this.pipelineSettings));
          this.pipelineHasRun = true;
          this.pipelineRunning = false;
          document.getElementById('cancelPipeline').disabled = true;
          this.updateEchoInfo(); // Re-enable run button
          break;

        case 'stageData':
          // Handle both live stage updates and explicit requests
          if (this.pendingStageResolve) {
            this.pendingStageResolve(data);
            this.pendingStageResolve = null;
          } else if (this.pipelineRunning) {
            // Live update during pipeline
            // displayNow defaults to true for backward compatibility
            const displayNow = data.displayNow !== false;
            if (displayNow) {
              this.displayLiveStageData(data);
            } else {
              // Just cache without displaying
              this.cacheStageData(data);
            }
          }
          break;
      }
    };

    this.worker.onerror = (e) => {
      this.updateOutput(`Worker error: ${e.message}`);
      console.error('Worker error:', e);
      this.pipelineRunning = false;
      document.getElementById('cancelPipeline').disabled = true;
      this.updateEchoInfo(); // Re-enable run button if valid
    };
  }

  async initializeWorker() {
    this.setupWorker();

    // Already initialized
    if (this.workerReady) return;

    // Already initializing - just wait for it
    if (this.workerInitializing) {
      return new Promise((resolve) => {
        const checkReady = setInterval(() => {
          if (this.workerReady) {
            clearInterval(checkReady);
            resolve();
          }
        }, 100);
      });
    }

    // Start initialization
    this.workerInitializing = true;
    this.updateOutput("Initializing WASM...");

    // Send init message to worker (no Python code needed)
    this.worker.postMessage({
      type: 'init',
      data: {}
    });

    // Wait for initialization
    return new Promise((resolve) => {
      const checkReady = setInterval(() => {
        if (this.workerReady) {
          clearInterval(checkReady);
          this.workerInitializing = false;
          resolve();
        }
      }, 100);
    });
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
    // Multi-echo file inputs
    document.getElementById('magnitudeFiles').addEventListener('change', (e) => {
      this.handleMultipleFiles(e, 'magnitude');
    });
    
    document.getElementById('phaseFiles').addEventListener('change', (e) => {
      this.handleMultipleFiles(e, 'phase');
    });
    
    document.getElementById('jsonFiles').addEventListener('change', (e) => {
      this.handleMultipleFiles(e, 'json');
    });

    // Echo time Tagify input
    this.setupEchoTagify();

    // Processing buttons
    document.getElementById('run').addEventListener('click', () => this.openPipelineSettingsModal());
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

    // Preview mask button
    const previewMaskBtn = document.getElementById('previewMask');
    if (previewMaskBtn) {
      previewMaskBtn.addEventListener('click', () => this.previewMask());
    }

    // BET brain extraction button - opens settings modal
    document.getElementById('runBET')?.addEventListener('click', () => this.openBetSettingsModal());

    // Auto threshold button (Otsu)
    document.getElementById('autoThreshold')?.addEventListener('click', () => this.autoDetectThreshold());

    // Mask Input Preparation
    document.getElementById('maskInputSource')?.addEventListener('change', (e) => {
      this.maskPrepSettings.source = e.target.value;
      this.maskPrepSettings.prepared = false;
      this.updatePrepareButtonState();
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
    document.getElementById('runPipelineWithSettings')?.addEventListener('click', () => this.runPipelineWithSettings());

    // BET settings modal
    document.getElementById('closeBetSettings')?.addEventListener('click', () => this.closeBetSettingsModal());
    document.getElementById('resetBetSettings')?.addEventListener('click', () => this.resetBetSettings());
    document.getElementById('runBetWithSettings')?.addEventListener('click', () => this.runBetWithSettings());

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

    // Close modals on overlay click
    document.getElementById('pipelineSettingsModal')?.addEventListener('click', (e) => {
      if (e.target.id === 'pipelineSettingsModal') this.closePipelineSettingsModal();
    });
    document.getElementById('betSettingsModal')?.addEventListener('click', (e) => {
      if (e.target.id === 'betSettingsModal') this.closeBetSettingsModal();
    });

    // Combined method dropdown - show/hide TGV settings and toggle other sections
    document.getElementById('combinedMethod')?.addEventListener('change', (e) => {
      this.updatePipelineModalVisibility();
    });

    // Method selection dropdowns - show/hide appropriate settings
    document.getElementById('unwrapMethod')?.addEventListener('change', (e) => {
      const isRomeo = e.target.value === 'romeo';
      document.getElementById('romeoSettings').style.display = isRomeo ? 'block' : 'none';
      document.getElementById('laplacianSettings').style.display = isRomeo ? 'none' : 'block';
    });

    // Single-echo unwrap method dropdown
    document.getElementById('singleEchoUnwrapMethod')?.addEventListener('change', (e) => {
      const isRomeo = e.target.value === 'romeo';
      document.getElementById('singleEchoRomeoSettings').style.display = isRomeo ? 'block' : 'none';
    });

    document.getElementById('bgRemovalMethod')?.addEventListener('change', (e) => {
      const method = e.target.value;
      document.getElementById('vsharpSettings').style.display = method === 'vsharp' ? 'block' : 'none';
      document.getElementById('sharpSettings').style.display = method === 'sharp' ? 'block' : 'none';
      document.getElementById('smvSettings').style.display = method === 'smv' ? 'block' : 'none';
      document.getElementById('ismvSettings').style.display = method === 'ismv' ? 'block' : 'none';
      document.getElementById('pdfSettings').style.display = method === 'pdf' ? 'block' : 'none';
      document.getElementById('lbvSettings').style.display = method === 'lbv' ? 'block' : 'none';
    });

    document.getElementById('dipoleMethod')?.addEventListener('change', (e) => {
      const method = e.target.value;
      document.getElementById('tkdSettings').style.display = method === 'tkd' ? 'block' : 'none';
      document.getElementById('tsvdSettings').style.display = method === 'tsvd' ? 'block' : 'none';
      document.getElementById('tikhonovSettings').style.display = method === 'tikhonov' ? 'block' : 'none';
      document.getElementById('tvSettings').style.display = method === 'tv' ? 'block' : 'none';
      document.getElementById('rtsSettings').style.display = method === 'rts' ? 'block' : 'none';
      document.getElementById('nltvSettings').style.display = method === 'nltv' ? 'block' : 'none';
      document.getElementById('mediSettings').style.display = method === 'medi' ? 'block' : 'none';
      document.getElementById('ilsqrSettings').style.display = method === 'ilsqr' ? 'block' : 'none';
    });

    // MEDI SMV checkbox toggle
    document.getElementById('mediSmv')?.addEventListener('change', (e) => {
      document.getElementById('mediSmvRadiusGroup').style.display = e.target.checked ? 'block' : 'none';
    });

    // Multi-echo method dropdown - show/hide appropriate settings
    document.getElementById('multiEchoMethod')?.addEventListener('change', (e) => {
      this.updatePipelineModalVisibility();
    });

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

  async handleMultipleFiles(event, type) {
    const files = Array.from(event.target.files);

    // Store files
    this.multiEchoFiles[type] = files.map(file => ({
      file: file,
      name: file.name
    }));

    // Update UI
    this.updateFileList(type, this.multiEchoFiles[type]);

    // Process JSON files immediately to extract echo times
    if (type === 'json') {
      await this.processJsonFiles(files);
    }

    // Enable preview buttons when files are loaded
    if (type === 'magnitude') {
      document.getElementById('vis_magnitude').disabled = files.length === 0;

      // Clear prepared state when magnitude files change
      this.maskPrepSettings.prepared = false;
      this.preparedMagnitudeData = null;
      this.preparedMagnitudeMax = 0;
      this.currentMaskData = null;
      this.originalMaskData = null;
      this.updatePrepareButtonState();  // This also calls updateMaskingControlsState

      if (files.length > 0) {
        const maskSection = document.getElementById('maskSection');
        if (maskSection) {
          maskSection.classList.remove('section-disabled');
        }
        // Load magnitude into viewer for visualization
        this.visualizeMagnitude();
      }
    }

    if (type === 'phase') {
      document.getElementById('vis_phase').disabled = files.length === 0;
    }

    // Update echo information
    this.updateEchoInfo();
  }

  updateFileList(type, fileList) {
    const listElement = document.getElementById(`${type}List`);
    const fileDrop = listElement?.closest('.upload-group')?.querySelector('.file-drop');

    if (!listElement) {
      console.error(`File list element not found: ${type}List`);
      return;
    }

    listElement.innerHTML = '';

    if (fileList.length > 0) {
      fileDrop?.classList.add('has-files');
      fileList.forEach((fileData, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
          <span>${fileData.name}</span>
          <button class="file-remove" onclick="app.removeFile('${type}', ${index})">×</button>
        `;
        listElement.appendChild(fileItem);
      });

      // Update drop label
      const label = fileDrop?.querySelector('.file-drop-label span');
      if (label) {
        label.textContent = `${fileList.length} file${fileList.length > 1 ? 's' : ''} selected`;
      }
    } else {
      fileDrop?.classList.remove('has-files');
      const label = fileDrop?.querySelector('.file-drop-label span');
      if (label) {
        const defaults = {
          'magnitude': 'Drop files or click',
          'phase': 'Drop files or click',
          'json': 'Drop files or click'
        };
        label.textContent = defaults[type] || 'Drop files or click';
      }
    }
  }

  removeFile(type, index) {
    this.multiEchoFiles[type].splice(index, 1);
    this.updateFileList(type, this.multiEchoFiles[type]);
    this.updateEchoInfo();

    // Re-disable buttons when files are removed
    if (type === 'magnitude') {
      document.getElementById('vis_magnitude').disabled = this.multiEchoFiles.magnitude.length === 0;
      if (this.multiEchoFiles.magnitude.length === 0) {
        const maskSection = document.getElementById('maskSection');
        if (maskSection) {
          maskSection.classList.add('section-disabled');
        }
        document.getElementById('previewMask')?.setAttribute('disabled', '');
        document.getElementById('runBET')?.setAttribute('disabled', '');
      }
    }

    if (type === 'phase') {
      document.getElementById('vis_phase').disabled = this.multiEchoFiles.phase.length === 0;
    }
  }

  async processJsonFiles(files) {
    console.log('Processing JSON files:', files);
    const echoTimes = [];
    let fieldStrength = null;

    for (const file of files) {
      try {
        const text = await file.text();
        const json = JSON.parse(text);

        console.log(`JSON file ${file.name} contents:`, json);
        console.log(`EchoTime field:`, json.EchoTime);
        console.log(`Available fields:`, Object.keys(json));

        // Extract echo time (in seconds, convert to ms)
        let echoTime = null;
        if (json.EchoTime) {
          echoTime = json.EchoTime * 1000; // Convert to ms
          console.log(`Found EchoTime: ${json.EchoTime}s -> ${echoTime}ms`);
        } else if (json.echo_time) {
          echoTime = json.echo_time * 1000;
          console.log(`Found echo_time: ${json.echo_time}s -> ${echoTime}ms`);
        } else if (json.TE) {
          echoTime = json.TE;
          console.log(`Found TE: ${json.TE}ms`);
        } else {
          console.warn(`No echo time found in ${file.name}. Available fields:`, Object.keys(json));
        }

        // Extract field strength (in Tesla) - only need to find it once
        if (fieldStrength === null) {
          if (json.MagneticFieldStrength) {
            fieldStrength = json.MagneticFieldStrength;
            console.log(`Found MagneticFieldStrength: ${fieldStrength}T`);
          } else if (json.FieldStrength) {
            fieldStrength = json.FieldStrength;
            console.log(`Found FieldStrength: ${fieldStrength}T`);
          } else if (json.field_strength) {
            fieldStrength = json.field_strength;
            console.log(`Found field_strength: ${fieldStrength}T`);
          }
        }

        if (echoTime !== null) {
          echoTimes.push({
            file: file.name,
            echoTime: echoTime,
            json: json
          });
          console.log(`Added echo time for ${file.name}: ${echoTime}ms`);
        } else {
          console.error(`Could not extract echo time from ${file.name}`);
        }
      } catch (error) {
        console.error(`Error parsing JSON file ${file.name}:`, error);
      }
    }

    // Sort by echo time
    echoTimes.sort((a, b) => a.echoTime - b.echoTime);
    this.multiEchoFiles.echoTimes = echoTimes;

    console.log(`Final processed echo times:`, echoTimes);
    console.log(`Stored in multiEchoFiles:`, this.multiEchoFiles.echoTimes);

    // Populate the editable inputs
    this.populateEchoTimeInputs(echoTimes.map(et => et.echoTime));

    // Populate field strength if found
    if (fieldStrength !== null) {
      const fieldInput = document.getElementById('magField');
      if (fieldInput) {
        fieldInput.value = fieldStrength;
        console.log(`Set field strength input to ${fieldStrength}T`);
        this.updateOutput(`Field strength: ${fieldStrength}T`);
      }
    }
  }

  updateEchoInfo() {
    const magCount = this.multiEchoFiles.magnitude.length;
    const phaseCount = this.multiEchoFiles.phase.length;
    const echoTimes = this.getEchoTimesFromInputs();
    const echoTimeCount = echoTimes.length;

    const runButton = document.getElementById('run');

    const isValid = magCount === phaseCount && magCount > 0;
    const hasEchoTimes = echoTimeCount > 0;
    const hasMask = this.currentMaskData !== null;
    const canRun = isValid && hasEchoTimes && hasMask;

    // Update run button
    if (runButton) {
      runButton.disabled = !canRun || this.pipelineRunning;
    }

    // Update pipeline section enabled state
    const pipelineSection = document.getElementById('pipelineSection');
    if (pipelineSection) {
      if (canRun || this.pipelineRunning) {
        pipelineSection.classList.remove('section-disabled');
      } else {
        pipelineSection.classList.add('section-disabled');
      }
    }
  }

  // Echo time Tagify management
  setupEchoTagify() {
    const input = document.getElementById('echoTimesTagify');
    if (!input || this.echoTagify) return;

    this.echoTagify = new Tagify(input, {
      delimiters: ',| ',
      pattern: /^[\d.]+$/,
      transformTag: (tagData) => {
        const num = parseFloat(tagData.value);
        if (!isNaN(num) && num > 0) {
          tagData.value = num.toFixed(2);
        }
      },
      validate: (tagData) => {
        const num = parseFloat(tagData.value);
        return !isNaN(num) && num > 0;
      },
      editTags: 1,
      placeholder: 'Type values...'
    });

    this.echoTagify.on('change', () => this.updateEchoInfo());
  }

  populateEchoTimeInputs(echoTimes) {
    if (!this.echoTagify) return;

    const tags = echoTimes.map(t => ({ value: t.toFixed(2) }));
    this.echoTagify.removeAllTags();
    this.echoTagify.addTags(tags);
    this.updateEchoInfo();
  }

  getEchoTimesFromInputs() {
    if (!this.echoTagify) return [];

    return this.echoTagify.value
      .map(tag => parseFloat(tag.value))
      .filter(n => !isNaN(n) && n > 0)
      .sort((a, b) => a - b);
  }

  async visualizeMagnitude() {
    if (this.multiEchoFiles.magnitude.length === 0) {
      this.updateOutput("No magnitude files uploaded");
      return;
    }

    this.currentViewType = 'magnitude';
    this.currentEchoIndex = 0;
    this.updateEchoNavigation();
    await this.visualizeCurrentEcho();
  }

  async visualizePhase() {
    if (this.multiEchoFiles.phase.length === 0) {
      this.updateOutput("No phase files uploaded");
      return;
    }

    this.currentViewType = 'phase';
    this.currentEchoIndex = 0;
    this.updateEchoNavigation();
    await this.visualizeCurrentEcho();
  }

  // Navigate to previous/next echo
  navigateEcho(direction) {
    if (!this.currentViewType) return;

    const files = this.multiEchoFiles[this.currentViewType];
    if (!files || files.length === 0) return;

    const newIndex = this.currentEchoIndex + direction;
    if (newIndex >= 0 && newIndex < files.length) {
      this.currentEchoIndex = newIndex;
      this.updateEchoNavigation();
      this.visualizeCurrentEcho();
    }
  }

  // Visualize the current echo based on currentViewType and currentEchoIndex
  async visualizeCurrentEcho() {
    if (!this.currentViewType) return;

    const files = this.multiEchoFiles[this.currentViewType];
    if (!files || files.length === 0) return;

    const file = files[this.currentEchoIndex].file;
    const typeName = this.currentViewType.charAt(0).toUpperCase() + this.currentViewType.slice(1);
    await this.loadAndVisualizeFile(file, `${typeName} (Echo ${this.currentEchoIndex + 1})`);
  }

  // Update echo navigation UI visibility and labels
  updateEchoNavigation() {
    const echoNav = document.getElementById('echoNav');
    const echoLabel = document.getElementById('echoLabel');
    const echoPrev = document.getElementById('echoPrev');
    const echoNext = document.getElementById('echoNext');

    if (!echoNav || !this.currentViewType) {
      if (echoNav) echoNav.style.display = 'none';
      return;
    }

    const files = this.multiEchoFiles[this.currentViewType];
    const numEchoes = files?.length || 0;

    if (numEchoes <= 1) {
      echoNav.style.display = 'none';
      return;
    }

    echoNav.style.display = 'flex';
    echoLabel.textContent = `Echo ${this.currentEchoIndex + 1}/${numEchoes}`;
    echoPrev.disabled = this.currentEchoIndex === 0;
    echoNext.disabled = this.currentEchoIndex >= numEchoes - 1;
  }

  // Hide echo navigation when viewing non-echo data (pipeline results)
  hideEchoNavigation() {
    this.currentViewType = null;
    const echoNav = document.getElementById('echoNav');
    if (echoNav) echoNav.style.display = 'none';
  }

  async loadAndVisualizeFile(file, description) {
    try {
      this.updateOutput(`Loading ${description}...`);

      // Create a blob URL for NiiVue to load
      const url = URL.createObjectURL(file);

      // Load using loadVolumes which handles the URL properly
      await this.nv.loadVolumes([{ url: url, name: file.name }]);

      // Clean up the blob URL
      URL.revokeObjectURL(url);

      // Hide overlay control (no overlay after loading new volume)
      this.showOverlayControl(false);

      // Enable download button
      this.updateDownloadVolumeButton();

      this.updateOutput(`${description} loaded`);
      this.currentFile = file;
    } catch (error) {
      this.updateOutput(`Error loading ${description}: ${error.message}`);
      console.error(error);
    }
  }

  /**
   * Update the Prepare button state based on current settings
   */
  updatePrepareButtonState() {
    const btn = document.getElementById('prepareMaskInput');
    const hasMagnitude = this.multiEchoFiles.magnitude.length > 0;

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

    // Preview Mask (Threshold) button - enabled when prepared
    const previewBtn = document.getElementById('previewMask');
    if (previewBtn) previewBtn.disabled = !prepared;

    // BET button - enabled when prepared
    const betBtn = document.getElementById('runBET');
    if (betBtn) betBtn.disabled = !prepared;

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
   */
  async prepareMaskInput() {
    if (this.multiEchoFiles.magnitude.length === 0) {
      this.updateOutput("No magnitude files uploaded");
      return;
    }

    const btn = document.getElementById('prepareMaskInput');
    const originalBtnHtml = btn?.innerHTML;

    if (btn) {
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> Preparing...';
    }

    try {
      // Debug: show current settings
      console.log('Prepare settings:', this.maskPrepSettings);
      this.updateOutput(`Preparing: source=${this.maskPrepSettings.source}, biasCorrection=${this.maskPrepSettings.biasCorrection}`);

      // Always load first file into NiiVue first - this handles gzip decompression
      // and gives us proper header access through the volume object
      const firstFile = this.multiEchoFiles.magnitude[0].file;
      const url = URL.createObjectURL(firstFile);
      await this.nv.loadVolumes([{ url: url, name: firstFile.name }]);
      URL.revokeObjectURL(url);

      // Get the uncompressed NIfTI data from NiiVue
      // NiiVue stores the raw (uncompressed) NIfTI in volume.fileObject or we can reconstruct
      // For header, we'll use the volume's hdr property
      const vol = this.nv.volumes[0];

      // Store header bytes from NiiVue's volume for createMaskNifti
      // We need to create a proper uncompressed NIfTI header
      this.magnitudeFileBytes = this.createNiftiHeaderFromVolume(vol);

      let magnitudeData;

      if (this.maskPrepSettings.source === 'combined') {
        // Load and combine all echoes with RSS
        this.updateOutput("Combining magnitude echoes (RSS)...");
        magnitudeData = await this.combineMagnitudeRSS();
      } else {
        // Just use first echo (already loaded)
        magnitudeData = new Float64Array(vol.img);
      }

      // Apply bias correction if enabled
      if (this.maskPrepSettings.biasCorrection) {
        this.updateOutput("Applying bias correction...");
        const beforeSum = magnitudeData.reduce((a, b) => a + b, 0);
        magnitudeData = await this.applyBiasCorrection(magnitudeData);
        const afterSum = magnitudeData.reduce((a, b) => a + b, 0);
        console.log(`Bias correction: before sum=${beforeSum.toExponential(3)}, after sum=${afterSum.toExponential(3)}`);
        this.updateOutput(`Bias correction applied (sum changed: ${(beforeSum !== afterSum)})`);
      }

      // Cache the prepared data
      this.preparedMagnitudeData = magnitudeData;
      this.magnitudeData = magnitudeData;

      // Calculate max
      let max = -Infinity;
      for (let i = 0; i < magnitudeData.length; i++) {
        if (magnitudeData[i] > max) max = magnitudeData[i];
      }
      this.preparedMagnitudeMax = max;
      this.magnitudeMax = max;

      this.maskPrepSettings.prepared = true;
      this.updatePrepareButtonState();

      // Hide echo navigation (we're viewing combined/processed data now)
      this.hideEchoNavigation();

      // Display the prepared magnitude as the base volume (no mask yet)
      await this.displayPreparedMagnitude();

      // Set threshold to Otsu-detected value by default
      this.autoDetectThreshold();

      this.updateOutput("Magnitude prepared. Click 'Threshold' to generate mask.");

    } catch (error) {
      this.updateOutput(`Error: ${error.message}`);
      console.error(error);
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = originalBtnHtml || 'Prepare';
        this.updatePrepareButtonState();
      }
    }
  }

  /**
   * Display the prepared magnitude data as the base volume
   */
  async displayPreparedMagnitude() {
    if (!this.preparedMagnitudeData || !this.magnitudeFileBytes) return;

    // Create NIfTI from prepared data (similar to createMaskNifti but with float64 data)
    const srcView = new DataView(this.magnitudeFileBytes);
    const voxOffset = srcView.getFloat32(108, true);
    const headerSize = Math.ceil(voxOffset);

    // Create buffer: header + data as float64
    const dataSize = this.preparedMagnitudeData.length * 8; // 8 bytes per float64
    const buffer = new ArrayBuffer(headerSize + dataSize);
    const destBytes = new Uint8Array(buffer);
    const destView = new DataView(buffer);

    // Copy header
    destBytes.set(new Uint8Array(this.magnitudeFileBytes).slice(0, headerSize));

    // Update datatype to FLOAT64 (64) at offset 70
    destView.setInt16(70, 64, true);
    // Update bitpix to 64 at offset 72
    destView.setInt16(72, 64, true);

    // Make it 3D
    destView.setInt16(40, 3, true);
    destView.setInt16(48, 1, true);

    // Copy data
    const dataView = new Float64Array(buffer, headerSize);
    dataView.set(this.preparedMagnitudeData);

    // Load into NiiVue
    const blob = new Blob([buffer], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);

    await this.nv.loadVolumes([{ url: url, name: 'prepared_magnitude.nii' }]);
    URL.revokeObjectURL(url);

    // Store reference to the volume for header info
    this.magnitudeVolume = this.nv.volumes[0];

    // Enable download button
    this.updateDownloadVolumeButton();
  }

  /**
   * Create a minimal NIfTI header buffer from a NiiVue volume
   * This is used when the original file was gzipped and we need uncompressed header
   */
  createNiftiHeaderFromVolume(vol) {
    // NIfTI-1 header is 348 bytes, data starts at 352 (vox_offset)
    const headerSize = 352;
    const buffer = new ArrayBuffer(headerSize);
    const view = new DataView(buffer);
    const hdr = vol.hdr;

    // sizeof_hdr (offset 0) - must be 348 for NIfTI-1
    view.setInt32(0, 348, true);

    // dim array (offset 40) - 8 int16 values
    const dims = hdr.dims || [3, vol.dims[1], vol.dims[2], vol.dims[3], 1, 1, 1, 1];
    for (let i = 0; i < 8; i++) {
      view.setInt16(40 + i * 2, dims[i] || 0, true);
    }

    // datatype (offset 70) - we'll use FLOAT32 = 16
    view.setInt16(70, 16, true);

    // bitpix (offset 72) - 32 bits for float32
    view.setInt16(72, 32, true);

    // pixdim array (offset 76) - 8 float32 values
    const pixdim = hdr.pixDims || [1, vol.pixDims[1] || 1, vol.pixDims[2] || 1, vol.pixDims[3] || 1, 1, 1, 1, 1];
    for (let i = 0; i < 8; i++) {
      view.setFloat32(76 + i * 4, pixdim[i] || 1, true);
    }

    // vox_offset (offset 108) - where data starts
    view.setFloat32(108, headerSize, true);

    // scl_slope (offset 112) and scl_inter (offset 116)
    view.setFloat32(112, hdr.scl_slope || 1, true);
    view.setFloat32(116, hdr.scl_inter || 0, true);

    // xyzt_units (offset 123) - typically 2 (mm) + 8 (sec) = 10
    view.setUint8(123, 10);

    // qform_code (offset 252) and sform_code (offset 254)
    view.setInt16(252, hdr.qform_code || 1, true);
    view.setInt16(254, hdr.sform_code || 1, true);

    // Affine matrix (srow_x, srow_y, srow_z at offsets 280, 296, 312)
    if (hdr.affine) {
      for (let i = 0; i < 4; i++) {
        view.setFloat32(280 + i * 4, hdr.affine[0][i] || 0, true);  // srow_x
        view.setFloat32(296 + i * 4, hdr.affine[1][i] || 0, true);  // srow_y
        view.setFloat32(312 + i * 4, hdr.affine[2][i] || 0, true);  // srow_z
      }
    }

    // magic (offset 344) - "n+1\0" for NIfTI-1
    view.setUint8(344, 0x6E);  // 'n'
    view.setUint8(345, 0x2B);  // '+'
    view.setUint8(346, 0x31);  // '1'
    view.setUint8(347, 0x00);  // '\0'

    return buffer;
  }

  /**
   * Combine multiple magnitude echoes using Root Sum of Squares (RSS)
   * @returns {Float64Array} RSS-combined magnitude
   */
  async combineMagnitudeRSS() {
    const nEchoes = this.multiEchoFiles.magnitude.length;
    if (nEchoes === 0) throw new Error("No magnitude files");

    if (nEchoes === 1) {
      // Single echo - just return it as Float64Array
      const file = this.multiEchoFiles.magnitude[0].file;
      const url = URL.createObjectURL(file);
      await this.nv.loadVolumes([{ url: url, name: file.name }]);
      URL.revokeObjectURL(url);
      return new Float64Array(this.nv.volumes[0].img);
    }

    // Load first echo to get dimensions
    const firstFile = this.multiEchoFiles.magnitude[0].file;
    const url = URL.createObjectURL(firstFile);
    await this.nv.loadVolumes([{ url: url, name: firstFile.name }]);
    URL.revokeObjectURL(url);

    const firstData = this.nv.volumes[0].img;
    const nTotal = firstData.length;

    // Initialize sum of squares
    const rssData = new Float64Array(nTotal);

    // Add squared values from first echo
    for (let i = 0; i < nTotal; i++) {
      rssData[i] = firstData[i] * firstData[i];
    }

    // Load and add remaining echoes
    for (let e = 1; e < nEchoes; e++) {
      const file = this.multiEchoFiles.magnitude[e].file;
      const echoUrl = URL.createObjectURL(file);
      await this.nv.loadVolumes([{ url: echoUrl, name: file.name }]);
      URL.revokeObjectURL(echoUrl);

      const echoData = this.nv.volumes[0].img;
      for (let i = 0; i < nTotal; i++) {
        rssData[i] += echoData[i] * echoData[i];
      }
    }

    // Take square root
    for (let i = 0; i < nTotal; i++) {
      rssData[i] = Math.sqrt(rssData[i]);
    }

    this.updateOutput(`Combined ${nEchoes} echoes with RSS`);
    return rssData;
  }

  /**
   * Apply bias field correction to magnitude data
   * @param {Float64Array} magnitudeData - Input magnitude
   * @returns {Float64Array} Bias-corrected magnitude
   */
  async applyBiasCorrection(magnitudeData) {
    // Ensure worker is ready
    if (!this.workerReady) {
      await this.ensureWorkerReady();
    }

    // Get dimensions from NIfTI header
    const srcView = new DataView(this.magnitudeFileBytes);
    const nx = srcView.getInt16(42, true);
    const ny = srcView.getInt16(44, true);
    const nz = srcView.getInt16(46, true);
    const vx = srcView.getFloat32(80, true) || 1;
    const vy = srcView.getFloat32(84, true) || 1;
    const vz = srcView.getFloat32(88, true) || 1;

    // Call WASM bias correction via worker
    return new Promise((resolve, reject) => {
      const messageHandler = (event) => {
        if (event.data.type === 'biasCorrection') {
          this.worker.removeEventListener('message', messageHandler);
          if (event.data.error) {
            reject(new Error(event.data.error));
          } else {
            resolve(new Float64Array(event.data.result));
          }
        }
      };

      this.worker.addEventListener('message', messageHandler);
      this.worker.postMessage({
        type: 'biasCorrection',
        data: {
          magnitude: magnitudeData,
          nx, ny, nz,
          vx, vy, vz,
          sigma_mm: 7.0,
          nbox: 15
        }
      });
    });
  }

  async previewMask() {
    if (!this.maskPrepSettings.prepared) {
      this.updateOutput("Please click 'Prepare' first");
      return;
    }

    if (!this.preparedMagnitudeData) {
      this.updateOutput("No prepared magnitude data available");
      return;
    }

    try {
      // Use the prepared data
      this.magnitudeData = this.preparedMagnitudeData;
      this.magnitudeMax = this.preparedMagnitudeMax;

      // Enable threshold slider since user chose threshold-based masking
      this.setThresholdSliderEnabled(true);

      await this.updateMaskPreview();
      this.updateOutput("Adjust threshold slider to refine mask");
    } catch (error) {
      this.updateOutput(`Error: ${error.message}`);
      console.error(error);
    }
  }

  async updateMaskPreview() {
    if (!this.magnitudeData || !this.nv.volumes.length || !this.magnitudeFileBytes) return;
    if (this.maskUpdating) return;  // Prevent concurrent updates

    this.maskUpdating = true;

    try {
      const threshold = (this.maskThreshold / 100) * this.magnitudeMax;
      const totalVoxels = this.magnitudeData.length;

      // Extract dimensions from NIfTI header
      const srcView = new DataView(this.magnitudeFileBytes);
      const nx = srcView.getInt16(42, true);  // dim[1]
      const ny = srcView.getInt16(44, true);  // dim[2]
      const nz = srcView.getInt16(46, true);  // dim[3]
      this.maskDims = [nx, ny, nz];

      // Extract voxel size from NIfTI header (pixdim[1-3] at offsets 80, 84, 88)
      const dx = srcView.getFloat32(80, true) || 1;
      const dy = srcView.getFloat32(84, true) || 1;
      const dz = srcView.getFloat32(88, true) || 1;
      this.voxelSize = [dx, dy, dz];

      // Create mask data from threshold
      const maskData = new Float32Array(totalVoxels);
      for (let i = 0; i < totalVoxels; i++) {
        maskData[i] = this.magnitudeData[i] > threshold ? 1 : 0;
      }

      // Store as both current and original (for reset)
      this.currentMaskData = maskData;
      this.originalMaskData = new Float32Array(maskData);

      // Display the mask
      await this.displayCurrentMask();

      // Show morphological operations panel
      const opsPanel = document.getElementById('maskOperations');
      if (opsPanel) opsPanel.style.display = 'block';

      // Update run button state (mask is now available)
      this.updateEchoInfo();

    } catch (error) {
      console.error('Error updating mask preview:', error);
    } finally {
      this.maskUpdating = false;
    }
  }

  async displayCurrentMask() {
    if (!this.currentMaskData) return;

    // Close any existing drawing layer
    if (this.nv.drawBitmap) {
      this.nv.closeDrawing();
    }

    // Remove ALL existing overlays (everything except base volume)
    while (this.nv.volumes.length > 1) {
      await this.nv.removeVolumeByIndex(1);
    }

    // Create mask NIfTI by copying header from original file
    const maskNifti = this.createMaskNifti(this.currentMaskData);
    const maskBlob = new Blob([maskNifti], { type: 'application/octet-stream' });
    const maskUrl = URL.createObjectURL(maskBlob);

    // Get current opacity from slider
    const opacitySlider = document.getElementById('overlayOpacity');
    const opacity = opacitySlider ? parseInt(opacitySlider.value) / 100 : 0.5;

    await this.nv.addVolumeFromUrl({
      url: maskUrl,
      name: 'mask_preview.nii',
      colormap: 'red',
      opacity: opacity
    });

    URL.revokeObjectURL(maskUrl);

    // Show overlay opacity control when overlay exists
    this.showOverlayControl(true);
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

  // 3D morphological erosion (6-connected)
  erodeMask3D() {
    if (!this.currentMaskData || !this.maskDims) return;

    const [nx, ny, nz] = this.maskDims;
    const src = this.currentMaskData;
    const dst = new Float32Array(src.length);

    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
          const idx = x + y * nx + z * nx * ny;

          // Check if all 6 neighbors are inside mask
          if (src[idx] > 0) {
            let allNeighbors = true;

            // Check 6-connected neighbors
            if (x > 0 && src[idx - 1] === 0) allNeighbors = false;
            if (x < nx - 1 && src[idx + 1] === 0) allNeighbors = false;
            if (y > 0 && src[idx - nx] === 0) allNeighbors = false;
            if (y < ny - 1 && src[idx + nx] === 0) allNeighbors = false;
            if (z > 0 && src[idx - nx * ny] === 0) allNeighbors = false;
            if (z < nz - 1 && src[idx + nx * ny] === 0) allNeighbors = false;

            dst[idx] = allNeighbors ? 1 : 0;
          }
        }
      }
    }

    this.currentMaskData = dst;
  }

  // 3D morphological dilation (6-connected)
  dilateMask3D() {
    if (!this.currentMaskData || !this.maskDims) return;

    const [nx, ny, nz] = this.maskDims;
    const src = this.currentMaskData;
    const dst = new Float32Array(src.length);

    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
          const idx = x + y * nx + z * nx * ny;

          // Check if any of 6 neighbors is in mask
          if (src[idx] > 0) {
            dst[idx] = 1;
          } else {
            let anyNeighbor = false;

            if (x > 0 && src[idx - 1] > 0) anyNeighbor = true;
            if (x < nx - 1 && src[idx + 1] > 0) anyNeighbor = true;
            if (y > 0 && src[idx - nx] > 0) anyNeighbor = true;
            if (y < ny - 1 && src[idx + nx] > 0) anyNeighbor = true;
            if (z > 0 && src[idx - nx * ny] > 0) anyNeighbor = true;
            if (z < nz - 1 && src[idx + nx * ny] > 0) anyNeighbor = true;

            dst[idx] = anyNeighbor ? 1 : 0;
          }
        }
      }
    }

    this.currentMaskData = dst;
  }

  // Fill holes in 3D mask using flood fill from edges
  fillHoles3D() {
    if (!this.currentMaskData || !this.maskDims) return;

    const [nx, ny, nz] = this.maskDims;
    const mask = this.currentMaskData;

    // Create a "visited from outside" array
    const outside = new Uint8Array(mask.length);

    // Use a queue for flood fill
    const queue = [];

    // Seed from all boundary voxels that are outside the mask
    // X boundaries
    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        const idx0 = 0 + y * nx + z * nx * ny;
        const idx1 = (nx - 1) + y * nx + z * nx * ny;
        if (mask[idx0] === 0 && !outside[idx0]) { outside[idx0] = 1; queue.push(idx0); }
        if (mask[idx1] === 0 && !outside[idx1]) { outside[idx1] = 1; queue.push(idx1); }
      }
    }
    // Y boundaries
    for (let z = 0; z < nz; z++) {
      for (let x = 0; x < nx; x++) {
        const idx0 = x + 0 * nx + z * nx * ny;
        const idx1 = x + (ny - 1) * nx + z * nx * ny;
        if (mask[idx0] === 0 && !outside[idx0]) { outside[idx0] = 1; queue.push(idx0); }
        if (mask[idx1] === 0 && !outside[idx1]) { outside[idx1] = 1; queue.push(idx1); }
      }
    }
    // Z boundaries
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const idx0 = x + y * nx + 0 * nx * ny;
        const idx1 = x + y * nx + (nz - 1) * nx * ny;
        if (mask[idx0] === 0 && !outside[idx0]) { outside[idx0] = 1; queue.push(idx0); }
        if (mask[idx1] === 0 && !outside[idx1]) { outside[idx1] = 1; queue.push(idx1); }
      }
    }

    // Flood fill from boundary
    const nxy = nx * ny;
    while (queue.length > 0) {
      const idx = queue.shift();
      const x = idx % nx;
      const y = Math.floor((idx % nxy) / nx);
      const z = Math.floor(idx / nxy);

      // Check 6-connected neighbors
      const neighbors = [];
      if (x > 0) neighbors.push(idx - 1);
      if (x < nx - 1) neighbors.push(idx + 1);
      if (y > 0) neighbors.push(idx - nx);
      if (y < ny - 1) neighbors.push(idx + nx);
      if (z > 0) neighbors.push(idx - nxy);
      if (z < nz - 1) neighbors.push(idx + nxy);

      for (const nidx of neighbors) {
        if (mask[nidx] === 0 && !outside[nidx]) {
          outside[nidx] = 1;
          queue.push(nidx);
        }
      }
    }

    // Fill holes: set all non-outside, non-mask voxels to 1
    const result = new Float32Array(mask.length);
    for (let i = 0; i < mask.length; i++) {
      result[i] = (mask[i] > 0 || !outside[i]) ? 1 : 0;
    }

    this.currentMaskData = result;
  }

  // Clear mask completely
  async clearMask() {
    this.currentMaskData = null;
    this.originalMaskData = null;

    // Close any drawing layer
    if (this.nv.drawBitmap) {
      this.nv.closeDrawing();
    }

    // Remove mask overlay (keep only base volume)
    while (this.nv.volumes.length > 1) {
      await this.nv.removeVolumeByIndex(1);
    }

    // Hide overlay control when no overlay
    this.showOverlayControl(false);

    // Update run button state (mask no longer available)
    this.updateEchoInfo();
  }

  // Toggle drawing mode on/off
  async toggleDrawingMode() {
    this.drawingEnabled = !this.drawingEnabled;

    const enableBtn = document.getElementById('enableDrawing');
    const addBtn = document.getElementById('brushAdd');
    const removeBtn = document.getElementById('brushRemove');
    const sizeControl = document.getElementById('brushSizeControl');
    const actionsDiv = document.getElementById('drawingActions');

    if (this.drawingEnabled) {
      // Need mask data and base volume
      if (!this.currentMaskData || this.nv.volumes.length === 0) {
        this.updateOutput("Please preview mask first before drawing");
        this.drawingEnabled = false;
        return;
      }

      // Enable drawing UI
      enableBtn.classList.add('active');
      addBtn.disabled = false;
      removeBtn.disabled = false;
      sizeControl.style.display = 'block';
      actionsDiv.style.display = 'grid';

      // Save current crosshair width and hide it
      this.savedCrosshairWidth = this.nv.opts.crosshairWidth;
      this.nv.opts.crosshairWidth = 0;
      this.nv.drawScene();  // Redraw to hide crosshair

      // Remove mask overlay - we'll use drawing layer instead
      while (this.nv.volumes.length > 1) {
        await this.nv.removeVolumeByIndex(1);
      }

      // Hide overlay control when in drawing mode (no overlay)
      this.showOverlayControl(false);

      // Create drawing and load current mask into it
      this.nv.createEmptyDrawing();

      // Copy current mask to drawing bitmap
      const totalVoxels = this.currentMaskData.length;
      for (let i = 0; i < totalVoxels; i++) {
        this.nv.drawBitmap[i] = this.currentMaskData[i] > 0 ? 1 : 0;
      }

      // Refresh the drawing display
      this.nv.refreshDrawing(true);

      // Enable drawing mode
      this.nv.setDrawingEnabled(true);
      this.nv.opts.penSize = this.brushSize;
      this.nv.setDrawOpacity(0.5);

      // Set pen value for add mode (1)
      this.nv.setPenValue(1, false);
      this.brushMode = 'add';

      // Update UI to show add mode active
      addBtn.classList.add('active');
      removeBtn.classList.remove('active');

      this.updateOutput("Draw mode: DRAG to add, switch to Remove & drag to erase. Click Apply when done.");
    } else {
      // Disable drawing
      enableBtn.classList.remove('active');
      addBtn.disabled = true;
      removeBtn.disabled = true;
      addBtn.classList.remove('active');
      removeBtn.classList.remove('active');
      sizeControl.style.display = 'none';
      actionsDiv.style.display = 'none';

      // Restore crosshair
      if (this.savedCrosshairWidth !== undefined) {
        this.nv.opts.crosshairWidth = this.savedCrosshairWidth;
      }

      this.nv.setDrawingEnabled(false);
      this.nv.closeDrawing();

      // Redisplay the mask as overlay
      await this.displayCurrentMask();

      this.updateOutput("Drawing mode disabled");
    }
  }

  // Set brush mode (add or remove)
  setBrushMode(mode) {
    this.brushMode = mode;

    const addBtn = document.getElementById('brushAdd');
    const removeBtn = document.getElementById('brushRemove');

    addBtn.classList.toggle('active', mode === 'add');
    removeBtn.classList.toggle('active', mode === 'remove');

    // Pen value: 1 for adding to mask, 0 for erasing from mask
    // NiiVue uses 0 as the erase value
    const penValue = mode === 'add' ? 1 : 0;
    this.nv.setPenValue(penValue, false);

    this.updateOutput(`Brush: ${mode === 'add' ? 'Add (paint)' : 'Remove (erase)'}`);
  }

  // Apply the drawing to the current mask
  async applyDrawingToMask() {
    if (!this.currentMaskData || !this.maskDims) {
      this.updateOutput("No mask data to apply drawing to");
      return;
    }

    try {
      const drawBitmap = this.nv.drawBitmap;

      if (!drawBitmap || drawBitmap.length === 0) {
        this.updateOutput("No drawing to apply");
        return;
      }

      // Copy drawing bitmap directly to mask
      // The drawing IS the mask now, so just copy it back
      const totalVoxels = this.currentMaskData.length;
      let maskCount = 0;

      for (let i = 0; i < Math.min(drawBitmap.length, totalVoxels); i++) {
        this.currentMaskData[i] = drawBitmap[i] > 0 ? 1 : 0;
        if (drawBitmap[i] > 0) maskCount++;
      }

      // Exit drawing mode and show the mask as overlay
      this.drawingEnabled = false;

      // Update UI
      document.getElementById('enableDrawing')?.classList.remove('active');
      document.getElementById('brushAdd').disabled = true;
      document.getElementById('brushRemove').disabled = true;
      document.getElementById('brushAdd')?.classList.remove('active');
      document.getElementById('brushRemove')?.classList.remove('active');
      document.getElementById('brushSizeControl').style.display = 'none';
      document.getElementById('drawingActions').style.display = 'none';

      // Restore crosshair
      if (this.savedCrosshairWidth !== undefined) {
        this.nv.opts.crosshairWidth = this.savedCrosshairWidth;
      }

      // Close drawing and display mask as overlay
      this.nv.closeDrawing();
      this.nv.setDrawingEnabled(false);
      await this.displayCurrentMask();

      const coverage = ((maskCount / totalVoxels) * 100).toFixed(1);
      this.updateOutput(`Mask updated: ${maskCount.toLocaleString()} voxels (${coverage}%)`);

      // Update run button state (mask may have been created/modified)
      this.updateEchoInfo();
    } catch (error) {
      this.updateOutput(`Error applying drawing: ${error.message}`);
      console.error(error);
    }
  }

  createMaskNifti(maskData) {
    // Copy header from original magnitude file and replace data
    const srcBytes = new Uint8Array(this.magnitudeFileBytes);
    const srcView = new DataView(this.magnitudeFileBytes);

    // Get vox_offset from original header (offset 108, float32)
    const voxOffset = srcView.getFloat32(108, true);
    const headerSize = Math.ceil(voxOffset);

    // Create new buffer: header + mask data (as float32)
    const dataSize = maskData.length * 4;
    const buffer = new ArrayBuffer(headerSize + dataSize);
    const destBytes = new Uint8Array(buffer);
    const destView = new DataView(buffer);

    // Copy entire header from source
    destBytes.set(srcBytes.slice(0, headerSize));

    // Update datatype to FLOAT32 (16) at offset 70
    destView.setInt16(70, 16, true);
    // Update bitpix to 32 at offset 72
    destView.setInt16(72, 32, true);

    // Make it 3D (remove 4th dimension if present)
    destView.setInt16(40, 3, true);  // ndim = 3
    destView.setInt16(48, 1, true);  // dim[4] = 1

    // Copy mask data
    const dataView = new Float32Array(buffer, headerSize);
    dataView.set(maskData);

    return buffer;
  }

  // Determine which pipeline stages can be skipped based on settings changes
  determineSkipStages() {
    if (!this.pipelineHasRun || !this.lastRunSettings) {
      return { skipUnwrap: false, skipBgRemoval: false };
    }

    const current = this.pipelineSettings;
    const last = this.lastRunSettings;

    // Check if unwrapping settings changed
    const unwrapChanged =
      current.unwrapMethod !== last.unwrapMethod ||
      (current.unwrapMethod === 'romeo' &&
       current.romeo.weighting !== last.romeo?.weighting);

    // Check if background removal settings changed
    const bgChanged =
      current.backgroundRemoval !== last.backgroundRemoval ||
      (current.backgroundRemoval === 'vsharp' && (
        current.vsharp.maxRadius !== last.vsharp?.maxRadius ||
        current.vsharp.minRadius !== last.vsharp?.minRadius ||
        current.vsharp.threshold !== last.vsharp?.threshold
      )) ||
      (current.backgroundRemoval === 'smv' &&
        current.smv.radius !== last.smv?.radius);

    // If unwrap changed, can't skip anything
    if (unwrapChanged) {
      return { skipUnwrap: false, skipBgRemoval: false };
    }

    // If only dipole inversion changed, can skip both unwrap and bg removal
    if (!bgChanged) {
      return { skipUnwrap: true, skipBgRemoval: true };
    }

    // If bg removal changed but not unwrap, can skip unwrap only
    return { skipUnwrap: true, skipBgRemoval: false };
  }

  async runRomeoQSM() {
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
      // Initialize worker if needed
      await this.initializeWorker();

      this.updateOutput("Starting ROMEO QSM Pipeline...");

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
      const skipStages = this.determineSkipStages();
      if (skipStages.skipUnwrap) {
        this.updateOutput("Reusing cached unwrapped phase data");
      }
      if (skipStages.skipBgRemoval) {
        this.updateOutput("Reusing cached background-removed data");
      }

      // Show phase image at the start
      await this.visualizePhase();

      // Clear previous results cache and disable all stage buttons
      this.clearResults();
      this.disableAllStageButtons();

      // Send to worker with pipeline settings
      // Include prepared magnitude if available (for MEDI gradient weighting and threshold mask)
      const preparedMagnitude = this.preparedMagnitudeData
        ? Array.from(this.preparedMagnitudeData)
        : null;

      this.worker.postMessage({
        type: 'run',
        data: {
          magnitudeBuffers,
          phaseBuffers,
          echoTimes,
          magField,
          maskThreshold: this.maskThreshold,
          customMaskBuffer,
          preparedMagnitude,
          pipelineSettings: this.pipelineSettings,
          skipStages
        }
      });

      // Enable cancel button and update state
      this.pipelineRunning = true;
      document.getElementById('cancelPipeline').disabled = false;
      document.getElementById('run').disabled = true;

    } catch (error) {
      this.updateOutput(`Error: ${error.message}`);
      this.setProgress(0, 'Failed');
      this.pipelineRunning = false;
      document.getElementById('cancelPipeline').disabled = true;
      this.updateEchoInfo(); // Re-enable run button if valid
      console.error(error);
    }
  }

  cancelPipeline() {
    if (!this.pipelineRunning) return;

    this.updateOutput("Cancelling pipeline...");

    // Terminate the worker to stop all processing
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
      this.workerReady = false;
      this.workerInitializing = false;
    }

    // Reset state
    this.pipelineRunning = false;
    this.setProgress(0, 'Cancelled');
    document.getElementById('cancelPipeline').disabled = true;
    this.updateEchoInfo(); // Re-enable run button if valid

    this.updateOutput("Pipeline cancelled. Worker will be reinitialized on next run.");
  }

  showStageButtons() {
    document.getElementById('stage-buttons').classList.remove('hidden');
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

    // Download button
    const downloadBtn = document.createElement('button');
    downloadBtn.className = 'btn btn-sm download-btn';
    downloadBtn.title = `Download ${displayName}`;
    downloadBtn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
        <polyline points="7 10 12 15 17 10"/>
        <line x1="12" y1="15" x2="12" y2="3"/>
      </svg>
    `;
    downloadBtn.addEventListener('click', () => this.downloadStage(stage));

    stageItem.appendChild(showBtn);
    stageItem.appendChild(downloadBtn);
    container.appendChild(stageItem);
  }

  // Get a user-friendly display name for a stage
  getStageDisplayName(stage, description) {
    // Map of stage IDs to short display names
    const nameMap = {
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
    this.stageOrder = [];
  }

  // Legacy method - now calls addStageButton
  enableStageButtons(stage) {
    this.addStageButton(stage);
  }

  // Legacy method - now calls clearStageButtons
  disableAllStageButtons() {
    this.clearStageButtons();
  }

  // Clear all cached results (called at pipeline start)
  clearResults() {
    this.results = {};
    this.stageOrder = [];
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
   */
  autoDetectThreshold() {
    if (!this.preparedMagnitudeData) {
      this.updateOutput("Please click Prepare first");
      return;
    }

    this.updateOutput("Computing optimal threshold (Otsu)...");

    const data = this.preparedMagnitudeData;
    const numBins = 256;

    // Find min/max
    let minVal = Infinity, maxVal = -Infinity;
    for (let i = 0; i < data.length; i++) {
      if (data[i] < minVal) minVal = data[i];
      if (data[i] > maxVal) maxVal = data[i];
    }

    if (maxVal - minVal < 1e-10) {
      this.updateOutput("Cannot compute threshold: constant image");
      return;
    }

    // Build histogram
    const histogram = new Array(numBins).fill(0);
    const binWidth = (maxVal - minVal) / numBins;

    for (let i = 0; i < data.length; i++) {
      let bin = Math.floor((data[i] - minVal) / binWidth);
      bin = Math.min(bin, numBins - 1);
      histogram[bin]++;
    }

    // Compute Otsu threshold
    const totalPixels = data.length;
    let sumTotal = 0;
    for (let i = 0; i < numBins; i++) {
      sumTotal += i * histogram[i];
    }

    let sumBackground = 0;
    let weightBackground = 0;
    let maxVariance = 0;
    let optimalThresholdBin = 0;

    for (let t = 0; t < numBins; t++) {
      weightBackground += histogram[t];
      if (weightBackground === 0) continue;

      const weightForeground = totalPixels - weightBackground;
      if (weightForeground === 0) break;

      sumBackground += t * histogram[t];

      const meanBackground = sumBackground / weightBackground;
      const meanForeground = (sumTotal - sumBackground) / weightForeground;

      const variance = weightBackground * weightForeground *
        Math.pow(meanBackground - meanForeground, 2);

      if (variance > maxVariance) {
        maxVariance = variance;
        optimalThresholdBin = t;
      }
    }

    // Convert bin to threshold value
    const thresholdValue = minVal + (optimalThresholdBin + 0.5) * binWidth;

    // Convert to percentage of max (what the slider uses)
    const thresholdPercent = Math.round((thresholdValue / maxVal) * 100);
    const clampedPercent = Math.max(1, Math.min(100, thresholdPercent));

    // Update slider and display
    const slider = document.getElementById('maskThreshold');
    if (slider) {
      slider.value = clampedPercent;
      this.maskThreshold = clampedPercent;
      document.getElementById('thresholdLabel').textContent = `Threshold (${clampedPercent}%)`;
    }

    this.updateOutput(`Otsu threshold: ${clampedPercent}% (${thresholdValue.toFixed(1)})`);

    // Only trigger mask preview if threshold slider is enabled (user has clicked Threshold button)
    const thresholdSlider = document.getElementById('maskThreshold');
    if (thresholdSlider && !thresholdSlider.disabled && this.magnitudeData && !this.maskUpdating) {
      this.updateMaskPreview();
    }
  }

  async runBET() {
    if (this.multiEchoFiles.magnitude.length === 0) {
      this.updateOutput("No magnitude files uploaded - please load magnitude data first");
      return;
    }

    // Disable threshold slider since user chose BET-based masking
    this.setThresholdSliderEnabled(false);

    try {
      this.updateOutput("Starting BET brain extraction...");
      this.setProgress(0.05, 'Initializing BET...');

      // Enable pipeline section during BET
      const pipelineSection = document.getElementById('pipelineSection');
      if (pipelineSection) {
        pipelineSection.classList.remove('section-disabled');
      }

      // Initialize worker if needed (must await to ensure WASM is loaded)
      await this.initializeWorker();

      // Load magnitude into NiiVue to get dimensions (handles gzip decompression)
      if (!this.magnitudeVolume || this.nv.volumes.length === 0) {
        const file = this.multiEchoFiles.magnitude[0].file;
        const url = URL.createObjectURL(file);
        await this.nv.loadVolumes([{ url: url, name: file.name }]);
        URL.revokeObjectURL(url);

        if (this.nv.volumes.length > 0) {
          this.magnitudeVolume = this.nv.volumes[0];
          this.magnitudeData = this.magnitudeVolume.img;
          let max = -Infinity;
          for (let i = 0; i < this.magnitudeData.length; i++) {
            if (this.magnitudeData[i] > max) max = this.magnitudeData[i];
          }
          this.magnitudeMax = max;
        }
      }

      // Create header from volume if not already done (handles gzipped files)
      if (!this.magnitudeFileBytes || this.magnitudeFileBytes.byteLength < 348) {
        this.magnitudeFileBytes = this.createNiftiHeaderFromVolume(this.magnitudeVolume);
      }

      // Extract dimensions from NIfTI header
      const srcView = new DataView(this.magnitudeFileBytes);
      const nx = srcView.getInt16(42, true);
      const ny = srcView.getInt16(44, true);
      const nz = srcView.getInt16(46, true);
      this.maskDims = [nx, ny, nz];

      // Get voxel size
      const dx = srcView.getFloat32(80, true);
      const dy = srcView.getFloat32(84, true);
      const dz = srcView.getFloat32(88, true);
      const voxelSize = [dz || 1, dy || 1, dx || 1]; // z, y, x order for Python

      this.updateOutput(`Image dimensions: ${nx}x${ny}x${nz}, voxel size: ${dx.toFixed(2)}x${dy.toFixed(2)}x${dz.toFixed(2)}mm`);

      // Create full NIfTI buffer with header + data for BET
      // Use prepared data if available, otherwise use raw magnitude data
      const magData = this.preparedMagnitudeData || this.magnitudeData;
      if (!magData) {
        throw new Error("No magnitude data available - run Prepare first");
      }

      // Create NIfTI from prepared/magnitude data (not raw volume)
      const voxOffset = srcView.getFloat32(108, true);
      const headerSize = Math.ceil(voxOffset);

      const dataSize = magData.length * 8; // 8 bytes per float64
      const buffer = new ArrayBuffer(headerSize + dataSize);
      const destBytes = new Uint8Array(buffer);
      const destView = new DataView(buffer);

      // Copy header
      destBytes.set(new Uint8Array(this.magnitudeFileBytes).slice(0, headerSize));

      // Update datatype to FLOAT64 (64) at offset 70
      destView.setInt16(70, 64, true);
      // Update bitpix to 64 at offset 72
      destView.setInt16(72, 64, true);

      // Make it 3D
      destView.setInt16(40, 3, true);
      destView.setInt16(48, 1, true);

      // Copy prepared magnitude data
      const dataView = new Float64Array(buffer, headerSize);
      dataView.set(magData);

      const magnitudeNifti = buffer;

      // Set up handler for BET messages
      const betHandler = (e) => {
        const { type, ...data } = e.data;

        switch (type) {
          case 'betProgress':
            this.setProgress(data.value, data.text);
            break;
          case 'betLog':
            this.updateOutput(data.message);
            break;
          case 'betComplete':
            this.worker.removeEventListener('message', betHandler);
            this.handleBETComplete(data);
            break;
          case 'betError':
            this.worker.removeEventListener('message', betHandler);
            this.updateOutput(`BET Error: ${data.message}`);
            this.setProgress(0, 'BET Failed');
            this.updateEchoInfo(); // Reset pipeline section state
            break;
        }
      };
      this.worker.addEventListener('message', betHandler);

      // Send BET request to worker (pure WASM, no Python code needed)
      this.worker.postMessage({
        type: 'runBET',
        data: {
          magnitudeBuffer: magnitudeNifti,
          voxelSize: voxelSize,
          fractionalIntensity: this.betSettings.fractionalIntensity,
          iterations: this.betSettings.iterations,
          subdivisions: this.betSettings.subdivisions
        }
      });

    } catch (error) {
      this.updateOutput(`BET Error: ${error.message}`);
      this.setProgress(0, 'Failed');
      this.updateEchoInfo(); // Reset pipeline section state
      console.error(error);
    }
  }

  async handleBETComplete(data) {
    try {
      this.updateOutput("BET completed, loading mask...");

      // Convert the mask data to Float32Array
      const maskData = new Float32Array(data.maskData);

      // Store as both current and original mask
      this.currentMaskData = maskData;
      this.originalMaskData = new Float32Array(maskData);

      // Display the mask
      await this.displayCurrentMask();

      // Show morphological operations panel
      const opsPanel = document.getElementById('maskOperations');
      if (opsPanel) opsPanel.style.display = 'block';

      // Update run button state (mask is now available)
      this.updateEchoInfo();

      this.setProgress(1.0, 'BET Complete');
      this.updateOutput(`BET brain extraction complete. Coverage: ${data.coverage}`);
    } catch (error) {
      this.updateOutput(`Error displaying BET mask: ${error.message}`);
      console.error(error);
    }
  }

  // Calculate dynamic defaults based on voxel size (matches QSM.jl)
  getVoxelBasedDefaults() {
    const vsz = this.voxelSize || [1, 1, 1];
    const minVsz = Math.min(...vsz);
    const maxVsz = Math.max(...vsz);

    // Calculate mask size for PDF maxit (if available)
    const maskSize = this.maskDims ? this.maskDims[0] * this.maskDims[1] * this.maskDims[2] : 100000;

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

  // Pipeline Settings Modal
  openPipelineSettingsModal() {
    // Get dynamic defaults based on voxel size
    const defaults = this.getVoxelBasedDefaults();

    // Populate form with current settings (or calculated defaults if null)

    // Combined method
    const combinedMethod = this.pipelineSettings.combinedMethod || 'none';
    document.getElementById('combinedMethod').value = combinedMethod;

    // TGV settings
    document.getElementById('tgvRegularization').value = this.pipelineSettings.tgv.regularization;
    document.getElementById('tgvIterations').value = this.pipelineSettings.tgv.iterations;
    document.getElementById('tgvErosions').value = this.pipelineSettings.tgv.erosions;

    // Multi-echo method
    const multiEchoMethod = this.pipelineSettings.multiEchoMethod || 'mcpc3ds';
    document.getElementById('multiEchoMethod').value = multiEchoMethod;

    // Phase unwrap method (for multi-echo OLS methods)
    const unwrapMethod = this.pipelineSettings.unwrapMethod || 'romeo';
    document.getElementById('unwrapMethod').value = unwrapMethod;
    document.getElementById('romeoSettings').style.display = unwrapMethod === 'romeo' ? 'block' : 'none';
    document.getElementById('laplacianSettings').style.display = unwrapMethod === 'laplacian' ? 'block' : 'none';
    document.getElementById('romeoWeighting').value = this.pipelineSettings.romeo.weighting;

    // Single-echo unwrap method (sync with multi-echo settings)
    document.getElementById('singleEchoUnwrapMethod').value = unwrapMethod;
    document.getElementById('singleEchoRomeoSettings').style.display = unwrapMethod === 'romeo' ? 'block' : 'none';
    document.getElementById('singleEchoRomeoWeighting').value = this.pipelineSettings.romeo.weighting;

    // MCPC-3D-S settings
    document.getElementById('mcpc3dsSigmaX').value = this.pipelineSettings.mcpc3ds?.sigma?.[0] ?? 10;
    document.getElementById('mcpc3dsSigmaY').value = this.pipelineSettings.mcpc3ds?.sigma?.[1] ?? 10;
    document.getElementById('mcpc3dsSigmaZ').value = this.pipelineSettings.mcpc3ds?.sigma?.[2] ?? 5;
    document.getElementById('mcpc3dsWeightType').value = this.pipelineSettings.mcpc3ds?.weightType ?? 'phase_snr';

    // Update section visibility based on current selections
    this.updatePipelineModalVisibility();

    // Background removal method
    const bgMethod = this.pipelineSettings.backgroundRemoval;
    document.getElementById('bgRemovalMethod').value = bgMethod;
    document.getElementById('vsharpSettings').style.display = bgMethod === 'vsharp' ? 'block' : 'none';
    document.getElementById('sharpSettings').style.display = bgMethod === 'sharp' ? 'block' : 'none';
    document.getElementById('smvSettings').style.display = bgMethod === 'smv' ? 'block' : 'none';
    document.getElementById('ismvSettings').style.display = bgMethod === 'ismv' ? 'block' : 'none';
    document.getElementById('pdfSettings').style.display = bgMethod === 'pdf' ? 'block' : 'none';
    document.getElementById('lbvSettings').style.display = bgMethod === 'lbv' ? 'block' : 'none';

    // V-SHARP settings (use calculated defaults if null)
    document.getElementById('vsharpMaxRadius').value = this.pipelineSettings.vsharp.maxRadius ?? defaults.vsharpMaxRadius;
    document.getElementById('vsharpMinRadius').value = this.pipelineSettings.vsharp.minRadius ?? defaults.vsharpMinRadius;
    document.getElementById('vsharpThreshold').value = this.pipelineSettings.vsharp.threshold;

    // SMV settings
    document.getElementById('smvRadius').value = this.pipelineSettings.smv.radius ?? defaults.smvRadius;

    // iSMV settings
    document.getElementById('ismvRadius').value = this.pipelineSettings.ismv.radius ?? defaults.ismvRadius;
    document.getElementById('ismvTol').value = this.pipelineSettings.ismv.tol;
    document.getElementById('ismvMaxit').value = this.pipelineSettings.ismv.maxit;

    // PDF settings
    document.getElementById('pdfTol').value = this.pipelineSettings.pdf.tol;
    document.getElementById('pdfMaxit').value = this.pipelineSettings.pdf.maxit ?? defaults.pdfMaxit;

    // LBV settings
    document.getElementById('lbvTol').value = this.pipelineSettings.lbv.tol;
    document.getElementById('lbvMaxit').value = this.pipelineSettings.lbv.maxit;

    // Dipole inversion method
    const dipoleMethod = this.pipelineSettings.dipoleInversion;
    document.getElementById('dipoleMethod').value = dipoleMethod;
    document.getElementById('tkdSettings').style.display = dipoleMethod === 'tkd' ? 'block' : 'none';
    document.getElementById('tsvdSettings').style.display = dipoleMethod === 'tsvd' ? 'block' : 'none';
    document.getElementById('tikhonovSettings').style.display = dipoleMethod === 'tikhonov' ? 'block' : 'none';
    document.getElementById('tvSettings').style.display = dipoleMethod === 'tv' ? 'block' : 'none';
    document.getElementById('rtsSettings').style.display = dipoleMethod === 'rts' ? 'block' : 'none';
    document.getElementById('nltvSettings').style.display = dipoleMethod === 'nltv' ? 'block' : 'none';
    document.getElementById('mediSettings').style.display = dipoleMethod === 'medi' ? 'block' : 'none';
    document.getElementById('ilsqrSettings').style.display = dipoleMethod === 'ilsqr' ? 'block' : 'none';

    // TKD settings
    document.getElementById('tkdThreshold').value = this.pipelineSettings.tkd.threshold;

    // TSVD settings
    document.getElementById('tsvdThreshold').value = this.pipelineSettings.tsvd.threshold;

    // Tikhonov settings
    document.getElementById('tikhLambda').value = this.pipelineSettings.tikhonov.lambda;
    document.getElementById('tikhReg').value = this.pipelineSettings.tikhonov.reg;

    // TV-ADMM settings
    document.getElementById('tvLambda').value = this.pipelineSettings.tv.lambda;
    document.getElementById('tvMaxIter').value = this.pipelineSettings.tv.maxIter;
    document.getElementById('tvTol').value = this.pipelineSettings.tv.tol;

    // RTS settings
    document.getElementById('rtsDelta').value = this.pipelineSettings.rts.delta;
    document.getElementById('rtsMu').value = this.pipelineSettings.rts.mu;
    document.getElementById('rtsRho').value = this.pipelineSettings.rts.rho;
    document.getElementById('rtsMaxIter').value = this.pipelineSettings.rts.maxIter;

    // NLTV settings
    document.getElementById('nltvLambda').value = this.pipelineSettings.nltv.lambda;
    document.getElementById('nltvMu').value = this.pipelineSettings.nltv.mu;
    document.getElementById('nltvMaxIter').value = this.pipelineSettings.nltv.maxIter;
    document.getElementById('nltvTol').value = this.pipelineSettings.nltv.tol;
    document.getElementById('nltvNewtonMaxIter').value = this.pipelineSettings.nltv.newtonMaxIter;

    // MEDI settings
    document.getElementById('mediLambda').value = this.pipelineSettings.medi.lambda;
    document.getElementById('mediPercentage').value = this.pipelineSettings.medi.percentage;
    document.getElementById('mediMaxIter').value = this.pipelineSettings.medi.maxIter;
    document.getElementById('mediCgMaxIter').value = this.pipelineSettings.medi.cgMaxIter;
    document.getElementById('mediSmv').checked = this.pipelineSettings.medi.smv;
    document.getElementById('mediSmvRadius').value = this.pipelineSettings.medi.smvRadius;
    document.getElementById('mediSmvRadiusGroup').style.display = this.pipelineSettings.medi.smv ? 'block' : 'none';
    document.getElementById('mediMerit').checked = this.pipelineSettings.medi.merit;

    // iLSQR settings
    document.getElementById('ilsqrTol').value = this.pipelineSettings.ilsqr?.tol || 0.01;
    document.getElementById('ilsqrMaxIter').value = this.pipelineSettings.ilsqr?.maxIter || 50;

    document.getElementById('pipelineSettingsModal').classList.add('active');
  }

  closePipelineSettingsModal() {
    document.getElementById('pipelineSettingsModal').classList.remove('active');
  }

  /**
   * Update visibility of pipeline modal sections based on:
   * - Number of echoes (single vs multi)
   * - QSM method (Standard vs TGV)
   * - Multi-echo method (OLS/OLS+offset vs MCPC-3D-S)
   */
  updatePipelineModalVisibility() {
    const combinedMethod = document.getElementById('combinedMethod').value;
    const multiEchoMethod = document.getElementById('multiEchoMethod').value;
    const isTgv = combinedMethod === 'tgv';
    const isQsmart = combinedMethod === 'qsmart';
    const isCombined = isTgv || isQsmart;  // Methods that handle BG removal + inversion together
    const isMcpc3ds = multiEchoMethod === 'mcpc3ds';

    // Count echoes from loaded files
    const nEchoes = this.multiEchoFiles?.phase?.filter(f => f.file)?.length || 0;
    const isMultiEcho = nEchoes > 1;

    // TGV settings - show only when TGV selected
    document.getElementById('tgvSettings').style.display = isTgv ? 'block' : 'none';

    // QSMART settings - show only when QSMART selected
    const qsmartSettings = document.getElementById('qsmartSettings');
    if (qsmartSettings) qsmartSettings.style.display = isQsmart ? 'block' : 'none';

    // Multi-echo section - show only when multi-echo data loaded AND standard pipeline
    // (TGV and QSMART have their own multi-echo handling)
    document.getElementById('multiEchoSection').style.display = (isMultiEcho && !isCombined) ? 'block' : 'none';

    // Within multi-echo section:
    // - Unwrap settings: show for OLS methods, hide for MCPC-3D-S
    // - MCPC-3D-S settings: show for MCPC-3D-S, hide for OLS methods
    const multiEchoUnwrapSettings = document.getElementById('multiEchoUnwrapSettings');
    const mcpc3dsSettings = document.getElementById('mcpc3dsSettings');

    if (multiEchoUnwrapSettings) multiEchoUnwrapSettings.style.display = !isMcpc3ds ? 'block' : 'none';
    if (mcpc3dsSettings) mcpc3dsSettings.style.display = isMcpc3ds ? 'block' : 'none';

    // Single-echo unwrap section - show only for single-echo + standard pipeline
    const singleEchoUnwrapSection = document.getElementById('singleEchoUnwrapSection');
    if (singleEchoUnwrapSection) {
      singleEchoUnwrapSection.style.display = (!isMultiEcho && !isCombined) ? 'block' : 'none';
    }

    // Background removal and dipole inversion - show only for standard pipeline
    // (TGV and QSMART handle these internally)
    document.getElementById('bgRemovalSection').style.display = isCombined ? 'none' : 'block';
    document.getElementById('dipoleInversionSection').style.display = isCombined ? 'none' : 'block';
  }

  resetPipelineSettings() {
    // Reset to defaults (using dynamic voxel-based values where applicable)
    const defaults = this.getVoxelBasedDefaults();

    // Combined method - reset to standard pipeline
    document.getElementById('combinedMethod').value = 'none';

    // TGV defaults
    document.getElementById('tgvRegularization').value = 2;
    document.getElementById('tgvIterations').value = 1000;
    document.getElementById('tgvErosions').value = 3;

    // QSMART defaults
    document.getElementById('qsmartSdfSigma1Stage1').value = 10;
    document.getElementById('qsmartSdfSigma2Stage1').value = 0;
    document.getElementById('qsmartSdfSigma1Stage2').value = 8;
    document.getElementById('qsmartSdfSigma2Stage2').value = 2;
    document.getElementById('qsmartSdfSpatialRadius').value = 8;
    document.getElementById('qsmartSdfLowerLim').value = 0.6;
    document.getElementById('qsmartSdfCurvConstant').value = 500;
    document.getElementById('qsmartVascSphereRadius').value = 8;
    document.getElementById('qsmartFrangiScaleMin').value = 1.0;
    document.getElementById('qsmartFrangiScaleMax').value = 10.0;
    document.getElementById('qsmartFrangiScaleRatio').value = 2.0;
    document.getElementById('qsmartFrangiC').value = 500;
    document.getElementById('qsmartIlsqrTol').value = 0.01;
    document.getElementById('qsmartIlsqrMaxIter').value = 50;

    // Multi-echo method - default to OLS with offset
    document.getElementById('multiEchoMethod').value = 'mcpc3ds';

    // Unwrap method (for both multi-echo and single-echo)
    document.getElementById('unwrapMethod').value = 'romeo';
    document.getElementById('romeoSettings').style.display = 'block';
    document.getElementById('laplacianSettings').style.display = 'none';
    document.getElementById('romeoWeighting').value = 'phase_snr';

    // Single-echo unwrap (sync with multi-echo)
    document.getElementById('singleEchoUnwrapMethod').value = 'romeo';
    document.getElementById('singleEchoRomeoSettings').style.display = 'block';
    document.getElementById('singleEchoRomeoWeighting').value = 'phase_snr';

    // MCPC-3D-S settings
    document.getElementById('mcpc3dsSigmaX').value = 10;
    document.getElementById('mcpc3dsSigmaY').value = 10;
    document.getElementById('mcpc3dsSigmaZ').value = 5;
    document.getElementById('mcpc3dsWeightType').value = 'phase_snr';

    // Update section visibility
    this.updatePipelineModalVisibility();

    // Background removal - default to V-SHARP
    document.getElementById('bgRemovalMethod').value = 'vsharp';
    document.getElementById('vsharpSettings').style.display = 'block';
    document.getElementById('sharpSettings').style.display = 'none';
    document.getElementById('smvSettings').style.display = 'none';
    document.getElementById('ismvSettings').style.display = 'none';
    document.getElementById('pdfSettings').style.display = 'none';
    document.getElementById('vsharpMaxRadius').value = defaults.vsharpMaxRadius;
    document.getElementById('vsharpMinRadius').value = defaults.vsharpMinRadius;
    document.getElementById('vsharpThreshold').value = 0.05;
    document.getElementById('sharpRadius').value = 6;
    document.getElementById('sharpThreshold').value = 0.05;
    document.getElementById('smvRadius').value = defaults.smvRadius;

    // iSMV defaults
    document.getElementById('ismvRadius').value = defaults.ismvRadius;
    document.getElementById('ismvTol').value = 0.001;
    document.getElementById('ismvMaxit').value = 500;

    // PDF defaults
    document.getElementById('pdfTol').value = 0.00001;
    document.getElementById('pdfMaxit').value = defaults.pdfMaxit;

    // LBV defaults
    document.getElementById('lbvSettings').style.display = 'none';
    document.getElementById('lbvTol').value = 0.001;
    document.getElementById('lbvMaxit').value = 500;

    // Dipole inversion - default to RTS
    document.getElementById('dipoleMethod').value = 'rts';
    document.getElementById('tkdSettings').style.display = 'none';
    document.getElementById('tsvdSettings').style.display = 'none';
    document.getElementById('tikhonovSettings').style.display = 'none';
    document.getElementById('tvSettings').style.display = 'none';
    document.getElementById('rtsSettings').style.display = 'block';
    document.getElementById('nltvSettings').style.display = 'none';
    document.getElementById('mediSettings').style.display = 'none';
    document.getElementById('ilsqrSettings').style.display = 'none';

    // TKD
    document.getElementById('tkdThreshold').value = 0.15;

    // TSVD
    document.getElementById('tsvdThreshold').value = 0.15;

    // Tikhonov
    document.getElementById('tikhLambda').value = 0.01;
    document.getElementById('tikhReg').value = 'identity';

    // TV-ADMM
    document.getElementById('tvLambda').value = 0.001;
    document.getElementById('tvMaxIter').value = 250;
    document.getElementById('tvTol').value = 0.001;

    // RTS
    document.getElementById('rtsDelta').value = 0.15;
    document.getElementById('rtsMu').value = 100000;
    document.getElementById('rtsRho').value = 10;
    document.getElementById('rtsMaxIter').value = 20;

    // NLTV
    document.getElementById('nltvLambda').value = 0.001;
    document.getElementById('nltvMu').value = 1;
    document.getElementById('nltvMaxIter').value = 250;
    document.getElementById('nltvTol').value = 0.001;
    document.getElementById('nltvNewtonMaxIter').value = 10;

    // MEDI
    document.getElementById('mediLambda').value = 1000;
    document.getElementById('mediPercentage').value = 0.9;
    document.getElementById('mediMaxIter').value = 10;
    document.getElementById('mediCgMaxIter').value = 100;
    document.getElementById('mediSmv').checked = false;
    document.getElementById('mediSmvRadius').value = 5;
    document.getElementById('mediSmvRadiusGroup').style.display = 'none';
    document.getElementById('mediMerit').checked = false;

    // iLSQR
    document.getElementById('ilsqrTol').value = 0.01;
    document.getElementById('ilsqrMaxIter').value = 50;
  }

  runPipelineWithSettings() {
    // Determine if multi-echo based on loaded files
    const nEchoes = this.multiEchoFiles?.phase?.filter(f => f.file)?.length || 0;
    const isMultiEcho = nEchoes > 1;

    // Get unwrap method from appropriate dropdown based on echo count
    const unwrapMethod = isMultiEcho
      ? document.getElementById('unwrapMethod').value
      : document.getElementById('singleEchoUnwrapMethod').value;

    // Get ROMEO weighting from appropriate dropdown
    const romeoWeighting = isMultiEcho
      ? document.getElementById('romeoWeighting').value
      : document.getElementById('singleEchoRomeoWeighting').value;

    // Save settings from form
    this.pipelineSettings = {
      combinedMethod: document.getElementById('combinedMethod').value,
      tgv: {
        regularization: parseInt(document.getElementById('tgvRegularization').value),
        iterations: parseInt(document.getElementById('tgvIterations').value),
        erosions: parseInt(document.getElementById('tgvErosions').value)
      },
      qsmart: {
        sdfSigma1Stage1: parseFloat(document.getElementById('qsmartSdfSigma1Stage1').value),
        sdfSigma2Stage1: parseFloat(document.getElementById('qsmartSdfSigma2Stage1').value),
        sdfSigma1Stage2: parseFloat(document.getElementById('qsmartSdfSigma1Stage2').value),
        sdfSigma2Stage2: parseFloat(document.getElementById('qsmartSdfSigma2Stage2').value),
        sdfSpatialRadius: parseInt(document.getElementById('qsmartSdfSpatialRadius').value),
        sdfLowerLim: parseFloat(document.getElementById('qsmartSdfLowerLim').value),
        sdfCurvConstant: parseFloat(document.getElementById('qsmartSdfCurvConstant').value),
        // Vasculature parameters in mm (auto-scaled to voxels based on image resolution)
        vascSphereRadiusMm: parseFloat(document.getElementById('qsmartVascSphereRadius').value),
        frangiScaleMinMm: parseFloat(document.getElementById('qsmartFrangiScaleMin').value),
        frangiScaleMaxMm: parseFloat(document.getElementById('qsmartFrangiScaleMax').value),
        frangiScaleRatioMm: parseFloat(document.getElementById('qsmartFrangiScaleRatio').value),
        frangiC: parseFloat(document.getElementById('qsmartFrangiC').value),
        ilsqrTol: parseFloat(document.getElementById('qsmartIlsqrTol').value),
        ilsqrMaxIter: parseInt(document.getElementById('qsmartIlsqrMaxIter').value)
      },
      unwrapMethod: unwrapMethod,
      multiEchoMethod: document.getElementById('multiEchoMethod').value,
      mcpc3ds: {
        sigma: [
          parseInt(document.getElementById('mcpc3dsSigmaX').value),
          parseInt(document.getElementById('mcpc3dsSigmaY').value),
          parseInt(document.getElementById('mcpc3dsSigmaZ').value)
        ],
        weightType: document.getElementById('mcpc3dsWeightType').value
      },
      romeo: {
        weighting: romeoWeighting
      },
      backgroundRemoval: document.getElementById('bgRemovalMethod').value,
      vsharp: {
        maxRadius: parseFloat(document.getElementById('vsharpMaxRadius').value),
        minRadius: parseFloat(document.getElementById('vsharpMinRadius').value),
        threshold: parseFloat(document.getElementById('vsharpThreshold').value)
      },
      sharp: {
        radius: parseFloat(document.getElementById('sharpRadius').value),
        threshold: parseFloat(document.getElementById('sharpThreshold').value)
      },
      smv: {
        radius: parseFloat(document.getElementById('smvRadius').value)
      },
      ismv: {
        radius: parseFloat(document.getElementById('ismvRadius').value),
        tol: parseFloat(document.getElementById('ismvTol').value),
        maxit: parseInt(document.getElementById('ismvMaxit').value)
      },
      pdf: {
        tol: parseFloat(document.getElementById('pdfTol').value),
        maxit: parseInt(document.getElementById('pdfMaxit').value)
      },
      lbv: {
        tol: parseFloat(document.getElementById('lbvTol').value),
        maxit: parseInt(document.getElementById('lbvMaxit').value)
      },
      dipoleInversion: document.getElementById('dipoleMethod').value,
      tkd: {
        threshold: parseFloat(document.getElementById('tkdThreshold').value)
      },
      tsvd: {
        threshold: parseFloat(document.getElementById('tsvdThreshold').value)
      },
      tikhonov: {
        lambda: parseFloat(document.getElementById('tikhLambda').value),
        reg: document.getElementById('tikhReg').value
      },
      tv: {
        lambda: parseFloat(document.getElementById('tvLambda').value),
        maxIter: parseInt(document.getElementById('tvMaxIter').value),
        tol: parseFloat(document.getElementById('tvTol').value)
      },
      rts: {
        delta: parseFloat(document.getElementById('rtsDelta').value),
        mu: parseFloat(document.getElementById('rtsMu').value),
        rho: parseFloat(document.getElementById('rtsRho').value),
        maxIter: parseInt(document.getElementById('rtsMaxIter').value)
      },
      nltv: {
        lambda: parseFloat(document.getElementById('nltvLambda').value),
        mu: parseFloat(document.getElementById('nltvMu').value),
        maxIter: parseInt(document.getElementById('nltvMaxIter').value),
        tol: parseFloat(document.getElementById('nltvTol').value),
        newtonMaxIter: parseInt(document.getElementById('nltvNewtonMaxIter').value)
      },
      medi: {
        lambda: parseFloat(document.getElementById('mediLambda').value),
        percentage: parseFloat(document.getElementById('mediPercentage').value),
        maxIter: parseInt(document.getElementById('mediMaxIter').value),
        cgMaxIter: parseInt(document.getElementById('mediCgMaxIter').value),
        cgTol: 0.01,
        tol: 0.1,
        smv: document.getElementById('mediSmv').checked,
        smvRadius: parseFloat(document.getElementById('mediSmvRadius').value),
        merit: document.getElementById('mediMerit').checked,
        dataWeighting: 1
      },
      ilsqr: {
        tol: parseFloat(document.getElementById('ilsqrTol').value),
        maxIter: parseInt(document.getElementById('ilsqrMaxIter').value)
      }
    };

    this.closePipelineSettingsModal();
    this.runRomeoQSM();
  }

  // BET Settings Modal
  openBetSettingsModal() {
    if (this.multiEchoFiles.magnitude.length === 0) {
      this.updateOutput("No magnitude files uploaded - please load magnitude data first");
      return;
    }

    // Populate form with current settings
    document.getElementById('betFractionalIntensity').value = this.betSettings.fractionalIntensity;
    document.getElementById('betFractionalIntensityValue').textContent = this.betSettings.fractionalIntensity;
    document.getElementById('betIterations').value = this.betSettings.iterations;
    document.getElementById('betSubdivisions').value = this.betSettings.subdivisions;

    document.getElementById('betSettingsModal').classList.add('active');
  }

  closeBetSettingsModal() {
    document.getElementById('betSettingsModal').classList.remove('active');
  }

  resetBetSettings() {
    // Reset to defaults
    document.getElementById('betFractionalIntensity').value = 0.5;
    document.getElementById('betFractionalIntensityValue').textContent = '0.5';
    document.getElementById('betIterations').value = 1000;
    document.getElementById('betSubdivisions').value = 4;
  }

  runBetWithSettings() {
    // Save settings from form
    this.betSettings = {
      fractionalIntensity: parseFloat(document.getElementById('betFractionalIntensity').value),
      iterations: parseInt(document.getElementById('betIterations').value),
      subdivisions: parseInt(document.getElementById('betSubdivisions').value)
    };

    this.closeBetSettingsModal();
    this.runBET();
  }
}

// Initialize the app - this will be called after NiiVue is loaded by the module script
function initQSMApp() {
  console.log('Initializing QSM App with NiiVue:', window.Niivue);
  window.app = new QSMApp();
}

// If the script loads after DOM is ready, initialize immediately
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    if (window.Niivue) {
      initQSMApp();
    } else {
      console.log('Waiting for NiiVue...');
      setTimeout(() => {
        if (window.Niivue) {
          initQSMApp();
        } else {
          document.getElementById("output").textContent = "Error: NiiVue library failed to load. Please refresh the page.";
        }
      }, 2000);
    }
  });
} else {
  // DOM already loaded
  if (window.Niivue) {
    initQSMApp();
  } else {
    setTimeout(() => {
      if (window.Niivue) {
        initQSMApp();
      } else {
        document.getElementById("output").textContent = "Error: NiiVue library failed to load. Please refresh the page.";
      }
    }, 1000);
  }
}