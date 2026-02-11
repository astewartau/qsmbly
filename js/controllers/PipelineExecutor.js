/**
 * PipelineExecutor
 *
 * Handles worker lifecycle, pipeline execution, and result caching.
 */

export class PipelineExecutor {
  constructor(options) {
    // Callbacks
    this.updateOutput = options.updateOutput || (() => {});
    this.setProgress = options.setProgress || (() => {});
    this.onStageData = options.onStageData || (() => {});
    this.onPipelineComplete = options.onPipelineComplete || (() => {});
    this.onPipelineError = options.onPipelineError || (() => {});
    this.onInitialized = options.onInitialized || (() => {});
    this.config = options.config;

    // Worker state
    this.worker = null;
    this.workerReady = false;
    this.workerInitializing = false;

    // Pipeline state
    this.pipelineRunning = false;
    this.results = {};
    this.stageOrder = [];
    this.pendingStageResolve = null;

    // Settings tracking for intelligent caching
    this.lastRunSettings = null;
    this.pipelineHasRun = false;
  }

  // ==================== State Accessors ====================

  isReady() {
    return this.workerReady;
  }

  isRunning() {
    return this.pipelineRunning;
  }

  hasResult(stage) {
    return !!this.results[stage]?.file;
  }

  getResult(stage) {
    return this.results[stage] || null;
  }

  getResults() {
    return this.results;
  }

  getStageOrder() {
    return this.stageOrder;
  }

  getLastRunSettings() {
    return this.lastRunSettings;
  }

  hasPipelineRun() {
    return this.pipelineHasRun;
  }

  // ==================== Worker Management ====================

  _setupWorker() {
    if (this.worker) return;

    this.worker = new Worker('js/qsm-worker-pure.js', { type: 'module' });

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
          this._handleError(data.message);
          break;

        case 'initialized':
          this.workerReady = true;
          this.updateOutput("WASM ready");
          this.onInitialized();
          break;

        case 'complete':
          this._handleComplete();
          break;

        case 'stageData':
          this._handleStageData(data);
          break;
      }
    };

    this.worker.onerror = (e) => {
      this.updateOutput(`Worker error: ${e.message}`);
      console.error('Worker error:', e);
      this._handleError(e.message);
    };
  }

  _handleError(message) {
    this.updateOutput(`Error: ${message}`);
    this.setProgress(0, 'Failed');
    this.pipelineRunning = false;
    this.onPipelineError(message);
  }

  _handleComplete() {
    this.updateOutput("Pipeline completed successfully!");
    this.pipelineHasRun = true;
    this.pipelineRunning = false;
    this.onPipelineComplete();
  }

  _handleStageData(data) {
    // Handle both live stage updates and explicit requests
    if (this.pendingStageResolve) {
      this.pendingStageResolve(data);
      this.pendingStageResolve = null;
    } else if (this.pipelineRunning) {
      // Track stage order
      if (!this.stageOrder.includes(data.stage)) {
        this.stageOrder.push(data.stage);
      }

      // Cache result
      const blob = new Blob([data.data], { type: 'application/octet-stream' });
      const file = new File([blob], `${data.stage}.nii`, { type: 'application/octet-stream' });
      this.results[data.stage] = {
        file: file,
        path: `${data.stage}.nii`,
        description: data.description
      };

      // Notify callback
      this.onStageData(data);
    }
  }

  async initialize() {
    this._setupWorker();

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

    // Send init message to worker
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

  // ==================== Pipeline Execution ====================

  async run(pipelineConfig) {
    const {
      magnitudeBuffers,
      phaseBuffers,
      echoTimes,
      magField,
      maskThreshold,
      customMaskBuffer,
      preparedMagnitude,
      pipelineSettings,
      skipStages
    } = pipelineConfig;

    try {
      await this.initialize();

      this.updateOutput("Starting ROMEO QSM Pipeline...");
      this.pipelineRunning = true;

      // Save settings for intelligent caching
      this.lastRunSettings = JSON.parse(JSON.stringify(pipelineSettings));

      this.worker.postMessage({
        type: 'run',
        data: {
          magnitudeBuffers,
          phaseBuffers,
          echoTimes,
          magField,
          maskThreshold,
          customMaskBuffer,
          preparedMagnitude,
          pipelineSettings,
          skipStages
        }
      });

      return true;
    } catch (error) {
      this._handleError(error.message);
      console.error(error);
      return false;
    }
  }

  cancel() {
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

    this.updateOutput("Pipeline cancelled. Worker will be reinitialized on next run.");
  }

  // ==================== Result Management ====================

  clearResults() {
    this.results = {};
    this.stageOrder = [];
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
    a.download = file.name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  // ==================== Skip Stage Logic ====================

  determineSkipStages(currentSettings) {
    // First run - can't skip anything
    if (!this.pipelineHasRun || !this.lastRunSettings) {
      return { skipUnwrap: false, skipBgRemoval: false };
    }

    // If key results are missing, can't skip
    if (!this.results['unwrapped'] || !this.results['localField']) {
      return { skipUnwrap: false, skipBgRemoval: false };
    }

    const last = this.lastRunSettings;
    const current = currentSettings;

    // Check if unwrapping settings changed
    const unwrapChanged =
      current.phaseUnwrapping !== last.phaseUnwrapping ||
      (current.phaseUnwrapping === 'romeo' && current.romeo?.useQualityMap !== last.romeo?.useQualityMap);

    // Check if background removal settings changed
    const bgChanged =
      current.backgroundRemoval !== last.backgroundRemoval ||
      (current.backgroundRemoval === 'vsharp' &&
        (current.vsharp?.minRadius !== last.vsharp?.minRadius ||
          current.vsharp?.maxRadius !== last.vsharp?.maxRadius)) ||
      (current.backgroundRemoval === 'pdf' &&
        (current.pdf?.tolerance !== last.pdf?.tolerance ||
          current.pdf?.iterations !== last.pdf?.iterations)) ||
      (current.backgroundRemoval === 'sharp' &&
        current.sharp?.radius !== last.sharp?.radius) ||
      (current.backgroundRemoval === 'lbv' &&
        current.lbv?.tolerance !== last.lbv?.tolerance) ||
      (current.backgroundRemoval === 'ismv' &&
        (current.ismv?.tolerance !== last.ismv?.tolerance ||
          current.ismv?.iterations !== last.ismv?.iterations)) ||
      (current.backgroundRemoval === 'smv' &&
        current.smv?.radius !== last.smv?.radius);

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

  // ==================== Worker Access (for mask controller) ====================

  getWorker() {
    return this.worker;
  }
}
