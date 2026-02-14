/**
 * DicomController - Handles DICOM to NIfTI conversion and classification.
 * Uses vendored @niivue/dcm2niix (WASM) for in-browser conversion.
 */

export class DicomController {
  constructor(options = {}) {
    this.onConversionComplete = options.onConversionComplete || (() => {});
    this.onFilesRetained = options.onFilesRetained || (() => {});
    this.updateOutput = options.updateOutput || console.log;

    this.dcm2niixModule = null; // Lazy-loaded module
    this.converting = false;

    // Accumulated classified results across batches
    this.classifiedFiles = {
      magnitude: [],  // [{file, name, echoTime, echoNumber}, ...]
      phase: [],
      jsonFiles: [],  // Raw JSON File objects
      fieldStrength: null,
      echoTimes: []   // Sorted ms values
    };
  }

  /**
   * Lazy-initialize the dcm2niix WASM module.
   * Creates a fresh Dcm2niix instance each time (worker FS doesn't reset).
   */
  async _createInstance() {
    if (!this.dcm2niixModule) {
      this.dcm2niixModule = await import('../../dcm2niix/index.js');
    }
    const dcm2niix = new this.dcm2niixModule.Dcm2niix();
    await dcm2niix.init();
    return dcm2niix;
  }

  /**
   * Convert DICOM files from a file input (webkitdirectory).
   * Files already have webkitRelativePath set by the browser.
   */
  async convertFiles(files) {
    if (!files || files.length === 0) return;

    this.converting = true;
    this.updateOutput(`Converting ${files.length} DICOM files...`);

    // Retain original files for dicompare validation
    this.onFilesRetained(files);

    try {
      const dcm2niix = await this._createInstance();
      const result = await dcm2niix.input(files).run();
      this._processResults(result);
    } catch (error) {
      console.error('DICOM conversion error:', error);
      this.updateOutput(`DICOM conversion failed: ${error.message}`);
    } finally {
      this.converting = false;
    }
  }

  /**
   * Convert DICOM files from a drag-and-drop event.
   * Traverses directory tree and sets _webkitRelativePath on each file.
   */
  async convertDropItems(dataTransferItems) {
    if (!dataTransferItems || dataTransferItems.length === 0) return;

    this.converting = true;
    this.updateOutput('Reading dropped files...');

    try {
      const files = [];
      for (let i = 0; i < dataTransferItems.length; i++) {
        const entry = dataTransferItems[i].webkitGetAsEntry?.();
        if (entry) {
          await this._traverseFileTree(entry, '', files);
        }
      }

      if (files.length === 0) {
        this.updateOutput('No DICOM files found in dropped items.');
        this.converting = false;
        return;
      }

      // Retain original files for dicompare validation
      this.onFilesRetained(files);

      this.updateOutput(`Converting ${files.length} DICOM files...`);
      const dcm2niix = await this._createInstance();
      const result = await dcm2niix.input(files).run();
      this._processResults(result);
    } catch (error) {
      console.error('DICOM conversion error:', error);
      this.updateOutput(`DICOM conversion failed: ${error.message}`);
    } finally {
      this.converting = false;
    }
  }

  /**
   * Recursively traverse a dropped directory tree.
   * Sets _webkitRelativePath on each File for dcm2niix.
   */
  _traverseFileTree(item, path, fileArray) {
    return new Promise((resolve) => {
      if (item.isFile) {
        item.file(file => {
          file._webkitRelativePath = path + file.name;
          fileArray.push(file);
          resolve();
        });
      } else if (item.isDirectory) {
        const dirReader = item.createReader();
        const readAllEntries = () => {
          dirReader.readEntries(entries => {
            if (entries.length > 0) {
              const promises = entries.map(entry =>
                this._traverseFileTree(entry, path + item.name + '/', fileArray)
              );
              Promise.all(promises).then(readAllEntries);
            } else {
              resolve();
            }
          });
        };
        readAllEntries();
      } else {
        resolve();
      }
    });
  }

  /**
   * Process dcm2niix output: separate NIfTI and JSON files, then classify.
   */
  async _processResults(resultFiles) {
    const niftiFiles = resultFiles.filter(f =>
      f.name.endsWith('.nii') || f.name.endsWith('.nii.gz')
    );
    const jsonFiles = resultFiles.filter(f => f.name.endsWith('.json'));

    if (niftiFiles.length === 0) {
      this.updateOutput('No NIfTI files produced. Are these valid DICOM files?');
      return;
    }

    await this._classifyFiles(niftiFiles, jsonFiles);
    this.onConversionComplete(this.classifiedFiles);
  }

  /**
   * Classify NIfTI files as magnitude or phase using JSON sidecar metadata.
   *
   * Strategy:
   * 1. Primary: Check ImageType array in JSON sidecar for "P"/"PHASE" (phase) or absence (magnitude)
   * 2. Fallback: Check filename for "_ph" suffix (dcm2niix convention)
   * 3. Default: Assume magnitude
   */
  async _classifyFiles(niftiFiles, jsonFiles) {
    // Parse all JSON sidecars first
    const jsonMap = new Map();
    for (const jsonFile of jsonFiles) {
      try {
        const text = await jsonFile.text();
        const json = JSON.parse(text);
        jsonMap.set(jsonFile.name, { file: jsonFile, data: json });
      } catch (error) {
        console.error(`Error parsing JSON sidecar ${jsonFile.name}:`, error);
      }
    }

    const newMagnitude = [];
    const newPhase = [];
    const newJsonFiles = [];
    let fieldStrength = this.classifiedFiles.fieldStrength;

    for (const niftiFile of niftiFiles) {
      // Find matching JSON sidecar by basename
      const baseName = niftiFile.name.replace(/\.nii(\.gz)?$/, '');
      const jsonEntry = jsonMap.get(baseName + '.json');

      let isPhase = false;
      let echoTime = null;
      let echoNumber = null;

      if (jsonEntry) {
        const json = jsonEntry.data;

        // Classify by ImageType
        const imageType = json.ImageType;
        if (Array.isArray(imageType)) {
          isPhase = imageType.some(t => t === 'P' || t === 'PHASE');
        }

        // Extract echo info
        if (json.EchoTime != null) {
          echoTime = json.EchoTime * 1000; // seconds → ms
        }
        if (json.EchoNumber != null) {
          echoNumber = json.EchoNumber;
        }

        // Extract field strength (only need it once)
        if (fieldStrength == null) {
          fieldStrength = json.MagneticFieldStrength
            || json.FieldStrength
            || json.field_strength
            || null;
        }

        newJsonFiles.push(jsonEntry.file);
      } else {
        // No JSON sidecar — fallback to filename convention
        isPhase = niftiFile.name.includes('_ph');
      }

      const entry = {
        file: niftiFile,
        name: niftiFile.name,
        echoTime,
        echoNumber
      };

      if (isPhase) {
        newPhase.push(entry);
      } else {
        newMagnitude.push(entry);
      }
    }

    // Sort by echo time (or echo number as tiebreaker)
    const sortByEcho = (a, b) => {
      if (a.echoTime != null && b.echoTime != null) {
        return a.echoTime - b.echoTime;
      }
      if (a.echoNumber != null && b.echoNumber != null) {
        return a.echoNumber - b.echoNumber;
      }
      return 0;
    };

    // Accumulate with previous batches
    this.classifiedFiles.magnitude.push(...newMagnitude);
    this.classifiedFiles.phase.push(...newPhase);
    this.classifiedFiles.jsonFiles.push(...newJsonFiles);
    this.classifiedFiles.fieldStrength = fieldStrength;

    // Re-sort accumulated results
    this.classifiedFiles.magnitude.sort(sortByEcho);
    this.classifiedFiles.phase.sort(sortByEcho);

    // Collect unique echo times from both groups
    const allEntries = [...this.classifiedFiles.magnitude, ...this.classifiedFiles.phase];
    const echoTimeSet = new Set();
    for (const entry of allEntries) {
      if (entry.echoTime != null) {
        echoTimeSet.add(entry.echoTime);
      }
    }
    this.classifiedFiles.echoTimes = [...echoTimeSet].sort((a, b) => a - b);

    const magCount = this.classifiedFiles.magnitude.length;
    const phaseCount = this.classifiedFiles.phase.length;
    this.updateOutput(
      `Found ${magCount} magnitude and ${phaseCount} phase image${magCount + phaseCount !== 1 ? 's' : ''}`
    );
  }

  /**
   * Get the current classified files.
   */
  getClassifiedFiles() {
    return this.classifiedFiles;
  }

  /**
   * Clear all accumulated results.
   */
  clearFiles() {
    this.classifiedFiles = {
      magnitude: [],
      phase: [],
      jsonFiles: [],
      fieldStrength: null,
      echoTimes: []
    };
  }
}
