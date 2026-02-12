/**
 * FileIOController
 *
 * Handles file input management, echo time extraction, and file list UI updates.
 * Supports multiple input modes: raw (magnitude+phase), totalField, and localField.
 */

export class FileIOController {
  constructor(options) {
    this.updateOutput = options.updateOutput || (() => {});
    this.onFilesChanged = options.onFilesChanged || (() => {});
    this.onMagnitudeFilesChanged = options.onMagnitudeFilesChanged || (() => {});
    this.onPhaseFilesChanged = options.onPhaseFilesChanged || (() => {});

    // Current input mode: 'raw', 'totalField', or 'localField'
    this.inputMode = 'raw';

    // File storage - raw mode
    this.multiEchoFiles = {
      magnitude: [],
      phase: [],
      json: [],
      echoTimes: [],
      combinedMagnitude: null,
      combinedPhase: null
    };

    // File storage - field map modes
    this.fieldMapFiles = {
      totalField: [],     // single file as array for consistency
      localField: [],     // single file as array for consistency
      magnitudeTF: [],    // optional magnitude for total field mode (multi-file)
      magnitudeLF: [],    // optional magnitude for local field mode (multi-file)
    };

    // Centralized mask file storage (used by all modes)
    this.maskFile = [];

    // Tagify instance for echo times
    this.echoTagify = null;
  }

  // ==================== Input Mode ====================

  getInputMode() {
    return this.inputMode;
  }

  setInputMode(mode) {
    this.inputMode = mode;
  }

  // ==================== State Accessors ====================

  getMagnitudeFiles() {
    return this.multiEchoFiles.magnitude;
  }

  getPhaseFiles() {
    return this.multiEchoFiles.phase;
  }

  getJsonFiles() {
    return this.multiEchoFiles.json;
  }

  getEchoCount() {
    return Math.max(
      this.multiEchoFiles.magnitude.length,
      this.multiEchoFiles.phase.length
    );
  }

  /**
   * Check if we have valid data for the current input mode.
   */
  hasValidData() {
    switch (this.inputMode) {
      case 'raw': {
        const magCount = this.multiEchoFiles.magnitude.length;
        const phaseCount = this.multiEchoFiles.phase.length;
        return magCount === phaseCount && magCount > 0;
      }
      case 'totalField':
        return this.fieldMapFiles.totalField.length > 0;
      case 'localField':
        return this.fieldMapFiles.localField.length > 0;
      default:
        return false;
    }
  }

  hasEchoTimes() {
    return this.getEchoTimesFromInputs().length > 0;
  }

  getMultiEchoFiles() {
    return this.multiEchoFiles;
  }

  // Field map mode accessors

  getTotalFieldFile() {
    return this.fieldMapFiles.totalField[0]?.file || null;
  }

  getLocalFieldFile() {
    return this.fieldMapFiles.localField[0]?.file || null;
  }

  getFieldMapMagnitudeFile() {
    return this.getFieldMapMagnitudeFiles()[0]?.file || null;
  }

  getFieldMapMagnitudeFiles() {
    if (this.inputMode === 'totalField') {
      return this.fieldMapFiles.magnitudeTF;
    } else if (this.inputMode === 'localField') {
      return this.fieldMapFiles.magnitudeLF;
    }
    return [];
  }

  getFieldMapMagnitudeCount() {
    return this.getFieldMapMagnitudeFiles().length;
  }

  hasFieldMapMagnitude() {
    return this.getFieldMapMagnitudeFile() !== null;
  }

  getMaskFile() {
    return this.maskFile[0]?.file || null;
  }

  hasMask() {
    return this.getMaskFile() !== null;
  }

  getFieldMapUnits() {
    const select = document.getElementById('fieldMapUnits');
    return select ? select.value : 'hz';
  }

  // ==================== File Handling ====================

  async handleFileInput(event, type) {
    const files = Array.from(event.target.files);

    // Determine which storage to use
    if (type in this.multiEchoFiles) {
      // Raw mode file types
      this.multiEchoFiles[type] = files.map(file => ({
        file: file,
        name: file.name
      }));
    } else if (type === 'mask') {
      // Centralized mask file (single file)
      this.maskFile = files.slice(0, 1).map(file => ({
        file: file,
        name: file.name
      }));
    } else if (type in this.fieldMapFiles) {
      // Field map mode file types
      // Single file for field maps, multi-file for magnitude
      const isSingleFileType = (type === 'totalField' || type === 'localField');
      const selectedFiles = isSingleFileType ? files.slice(0, 1) : files;
      this.fieldMapFiles[type] = selectedFiles.map(file => ({
        file: file,
        name: file.name
      }));
    }

    // Update UI
    this.updateFileList(type, this._getFileList(type));

    // Process JSON files immediately to extract echo times
    if (type === 'json') {
      await this.processJsonFiles(files);
    }

    // Notify listeners
    if (type === 'magnitude') {
      this.onMagnitudeFilesChanged(files);
    } else if (type === 'phase') {
      this.onPhaseFilesChanged(files);
    }

    this.onFilesChanged(type, files);
  }

  _getFileList(type) {
    if (type in this.multiEchoFiles) {
      return this.multiEchoFiles[type];
    } else if (type === 'mask') {
      return this.maskFile;
    } else if (type in this.fieldMapFiles) {
      return this.fieldMapFiles[type];
    }
    return [];
  }

  removeFile(type, index) {
    if (type in this.multiEchoFiles) {
      this.multiEchoFiles[type].splice(index, 1);
    } else if (type === 'mask') {
      this.maskFile.splice(index, 1);
    } else if (type in this.fieldMapFiles) {
      this.fieldMapFiles[type].splice(index, 1);
    }
    this.updateFileList(type, this._getFileList(type));

    // Notify listeners
    const files = this._getFileList(type).map(f => f.file);
    if (type === 'magnitude') {
      this.onMagnitudeFilesChanged(files);
    } else if (type === 'phase') {
      this.onPhaseFilesChanged(files);
    }

    this.onFilesChanged(type, files);
  }

  clearFiles(type) {
    if (type in this.multiEchoFiles) {
      this.multiEchoFiles[type] = [];
    } else if (type === 'mask') {
      this.maskFile = [];
    } else if (type in this.fieldMapFiles) {
      this.fieldMapFiles[type] = [];
    }
    this.updateFileList(type, []);

    if (type === 'magnitude') {
      this.onMagnitudeFilesChanged([]);
    } else if (type === 'phase') {
      this.onPhaseFilesChanged([]);
    }

    this.onFilesChanged(type, []);
  }

  clearAllFiles() {
    this.clearFiles('magnitude');
    this.clearFiles('phase');
    this.clearFiles('json');
    this.multiEchoFiles.echoTimes = [];
    this.multiEchoFiles.combinedMagnitude = null;
    this.multiEchoFiles.combinedPhase = null;

    // Clear field map files
    for (const key of Object.keys(this.fieldMapFiles)) {
      this.fieldMapFiles[key] = [];
      this.updateFileList(key, []);
    }

    // Clear centralized mask
    this.maskFile = [];
    this.updateFileList('mask', []);
  }

  // ==================== File List UI ====================

  updateFileList(type, fileList) {
    const listElement = document.getElementById(`${type}List`);
    const fileDrop = listElement?.closest('.upload-group')?.querySelector('.file-drop');

    if (!listElement) {
      // Not an error - some list elements may not exist for all modes
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
        label.textContent = 'Drop or click';
      }
    }
  }

  // ==================== JSON Processing ====================

  async processJsonFiles(files) {
    console.log('Processing JSON files:', files);
    const echoTimes = [];
    let fieldStrength = null;

    for (const file of files) {
      try {
        const text = await file.text();
        const json = JSON.parse(text);

        console.log(`JSON file ${file.name} contents:`, json);

        // Extract echo time (in seconds, convert to ms)
        let echoTime = null;
        if (json.EchoTime) {
          echoTime = json.EchoTime * 1000; // Convert to ms
        } else if (json.echo_time) {
          echoTime = json.echo_time * 1000;
        } else if (json.TE) {
          echoTime = json.TE;
        }

        // Extract field strength (in Tesla) - only need to find it once
        if (fieldStrength === null) {
          if (json.MagneticFieldStrength) {
            fieldStrength = json.MagneticFieldStrength;
          } else if (json.FieldStrength) {
            fieldStrength = json.FieldStrength;
          } else if (json.field_strength) {
            fieldStrength = json.field_strength;
          }
        }

        if (echoTime !== null) {
          echoTimes.push({
            file: file.name,
            echoTime: echoTime,
            json: json
          });
        }
      } catch (error) {
        console.error(`Error parsing JSON file ${file.name}:`, error);
      }
    }

    // Sort by echo time
    echoTimes.sort((a, b) => a.echoTime - b.echoTime);
    this.multiEchoFiles.echoTimes = echoTimes;

    // Populate the editable inputs
    this.populateEchoTimeInputs(echoTimes.map(et => et.echoTime));

    // Populate field strength if found
    if (fieldStrength !== null) {
      const fieldInput = document.getElementById('magField');
      if (fieldInput) {
        fieldInput.value = fieldStrength;
        this.updateOutput(`Field strength: ${fieldStrength}T`);
      }
    }
  }

  getFieldStrength() {
    const fieldInput = document.getElementById('magField');
    return fieldInput ? parseFloat(fieldInput.value) : null;
  }

  // ==================== Echo Time Management ====================

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

    this.echoTagify.on('change', () => this.onFilesChanged('echoTimes', null));
  }

  populateEchoTimeInputs(echoTimes) {
    if (!this.echoTagify) return;

    const tags = echoTimes.map(t => ({ value: t.toFixed(2) }));
    this.echoTagify.removeAllTags();
    this.echoTagify.addTags(tags);
  }

  getEchoTimesFromInputs() {
    if (!this.echoTagify) return [];

    return this.echoTagify.value
      .map(tag => parseFloat(tag.value))
      .filter(n => !isNaN(n) && n > 0)
      .sort((a, b) => a - b);
  }

  // ==================== Combined Data ====================

  setCombinedMagnitude(data) {
    this.multiEchoFiles.combinedMagnitude = data;
  }

  setCombinedPhase(data) {
    this.multiEchoFiles.combinedPhase = data;
  }

  getCombinedMagnitude() {
    return this.multiEchoFiles.combinedMagnitude;
  }

  getCombinedPhase() {
    return this.multiEchoFiles.combinedPhase;
  }
}
