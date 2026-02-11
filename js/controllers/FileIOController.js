/**
 * FileIOController
 *
 * Handles file input management, echo time extraction, and file list UI updates.
 */

export class FileIOController {
  constructor(options) {
    this.updateOutput = options.updateOutput || (() => {});
    this.onFilesChanged = options.onFilesChanged || (() => {});
    this.onMagnitudeFilesChanged = options.onMagnitudeFilesChanged || (() => {});
    this.onPhaseFilesChanged = options.onPhaseFilesChanged || (() => {});

    // File storage
    this.multiEchoFiles = {
      magnitude: [],
      phase: [],
      json: [],
      echoTimes: [],
      combinedMagnitude: null,
      combinedPhase: null
    };

    // Tagify instance for echo times
    this.echoTagify = null;
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

  hasValidData() {
    const magCount = this.multiEchoFiles.magnitude.length;
    const phaseCount = this.multiEchoFiles.phase.length;
    return magCount === phaseCount && magCount > 0;
  }

  hasEchoTimes() {
    return this.getEchoTimesFromInputs().length > 0;
  }

  getMultiEchoFiles() {
    return this.multiEchoFiles;
  }

  // ==================== File Handling ====================

  async handleFileInput(event, type) {
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

    // Notify listeners
    if (type === 'magnitude') {
      this.onMagnitudeFilesChanged(files);
    } else if (type === 'phase') {
      this.onPhaseFilesChanged(files);
    }

    this.onFilesChanged(type, files);
  }

  removeFile(type, index) {
    this.multiEchoFiles[type].splice(index, 1);
    this.updateFileList(type, this.multiEchoFiles[type]);

    // Notify listeners
    const files = this.multiEchoFiles[type].map(f => f.file);
    if (type === 'magnitude') {
      this.onMagnitudeFilesChanged(files);
    } else if (type === 'phase') {
      this.onPhaseFilesChanged(files);
    }

    this.onFilesChanged(type, files);
  }

  clearFiles(type) {
    this.multiEchoFiles[type] = [];
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
  }

  // ==================== File List UI ====================

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
