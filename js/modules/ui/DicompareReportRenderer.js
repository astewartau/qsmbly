/**
 * DicompareReportRenderer - Renders compliance results into DOM elements
 * and generates standalone HTML for printing.
 *
 * Adapted from dicompare-web's FieldTable and printReportGenerator for vanilla JS.
 */

export class DicompareReportRenderer {

  /**
   * Render the full report into a container element.
   * @param {HTMLElement} container - The modal body element
   * @param {Object} data - { acquisitions, complianceResults, schema }
   */
  render(container, data) {
    container.innerHTML = '';

    const { acquisitions, complianceResults, schema } = data;

    if (!acquisitions || acquisitions.length === 0) {
      container.innerHTML = '<p class="dicompare-empty">No acquisitions found in DICOM files.</p>';
      return;
    }

    // Schema info header
    container.appendChild(this._createSchemaHeader(schema));

    // Summary badges
    container.appendChild(this._createSummary(complianceResults));

    // Per-acquisition results
    for (let i = 0; i < complianceResults.length; i++) {
      const compliance = complianceResults[i];
      const acquisition = acquisitions[i] || null;
      container.appendChild(this._createAcquisitionSection(compliance, schema, acquisition));
    }
  }

  _createSchemaHeader(schema) {
    const header = document.createElement('div');
    header.className = 'dicompare-schema-header';

    const title = document.createElement('h4');
    title.textContent = schema?.name || 'QSM Consensus Guidelines';
    header.appendChild(title);

    if (schema?.version) {
      const version = document.createElement('span');
      version.className = 'dicompare-schema-version';
      version.textContent = `v${schema.version}`;
      title.appendChild(document.createTextNode(' '));
      title.appendChild(version);
    }

    if (schema?.description) {
      const desc = document.createElement('p');
      // Show first sentence only for brevity
      const firstSentence = schema.description.split('\n')[0];
      desc.textContent = firstSentence;
      header.appendChild(desc);
    }

    if (schema?.authors?.length) {
      const authors = document.createElement('p');
      authors.className = 'dicompare-schema-authors';
      authors.textContent = `Authors: ${schema.authors.join(', ')}`;
      header.appendChild(authors);
    }

    return header;
  }

  _createSummary(complianceResults) {
    const summary = document.createElement('div');
    summary.className = 'dicompare-summary';

    let pass = 0, fail = 0, warning = 0, na = 0;
    for (const compliance of complianceResults) {
      for (const r of (compliance.results || [])) {
        const status = r.status || r.complianceStatus;
        if (status === 'pass' || status === 'ok') pass++;
        else if (status === 'fail' || status === 'error') fail++;
        else if (status === 'warning') warning++;
        else na++;
      }
    }

    const badges = [
      { label: `${pass} Passed`, cls: 'pass' },
      { label: `${fail} Failed`, cls: 'fail' },
      { label: `${warning} Warning${warning !== 1 ? 's' : ''}`, cls: 'warning' },
    ];
    if (na > 0) {
      badges.push({ label: `${na} N/A`, cls: 'na' });
    }

    for (const { label, cls } of badges) {
      const badge = document.createElement('span');
      badge.className = `dicompare-summary-badge ${cls}`;
      badge.textContent = label;
      summary.appendChild(badge);
    }

    return summary;
  }

  _createAcquisitionSection(compliance, schema, acquisition) {
    const section = document.createElement('div');
    section.className = 'dicompare-acquisition';

    const title = document.createElement('h4');
    title.textContent = compliance.acquisitionName || 'Acquisition';
    section.appendChild(title);

    if (compliance.error) {
      const errorEl = document.createElement('p');
      errorEl.className = 'dicompare-error';
      errorEl.textContent = `Validation error: ${compliance.error}`;
      section.appendChild(errorEl);
      return section;
    }

    const results = compliance.results || [];
    if (results.length === 0) {
      const empty = document.createElement('p');
      empty.className = 'dicompare-empty';
      empty.textContent = 'No validation results for this acquisition.';
      section.appendChild(empty);
      return section;
    }

    // Split into field results and rule results
    const fieldResults = results.filter(r => r.validationType !== 'rule' && !r.rule_name);
    const ruleResults = results.filter(r => r.validationType === 'rule' || r.rule_name);

    if (fieldResults.length > 0) {
      const label = document.createElement('div');
      label.className = 'dicompare-section-label';
      label.textContent = 'Field Checks';
      section.appendChild(label);
      section.appendChild(this._createFieldTable(fieldResults));
    }

    if (ruleResults.length > 0) {
      const label = document.createElement('div');
      label.className = 'dicompare-section-label';
      label.textContent = 'Validation Rules';
      section.appendChild(label);
      section.appendChild(this._createRuleTable(ruleResults, schema));
    }

    // Unchecked fields: fields in data but not validated by the schema
    const uncheckedFields = this._getUncheckedFields(acquisition, schema);
    if (uncheckedFields.length > 0) {
      section.appendChild(this._createUncheckedFieldsSection(uncheckedFields));
    }

    return section;
  }

  /**
   * Find fields in the acquisition data that are not covered by the schema.
   */
  _getUncheckedFields(acquisition, schema) {
    if (!acquisition || !schema?.acquisitions) return [];

    const dataFields = acquisition.acquisitionFields || [];
    if (dataFields.length === 0) return [];

    // Collect all field identifiers from the schema
    const schemaFieldIds = new Set();
    const schemaKeywords = new Set();
    for (const acqData of Object.values(schema.acquisitions)) {
      for (const f of (acqData.fields || [])) {
        if (f.tag) schemaFieldIds.add(f.tag.replace(/\s/g, ''));
        if (f.field) schemaKeywords.add(f.field.toLowerCase());
      }
      // Also include fields referenced by rules
      for (const rule of (acqData.rules || [])) {
        for (const fieldName of (rule.fields || [])) {
          schemaKeywords.add(fieldName.toLowerCase());
        }
      }
    }

    return dataFields.filter(f => {
      const tag = f.tag ? f.tag.replace(/\s/g, '') : '';
      const keyword = (f.keyword || f.name || '').toLowerCase();
      return !schemaFieldIds.has(tag) && !schemaKeywords.has(keyword);
    });
  }

  /**
   * Create a collapsible section showing unchecked fields.
   */
  _createUncheckedFieldsSection(uncheckedFields) {
    const wrapper = document.createElement('div');
    wrapper.className = 'dicompare-unchecked';

    // Toggle button
    const toggle = document.createElement('button');
    toggle.className = 'dicompare-unchecked-toggle';
    toggle.innerHTML = `<span><strong>${uncheckedFields.length} field${uncheckedFields.length !== 1 ? 's' : ''}</strong> in data not validated by schema</span><span class="dicompare-chevron">&#9660;</span>`;
    wrapper.appendChild(toggle);

    // Content (hidden by default)
    const content = document.createElement('div');
    content.className = 'dicompare-unchecked-content';
    content.style.display = 'none';

    const table = document.createElement('table');
    table.className = 'dicompare-table';
    const thead = document.createElement('thead');
    thead.innerHTML = '<tr><th>Field</th><th>Value in Data</th></tr>';
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    for (const f of uncheckedFields) {
      const tr = document.createElement('tr');

      const tdField = document.createElement('td');
      const name = document.createElement('span');
      name.className = 'dicompare-field-name';
      name.textContent = f.keyword || f.name || '';
      tdField.appendChild(name);
      if (f.tag) {
        const tag = document.createElement('span');
        tag.className = 'dicompare-field-tag';
        tag.textContent = ` (${f.tag})`;
        tdField.appendChild(tag);
      }
      tr.appendChild(tdField);

      const tdValue = document.createElement('td');
      tdValue.textContent = this._formatValue(f.value);
      tr.appendChild(tdValue);

      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    content.appendChild(table);
    wrapper.appendChild(content);

    // Toggle visibility
    toggle.addEventListener('click', () => {
      const isHidden = content.style.display === 'none';
      content.style.display = isHidden ? '' : 'none';
      toggle.querySelector('.dicompare-chevron').style.transform = isHidden ? 'rotate(180deg)' : '';
    });

    return wrapper;
  }

  _createFieldTable(results) {
    const table = document.createElement('table');
    table.className = 'dicompare-table';

    // Header
    const thead = document.createElement('thead');
    thead.innerHTML = `
      <tr>
        <th>Field</th>
        <th>Expected</th>
        <th>Actual</th>
        <th>Status</th>
      </tr>
    `;
    table.appendChild(thead);

    // Body
    const tbody = document.createElement('tbody');
    for (const r of results) {
      const tr = document.createElement('tr');

      // Field name + tag
      const tdField = document.createElement('td');
      const fieldName = document.createElement('span');
      fieldName.className = 'dicompare-field-name';
      fieldName.textContent = r.fieldName || r.field || '';
      tdField.appendChild(fieldName);
      if (r.fieldPath) {
        const tag = document.createElement('span');
        tag.className = 'dicompare-field-tag';
        tag.textContent = `(${r.fieldPath})`;
        tdField.appendChild(document.createTextNode(' '));
        tdField.appendChild(tag);
      }
      tr.appendChild(tdField);

      // Expected
      const tdExpected = document.createElement('td');
      tdExpected.textContent = this._formatValue(r.expectedValue ?? r.expected ?? '');
      tr.appendChild(tdExpected);

      // Actual
      const tdActual = document.createElement('td');
      tdActual.textContent = this._formatValue(r.actualValue ?? r.value ?? '');
      tr.appendChild(tdActual);

      // Status
      const tdStatus = document.createElement('td');
      tdStatus.appendChild(this._createStatusBadge(r.status || r.complianceStatus));
      if (r.message) {
        const msg = document.createElement('div');
        msg.className = 'dicompare-status-message';
        msg.textContent = r.message;
        tdStatus.appendChild(msg);
      }
      tr.appendChild(tdStatus);

      tbody.appendChild(tr);
    }
    table.appendChild(tbody);

    return table;
  }

  _createRuleTable(results, schema) {
    const table = document.createElement('table');
    table.className = 'dicompare-table';

    const thead = document.createElement('thead');
    thead.innerHTML = `
      <tr>
        <th>Rule</th>
        <th>Status</th>
      </tr>
    `;
    table.appendChild(thead);

    // Get rule descriptions from schema
    const schemaRules = this._getSchemaRules(schema);

    const tbody = document.createElement('tbody');
    for (const r of results) {
      const tr = document.createElement('tr');

      // Rule name + description
      const tdRule = document.createElement('td');
      const ruleName = document.createElement('div');
      ruleName.className = 'dicompare-field-name';
      ruleName.textContent = r.rule_name || r.fieldName || 'Rule';
      tdRule.appendChild(ruleName);

      // Find description from schema rules
      const schemaRule = schemaRules.find(sr =>
        sr.name === (r.rule_name || r.fieldName)
      );
      if (schemaRule?.description) {
        const desc = document.createElement('div');
        desc.className = 'dicompare-rule-description';
        desc.textContent = schemaRule.description;
        tdRule.appendChild(desc);
      }
      tr.appendChild(tdRule);

      // Status + message
      const tdStatus = document.createElement('td');
      tdStatus.appendChild(this._createStatusBadge(r.status || r.complianceStatus));
      if (r.message) {
        const msg = document.createElement('div');
        msg.className = 'dicompare-status-message';
        msg.textContent = r.message;
        tdStatus.appendChild(msg);
      }
      tr.appendChild(tdStatus);

      tbody.appendChild(tr);
    }
    table.appendChild(tbody);

    return table;
  }

  _getSchemaRules(schema) {
    if (!schema?.acquisitions) return [];
    const rules = [];
    for (const acqData of Object.values(schema.acquisitions)) {
      if (acqData.rules) {
        rules.push(...acqData.rules);
      }
    }
    return rules;
  }

  _createStatusBadge(status) {
    const normalized = this._normalizeStatus(status);
    const badge = document.createElement('span');
    badge.className = `dicompare-status dicompare-status-${normalized}`;
    badge.textContent = normalized === 'pass' ? 'Pass'
      : normalized === 'fail' ? 'Fail'
      : normalized === 'warning' ? 'Warning'
      : normalized === 'na' ? 'N/A'
      : 'Unknown';
    return badge;
  }

  _normalizeStatus(status) {
    if (status === 'ok' || status === 'pass') return 'pass';
    if (status === 'error' || status === 'fail') return 'fail';
    if (status === 'warning') return 'warning';
    if (status === 'na') return 'na';
    return 'unknown';
  }

  _formatValue(value) {
    if (value === null || value === undefined || value === '') return '\u2014';
    if (Array.isArray(value)) {
      return value.map(v => String(v)).join(', ');
    }
    return String(value);
  }

  // --- Print HTML generation ---

  /**
   * Generate standalone HTML for printing (opens in new window).
   * @param {Object} data - { acquisitions, complianceResults, schema }
   * @returns {string} Complete HTML document
   */
  generatePrintHtml(data) {
    const { acquisitions, complianceResults, schema } = data;
    const schemaName = schema?.name || 'QSM Consensus Guidelines';
    const schemaVersion = schema?.version || '';
    const schemaAuthors = schema?.authors || [];
    const schemaDesc = schema?.description?.split('\n')[0] || '';

    let fieldsHtml = '';
    let rulesHtml = '';
    let uncheckedHtml = '';
    const schemaRules = this._getSchemaRules(schema);

    for (let i = 0; i < complianceResults.length; i++) {
      const compliance = complianceResults[i];
      const acquisition = acquisitions?.[i] || null;
      const results = compliance.results || [];
      const fieldResults = results.filter(r => r.validationType !== 'rule' && !r.rule_name);
      const ruleResults = results.filter(r => r.validationType === 'rule' || r.rule_name);

      const acqHeader = `<h3>${this._escapeHtml(compliance.acquisitionName || 'Acquisition')}</h3>`;

      if (fieldResults.length > 0) {
        const rows = fieldResults.map(r => {
          const status = this._normalizeStatus(r.status || r.complianceStatus);
          return `<tr>
            <td><span class="field-name">${this._escapeHtml(r.fieldName || r.field || '')}</span>${r.fieldPath ? ` <code>${this._escapeHtml(r.fieldPath)}</code>` : ''}</td>
            <td>${this._escapeHtml(this._formatValue(r.expectedValue ?? r.expected))}</td>
            <td>${this._escapeHtml(this._formatValue(r.actualValue ?? r.value))}</td>
            <td class="${status}">${this._escapeHtml(r.message || status)}</td>
          </tr>`;
        }).join('');

        fieldsHtml += `${acqHeader}
          <table>
            <thead><tr><th>Field</th><th>Expected</th><th>Actual</th><th>Status</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>`;
      }

      if (ruleResults.length > 0) {
        const rows = ruleResults.map(r => {
          const status = this._normalizeStatus(r.status || r.complianceStatus);
          const schemaRule = schemaRules.find(sr => sr.name === (r.rule_name || r.fieldName));
          return `<tr>
            <td><span class="field-name">${this._escapeHtml(r.rule_name || r.fieldName || 'Rule')}</span>
            ${schemaRule?.description ? `<div class="rule-desc">${this._escapeHtml(schemaRule.description)}</div>` : ''}</td>
            <td class="${status}">${this._escapeHtml(r.message || status)}</td>
          </tr>`;
        }).join('');

        rulesHtml += `<h2>Validation Rules</h2>
          <table>
            <thead><tr><th>Rule</th><th>Status</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>`;
      }

      // Unchecked fields for print
      const uncheckedFields = this._getUncheckedFields(acquisition, schema);
      if (uncheckedFields.length > 0) {
        const rows = uncheckedFields.map(f => `<tr>
          <td><span class="field-name">${this._escapeHtml(f.keyword || f.name || '')}</span>${f.tag ? ` <code>${this._escapeHtml(f.tag)}</code>` : ''}</td>
          <td>${this._escapeHtml(this._formatValue(f.value))}</td>
        </tr>`).join('');

        uncheckedHtml += `
          <div class="unchecked-section">
            <h2 class="unchecked-header">${uncheckedFields.length} field${uncheckedFields.length !== 1 ? 's' : ''} in data not validated by schema</h2>
            <table class="unchecked-table">
              <thead><tr><th>Field</th><th>Value in Data</th></tr></thead>
              <tbody>${rows}</tbody>
            </table>
          </div>`;
      }
    }

    // Summary counts
    let pass = 0, fail = 0, warning = 0;
    for (const c of complianceResults) {
      for (const r of (c.results || [])) {
        const s = this._normalizeStatus(r.status || r.complianceStatus);
        if (s === 'pass') pass++;
        else if (s === 'fail') fail++;
        else if (s === 'warning') warning++;
      }
    }

    return `<!DOCTYPE html>
<html>
<head>
  <title>${this._escapeHtml(schemaName)} - dicompare Report</title>
  <style>${this._getPrintStyles()}</style>
</head>
<body>
  <div class="header-section">
    <div class="header-item schema">
      <div class="header-label">Reference Schema</div>
      <div class="header-title">${this._escapeHtml(schemaName)}${schemaVersion ? ` <span class="version">v${this._escapeHtml(schemaVersion)}</span>` : ''}</div>
      ${schemaDesc ? `<div class="header-subtitle">${this._escapeHtml(schemaDesc)}</div>` : ''}
      ${schemaAuthors.length > 0 ? `<div class="header-authors">Authors: ${schemaAuthors.map(a => this._escapeHtml(a)).join(', ')}</div>` : ''}
    </div>
  </div>
  <div class="summary">
    <span class="badge pass">${pass} Passed</span>
    <span class="badge fail">${fail} Failed</span>
    <span class="badge warning">${warning} Warning${warning !== 1 ? 's' : ''}</span>
  </div>
  <h2>Field Checks</h2>
  ${fieldsHtml || '<p>No field checks available.</p>'}
  ${rulesHtml}
  ${uncheckedHtml}
  <div class="print-date">Generated on ${new Date().toLocaleDateString()} by QSMbly + dicompare</div>
</body>
</html>`;
  }

  _escapeHtml(str) {
    const s = String(str ?? '');
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  _getPrintStyles() {
    return `
      * { box-sizing: border-box; }
      body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        padding: 40px;
        max-width: 1000px;
        margin: 0 auto;
        color: #1a1a1a;
      }
      .header-section { margin-bottom: 24px; padding-bottom: 20px; border-bottom: 2px solid #e5e5e5; }
      .header-item.schema { border-left: 3px solid #2563eb; padding-left: 12px; }
      .header-label { font-size: 11px; font-weight: 600; text-transform: uppercase; color: #666; margin-bottom: 4px; }
      .header-title { font-size: 20px; font-weight: 600; margin-bottom: 4px; }
      .header-subtitle { font-size: 13px; color: #666; margin-bottom: 4px; }
      .header-authors { font-size: 12px; color: #888; }
      .version { font-size: 14px; font-weight: 400; color: #888; }
      .summary { display: flex; gap: 10px; margin-bottom: 20px; }
      .badge { display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 600; }
      .badge.pass { background: #dff0d8; color: #3c763d; }
      .badge.fail { background: #f2dede; color: #a94442; }
      .badge.warning { background: #fcf8e3; color: #8a6d3b; }
      h2 { font-size: 16px; margin-top: 24px; margin-bottom: 12px; border-bottom: 1px solid #ddd; padding-bottom: 8px; color: #333; }
      h3 { font-size: 14px; margin-top: 20px; margin-bottom: 8px; color: #444; }
      table { width: 100%; border-collapse: collapse; margin-bottom: 16px; font-size: 12px; }
      th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; vertical-align: top; }
      th { background: #f5f5f5; font-weight: 600; }
      code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-family: monospace; font-size: 10px; color: #666; }
      .field-name { font-weight: 500; }
      .rule-desc { font-size: 11px; color: #666; margin-top: 4px; }
      .pass { color: #16a34a; font-weight: 500; }
      .fail { color: #dc2626; font-weight: 500; }
      .warning { color: #ca8a04; font-weight: 500; }
      .unknown { color: #9ca3af; font-style: italic; }
      .na { color: #9ca3af; }
      .unchecked-section { margin-top: 24px; }
      .unchecked-header { color: #666; font-size: 14px; border-bottom: 1px dashed #ccc; }
      .unchecked-table th { background: #f9fafb; }
      .print-date { color: #999; font-size: 11px; margin-top: 40px; text-align: center; }
      @media print {
        body { padding: 20px; }
        h2 { page-break-after: avoid; }
        table { page-break-inside: auto; }
        tr { page-break-inside: avoid; }
        thead { display: table-header-group; }
      }
    `;
  }
}
