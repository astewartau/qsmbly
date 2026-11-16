/**
 * Number formatting for the workspace.
 */

/**
 * Binary multiples, so the 4 GB WASM heap reads as 4.0 GB.
 * @param {number} bytes
 * @returns {string} e.g. `1.8 GB`, `19.9 MB`
 */
export function formatBytes(bytes) {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i++;
  }
  return `${i === 0 ? value : value.toFixed(1)} ${units[i]}`;
}

/**
 * @param {number} seconds
 * @returns {string} milliseconds, e.g. `5`, `11.5`
 */
export function formatMs(seconds) {
  return String(Math.round(seconds * 1e4) / 10);
}
