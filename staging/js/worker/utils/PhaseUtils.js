/**
 * Phase Processing Utilities
 *
 * The ppm → phase conversion TGV needs for a field-map input. Phase scaling, field mapping
 * and unit conversions are qsm-core's (see the worker).
 */

/**
 * Convert a B0 field map in ppm to the equivalent phase in radians at one echo time.
 *
 * qsm-core's field-mapping stage returns ppm; TGV takes phase and divides this
 * straight back out. `gyromagneticRatio` must be the same gamma qsm-core's
 * hz_to_ppm used (42.576e6 Hz/T), or the round trip does not close.
 *
 * @param {Float64Array} b0Ppm - B0 field map in ppm
 * @param {number} fieldStrength - B0 in Tesla
 * @param {number} te - Echo time in SECONDS
 * @param {number} gyromagneticRatio - Proton gamma in Hz/T
 * @returns {Float64Array} Phase in radians
 */
export function ppmFieldToPhase(b0Ppm, fieldStrength, te, gyromagneticRatio) {
  const ppmToHz = (gyromagneticRatio * fieldStrength) / 1e6;
  const scale = 2 * Math.PI * ppmToHz * te;
  const phase = new Float64Array(b0Ppm.length);
  for (let i = 0; i < b0Ppm.length; i++) {
    phase[i] = b0Ppm[i] * scale;
  }
  return phase;
}
