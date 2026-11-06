/**
 * PhaseUtils Tests
 */
import { ppmFieldToPhase } from './PhaseUtils.js';

describe('ppmFieldToPhase', () => {
  // Proton gamma, matching qsm-core's hz_to_ppm. The round trip only closes if both
  // sides use the same constant.
  const GAMMA = 42.576e6;

  test('inverts the Hz→ppm the field-mapping stage applied', () => {
    const fieldStrength = 3.0;
    const te = 0.012;
    const b0Hz = [-120.5, 0, 0.5, 999.25];

    // What qsm-core's hz_to_ppm produced from those Hz values.
    const ppm = Float64Array.from(b0Hz, hz => (hz * 1e6) / (GAMMA * fieldStrength));
    const phase = ppmFieldToPhase(ppm, fieldStrength, te, GAMMA);

    // Must match converting straight from Hz: phase = 2*pi*B0*TE.
    b0Hz.forEach((hz, i) => {
      expect(phase[i]).toBeCloseTo(2 * Math.PI * hz * te, 12);
    });
  });

  test('scales linearly with field strength and echo time', () => {
    const ppm = Float64Array.from([1.0]);

    const base = ppmFieldToPhase(ppm, 3.0, 0.01, GAMMA)[0];
    expect(ppmFieldToPhase(ppm, 7.0, 0.01, GAMMA)[0]).toBeCloseTo(base * (7 / 3), 12);
    expect(ppmFieldToPhase(ppm, 3.0, 0.02, GAMMA)[0]).toBeCloseTo(base * 2, 12);
  });

  test('returns a Float64Array of the same length and maps zero to zero', () => {
    const phase = ppmFieldToPhase(new Float64Array([0, 0, 0]), 3.0, 0.01, GAMMA);

    expect(phase).toBeInstanceOf(Float64Array);
    expect(phase.length).toBe(3);
    expect(Array.from(phase)).toEqual([0, 0, 0]);
  });
});
