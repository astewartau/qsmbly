/**
 * PhaseUtils Tests
 */
import { jest } from '@jest/globals';
import { scalePhase, computeB0FromUnwrapped, computeWeightedEchoFit, ppmFieldToPhase } from './PhaseUtils.js';

describe('PhaseUtils', () => {
  describe('scalePhase', () => {
    test('should pass through data already in [-π, π]', () => {
      const input = new Float64Array([0, Math.PI / 2, -Math.PI / 2, Math.PI * 0.9]);
      const result = scalePhase(input);

      // Should be close to input (wrapped)
      expect(result[0]).toBeCloseTo(0, 5);
      expect(result[1]).toBeCloseTo(Math.PI / 2, 5);
      expect(result[2]).toBeCloseTo(-Math.PI / 2, 5);
    });

    test('should scale data with range > 2π', () => {
      // Input spans [0, 4π]
      const input = new Float64Array([0, Math.PI, 2 * Math.PI, 3 * Math.PI, 4 * Math.PI]);
      const result = scalePhase(input);

      // Should be scaled to [-π, π]
      expect(result[0]).toBeCloseTo(-Math.PI, 5);
      expect(result[4]).toBeCloseTo(Math.PI, 5);
      expect(result[2]).toBeCloseTo(0, 5);  // Middle should be ~0
    });

    test('should scale integer phase values (e.g., 0-4095)', () => {
      // Simulate 12-bit phase data
      const input = new Float64Array([0, 1024, 2048, 3072, 4095]);
      const result = scalePhase(input);

      // Should be scaled to [-π, π]
      expect(result[0]).toBeCloseTo(-Math.PI, 5);
      expect(result[4]).toBeCloseTo(Math.PI, 5);
    });

    test('should handle single value', () => {
      const input = new Float64Array([0.5]);
      const result = scalePhase(input);
      expect(result.length).toBe(1);
    });

    test('should handle all zeros', () => {
      const input = new Float64Array([0, 0, 0, 0]);
      const result = scalePhase(input);
      // All zeros should remain zeros (wrapped)
      for (let i = 0; i < result.length; i++) {
        expect(result[i]).toBeCloseTo(0, 5);
      }
    });
  });

  describe('computeB0FromUnwrapped', () => {
    const nx = 3, ny = 3, nz = 3;
    const voxelCount = nx * ny * nz;

    test('should compute B0 for single echo', () => {
      const echoTimes = [20];  // 20ms
      const phase = new Float64Array(voxelCount).fill(Math.PI);  // π radians

      const b0 = computeB0FromUnwrapped(phase, echoTimes, nx, ny, nz);

      // B0 = phase / (2π * TE) = π / (2π * 0.02) = 25 Hz
      expect(b0[0]).toBeCloseTo(25, 3);
    });

    test('should compute B0 for multi-echo with OLS', () => {
      const echoTimes = [10, 20, 30];  // ms
      const slope = 100;  // rad/s (B0 = 100/(2π) ≈ 15.9 Hz)

      // Create phase that increases linearly with TE
      const phase = new Float64Array(3 * voxelCount);
      for (let e = 0; e < 3; e++) {
        const phaseVal = slope * echoTimes[e] / 1000;  // phase = slope * TE
        for (let v = 0; v < voxelCount; v++) {
          phase[e * voxelCount + v] = phaseVal;
        }
      }

      const b0 = computeB0FromUnwrapped(phase, echoTimes, nx, ny, nz, 'ols');

      // B0 should be slope / (2π)
      const expectedB0 = slope / (2 * Math.PI);
      expect(b0[0]).toBeCloseTo(expectedB0, 3);
    });

    test('should handle phase offset with ols_offset method', () => {
      const echoTimes = [10, 20, 30];
      const slope = 100;  // rad/s
      const offset = 0.5; // rad

      // Create phase with offset: phase = offset + slope * TE
      const phase = new Float64Array(3 * voxelCount);
      for (let e = 0; e < 3; e++) {
        const phaseVal = offset + slope * echoTimes[e] / 1000;
        for (let v = 0; v < voxelCount; v++) {
          phase[e * voxelCount + v] = phaseVal;
        }
      }

      const b0 = computeB0FromUnwrapped(phase, echoTimes, nx, ny, nz, 'ols_offset');

      // Should correctly estimate slope despite offset
      const expectedB0 = slope / (2 * Math.PI);
      expect(b0[0]).toBeCloseTo(expectedB0, 3);
    });

    test('should return Float64Array', () => {
      const phase = new Float64Array(voxelCount);
      const b0 = computeB0FromUnwrapped(phase, [20], nx, ny, nz);

      expect(b0).toBeInstanceOf(Float64Array);
      expect(b0.length).toBe(voxelCount);
    });
  });
});

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

  describe('computeWeightedEchoFit', () => {
    const nx = 4, ny = 4, nz = 3;
    const n = nx * ny * nz;
    const echoTimes = [10, 20, 30]; // ms
    const mask = new Uint8Array(n).fill(1);
    const magnitude4d = echoTimes.map(() => new Float64Array(n).fill(100));

    beforeEach(() => jest.spyOn(console, 'log').mockImplementation(() => {}));
    afterEach(() => console.log.mockRestore());

    const linearPhase = (hz) => {
      const phase = new Float64Array(echoTimes.length * n);
      echoTimes.forEach((te, e) => phase.fill(2 * Math.PI * hz * te / 1000, e * n, (e + 1) * n));
      return phase;
    };

    test('recovers the frequency of noiseless linear phase and keeps every voxel reliable', () => {
      const { tfs, R_0 } = computeWeightedEchoFit(
        linearPhase(25), magnitude4d, echoTimes, nx, ny, nz, [1, 1, 1], mask,
      );
      for (let i = 0; i < n; i++) {
        expect(tfs[i]).toBeCloseTo(25, 8);
        expect(R_0[i]).toBe(1);
      }
    });

    test('flags a voxel whose blurred fit residual exceeds the threshold', () => {
      const phase = linearPhase(25);
      const bad = 1 + nx + nx * ny;
      phase[2 * n + bad] += 3; // last echo off the line
      const { R_0 } = computeWeightedEchoFit(
        phase, magnitude4d, echoTimes, nx, ny, nz, [1, 1, 1], mask, 0.1,
      );
      expect(R_0[bad]).toBe(0);
      // The 3x3x3 residual blur spreads the outlier to its neighbours but not to the far corner.
      expect(R_0[bad + 1]).toBe(0);
      expect(R_0[n - 1]).toBe(1);
    });
  });
});
