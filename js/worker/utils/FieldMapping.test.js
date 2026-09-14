/**
 * FieldMapping Tests
 */
import { computeFieldMap } from './FieldMapping.js';

/**
 * Build a stub WASM module that records the arguments handed to
 * mcpc3ds_b0_pipeline_wasm and returns a well-formed result buffer.
 */
function stubWasm(voxelCount, nEchoes) {
  const calls = [];
  return {
    calls,
    mcpc3ds_b0_pipeline_wasm(phasesFlat, magsFlat, tes, mask, ...rest) {
      calls.push({ phasesFlat, magsFlat, tes, mask, rest });
      // b0, phase_offset, then the corrected phases
      return new Float64Array((2 + nEchoes) * voxelCount);
    },
  };
}

function runMcpc3ds({ echoTimes, dims = [2, 2, 2] }) {
  const voxelCount = dims[0] * dims[1] * dims[2];
  const nEchoes = echoTimes.length;
  const wasm = stubWasm(voxelCount, nEchoes);

  const phase4d = [];
  const magnitude4d = [];
  for (let e = 0; e < nEchoes; e++) {
    phase4d.push(new Float64Array(voxelCount));
    magnitude4d.push(new Float64Array(voxelCount).fill(1));
  }

  computeFieldMap(wasm, {
    phase4d, magnitude4d, echoTimes,
    mask: new Uint8Array(voxelCount).fill(1),
    dims,
    voxelSize: [1, 1, 1],
    affine: new Float64Array(16),
    settings: { phase_offset_method: 'mcpc3ds' },
    postLog: () => {},
    postProgress: () => {},
    sendStageData: null,
  });

  return wasm.calls[0];
}

describe('computeFieldMap', () => {
  describe('echo-time units on the MCPC-3D-S path', () => {
    // qsm-core's calculate_b0_weighted documents echo times in SECONDS and applies
    // scale = 1/2pi with no ms factor. computeFieldMap takes them in ms, so the
    // boundary must convert. Passing ms straight through makes B0 1000x too small.
    test('converts echo times from ms to seconds before calling WASM', () => {
      const call = runMcpc3ds({ echoTimes: [4, 12, 20] });

      expect(Array.from(call.tes)).toEqual([0.004, 0.012, 0.020]);
    });

    test('passes echo times as a Float64Array', () => {
      const call = runMcpc3ds({ echoTimes: [5, 10] });

      expect(call.tes).toBeInstanceOf(Float64Array);
      expect(call.tes.length).toBe(2);
    });

    test('agrees with the seconds convention used elsewhere in the worker', () => {
      // The TGV caller converts the B0 map back to phase with te = echoTimes[0] / 1000,
      // so the forward direction must use the same scale factor.
      const echoTimes = [7.5, 15];
      const call = runMcpc3ds({ echoTimes });

      expect(call.tes[0]).toBeCloseTo(echoTimes[0] / 1000, 12);
    });

    test('does not mutate the caller-supplied echo times', () => {
      const echoTimes = [4, 12, 20];
      runMcpc3ds({ echoTimes });

      expect(echoTimes).toEqual([4, 12, 20]);
    });
  });
});
