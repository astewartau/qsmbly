/**
 * EngineRelay Tests
 */
import {
  createVolumeStore, engineSidecars, engineConfigJson, outputStage, createPlanProgress, SUPPLIED_MASK_TOOL,
} from './EngineRelay.js';

describe('createVolumeStore', () => {
  test('keeps, copies and forgets volumes by the engine\'s names for them', () => {
    const store = createVolumeStore();
    const phase = { nifti: new Uint8Array(10) };
    store.put('AcquiredPhase(1)', phase);
    store.put('Chi', new Float32Array(4));
    expect(store.has('AcquiredPhase(1)')).toBe(true);
    expect(store.get('AcquiredPhase(1)')).toBe(phase);
    expect(store.get('FieldPpm')).toBeUndefined();

    expect(store.copy('Chi', 'RawChi(Main)')).toBe(true);
    expect(store.get('RawChi(Main)')).toBe(store.get('Chi'));
    expect(store.copy('FieldPpm', 'Phase(1)')).toBe(false);
    expect(store.has('Phase(1)')).toBe(false);

    // A volume kept under two names is held once.
    expect(store.bytes()).toBe(10 + 16);
    expect(store.remove('Chi')).toBe(true);
    expect(store.remove('Chi')).toBe(false);
  });

  test('learns the run\'s grid, and keeps what the engine recorded', () => {
    const store = createVolumeStore();
    expect(store.grid).toBeNull();
    expect(store.recorded).toBeNull();
    store.record('{"steps":{}}');
    expect(store.recorded).toBe('{"steps":{}}');
    store.begin(JSON.stringify({ dims: [2, 3, 4], voxel_size: [1, 1, 2], affine: new Array(16).fill(0) }));
    expect(store.grid.dims).toEqual([2, 3, 4]);
  });
});

describe('engineSidecars', () => {
  test('keeps the scan\'s sidecars and takes the echo times and field strength from the UI', () => {
    const sidecars = engineSidecars([5, 10], 2.9, [{ B0_dir: [0, 0, 1], EchoTime: 0.004 }]);
    expect(sidecars).toEqual([
      { B0_dir: [0, 0, 1], EchoTime: 0.005, MagneticFieldStrength: 2.9 },
      { EchoTime: 0.01, MagneticFieldStrength: 2.9 },
    ]);
  });
});

describe('engineConfigJson', () => {
  test('is the exported config, naming the supplied mask when there is one', () => {
    const settings = { dipole_inversion: 'tkd', bf_algorithm: 'pdf' };
    const plain = JSON.parse(engineConfigJson(settings));
    expect(plain.inversion.algorithm).toBe('tkd');
    expect(plain.bg_removal.algorithm).toBe('pdf');
    expect(plain.masking.custom_mask_tool).toBeUndefined();

    const supplied = JSON.parse(engineConfigJson(settings, { suppliedMask: true }));
    expect(supplied.masking.custom_mask_tool).toBe(SUPPLIED_MASK_TOOL);
    expect(supplied.masking.inhomogeneity_correction).toBe(plain.masking.inhomogeneity_correction);
  });
});

describe('outputStage', () => {
  test('lists the outputs today\'s viewer shows under their stage names', () => {
    expect(outputStage('Chi')).toEqual({ stage: 'final', description: 'QSM Result (ppm)', display: true });
    expect(outputStage('FieldPpm').stage).toBe('B0');
    expect(outputStage('LocalField(Main)').stage).toBe('bgRemoved');
    expect(outputStage('QsmartIntermediate(Main, "vessel_mask")').stage).toBe('vascDetect');
    expect(outputStage('QsmartIntermediate(Main, "chi_stage2")').stage).toBe('chiStage2');
  });

  test('leaves out per-echo volumes, working masks and unknown names', () => {
    for (const key of ['Phase(1)', 'Magnitude(3)', 'LocalFieldMask(Main)', 'TotalFieldHz', 'LocalField(Reliable)',
      'QsmartIntermediate(Reliable, "chi_stage1")', 'QsmartIntermediate(Main, "unknown")']) {
      expect(outputStage(key)).toBeNull();
    }
  });

  test('lists the brain mask only when the engine computed it', () => {
    expect(outputStage('BrainMask')).toBeNull();
    expect(outputStage('BrainMask', { computedMask: true }).stage).toBe('mask');
  });
});

describe('createPlanProgress', () => {
  const plan = { steps: ['load', 'scale_phase', 'mask', 'invert'].map(name => ({ name })) };

  test('gives each computed step an equal share, moved through by its iterations', () => {
    const progress = createPlanProgress(plan, 0, 0.9);
    expect(progress.step('scale_phase')).toBeCloseTo(0);
    expect(progress.step('mask')).toBeCloseTo(0.3);
    expect(progress.within(1, 2)).toBeCloseTo(0.45);
    expect(progress.step('invert')).toBeCloseTo(0.6);
    expect(progress.within(5, 5)).toBeCloseTo(0.9);
  });

  test('stays on the last known step for a step the plan does not name', () => {
    const progress = createPlanProgress(plan, 0, 0.9);
    progress.step('mask');
    expect(progress.step('load')).toBeCloseTo(0.3);
  });
});
