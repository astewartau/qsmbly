import { sortFiles, echoes, scanInputs, inputKeys, outputPrefix, echoFromName, guessRole } from './inputs.js';

const bytes = (text = '') => new TextEncoder().encode(text);
const json = (o) => bytes(JSON.stringify(o));
const grid = { dims: [4, 4, 4], voxel_size: [1, 1, 1], affine: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] };

describe('sorting files into roles', () => {
  test('BIDS names give the part and echo, and sidecars sit beside their images', () => {
    const files = sortFiles([
      { name: 'sub-1_echo-2_part-phase_MEGRE.nii.gz', bytes: bytes() },
      { name: 'sub-1_echo-1_part-mag_MEGRE.nii.gz', bytes: bytes() },
      { name: 'sub-1_echo-1_part-phase_MEGRE.nii.gz', bytes: bytes() },
      { name: 'sub-1_echo-1_part-phase_MEGRE.json', bytes: json({ EchoTime: 0.005 }) },
      { name: 'sub-1_echo-2_part-phase_MEGRE.json', bytes: json({ EchoTime: 0.01 }) },
      { name: 'notes.txt', bytes: bytes('x') },
    ]);
    const by = Object.fromEntries(files.map((f) => [f.name, f]));
    expect(by['sub-1_echo-2_part-phase_MEGRE.nii.gz']).toMatchObject({ role: 'phase', echo: 2, type: 'nifti' });
    expect(by['sub-1_echo-1_part-mag_MEGRE.nii.gz']).toMatchObject({ role: 'magnitude', echo: 1 });
    expect(by['sub-1_echo-2_part-phase_MEGRE.json']).toMatchObject({ role: 'sidecar', echo: 2, json: { EchoTime: 0.01 } });
    expect(by['notes.txt']).toMatchObject({ type: 'other', role: 'ignore' });
    const e = echoes(files);
    expect(e.map((x) => [x.echo, x.phase?.name, x.magnitude?.name ?? null, x.sidecar?.name])).toEqual([
      [1, 'sub-1_echo-1_part-phase_MEGRE.nii.gz', 'sub-1_echo-1_part-mag_MEGRE.nii.gz', 'sub-1_echo-1_part-phase_MEGRE.json'],
      [2, 'sub-1_echo-2_part-phase_MEGRE.nii.gz', null, 'sub-1_echo-2_part-phase_MEGRE.json'],
    ]);
  });

  test('dcm2niix names, and echoes numbered by their echo times where the names do not say', () => {
    expect(echoFromName('gre_e3_ph.nii')).toBe(3);
    expect(echoFromName('gre_e2.nii.gz')).toBe(2);
    expect(echoFromName('scan.nii')).toBe(null);
    expect(guessRole('gre_e1_ph.nii', null)).toBe('phase');
    expect(guessRole('scan.nii', { ImageType: ['ORIGINAL', 'PRIMARY', 'P', 'ND'] })).toBe('phase');
    expect(guessRole('scan.nii', null)).toBe('ignore');
    const files = sortFiles([
      { name: 'b.nii', bytes: bytes() },
      { name: 'b.json', bytes: json({ EchoTime: 0.02, ImageType: ['P'] }) },
      { name: 'a.nii', bytes: bytes() },
      { name: 'a.json', bytes: json({ EchoTime: 0.01, ImageType: ['P'] }) },
    ]);
    const phase = files.filter((f) => f.role === 'phase').sort((x, y) => (x.echo ?? 0) - (y.echo ?? 0));
    expect(phase.map((f) => f.name)).toEqual(['a.nii', 'b.nii']);
  });

  test('a sidecar that is not JSON is an error on the file', () => {
    const [f] = sortFiles([{ name: 'x.json', bytes: bytes('{') }]);
    expect(f.error).toMatch(/Not valid JSON/);
  });
});

describe('what the engine is given', () => {
  test('ScanInputs: each echo\'s headers and sidecar, and only the typed values that are set', () => {
    const files = sortFiles([
      { name: 'sub-2_ses-a_echo-1_part-phase_MEGRE.nii', bytes: bytes() },
      { name: 'sub-2_ses-a_echo-1_part-mag_MEGRE.nii', bytes: bytes() },
      { name: 'sub-2_ses-a_echo-1_part-phase_MEGRE.json', bytes: json({ EchoTime: 0.004 }) },
    ]);
    files[0].geometry = grid;
    files[1].geometry = grid;
    expect(scanInputs(files, { field_strength: 3, echo_times: null })).toEqual({
      echoes: [{ phase: grid, magnitude: grid, sidecar: { EchoTime: 0.004 } }],
      overrides: { field_strength: 3 },
    });
    const reported = [
      { key: 'AcquiredPhase(1)', echo: 1, coil: null, part: 'phase' },
      { key: 'AcquiredMagnitude(1)', echo: 1, coil: null, part: 'magnitude' },
    ];
    expect(inputKeys(files, reported).map(([k, f]) => [k, f.name])).toEqual([
      ['AcquiredPhase(1)', 'sub-2_ses-a_echo-1_part-phase_MEGRE.nii'],
      ['AcquiredMagnitude(1)', 'sub-2_ses-a_echo-1_part-mag_MEGRE.nii'],
    ]);
    expect(outputPrefix(files)).toBe('sub-2_ses-a');
    expect(outputPrefix([])).toBe('qsmbly');
  });

  test('a file set aside is not given; an echo without its phase still counts', () => {
    const files = sortFiles([
      { name: 'sub-1_echo-1_part-phase_MEGRE.nii', bytes: bytes() },
      { name: 'sub-1_echo-2_part-mag_MEGRE.nii', bytes: bytes() },
    ]);
    expect(scanInputs(files).echoes).toEqual([
      { phase: null, magnitude: null, sidecar: null },
      { phase: null, magnitude: null, sidecar: null },
    ]);
    files[1].role = 'ignore';
    expect(scanInputs(files).echoes).toHaveLength(1);
  });
});

describe('real and imaginary parts, uncombined coils and a MESE', () => {
  test('a real/imaginary scan: its parts by BIDS part-, given as real and imaginary per echo', () => {
    const files = sortFiles([
      { name: 'sub-1_echo-1_part-real_MEGRE.nii', bytes: bytes() },
      { name: 'sub-1_echo-1_part-imag_MEGRE.nii', bytes: bytes() },
      { name: 'sub-1_echo-1_part-real_MEGRE.json', bytes: json({ EchoTime: 0.004, Manufacturer: 'GE' }) },
      { name: 'sub-1_echo-2_part-real_MEGRE.nii', bytes: bytes() },
      { name: 'sub-1_echo-2_part-imag_MEGRE.nii', bytes: bytes() },
    ]);
    expect(files.filter((f) => f.type === 'nifti').map((f) => [f.role, f.echo])).toEqual([['real', 1], ['imaginary', 1], ['real', 2], ['imaginary', 2]]);
    for (const f of files) if (f.type === 'nifti') f.geometry = grid;
    expect(scanInputs(files).echoes[0]).toEqual({ phase: null, magnitude: null, real: grid, imaginary: grid, sidecar: { EchoTime: 0.004, Manufacturer: 'GE' } });
    expect(outputPrefix(files)).toBe('sub-1');
    const reported = [1, 2].flatMap((e) => [
      { key: `AcquiredReal(${e})`, echo: e, coil: null, part: 'real' },
      { key: `AcquiredImaginary(${e})`, echo: e, coil: null, part: 'imaginary' },
    ]);
    expect(inputKeys(files, reported).map(([k]) => k)).toEqual(['AcquiredReal(1)', 'AcquiredImaginary(1)', 'AcquiredReal(2)', 'AcquiredImaginary(2)']);
  });

  test('a sidecar\'s ComplexImageComponent says real or imaginary too', () => {
    expect(guessRole('x.nii', { ComplexImageComponent: 'REAL' })).toBe('real');
    expect(guessRole('x.nii', { ImageType: ['ORIGINAL', 'PRIMARY', 'IMAGINARY'] })).toBe('imaginary');
  });

  test('uncombined coils: each coil\'s images by BIDS coil-, in coil order per echo', () => {
    const files = sortFiles([2, 1].flatMap((c) => [1, 2].flatMap((e) => [
      { name: `sub-1_echo-${e}_coil-${c}_part-phase_MEGRE.nii`, bytes: bytes() },
      { name: `sub-1_echo-${e}_coil-${c}_part-mag_MEGRE.nii`, bytes: bytes() },
    ])));
    for (const f of files) f.geometry = grid;
    const inputs = scanInputs(files);
    expect(inputs.echoes).toHaveLength(2);
    expect(inputs.echoes[0].coils).toEqual([{ phase: grid, magnitude: grid }, { phase: grid, magnitude: grid }]);
    expect(inputs.echoes[0].phase).toBeNull();
    expect(outputPrefix(files)).toBe('sub-1');
    const reported = [1, 2].flatMap((e) => [1, 2].flatMap((c) => [
      { key: `CoilPhase(${c}, ${e})`, echo: e, coil: c, part: 'phase' },
      { key: `CoilMagnitude(${c}, ${e})`, echo: e, coil: c, part: 'magnitude' },
    ]));
    const keys = Object.fromEntries(inputKeys(files, reported).map(([k, f]) => [f.name, k]));
    expect(keys['sub-1_echo-2_coil-1_part-phase_MEGRE.nii']).toBe('CoilPhase(1, 2)');
    expect(keys['sub-1_echo-1_coil-2_part-mag_MEGRE.nii']).toBe('CoilMagnitude(2, 1)');
  });

  test('a MESE beside the scan: its echoes, numbered by echo time, with their sidecars', () => {
    const files = sortFiles([
      { name: 'sub-1_echo-1_part-phase_MEGRE.nii', bytes: bytes() },
      { name: 'sub-1_run-1_echo-2_MESE.nii', bytes: bytes() },
      { name: 'sub-1_run-1_echo-1_MESE.nii', bytes: bytes() },
      { name: 'sub-1_run-1_echo-1_MESE.json', bytes: json({ EchoTime: 0.01 }) },
    ]);
    expect(files.filter((f) => f.role === 'mese').map((f) => f.echo).sort()).toEqual([1, 2]);
    for (const f of files) if (f.type === 'nifti') f.geometry = grid;
    const inputs = scanInputs(files);
    expect(inputs.echoes).toHaveLength(1);
    expect(inputs.mese).toEqual({ name: 'sub-1_run-1_MESE', echoes: [{ magnitude: grid, sidecar: { EchoTime: 0.01 } }, { magnitude: grid, sidecar: null }] });
    const reported = [
      { key: 'AcquiredPhase(1)', echo: 1, coil: null, part: 'phase' },
      { key: 'MeseMagnitude(1)', echo: 1, coil: null, part: 'mese_magnitude' },
      { key: 'MeseMagnitude(2)', echo: 2, coil: null, part: 'mese_magnitude' },
    ];
    expect(inputKeys(files, reported).map(([k]) => k)).toEqual(['AcquiredPhase(1)', 'MeseMagnitude(1)', 'MeseMagnitude(2)']);
  });
});
