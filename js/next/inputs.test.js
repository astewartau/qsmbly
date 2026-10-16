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
    expect(inputKeys(files).map(([k, f]) => [k, f.name])).toEqual([
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
