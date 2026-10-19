// The export bundle: which outputs the engine is told about, and the bundle's layout.
import { VolumeStore } from './store/volumes.js';
import { heldOutputs, publishedFiles, bundleEntries, bundleZip, DERIVATIVES } from './bundle.js';
import { unzip } from '../../tests/e2e/unzip.js';

const grid = { dims: [2, 2, 2], voxel_size: [1, 1, 1], affine: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] };

function store() {
  const s = new VolumeStore({ shared: false });
  s.putInput('AcquiredPhase(1)', new Uint8Array([1, 2, 3]));
  s.put('Chi', new Float32Array(8), { grid, step: 'reference' });
  s.put('BrainMask', new Uint8Array(8), { grid, step: 'mask' });
  s.put('LocalField(Main)', new Float32Array(8), { grid, step: 'bgremove' });
  s.record(JSON.stringify({ steps: {}, annotations: { Chi: { QsmReference: 'mean', QsmReferenceOffsetPpm: 0.01 } } }));
  return s;
}

describe('bundle', () => {
  test('the engine hears of every computed output, the step that wrote it and what the run recorded', () => {
    expect(heldOutputs(store())).toEqual([
      { key: 'Chi', step: 'reference', annotations: { QsmReference: 'mean', QsmReferenceOffsetPpm: 0.01 } },
      { key: 'BrainMask', step: 'mask', annotations: null },
      { key: 'LocalField(Main)', step: 'bgremove', annotations: null },
    ]);
  });

  test('the run is named after the first echo\'s phase', async () => {
    /** @type {any} */
    let asked = null;
    const engine = /** @type {any} */ ({ outputFiles: async (r) => { asked = r; return { run: 'sub-1', files: [] }; } });
    const session = /** @type {any} */ ({
      facts: { peek: () => ({ n_echoes: 1 }) },
      files: { peek: () => [{ name: 'sub-1_echo-1_part-phase_MEGRE.nii.gz', role: 'phase', echo: 1, type: 'nifti' }] },
    });
    await publishedFiles({ session, store: store(), engine });
    expect(asked.scan_file).toBe('sub-1_echo-1_part-phase_MEGRE.nii.gz');
    expect(asked.outputs.map((o) => o.key)).toEqual(['Chi', 'BrainMask', 'LocalField(Main)']);
  });

  const texts = (text) => ({ sentences: [{ text, steps: [], citations: [] }], references: [] });
  const summary = (run = null, changed = null) => ({ methods: texts('The dataset.'), mermaid: 'dataset graph', run, changed });

  test("the dataset's methods and graph, and the run's own beside it, as qsmxt writes them once its runs are done", async () => {
    const parts = { command: '', toml: '', run: 'sub-1', files: [], nifti: async () => new Uint8Array() };
    const at = (entries, path) => entries.find((e) => e.path === `${DERIVATIVES}/${path}`)?.bytes;
    const run = { methods: texts('The run.'), planned: 'planned', after: 'after', differs: false };

    const planned = await bundleEntries({ ...parts, summary: summary() });
    expect(planned.map((e) => e.path).filter((p) => p.includes('workflow') || p.includes('changes'))).toEqual([]);

    const ran = await bundleEntries({ ...parts, summary: summary(run, false) });
    expect(at(ran, 'methods.md')).toBe('# Methods\n\nThe dataset.\n\n');
    expect(at(ran, 'pipeline.mmd')).toBe('dataset graph');
    expect(at(ran, 'workflow/sub-1/sub-1_methods.md')).toBe('# Methods\n\nThe run.\n\n');
    expect(at(ran, 'workflow/sub-1/sub-1_pipeline.mmd')).toBe('planned');
    expect(at(ran, 'workflow/sub-1/sub-1_desc-ran_pipeline.mmd')).toBeUndefined();

    const differed = await bundleEntries({ ...parts, summary: summary({ ...run, differs: true }, true) });
    expect(at(differed, 'workflow/sub-1/sub-1_desc-ran_pipeline.mmd')).toBe('after');
    expect(at(differed, 'changes/sub-1_methods.md')).toBe('# Methods\n\nThe run.\n\n');
    expect(at(differed, 'changes/sub-1_desc-ran_pipeline.mmd')).toBe('after');
  });

  test('laid out as qsmxt run writes its output directory, under one folder', async () => {
    const files = [
      { key: 'Chi', path: 'sub-1/anat/sub-1_Chimap.nii', sidecar_path: 'sub-1/anat/sub-1_Chimap.json', sidecar: '{\n  "QsmReference": "mean"\n}' },
      { key: 'RegionStats', path: 'sub-1/anat/sub-1_desc-segmentation_stats.tsv', sidecar_path: null, sidecar: null },
    ];
    const entries = await bundleEntries({
      command: 'qsmxt run <bids_dir> <output_dir>', toml: '[pipeline]\n', summary: summary(), run: 'sub-1',
      files, nifti: async (key) => new TextEncoder().encode(`nifti of ${key}`),
      extra: [{ path: 'derivatives/qsmbly/sub-1/anat/sub-1_mask.nii.gz', bytes: new Uint8Array([1]) }],
    });
    expect(entries.map((e) => e.path)).toEqual([
      'command.txt', `${DERIVATIVES}/methods.md`, `${DERIVATIVES}/pipeline_config.toml`, `${DERIVATIVES}/pipeline.mmd`,
      `${DERIVATIVES}/sub-1/anat/sub-1_Chimap.nii`, `${DERIVATIVES}/sub-1/anat/sub-1_Chimap.json`,
      `${DERIVATIVES}/sub-1/anat/sub-1_desc-segmentation_stats.tsv`,
      'derivatives/qsmbly/sub-1/anat/sub-1_mask.nii.gz',
    ]);
    const back = await unzip(new Uint8Array(await (await bundleZip('sub-1', entries)).arrayBuffer()));
    expect([...back.keys()].every((p) => p.startsWith('qsmbly-sub-1/'))).toBe(true);
    expect(new TextDecoder().decode(back.get(`qsmbly-sub-1/${DERIVATIVES}/sub-1/anat/sub-1_Chimap.json`))).toContain('"QsmReference": "mean"');
    expect(new TextDecoder().decode(back.get('qsmbly-sub-1/command.txt'))).toBe('qsmxt run <bids_dir> <output_dir>\n');
  });
});
