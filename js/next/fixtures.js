/**
 * Stand-ins for the engine until the workspace runs on it (U2): the engine's own answers about
 * the example data (fixtures/*.json, from scripts/next-fixtures.mjs), a planner that picks among
 * them, and volumes to look at; and the engine's settings schema, browser profile and recorded
 * settings views (from scripts/next-schema-fixtures.mjs).
 *
 * @typedef {import('./protocol.js').Plan} Plan
 * @typedef {import('./protocol.js').ScanFacts} ScanFacts
 * @typedef {import('./protocol.js').ModelEntry} ModelEntry
 * @typedef {import('./protocol.js').RunMetadata} RunMetadata
 * @typedef {import('./store/volumes.js').VolumeStore} VolumeStore
 */
import { readNifti } from './nifti.js';

/**
 * @typedef {object} Fixtures
 * @property {ScanFacts} facts - the example scan
 * @property {ModelEntry[]} models
 * @property {Record<string, any>} browserConfig - the browser profile: the session's defaults
 * @property {import('./schema.js').Schema} schema
 * @property {import('./settings-view.js').RecordedViews} views
 * @property {RecordedPlans} plans
 * @property {import('./mask.js').MaskKit} mask - the mask recipe's building blocks
 */

/**
 * @param {string} [base] - where fixtures/ is
 * @returns {Promise<Fixtures>}
 */
export async function loadFixtures(base = new URL('./fixtures/', import.meta.url).href) {
  const get = async (name) => {
    const res = await fetch(new URL(name, base));
    if (!res.ok) throw new Error(`fixture ${name}: HTTP ${res.status}`);
    return res.json();
  };
  const [facts, models, browserConfig, schema, views, plans, mask] = await Promise.all([
    get('facts-example.json'),
    get('models.json'),
    get('config-browser.json'),
    get('schema.json'),
    get('settings-views.json'),
    get('plans.json'),
    get('mask.json'),
  ]);
  return { facts, models, browserConfig, schema, views, plans, mask };
}

/**
 * Plans the engine recorded (fixtures/plans.json, from scripts/next-schema-fixtures.mjs).
 * @typedef {object} RecordedPlans
 * @property {ScanFacts} scan
 * @property {Record<string, { stage: any, label: string }>} steps - each step name's stage and label
 * @property {({ name?: string, edits: [string, unknown][], scan?: Partial<ScanFacts>, steps: ([string] | [string, string[]])[] }
 *   & Omit<Plan, 'steps'>)[]} plans - plans[0] is the browser profile's
 */

/**
 * A planner over recorded plans (TEMPORARY, until the workspace plans through the bindings in
 * U2): the engine's plan for a recorded config, else the browser profile's plan with the steps
 * each changed choice adds or removes on its own.
 * @param {RecordedPlans} recorded
 * @param {Record<string, any>} defaults - the browser profile
 * @returns {(config: Record<string, any>, facts: ScanFacts) => Plan}
 */
export function recordedPlanner(recorded, defaults) {
  const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
  const get = (o, p) => p.split('.').reduce((v, k) => (v == null ? undefined : v[k]), o);
  const paths = [...new Set(recorded.plans.flatMap((p) => p.edits.map(([path]) => path)))];
  const key = (edits) => edits.map(([p, v]) => `${p}=${JSON.stringify(v ?? null)}`).sort().join('&');
  /** @param {RecordedPlans['plans'][number]} p @returns {Plan} */
  const expand = (p) => {
    const { steps, models, peak_memory_bytes, check, notes, two_pass, motion_corrected } = p;
    return { models, peak_memory_bytes, check, notes, two_pass, motion_corrected, steps: steps.map(([name, models = []]) => ({ ...recorded.steps[name], name, models, host: false })) };
  };
  const base = expand(recorded.plans[0]);
  return (config, facts) => {
    const changes = paths.map((p) => /** @type {[string, unknown]} */ ([p, get(config, p) ?? null])).filter(([p, v]) => !same(v, get(defaults, p)));
    const scanDiff = (p) => Object.entries(p.scan ?? {}).every(([k, v]) => facts?.[k] === v);
    const noMagnitude = facts && facts.has_magnitude === false;
    const exact = recorded.plans.find((p) => key(p.edits) === key(changes) && scanDiff(p) && (!noMagnitude || p.scan?.has_magnitude === false));
    if (exact) return expand(exact);
    const singles = changes.map((c) => recorded.plans.find((p) => p.edits.length === 1 && !p.scan && key(p.edits) === key([c]))).filter(Boolean);
    let steps = [...base.steps];
    const result = { ...base, check: { ...base.check, errors: [...base.check.errors], warnings: [...base.check.warnings], problems: [...(base.check.problems ?? [])] }, notes: [...base.notes] };
    for (const single of singles) {
      const s = expand(/** @type {any} */ (single));
      const names = new Set(s.steps.map((x) => x.name));
      const before = new Set(base.steps.map((x) => x.name));
      steps = steps.filter((x) => !before.has(x.name) || names.has(x.name));
      s.steps.forEach((x, i) => {
        const at = steps.findIndex((y) => y.name === x.name);
        if (at >= 0) { if (x.models.length) steps[at] = x; return; }
        if (before.has(x.name)) return; // a step this choice keeps, which another one removed
        // After the step it follows in its own plan, and after steps other choices added there.
        const prev = s.steps.slice(0, i).reverse().find((y) => steps.some((z) => z.name === y.name));
        let k = prev ? steps.findIndex((z) => z.name === prev.name) + 1 : 0;
        while (k < steps.length && !before.has(steps[k].name) && !names.has(steps[k].name)) k++;
        steps.splice(k, 0, x);
      });
      result.peak_memory_bytes = Math.max(result.peak_memory_bytes, s.peak_memory_bytes);
      result.two_pass ||= s.two_pass;
      result.motion_corrected ||= s.motion_corrected;
      for (const e of s.check.errors) if (!result.check.errors.includes(e)) result.check.errors.push(e);
      for (const w of s.check.warnings) if (!result.check.warnings.includes(w)) result.check.warnings.push(w);
      for (const p of s.check.problems ?? []) if (!result.check.problems?.some((q) => same(q, p))) result.check.problems?.push(p);
      for (const n of s.notes) if (!result.notes.includes(n)) result.notes.push(n);
    }
    result.steps = steps;
    result.models = [...new Set(steps.flatMap((x) => x.models))];
    return result;
  };
}

/**
 * The run grid of a scan on a 1 mm grid centred on the origin.
 * @param {[number, number, number]} dims
 * @param {ScanFacts} facts
 * @returns {RunMetadata}
 */
export function centredGrid(dims, facts) {
  const [nx, ny, nz] = dims;
  return {
    dims,
    voxel_size: [1, 1, 1],
    affine: [1, 0, 0, -nx / 2, 0, 1, 0, -ny / 2, 0, 0, 1, -nz / 2, 0, 0, 0, 1],
    n_echoes: facts.n_echoes,
    echo_times: facts.echo_times,
    b0_direction: facts.b0_dir ?? [0, 0, 1],
    field_strength: facts.field_strength,
    has_magnitude: facts.has_magnitude,
  };
}

/**
 * A synthetic head on the example's grid, so the viewer has something to show without a
 * download: a magnitude-like image and a χ-like map (ppm), as float32, plus its mask.
 * @param {ScanFacts} facts
 * @param {VolumeStore} store
 * @returns {string[]} the keys it stored
 */
export function putPhantom(facts, store) {
  const dims = facts.dims;
  const [nx, ny, nz] = dims;
  const n = nx * ny * nz;
  const magnitude = new Float32Array(store.allocate(n * 4));
  const chi = new Float32Array(store.allocate(n * 4));
  const mask = new Uint8Array(store.allocate(n));
  // Ellipsoids as fractions of the field of view: [cx, cy, cz, rx, ry, rz, magnitude, chi].
  const blobs = [
    [0, 0.05, 0, 0.30, 0.36, 0.34, 0.75, 0],  // brain
    [-0.08, 0.02, 0, 0.05, 0.11, 0.08, 0.95, -0.03], // ventricles
    [0.08, 0.02, 0, 0.05, 0.11, 0.08, 0.95, -0.03],
    [-0.11, -0.04, -0.06, 0.04, 0.05, 0.05, 0.45, 0.12], // deep grey nuclei
    [0.11, -0.04, -0.06, 0.04, 0.05, 0.05, 0.45, 0.12],
    [0, -0.18, -0.12, 0.03, 0.03, 0.04, 0.35, 0.09],
  ];
  let i = 0;
  for (let z = 0; z < nz; z++) {
    const fz = z / nz - 0.5;
    for (let y = 0; y < ny; y++) {
      const fy = y / ny - 0.5;
      for (let x = 0; x < nx; x++, i++) {
        const fx = x / nx - 0.5;
        const head = (fx / 0.36) ** 2 + (fy / 0.43) ** 2 + (fz / 0.40) ** 2;
        if (head > 1) continue;
        let m = head > 0.8 ? 0.35 : 0.2; // scalp and skull
        let c = 0;
        for (const [cx, cy, cz, rx, ry, rz, bm, bc] of blobs) {
          if (((fx - cx) / rx) ** 2 + ((fy - cy) / ry) ** 2 + ((fz - cz) / rz) ** 2 <= 1) {
            m = bm;
            c = bc;
            mask[i] = 1;
          }
        }
        // Some smooth texture, so slices are not flat.
        const texture = mask[i] ? Math.sin(14 * fx + 3 * fz) * Math.sin(11 * fy) * Math.cos(9 * fz) : 0;
        magnitude[i] = 1000 * (m + 0.05 * texture);
        chi[i] = mask[i] ? c + 0.02 * texture : 0;
      }
    }
  }
  const grid = centredGrid(dims, facts);
  store.adopt('CombinedMagnitude', magnitude, { grid, step: 'fixture' });
  store.adopt('BrainMask', mask, { grid, step: 'fixture' });
  store.adopt('Chi', chi, { grid, step: 'fixture' });
  return ['CombinedMagnitude', 'BrainMask', 'Chi'];
}

/**
 * Download the example's first echo (magnitude and phase) and keep both as inputs: the file, as
 * the engine reads it, and float32 voxels for the viewer.
 * @param {VolumeStore} store
 * @param {(text: string) => void} [onProgress]
 * @returns {Promise<string[]>} the keys it stored
 */
export async function putExampleEcho(store, onProgress = () => {}) {
  const parts = [
    ['AcquiredMagnitude(1)', 'sub-1_echo-1_part-mag_MEGRE.nii.gz'],
    ['AcquiredPhase(1)', 'sub-1_echo-1_part-phase_MEGRE.nii.gz'],
  ];
  // Shared with today's UI; imported by URL so the type check (js/next only) leaves it out.
  const { EXAMPLE_DATA } = await import(new URL('../app/config.js', import.meta.url).href);
  const keys = [];
  for (const [key, file] of parts) {
    onProgress(`Downloading ${file}`);
    const res = await fetch(`${EXAMPLE_DATA.baseUrl}/${file}`);
    if (!res.ok) throw new Error(`${file}: HTTP ${res.status}`);
    const bytes = new Uint8Array(await res.arrayBuffer());
    const vol = await readNifti(bytes);
    const decoded = new Float32Array(store.allocate(vol.data.byteLength));
    decoded.set(vol.data);
    /** @type {RunMetadata} */
    const grid = {
      dims: vol.dims,
      voxel_size: vol.voxelSize,
      affine: vol.affine,
      n_echoes: 1,
      echo_times: [],
      b0_direction: [0, 0, 1],
      field_strength: 0,
      has_magnitude: true,
    };
    store.putInput(key, bytes, { decoded, grid });
    keys.push(key);
  }
  onProgress('');
  return keys;
}
