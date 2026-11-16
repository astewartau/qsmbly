/**
 * The deep-learning weights a plan needs (`plan.models`, QSM.rs's registry for their files, sizes
 * and licences): which are already in the browser's cache, and downloading the rest before a run.
 * The cache and the download are today's UI's (js/modules/ModelWeights.js), so a model fetched in
 * either interface is fetched once.
 *
 * @typedef {import('./protocol.js').ModelEntry} ModelEntry
 *
 * @typedef {object} ModelDownload
 * @property {string} id
 * @property {string} name
 * @property {number} bytes - all of its files
 * @property {number | null} missing - bytes not yet cached; null until the cache was asked
 * @property {string} license - '' where the registry gives none
 */

/** @returns {Promise<any>} ModelWeights.js, imported by URL so the type check leaves it out. */
const weights = () => import(new URL('../modules/ModelWeights.js', import.meta.url).href);

/** Where the weights are fetched from: the configured mirror, absolute ('' for the registry's URLs). */
export async function weightBase() {
  try {
    const { MODEL_WEIGHT_BASE_URL: base = '' } = await import(new URL('../app/config.js', import.meta.url).href);
    if (!base) return '';
    return /^https?:\/\//i.test(base) ? base : new URL(base, new URL('../../', import.meta.url)).href;
  } catch {
    return '';
  }
}

/**
 * The models of a plan, from the registry.
 * @param {string[]} ids
 * @param {ModelEntry[]} registry
 * @returns {ModelDownload[]}
 */
export function planModels(ids, registry) {
  return ids.map((id) => {
    const entry = registry.find((m) => m.id === id);
    const bytes = entry ? entry.files.reduce((t, f) => t + (Number(f.bytes) || 0), 0) : 0;
    return { id, name: entry?.name ?? id, bytes, missing: null, license: entry?.license ?? '' };
  });
}

/**
 * The bytes of each model not yet in the cache.
 * @param {ModelDownload[]} models
 * @param {ModelEntry[]} registry
 * @returns {Promise<ModelDownload[]>}
 */
export async function withCacheState(models, registry) {
  const { uncachedBytes } = await weights();
  return Promise.all(models.map(async (m) => {
    const entry = registry.find((r) => r.id === m.id);
    try {
      return { ...m, missing: entry ? await uncachedBytes(entry) : 0 };
    } catch {
      return { ...m, missing: m.bytes }; // no cache here (private browsing): all of it
    }
  }));
}

/**
 * Download every model not cached yet into the cache.
 * @param {string[]} ids
 * @param {ModelEntry[]} registry
 * @param {(progress: { model: string, done: number, total: number }) => void} [onProgress] - bytes over all the models
 */
export async function downloadModels(ids, registry, onProgress = () => {}) {
  const { fetchModelWeights } = await weights();
  const base = await weightBase();
  const entries = ids.map((id) => {
    const entry = registry.find((m) => m.id === id);
    if (!entry) throw new Error(`${id} is not in the model registry`);
    return entry;
  });
  const total = entries.reduce((t, e) => t + e.files.reduce((s, f) => s + (Number(f.bytes) || 0), 0), 0);
  let before = 0;
  for (const entry of entries) {
    const sizes = entry.files.map((f) => Number(f.bytes) || 0);
    await fetchModelWeights(entry, (i, _name, done) => {
      const finished = sizes.slice(0, i).reduce((a, b) => a + b, 0);
      onProgress({ model: entry.name, done: before + finished + done, total });
    }, base);
    before += sizes.reduce((a, b) => a + b, 0);
  }
}
