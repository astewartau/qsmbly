/**
 * The engine worker of the new workspace: answers what the page asks QSMxT's engine (ops.js)
 * off the main thread, so planning and the settings view never block typing.
 *
 * It is a worker of its own, not the compute worker's: a run holds the compute worker for minutes
 * (`Engine.run` is synchronous) and is cancelled by terminating it, while the settings view and
 * the plan must keep answering during a run. It loads only the base bundle and starts no thread
 * pool: these answers are milliseconds of work.
 *
 * Protocol: `{ id, op, args }` in, `{ id, ok: true, result }` or `{ id, ok: false, error }` out.
 * `{ id, op: 'ready' }` answers once the bundle is loaded, with its QSMxT and QSM.rs versions.
 */
import { engineOps } from './ops.js';

/** The site root (where wasm/ is), from this file's place in js/next/engine/. */
const root = new URL('../../../', import.meta.url);
/** Cache-busts the bundle by the page's version (`?v=` on the worker's own URL), if it has one. */
const version = new URL(import.meta.url).searchParams.get('v');
const asset = (name) => {
  const url = new URL(`wasm/${name}`, root);
  if (version) url.searchParams.set('v', version);
  return url.href;
};

const ready = (async () => {
  const wasm = await import(asset('qsm_wasm.js'));
  await wasm.default({ module_or_path: asset('qsm_wasm_bg.wasm') });
  return { wasm, ops: engineOps(wasm) };
})();

self.onmessage = async (/** @type {MessageEvent} */ event) => {
  const { id, op, args = [] } = event.data ?? {};
  try {
    const { wasm, ops } = await ready;
    let result;
    if (op === 'ready') result = JSON.parse(wasm.get_dependency_versions_wasm());
    else if (Object.hasOwn(ops, op)) result = ops[op](...args);
    else throw new Error(`no engine op ${op}`);
    self.postMessage({ id, ok: true, result });
  } catch (e) {
    self.postMessage({ id, ok: false, error: e instanceof Error ? e.message : String(e) });
  }
};
