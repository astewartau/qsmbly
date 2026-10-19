/**
 * Calling QSMxT's engine (rust-wasm/src/engine.rs `Engine`) so that the page is told what went
 * wrong, not what went wrong afterwards.
 *
 * - A failed run (the engine's own error) throws an `Error` with its `kind` and `step`; that is
 *   reported as it is.
 * - A call that dies inside the WASM module (a Rust panic, which aborts: the build is
 *   `panic=abort`; or memory the heap cannot grow to) throws a `WebAssembly.RuntimeError`
 *   ("unreachable"). The panic's own message is asked of the module (`take_panic_wasm`), and a
 *   heap at the browser's limit is named. Such a call never returns, so the engine is left
 *   borrowed: freeing it then throws "attempted to take ownership of Rust value while it was
 *   borrowed", which must not replace the error. The module's state is not to be trusted after
 *   it, so the failure is `fatal`: the page starts a fresh worker for the next job.
 * - The page's callbacks (the observer's `stage`, `progress`, `log`, `output`) are called from
 *   inside the run, where a throw is dropped by the bindings; {@link reporting} reports it instead.
 */

import { WASM_HEAP_BYTES as HEAP_LIMIT } from '../protocol.js';
import { formatBytes } from '../format.js';

/** A heap within this of {@link HEAP_LIMIT} is taken to have run out. */
const NEAR_LIMIT = 256 * 1024 ** 2;

/**
 * @typedef {{ message: string, kind: string | null, step: string | null, fatal: boolean }} Failure
 * @typedef {{ take_panic_wasm?: () => string | undefined }} PanicSource
 */

/**
 * What a thrown value says, as the page reports it.
 * @param {unknown} e
 * @param {{ mod?: PanicSource | null, memory?: { buffer: { byteLength: number } } | null, step?: string | null }} [context]
 *   the module the call was into (for its panic message), its memory (for its size), and the
 *   step under way (for a failure that names none)
 * @returns {Failure}
 */
export function describeFailure(e, { mod = null, memory = null, step = null } = {}) {
  const err = /** @type {any} */ (e);
  const trapped = typeof WebAssembly !== 'undefined' && e instanceof WebAssembly.RuntimeError;
  if (!trapped) {
    return {
      message: err instanceof Error ? err.message : String(err),
      kind: err?.kind ?? null,
      step: err?.step ?? step,
      fatal: false,
    };
  }
  let panic;
  try {
    panic = mod?.take_panic_wasm?.();
  } catch {
    panic = undefined;
  }
  const heap = memory?.buffer.byteLength ?? null;
  let message;
  if (panic) message = `The engine panicked: ${panic}`;
  else if (heap === null) message = `The engine stopped (${err.message})`;
  else if (heap >= HEAP_LIMIT - NEAR_LIMIT) {
    message = `The engine ran out of memory: its heap had grown to ${formatBytes(heap)} of the ${formatBytes(HEAP_LIMIT)} a browser allows (${err.message})`;
  } else message = `The engine stopped (${err.message}) with its heap at ${formatBytes(heap)}`;
  return { message, kind: panic ? 'panic' : 'abort', step, fatal: true };
}

/**
 * Call `use(engine)`, then free the engine. A call that fails is reported as it failed
 * ({@link describeFailure}, thrown as an `Error` carrying `kind`, `step` and `fatal`); freeing an
 * engine the failure left borrowed is not tried twice and never replaces it.
 * @template T
 * @param {{ free: () => void }} engine
 * @param {(engine: any) => T} use
 * @param {Parameters<typeof describeFailure>[1]} [context]
 * @returns {T}
 */
export function withEngine(engine, use, context = {}) {
  let result;
  try {
    result = use(engine);
  } catch (e) {
    const failure = describeFailure(e, context);
    // After a trap the engine is still borrowed and free() throws; after an engine error it
    // frees as usual. Either way, the failure is what is reported.
    try {
      engine.free();
    } catch {
      // the engine stays allocated; the worker is replaced (fatal) or the leak is one object
    }
    throw Object.assign(new Error(failure.message), { kind: failure.kind, step: failure.step, fatal: failure.fatal });
  }
  engine.free();
  return result;
}

/**
 * `observer` with each method reporting what it throws through `report` instead of throwing it:
 * the bindings drop a throw from a callback, so a broken callback would otherwise go unseen.
 * @template {Record<string, (...args: any[]) => any>} O
 * @param {O} observer
 * @param {(method: string, error: unknown) => void} report
 * @returns {O}
 */
export function reporting(observer, report) {
  /** @type {Record<string, (...args: any[]) => any>} */
  const out = {};
  for (const [name, fn] of Object.entries(observer)) {
    out[name] = (...args) => {
      try {
        return fn(...args);
      } catch (e) {
        try {
          report(name, e);
        } catch {
          // nothing left to tell
        }
        return undefined;
      }
    };
  }
  return /** @type {O} */ (out);
}
