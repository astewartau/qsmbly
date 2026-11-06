import { describeFailure, withEngine, reporting } from './engine-call.js';
import { WASM_HEAP_BYTES as HEAP_LIMIT } from '../protocol.js';

/**
 * An engine like the bindings': `free()` throws while a call holds it borrowed, as wasm-bindgen's
 * does after a call that trapped and so never released its borrow.
 * @param {() => any} run
 */
function fakeEngine(run) {
  let borrowed = false;
  let freed = 0;
  return {
    run() {
      borrowed = true;
      const out = run();
      borrowed = false;
      return out;
    },
    free() {
      if (borrowed) throw new Error('attempted to take ownership of Rust value while it was borrowed');
      freed += 1;
    },
    get freed() {
      return freed;
    },
  };
}

const memory = (bytes) => ({ buffer: { byteLength: bytes } });

describe('calling the engine', () => {
  test('a run that returns is freed and its result returned', () => {
    const engine = fakeEngine(() => '{"steps":{}}');
    expect(withEngine(engine, (e) => e.run())).toBe('{"steps":{}}');
    expect(engine.freed).toBe(1);
  });

  test('a run that traps reports the panic, not the borrowed engine free() then throws about', () => {
    const engine = fakeEngine(() => {
      throw new WebAssembly.RuntimeError('unreachable');
    });
    const mod = { take_panic_wasm: () => 'index out of bounds: the len is 3 but the index is 7 (at src/segment/synthseg.rs:140)' };
    let thrown;
    try {
      withEngine(engine, (e) => e.run(), { mod, memory: memory(1e9), step: 'segmentation' });
    } catch (e) {
      thrown = e;
    }
    expect(thrown).toBeInstanceOf(Error);
    expect(thrown.message).toBe('The engine panicked: index out of bounds: the len is 3 but the index is 7 (at src/segment/synthseg.rs:140)');
    expect(thrown.message).not.toMatch(/ownership/);
    expect(thrown).toMatchObject({ kind: 'panic', step: 'segmentation', fatal: true });
    expect(engine.freed).toBe(0);
  });

  test('a trap with no panic names a heap at the browser\'s limit as running out of memory', () => {
    const engine = fakeEngine(() => {
      throw new WebAssembly.RuntimeError('unreachable');
    });
    expect(() => withEngine(engine, (e) => e.run(), { mod: { take_panic_wasm: () => undefined }, memory: memory(HEAP_LIMIT - 1024) }))
      .toThrow(/^The engine ran out of memory: its heap had grown to 4\.0 GB of the 4\.0 GB a browser allows \(unreachable\)$/);
  });

  test('the engine\'s own error is reported as it is, with its kind and step, and the engine is freed', () => {
    const engine = fakeEngine(() => null);
    // The bindings release the borrow when a call returns an error.
    engine.run = () => {
      throw Object.assign(new Error('the local field is empty'), { kind: 'invalid_input', step: 'invert' });
    };
    expect(() => withEngine(engine, (e) => e.run())).toThrow(expect.objectContaining({ message: 'the local field is empty', kind: 'invalid_input', step: 'invert', fatal: false }));
    expect(engine.freed).toBe(1);
  });

  test('a panic message that cannot be read leaves the trap\'s own', () => {
    const f = describeFailure(new WebAssembly.RuntimeError('unreachable'), { mod: { take_panic_wasm: () => { throw new Error('dead'); } }, memory: memory(2e9) });
    expect(f).toEqual({ message: 'The engine stopped (unreachable) with its heap at 1.9 GB', kind: 'abort', step: null, fatal: true });
  });

  test('a trap with the heap unknown says only that the engine stopped', () => {
    expect(describeFailure(new WebAssembly.RuntimeError('unreachable')).message).toBe('The engine stopped (unreachable)');
  });

  test('a callback that throws is reported, and the run is told nothing', () => {
    const reported = [];
    const observer = reporting({
      stage() {
        throw new Error('DataCloneError: could not be cloned');
      },
      progress: (done) => done * 2,
    }, (method, e) => reported.push([method, e.message]));
    expect(observer.stage('mask', null, 'started', 'Creating mask')).toBeUndefined();
    expect(observer.progress(3)).toBe(6);
    expect(reported).toEqual([['stage', 'DataCloneError: could not be cloned']]);
  });
});
