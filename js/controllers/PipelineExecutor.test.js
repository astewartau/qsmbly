/**
 * Cancelling is a hard `worker.terminate()`, so a job waiting on a worker message never gets a
 * reply. These cover the registry that lets such a job settle instead of hanging the UI.
 */

import { jest } from '@jest/globals';
import { PipelineExecutor } from './PipelineExecutor.js';

function makeExecutor() {
  const ex = new PipelineExecutor({ updateOutput: () => {}, setProgress: () => {} });
  ex.worker = { terminate: () => { ex.worker.terminated = true; }, terminated: false };
  ex.pipelineRunning = true;
  return ex;
}

describe('PipelineExecutor cancellation', () => {
  test('cancel runs registered handlers and terminates the worker', () => {
    const ex = makeExecutor();
    const worker = ex.worker;
    let called = 0;
    ex.onCancel(() => called++);

    ex.cancel();

    expect(called).toBe(1);
    expect(worker.terminated).toBe(true);
    expect(ex.worker).toBeNull();
    expect(ex.pipelineRunning).toBe(false);
    expect(ex.cancelHandlers.size).toBe(0);
  });

  test('a settled job unregisters, so a later cancel does not touch it', () => {
    const ex = makeExecutor();
    let called = 0;
    const unregister = ex.onCancel(() => called++);

    unregister();
    ex.cancel();

    expect(called).toBe(0);
  });

  test('one failing handler does not stop the others or the terminate', () => {
    const ex = makeExecutor();
    const worker = ex.worker;
    const spy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    let second = 0;
    ex.onCancel(() => { throw new Error('boom'); });
    ex.onCancel(() => second++);

    ex.cancel();

    expect(second).toBe(1);
    expect(worker.terminated).toBe(true);
    spy.mockRestore();
  });

  test('cancel is inert when nothing is running', () => {
    const ex = makeExecutor();
    ex.pipelineRunning = false;
    const worker = ex.worker;
    let called = 0;
    ex.onCancel(() => called++);

    ex.cancel();

    expect(called).toBe(0);
    expect(worker.terminated).toBe(false);
  });
});
