/**
 * @jest-environment jsdom
 */
import { handleUndoKeys, installUndoKeys } from './shortcuts.js';

function spy() {
  const calls = [];
  return { calls, undo: () => calls.push('undo'), redo: () => calls.push('redo') };
}
const key = (target, init) => {
  const e = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ...init });
  target.dispatchEvent(e);
  return e;
};

describe('undo and redo shortcuts', () => {
  test('Ctrl+Z undoes, Ctrl+Shift+Z and Ctrl+Y redo, ⌘Z too', () => {
    const s = spy();
    const stop = installUndoKeys(window, s);
    expect(key(document.body, { key: 'z', ctrlKey: true }).defaultPrevented).toBe(true);
    key(document.body, { key: 'Z', ctrlKey: true, shiftKey: true });
    key(document.body, { key: 'y', ctrlKey: true });
    key(document.body, { key: 'z', metaKey: true });
    key(document.body, { key: 'z' });
    expect(s.calls).toEqual(['undo', 'redo', 'redo', 'undo']);
    stop();
    key(document.body, { key: 'z', ctrlKey: true });
    expect(s.calls).toHaveLength(4);
  });

  test('a field with its value committed undoes the settings; one mid-edit keeps its own undo', () => {
    const s = spy();
    const input = document.createElement('input');
    input.type = 'text';
    input.dataset.committed = '12';
    input.value = '12';
    document.body.append(input);
    const stop = installUndoKeys(window, s);
    key(input, { key: 'z', ctrlKey: true });
    expect(s.calls).toEqual(['undo']);
    input.value = '15';
    const e = key(input, { key: 'z', ctrlKey: true });
    expect(e.defaultPrevented).toBe(false);
    expect(s.calls).toEqual(['undo']);
    stop();
    input.remove();
  });

  test('a field with no committed value (search) keeps its own undo; a checkbox does not', () => {
    const s = spy();
    const search = Object.assign(document.createElement('input'), { type: 'search' });
    const box = Object.assign(document.createElement('input'), { type: 'checkbox' });
    expect(handleUndoKeys(/** @type {any} */ ({ key: 'z', ctrlKey: true, target: search, preventDefault() {} }), s)).toBe(false);
    expect(handleUndoKeys(/** @type {any} */ ({ key: 'z', ctrlKey: true, target: box, preventDefault() {} }), s)).toBe(true);
    expect(s.calls).toEqual(['undo']);
  });
});
