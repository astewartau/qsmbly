/**
 * Keyboard shortcuts for the workspace: Ctrl+Z (⌘Z) undoes the last settings change, and
 * Ctrl+Shift+Z or Ctrl+Y redoes it, wherever the focus is. A text field with an edit not yet
 * committed keeps its own undo.
 *
 * @typedef {{ undo: () => void, redo: () => void }} Undoable
 */

/**
 * Whether the event's target is a text field holding uncommitted typing (its value differs from
 * the value it was rendered with, `data-committed`). Fields without that mark (search) always
 * keep their own undo.
 * @param {EventTarget | null} target
 */
function editingText(target) {
  if (!target || typeof target !== 'object' || !('tagName' in target)) return false;
  const el = /** @type {HTMLElement} */ (target);
  if (el.tagName === 'TEXTAREA' || el.isContentEditable) return true;
  if (el.tagName !== 'INPUT') return false;
  const input = /** @type {HTMLInputElement} */ (el);
  if (!['text', 'search', 'number', 'url', 'email', ''].includes(input.type)) return false;
  const committed = input.dataset.committed;
  return committed === undefined || input.value !== committed;
}

/**
 * Handle a keydown: undo or redo the session, unless a text field is mid-edit.
 * @param {KeyboardEvent} event
 * @param {Undoable} session
 * @returns {boolean} whether it was handled
 */
export function handleUndoKeys(event, session) {
  if (!(event.ctrlKey || event.metaKey) || event.altKey) return false;
  const key = event.key.toLowerCase();
  const redo = (key === 'z' && event.shiftKey) || (key === 'y' && !event.shiftKey);
  const undo = key === 'z' && !event.shiftKey;
  if (!undo && !redo) return false;
  if (editingText(event.target)) return false;
  event.preventDefault();
  if (redo) session.redo();
  else session.undo();
  return true;
}

/**
 * Listen for the shortcuts on a target (the window).
 * @param {EventTarget} target
 * @param {Undoable} session
 * @returns {() => void} stops listening
 */
export function installUndoKeys(target, session) {
  const listener = (e) => handleUndoKeys(/** @type {KeyboardEvent} */ (e), session);
  target.addEventListener('keydown', listener);
  return () => target.removeEventListener('keydown', listener);
}
