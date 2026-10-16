/**
 * Helpers for the Pipeline and Command tabs: a provider's answer kept until the next one arrives
 * (so a change never blanks the tab), copying to the clipboard and saving a file.
 */
import { useEffect, useState } from 'preact/hooks';

/**
 * The latest answer of `ask()`, asked again whenever `deps` change. The previous answer stays
 * until the new one arrives; an answer to an older question is dropped.
 * @template T
 * @param {() => Promise<T>} ask
 * @param {unknown[]} deps
 * @returns {{ value: T | null, error: string | null }}
 */
export function useProvided(ask, deps) {
  const [state, setState] = useState(/** @type {{ value: T | null, error: string | null }} */ ({ value: null, error: null }));
  useEffect(() => {
    let current = true;
    ask().then(
      (value) => { if (current) setState({ value, error: null }); },
      (e) => { if (current) setState((s) => ({ value: s.value, error: e instanceof Error ? e.message : String(e) })); },
    );
    return () => { current = false; };
  }, deps);
  return state;
}

/**
 * @param {string} text
 * @returns {Promise<boolean>} whether it was copied
 */
export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

/**
 * Save `text` as a file through the browser's download.
 * @param {string} name
 * @param {string} text
 * @param {string} type
 */
export function saveFile(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * A copy button's label: "Copy", then "Copied" (or "Could not copy") for a moment.
 * @param {string} idle
 * @returns {[string, (text: string) => Promise<void>]}
 */
export function useCopy(idle = 'Copy') {
  const [label, setLabel] = useState(idle);
  useEffect(() => {
    if (label === idle) return undefined;
    const t = setTimeout(() => setLabel(idle), 1500);
    return () => clearTimeout(t);
  }, [label]);
  return [label, async (text) => setLabel((await copyText(text)) ? 'Copied' : 'Could not copy')];
}
