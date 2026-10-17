/**
 * What the engine's items are (qsmxt-engine `Item::meta()`): their names, units and kinds, for
 * the viewer and the outputs. Nothing here names an item: the plan brings the meta of every item
 * it touches (`plan.items`), and any other key is asked of the engine once.
 *
 * @typedef {import('./protocol.js').ItemKey} ItemKey
 * @typedef {import('./protocol.js').ItemMeta} ItemMeta
 */
import { signal } from '@preact/signals-core';

/** @type {Map<ItemKey, ItemMeta>} */
const known = new Map();
/** Bumped when an item's meta arrives, so what showed its key re-renders with its name. */
export const itemsVersion = signal(0);
/** @type {((key: ItemKey) => Promise<ItemMeta | null>) | null} */
let lookup = null;
/** @type {Set<ItemKey>} */
const asked = new Set();

/**
 * Where meta the workspace has not seen comes from (the engine client's `itemMeta`).
 * @param {((key: ItemKey) => Promise<ItemMeta | null>) | null} fn
 */
export function setItemLookup(fn) {
  lookup = fn;
  asked.clear();
}

/**
 * Keep the meta of these items (a plan's `items`).
 * @param {ItemMeta[] | undefined} items
 */
export function learnItems(items) {
  let added = false;
  for (const meta of items ?? []) {
    if (!known.has(meta.key)) added = true;
    known.set(meta.key, meta);
  }
  if (added) itemsVersion.value++;
}

/**
 * An item's meta, with its echo and coil in its label (`Phase (as acquired), echo 2`). Until the
 * engine has said what a key is, its key stands in.
 * @param {ItemKey} key
 * @returns {ItemMeta}
 */
export function itemMeta(key) {
  itemsVersion.value;
  const meta = known.get(key);
  if (!meta) {
    if (lookup && !asked.has(key)) {
      asked.add(key);
      lookup(key).then((m) => { if (m) learnItems([m]); }, () => {});
    }
    return { key, label: key, units: null, kind: 'other' };
  }
  const parts = [meta.label];
  if (meta.echo != null) parts.push(`echo ${meta.echo}`);
  if (meta.coil != null) parts.push(`coil ${meta.coil}`);
  return { ...meta, label: parts.join(', ') };
}
