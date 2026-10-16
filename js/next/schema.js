/**
 * The engine's settings schema (qsmxt-engine `schema::schema_json()`), and how the inspector
 * turns a setting into a control. Every label, help text, option, range and default comes from
 * the schema; nothing here names a setting. Whether a value is in range is the engine's to say
 * (`settings_view`'s `error`); the checks here only parse what was typed.
 *
 * @typedef {Record<string, any>} PipelineConfig
 *
 * @typedef {object} SchemaOption
 * @property {unknown} value - as written in the config
 * @property {string} label - the TUI's (usually the `qsmxt run` value)
 * @property {string} name - its display name (`iLSQR`)
 * @property {string} help
 * @property {string} [url]
 * @property {'named' | 'echo-number'} [kind] - an echo reference's: a named echo, or any echo by
 *   its number (the setting's value is then the number as a string)
 * @property {number} [min] - for `echo-number`: the first echo number
 *
 * @typedef {'bool' | 'int' | 'float' | 'int-vec3' | 'float-vec3' | 'text' | 'enum' | 'echo-reference' | 'mask-sections' | 'mask-ops'} SettingType
 *
 * @typedef {object} Setting - qsmxt-engine `schema::Setting`, as JSON
 * @property {string} path
 * @property {string} label - the TUI's short label
 * @property {string} full_label - the label in full, for a front end with room
 * @property {string} [context] - the method it belongs to, where the label alone could be another's
 * @property {string} help
 * @property {string} [value_label] - for a checkbox with a value (`off` set): the value's label
 * @property {string} [value_help]
 * @property {SettingType} type
 * @property {SchemaOption[]} [options] - enum and echo-reference
 * @property {boolean} optional - the value may be null
 * @property {unknown} [off] - the value that turns a checkbox-with-a-value setting off
 * @property {string} [units]
 * @property {number} [min]
 * @property {number} [max]
 * @property {boolean} [exclusive_min] - `min` itself is out of range (QSMxT #368)
 * @property {boolean} [exclusive_max] - `max` itself is out of range
 * @property {number} [step]
 * @property {unknown} default - QSMxT's
 * @property {unknown} browser_default
 * @property {'basic' | 'advanced'} level
 * @property {string} tab
 * @property {string} section
 * @property {string[]} [also_in]
 * @property {number} order
 * @property {string} [parent]
 * @property {string} [cli_flag]
 * @property {'available' | 'supplied-file' | 'unavailable'} browser
 * @property {string} [item] - a supplied-file setting's: the engine item the file fills
 * @property {string} [browser_help] - a supplied-file setting's help for its file input
 *
 * @typedef {object} Schema
 * @property {{ id: string, label: string }[]} tabs
 * @property {{ id: string, label: string, tab: string }[]} sections
 * @property {Setting[]} settings
 * @property {import('./mask.js').MaskSchema} mask - the mask recipe's parts
 */
import { getPath } from './store/session.js';

/** The derivatives "tool" a supplied file is named by (qsmxt-engine `BROWSER_SUPPLIED_TOOL`). */
export const SUPPLIED_TOOL = 'qsmbly';

const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

/**
 * The settings shown in a schema section (its own and those `also_in` it), in QSMxT's order.
 * @param {Schema} schema
 * @param {string} section
 * @returns {Setting[]}
 */
export function settingsIn(schema, section) {
  return schema.settings
    .filter((s) => s.section === section || s.also_in?.includes(section))
    .sort((a, b) => a.order - b.order);
}

/**
 * The setting's current value: the config's, with a missing optional value read as null.
 * @param {Setting} setting
 * @param {PipelineConfig} config
 */
export function valueOf(setting, config) {
  return getPath(config, setting.path) ?? null;
}

/**
 * Whether the value differs from the browser default (which is what "default" means here).
 * @param {Setting} setting
 * @param {unknown} value
 */
export function isChanged(setting, value) {
  return !same(value, setting.browser_default);
}

/**
 * Where the browser's default departs from QSMxT's, the two, for the badge.
 * @param {Setting} setting
 * @returns {{ native: string, browser: string } | null}
 */
export function browserDefaultNote(setting) {
  if (same(setting.default, setting.browser_default)) return null;
  return { native: formatValue(setting, setting.default), browser: formatValue(setting, setting.browser_default) };
}

/**
 * A value as the inspector writes it in text (badges, the changes list).
 * @param {Setting} setting
 * @param {unknown} value
 * @returns {string}
 */
export function formatValue(setting, value) {
  if (setting.off !== undefined && same(value, setting.off)) return 'off';
  if (value === null || value === undefined) return 'unset';
  if (setting.type === 'bool') return value ? 'on' : 'off';
  if (setting.type === 'enum' || setting.type === 'echo-reference') {
    const { named, echo } = echoChoice(setting, value);
    if (echo !== null) return `echo ${echo}`;
    const option = (setting.type === 'enum' ? setting.options ?? [] : named).find((o) => same(o.value, value));
    if (option) return option.name;
  }
  if (Array.isArray(value)) {
    if (setting.type === 'mask-sections' || setting.type === 'mask-ops') return `${value.length} step${value.length === 1 ? '' : 's'}`;
    return value.join(' × ');
  }
  if (value === '') return 'empty';
  return `${value}${setting.units && typeof value === 'number' ? ` ${setting.units}` : ''}`;
}

/**
 * @typedef {object} Control
 * @property {'checkbox' | 'number' | 'vec3' | 'select' | 'echo' | 'text' | 'file' | 'recipe' | 'unavailable'} kind
 * @property {boolean} toggled - shown as a checkbox with a value under it (`off` set)
 * @property {boolean} nullable - the value may be left unset (an empty field)
 * @property {boolean} integer
 * @property {{ min?: number, max?: number, step?: number | 'any' }} [range] - input constraints
 */

/**
 * How a setting is edited.
 * @param {Setting} setting
 * @returns {Control}
 */
export function controlFor(setting) {
  const toggled = setting.off !== undefined;
  const integer = setting.type === 'int' || setting.type === 'int-vec3';
  const base = { toggled, nullable: setting.optional && !toggled, integer };
  /** @type {Control['range']} */
  const range = { min: setting.min, max: setting.max, step: setting.step ?? (integer ? 1 : 'any') };
  if (setting.browser === 'unavailable') return { ...base, kind: 'unavailable' };
  if (setting.browser === 'supplied-file') return { ...base, kind: 'file' };
  switch (setting.type) {
    case 'bool': return { ...base, kind: 'checkbox' };
    case 'int':
    case 'float': return { ...base, kind: 'number', range };
    case 'int-vec3':
    case 'float-vec3': return { ...base, kind: 'vec3', range };
    case 'enum': return { ...base, kind: 'select' };
    case 'echo-reference': return { ...base, kind: 'echo', range: { min: setting.options?.find((o) => o.kind === 'echo-number')?.min, step: 1 } };
    case 'mask-sections':
    case 'mask-ops': return { ...base, kind: 'recipe' };
    default: return { ...base, kind: 'text' };
  }
}

/**
 * Why `value` cannot be a value of the setting at all, or null: the wrong type, not a number, not
 * a whole number, not one of the choices. Such a value is not written to the config. Whether a
 * value is in range, or allowed by the setting's rules, is the engine's to say: it is written, and
 * `settings_view` reports it (`error`).
 * @param {Setting} setting
 * @param {unknown} value
 * @returns {string | null}
 */
export function validate(setting, value) {
  if (value === null || value === undefined) {
    return setting.optional || (setting.off === null) ? null : 'A value is needed.';
  }
  if (setting.off !== undefined && same(value, setting.off)) return null;
  const number = (v) => {
    if (typeof v !== 'number' || !Number.isFinite(v)) return 'Enter a number.';
    if ((setting.type === 'int' || setting.type === 'int-vec3') && !Number.isInteger(v)) return 'Enter a whole number.';
    return null;
  };
  switch (setting.type) {
    case 'bool': return typeof value === 'boolean' ? null : 'Must be on or off.';
    case 'int':
    case 'float': return number(value);
    case 'int-vec3':
    case 'float-vec3':
      if (!Array.isArray(value) || value.length !== 3) return 'Enter three values.';
      for (const v of value) {
        const e = number(v);
        if (e) return e;
      }
      return null;
    case 'enum': return setting.options?.some((o) => same(o.value, value)) ? null : 'Not one of the choices.';
    case 'echo-reference': {
      const { named, numbered, echo } = echoChoice(setting, value);
      if (named.some((o) => o.value === value)) return null;
      return numbered && echo !== null && Number.isInteger(echo) ? null : 'Enter an echo number.';
    }
    case 'text': return typeof value === 'string' ? null : 'Enter text.';
    default: return null;
  }
}

/**
 * The value a field's text stands for. An empty field is unset (null) where the setting may be.
 * @param {Setting} setting
 * @param {string} text
 * @returns {{ value: unknown, error: string | null }}
 */
export function parseField(setting, text) {
  const t = text.trim();
  if (setting.type === 'text') {
    const value = t === '' && setting.optional ? null : text;
    return { value, error: validate(setting, value) };
  }
  if (t === '') {
    const value = setting.optional ? null : undefined;
    return { value, error: value === undefined ? 'A value is needed.' : validate(setting, value) };
  }
  const value = Number(t);
  return { value, error: validate(setting, value) };
}

/**
 * The value a checkbox-with-a-value setting takes when it is switched on: its default, or failing
 * that its other default, or the smallest value it allows.
 * @param {Setting} setting
 */
export function onValue(setting) {
  for (const v of [setting.browser_default, setting.default]) {
    if (!same(v, setting.off) && v !== null && v !== undefined) return v;
  }
  if (setting.type === 'text') return '';
  return setting.min !== undefined && setting.min > 0 ? setting.min : 1;
}

/**
 * The echo-reference choices: the named ones, and the echo-number one, which stands for any echo
 * by its number. A value that is no named choice is an echo number, written as a string.
 * @param {Setting} setting
 * @param {unknown} value
 * @returns {{ named: SchemaOption[], numbered: SchemaOption | undefined, echo: number | null }}
 */
export function echoChoice(setting, value) {
  const options = setting.options ?? [];
  const named = options.filter((o) => o.kind !== 'echo-number');
  const numbered = options.find((o) => o.kind === 'echo-number');
  const isNamed = named.some((o) => o.value === value);
  const echo = !isNamed && numbered && typeof value === 'string' && value.trim() !== '' && Number.isFinite(Number(value)) ? Number(value) : null;
  return { named, numbered, echo };
}

/**
 * Settings matching a search, by label, method, help, option, `qsmxt run` flag or config path.
 * @param {Schema} schema
 * @param {string} query
 * @returns {Setting[]}
 */
export function searchSettings(schema, query) {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return [];
  return schema.settings.filter((s) => {
    const text = [s.label, s.full_label, s.context, s.help, s.value_label, s.value_help, s.cli_flag, s.path,
      ...(s.options ?? []).flatMap((o) => [o.label, o.name, o.help])].filter(Boolean).join(' ').toLowerCase();
    return words.every((w) => text.includes(w));
  });
}
