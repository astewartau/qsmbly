/**
 * The pipeline rules' findings in the workspace, as QSMxT's TUI shows them (src/tui/recommend.rs):
 * every finding, reference value, marker and method's reference pipeline is the engine's
 * (`settings_view`, and qsmxt-engine's `recommendations` functions, from QSM.rs's rules); nothing
 * here knows a rule or a method.
 *
 * - **Two kinds, labelled apart** (the engine's `kind`, QSMxT#486): a `reasoned` finding (required,
 *   recommended, discouraged) gives a reason why its value is better, and its change is
 *   preselected; a `reference` one is the original authors' setting, with no such reason, so Keep
 *   mine is preselected (QSMxT#501). Enter on a pop-up that holds only those changes nothing.
 * - **A method's reference pipeline** is offered when a change chooses a method that has one (the
 *   engine's `chosen_methods`, before and after the change) and the config differs from it
 *   (`reference_differences`): one pop-up, "MEDI's reference pipeline", with a row per setting
 *   (current → reference, why, source) and Apply the authors' settings / Choose per setting / Keep
 *   mine, Keep mine preselected. Nothing changes unless applied; Close (or Escape) changes nothing.
 *   Keep mine records nothing for an authors' setting, so it is offered again when the method is
 *   chosen again.
 * - **Findings chosen together** are settled by the engine (`settle`, QSMxT#498), as on every host:
 *   on each setting the strongest takes it, and two as strong that want different values leave it
 *   as it is, marked. Choose per setting settles only the rows given a value, each wanting the value
 *   picked for it: a row on Keep mine asks for nothing, so it cannot outrank a pick. A notice says
 *   what was left and why.
 * - **A recommendation** (a reasoned finding a change creates, not dismissed) is offered in the
 *   same pop-up, first; Apply recommended is then preselected, and keeps every authors' setting.
 *   Keep mine is kept in the config's `dismissed_recommendations`, so it is not offered again for
 *   the same values. A required one, or two that disagree, get a pop-up of their own (Apply
 *   preselected, the other allowed values, Keep mine unless required), and the reference
 *   pipeline's follows it. With no authors' settings to offer, each recommendation gets its own
 *   pop-up, one after another, as in the TUI.
 * - **A loaded scan** can bring findings of a method already chosen (a rule on the manufacturer or
 *   the field strength, say). What it adds that was not there before is offered as if the change
 *   had created it, recommendations and authors' settings alike, so what is offered does not
 *   depend on whether the scan or the method came first. Each is offered once while a scan is
 *   loaded: not again for a sidecar or a further file.
 * - **Markers**: a setting that departs from a rule shows the engine's marker, ⚑ when a finding is
 *   offered as a change and not kept, ⓘ otherwise, with whose rule and what it uses ("MEDI:
 *   BET"). Either opens the setting's details: every rule it departs from, offered ones first.
 * - **Dismissals** hold while the setting keeps the value they were made for: after each change
 *   the engine drops those that no longer do (`prune_dismissals`), so going back to a value, or
 *   choosing a method again, offers it again.
 *
 * @typedef {import('./protocol.js').Recommendation} Recommendation
 * @typedef {import('./protocol.js').SettingView} SettingView
 * @typedef {import('./protocol.js').Trigger} Trigger
 * @typedef {import('./protocol.js').ScanFacts} ScanFacts
 * @typedef {import('./store/session.js').Session} Session
 *
 * @typedef {{ kind: 'apply', rule: string, value?: unknown, label: string } | { kind: 'ignore', label: string }} Choice
 *
 * @typedef {object} RecModal - a recommendation's pop-up, or a setting's details
 * @property {'finding'} kind
 * @property {Recommendation[]} findings - one, or equally strong ones on one setting that disagree;
 *   for the details, every rule the setting departs from
 * @property {Choice[]} choices
 * @property {number | null} preselected - the choice offered first: the first finding's change for
 *   a recommendation, Keep mine for an authors' setting, none when recommendations disagree
 * @property {string} title
 * @property {boolean} details - a marked setting's details, not a pop-up a change opened
 * @property {string[]} [whose] - for the details, whose rule each finding is ("MEDI's reference
 *   pipeline uses PDF")
 *
 * @typedef {object} RefRow - one setting of a reference pipeline pop-up
 * @property {string} setting
 * @property {string} label
 * @property {'reasoned' | 'reference'} kind - a recommendation, or the original authors' setting
 * @property {string} current - its value now
 * @property {{ value: unknown, name: string }} reference - what the methods' rules use
 * @property {Recommendation[]} findings - each method's rule with that value (merged)
 * @property {string[]} methods - their methods ("MEDI", "PDF")
 * @property {{ value: unknown, name: string }[]} alternatives - other values the rules allow
 *
 * @typedef {object} RefModal - a chosen method's reference pipeline, and any recommendations the
 *   same change (or scan) created
 * @property {'reference'} kind
 * @property {string} title
 * @property {RefRow[]} recommendations - the reasoned rows, offered first
 * @property {RefRow[]} rows - the differences from the authors' settings
 *
 * @typedef {import('./protocol.js').Settled} Settled
 * @typedef {'recommended' | 'all' | 'keep' | { picks: RefPick[] }} RefAnswer
 * @typedef {'reference' | 'keep' | number} RefPick - a row's value, its current value, or the index
 *   of one of its alternatives
 */
import { signal, effect } from '@preact/signals-core';

const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

/**
 * A finding offered as a change and not dismissed: what opens the pop-up or shows the ⚑ badge.
 * @param {Recommendation} r
 */
export const isOpen = (r) => r.departs && r.kind !== 'reference' && !r.dismissed;

/**
 * What kind of finding it is, as its pop-up labels it: the original authors' setting, or a reasoned
 * one by its strength.
 * @param {Recommendation} r
 * @returns {string}
 */
export function kindLabel(r) {
  if (r.kind === 'reference') return "Original authors' setting (not a recommendation)";
  return r.strength === 'required' ? 'Required' : r.strength === 'discouraged' ? 'Discouraged' : 'Recommended';
}

/**
 * Every finding in a settings view.
 * @param {Map<string, SettingView> | null} view
 * @returns {Recommendation[]}
 */
export function findingsOf(view) {
  return view ? [...view.values()].flatMap((v) => v.recommendations ?? []) : [];
}

/**
 * The marker a setting shows, from the engine's (`SettingView.marker`): `open` (⚑, a finding to
 * review), `info` (ⓘ, information only), or null.
 * @param {SettingView | undefined} v
 * @returns {'open' | 'info' | null}
 */
export function marker(v) {
  if (!v?.marker) return null;
  return v.marker.actionable ? 'open' : 'info';
}

/**
 * How a setting's reference values read, one per rule covering it, as the TUI's help line has
 * them: what the rule says of the current value, the value, and the method and source.
 * @param {Recommendation} r
 * @returns {{ what: string, value: string, trigger: string, source: string }}
 */
export function referenceLine(r) {
  const authors = r.kind === 'reference';
  const what = !r.departs ? (authors ? "Authors' setting" : 'Recommended')
    : authors ? "Differs from the authors' setting"
      : r.dismissed ? 'Kept'
        : r.strength === 'required' ? 'Required'
          : r.strength === 'discouraged' ? 'Discouraged' : 'Recommended';
  const value = r.strength === 'discouraged' && r.departs ? `not ${currentText(r)}` : r.recommended.name;
  return { what, value, trigger: r.trigger.name, source: r.source };
}

/** The current value(s), named once each ("V-SHARP"). @param {Recommendation} r */
export function currentText(r) {
  return [...new Set(r.current.map((n) => n.name))].join(', ');
}

/** What a finding is for, to tell the findings apart: its rule and the value it departs at. */
const key = (r) => `${r.rule}=${JSON.stringify(r.current.map((n) => n.value))}`;
const methodKey = (/** @type {Trigger} */ m) => JSON.stringify([m.setting, m.value]);

/**
 * `first`, with the open findings as strong on its setting that disagree with it.
 * @param {Recommendation[]} all
 * @param {Recommendation} first
 */
export function group(all, first) {
  if (!first.conflict) return [first];
  return all.filter((r) => r.setting === first.setting && r.strength === first.strength && isOpen(r));
}

/** What the rule allows besides its recommended value. @param {Recommendation} r */
function alternatives(r) {
  return r.allowed.test === 'one_of' ? r.allowed.values.filter((v) => !same(v.value, r.recommended.value)) : [];
}

/** Offered findings first, then kept ones, then the authors' settings. */
const STRENGTH_ORDER = { required: 0, recommended: 1, discouraged: 2, reference: 3 };
const offeredFirst = (a, b) => (Number(!isOpen(a)) - Number(!isOpen(b)))
  || (STRENGTH_ORDER[a.strength] - STRENGTH_ORDER[b.strength]);

/**
 * The pop-up for `findings` a change created: Apply preselected, the other allowed values, Keep
 * mine (not for a required one). Two that disagree preselect nothing.
 * @param {Recommendation[]} findings
 * @returns {RecModal}
 */
export function modalFor(findings) {
  /** @type {Choice[]} */
  const choices = [];
  const several = findings.length > 1;
  for (const r of findings) {
    const who = several ? ` (${r.trigger.name})` : '';
    choices.push({ kind: 'apply', rule: r.rule, label: `Apply: ${r.recommended.name}${who}` });
    for (const alt of alternatives(r)) choices.push({ kind: 'apply', rule: r.rule, value: alt.value, label: `Use ${alt.name} instead${who}` });
  }
  if (findings.every((r) => r.strength !== 'required')) choices.push({ kind: 'ignore', label: 'Keep mine' });
  const r = findings[0];
  const title = r.strength === 'required' ? `${r.trigger.name} requires a change`
    : r.strength === 'discouraged' ? `Discouraged with ${r.trigger.name}`
      : several ? 'Recommendations disagree' : `${r.trigger.name} recommends a change`;
  return { kind: 'finding', findings, choices, preselected: several ? null : 0, title, details: false };
}

/**
 * Whose rule a finding is, as the details list it after its label (`kindLabel`): "MEDI's reference
 * pipeline uses PDF" for an authors' setting (with the engine's title for the method's reference),
 * "NDI recommends First echo time", "MEDI requires ...", "MEDI discourages Off".
 * @param {Recommendation} r
 * @param {string} referenceTitle - the engine's `reference_title` of its trigger
 */
export function whoSays(r, referenceTitle) {
  if (r.kind === 'reference') return `${referenceTitle} uses ${r.recommended.name}`;
  if (r.strength === 'discouraged') return `${r.trigger.name} discourages ${currentText(r)}`;
  return `${r.trigger.name} ${r.strength === 'required' ? 'requires' : 'recommends'} ${r.recommended.name}`;
}

/**
 * A marked setting's details: every rule it departs from (`findings`, offered ones first), whose
 * each is, and Apply, or Use the authors' setting for a value only the authors' settings have (one
 * per value, naming the methods when several use it), the other allowed values and Keep mine (not
 * when a finding is required). Apply is preselected for a recommendation, Keep mine for an authors'
 * setting, and nothing when recommendations on offer disagree.
 * @param {Recommendation[]} findings
 * @param {string[]} titles - the engine's reference title of each finding's trigger
 * @returns {RecModal}
 */
export function detailsFor(findings, titles) {
  /** @type {Choice[]} */
  const choices = [];
  /** @type {{ value: unknown, name: string, rule: string, methods: string[], authors: boolean }[]} */
  const values = [];
  for (const r of findings) {
    const v = values.find((x) => same(x.value, r.recommended.value));
    if (v) {
      v.methods.push(r.trigger.name);
      v.authors &&= r.kind === 'reference';
    } else values.push({ ...r.recommended, rule: r.rule, methods: [r.trigger.name], authors: r.kind === 'reference' });
  }
  const severalMethods = new Set(findings.map((f) => f.trigger.name)).size > 1;
  for (const v of values) {
    const lead = v.authors ? "Use the authors' setting" : 'Apply';
    choices.push({ kind: 'apply', rule: v.rule, label: `${lead}: ${v.name}${severalMethods ? ` (${v.methods.join(', ')})` : ''}` });
  }
  for (const r of findings) {
    for (const alt of alternatives(r)) {
      if (values.some((v) => same(v.value, alt.value)) || choices.some((c) => c.kind === 'apply' && 'value' in c && same(c.value, alt.value))) continue;
      choices.push({ kind: 'apply', rule: r.rule, value: alt.value, label: `Use ${alt.name} instead` });
    }
  }
  if (findings.every((r) => r.strength !== 'required')) choices.push({ kind: 'ignore', label: 'Keep mine' });
  const r = findings[0];
  const methods = [...new Set(findings.map((f) => f.trigger.name))];
  const title = r.kind === 'reference'
    ? (methods.length === 1 ? `Original authors' setting: ${titles[0]}` : `Original authors' settings: ${r.label} of ${methods.join(', ')}`)
    : findings.length > 1 ? `${r.label}: ${findings.length} differences` : modalFor([r]).title;
  // The first choice is the first finding's value, and reasoned findings come before authors' ones.
  const keep = choices.findIndex((c) => c.kind === 'ignore');
  const preselected = findings.some((f) => f.conflict && isOpen(f)) ? null
    : r.kind !== 'reference' ? 0
      : keep >= 0 ? keep : null;
  return { kind: 'finding', findings, choices, preselected, title, details: true, whose: findings.map((f, i) => whoSays(f, titles[i])) };
}

/**
 * Rows for a pop-up: one per setting and value, merging the rules of several methods that use the
 * same value ("(MEDI, PDF)").
 * @param {Recommendation[]} findings
 * @returns {RefRow[]}
 */
export function rowsOf(findings) {
  /** @type {RefRow[]} */
  const rows = [];
  for (const r of findings) {
    const row = rows.find((x) => x.setting === r.setting && same(x.reference.value, r.recommended.value));
    if (row) {
      row.findings.push(r);
      if (!row.methods.includes(r.trigger.name)) row.methods.push(r.trigger.name);
      continue;
    }
    rows.push({
      setting: r.setting, label: r.label, kind: r.kind, current: currentText(r), reference: r.recommended,
      findings: [r], methods: [r.trigger.name], alternatives: alternatives(r),
    });
  }
  return rows;
}

/**
 * The title of a pop-up offering several methods' reference pipelines: "MEDI's reference
 * pipeline", "MEDI's and PDF's reference pipelines".
 * @param {string[]} titles - the engine's, one per method
 * @param {Trigger[]} methods
 */
export function referencesTitle(titles, methods) {
  if (titles.length === 1) return titles[0];
  const suffix = "'s reference pipeline";
  if (titles.every((t, i) => t === `${methods[i].name}${suffix}`)) {
    return `${methods.map((m) => `${m.name}'s`).join(' and ')} reference pipelines`;
  }
  return titles.join(' · ');
}

/**
 * The buttons that apply a reference pipeline pop-up's rows, the first preselected when it holds
 * recommendations. An authors' setting is applied only when asked for, so with only those no such
 * button is preselected: Keep mine is (`keepPreselected`).
 * @param {RefModal} modal
 * @returns {{ answer: 'recommended' | 'all', label: string, preselected: boolean }[]}
 */
export function applyChoices(modal) {
  return modal.recommendations.length
    ? [
      { answer: 'recommended', label: 'Apply recommended', preselected: true },
      { answer: 'all', label: "Apply all, with the authors' settings", preselected: false },
    ]
    : [{ answer: 'all', label: "Apply the authors' settings", preselected: false }];
}

/**
 * Whether a reference pipeline pop-up preselects Keep mine: it holds only authors' settings.
 * @param {RefModal} modal
 * @returns {boolean}
 */
export const keepPreselected = (modal) => !modal.recommendations.length;

/**
 * What a settling left, for the notice under the header: the settings it left because the
 * suggestions disagree, and those where a stronger suggestion wanted another value. Null when it
 * left nothing.
 * @param {Settled} settled
 * @returns {string | null}
 */
export function leftNotice(settled) {
  const labels = (/** @type {Recommendation[]} */ list) => [...new Set(list.map((r) => r.label))];
  const disagree = labels(settled.conflicting);
  const outranked = labels(settled.outranked).filter((l) => !disagree.includes(l));
  const lines = [];
  if (disagree.length) {
    lines.push(`Left ${disagree.join(', ')} as set: the suggestions disagree. ${disagree.length === 1 ? 'Its marker has' : 'Their markers have'} the choices.`);
  }
  if (outranked.length) lines.push(`One suggestion for ${outranked.join(', ')} was not applied: a stronger one wants another value.`);
  return lines.length ? lines.join(' ') : null;
}

/**
 * What Choose per setting starts on, one per row in the order shown: a recommendation on its value,
 * an authors' setting on keeping the current value.
 * @param {RefModal} modal
 * @returns {RefPick[]}
 */
export function startingPicks(modal) {
  return [...modal.recommendations, ...modal.rows].map((row) => (row.kind === 'reference' ? 'keep' : 'reference'));
}

/**
 * @param {object} options
 * @param {Session} options.session
 * @param {{
 *   applyRecommendation: (config: any, rule: string, pick?: unknown) => Promise<any>,
 *   dismissRecommendation: (config: any, rule: string, facts: any, scan?: boolean) => Promise<any>,
 *   pruneDismissals: (config: any, facts: any, scan?: boolean) => Promise<any>,
 *   chosenMethods: (config: any, facts: any, scan?: boolean) => Promise<Trigger[]>,
 *   referenceOffers: (config: any, facts: any, scan: boolean, methods: Trigger[]) => Promise<{ method: Trigger, title: string, differences: Recommendation[] }[]>,
 *   referenceTitles: (methods: Trigger[]) => Promise<string[]>,
 *   keepFindings: (config: any, findings: Recommendation[]) => Promise<any>,
 *   applyAll: (config: any, findings: Recommendation[]) => Promise<{ config: any, settled: Settled }>,
 *   settle: (findings: Recommendation[]) => Promise<Settled>,
 * }} options.engine
 */
export function createRecommendations({ session, engine }) {
  const modal = signal(/** @type {RecModal | RefModal | null} */ (null));
  /** Pop-ups waiting for the open one to close (a reference pipeline after a required finding). */
  let queue = /** @type {(RecModal | RefModal)[]} */ ([]);
  const error = signal(/** @type {string | null} */ (null));
  /** Checks of a new view still asking the engine (what a change chose, and what to offer). */
  const pending = signal(0);
  /** The open findings the last view had, and the edits it was asked at. */
  let seen = /** @type {Set<string> | null} */ (null);
  let seenEdits = 0;
  /**
   * The rules the config departs from, or has departed from, since the scan was loaded: those
   * there before it, those offered, and those kept. A scan offers only what is not among them.
   */
  let knownFindings = /** @type {Set<string> | null} */ (null);
  /** The facts of the last view: only new facts can bring anything. */
  let lastFacts = /** @type {ScanFacts | null} */ (null);
  /** The methods with a reference pipeline the config had at the last view. */
  let methods = /** @type {Set<string> | null} */ (null);
  /** Settings a pop-up's own Apply just decided: not offered again by the methods it chose. */
  let decided = /** @type {Set<string>} */ (new Set());
  let checking = 0;

  const show = (/** @type {RecModal | RefModal} */ m) => {
    if (modal.peek()) queue.push(m);
    else modal.value = m;
  };

  /**
   * After each view: what the change that asked for it created. A view that follows an undo or a
   * redo opens nothing, and one that follows the scan offers only the findings it brought; the
   * other findings show as markers.
   */
  async function check(view, edit) {
    pending.value = pending.peek() + 1;
    try {
      await offer(view, edit);
    } finally {
      pending.value = pending.peek() - 1;
    }
  }

  /** What the change that asked for `view` created: the pop-ups to show. */
  async function offer(view, edit) {
    const id = ++checking;
    const config = session.config.peek();
    const facts = session.facts.peek();
    const loaded = session.loaded.peek();
    const all = findingsOf(view);
    const now = all.filter(isOpen);
    const created = seen && edit ? now.filter((r) => !seen?.has(key(r))) : [];
    seen = new Set(now.map(key));
    // As the TUI: a pop-up an earlier answer has resolved does not stay, or open after the next.
    const live = (/** @type {RecModal | RefModal} */ m) => m.kind !== 'finding' || m.details
      || m.findings.some((f) => now.some((r) => r.rule === f.rule));
    queue = queue.filter(live);
    const showing = modal.peek();
    if (showing && !live(showing)) next();
    let chosen = /** @type {Trigger[]} */ ([]);
    try {
      chosen = await engine.chosenMethods(config, facts, loaded);
    } catch {
      chosen = [];
    }
    if (id !== checking) return;
    const before = methods;
    methods = new Set(chosen.map(methodKey));
    const fresh = before && edit ? chosen.filter((m) => !before.has(methodKey(m))) : [];
    const differing = all.filter((r) => r.kind === 'reference' && r.departs);
    const viewFacts = session.viewFacts.peek();
    const scanBrought = (/** @type {Recommendation} */ r) => loaded && knownFindings && viewFacts !== lastFacts
      && !r.dismissed && !knownFindings.has(r.rule);
    const brought = differing.filter(scanBrought);
    const reasoned = [...created, ...now.filter((r) => scanBrought(r) && !created.some((c) => key(c) === key(r)))];
    const bringing = brought.map((r) => r.trigger)
      .filter((m, i, list) => !fresh.some((f) => methodKey(f) === methodKey(m)) && list.findIndex((x) => methodKey(x) === methodKey(m)) === i);
    let offers = /** @type {{ method: Trigger, title: string, differences: Recommendation[] }[]} */ ([]);
    if (fresh.length || bringing.length) {
      try {
        offers = await engine.referenceOffers(config, facts, loaded, [...fresh, ...bringing]);
      } catch {
        offers = [];
      }
      if (id !== checking) return;
    }
    // Only now: a view that replaced this one while the engine answered still has it all to offer.
    lastFacts = viewFacts;
    knownFindings = new Set([...(loaded ? knownFindings ?? [] : []), ...all.filter((r) => r.departs).map((r) => r.rule)]);
    const skip = edit ? decided : new Set();
    if (edit) decided = new Set();
    // A method the change chose offers all it differs by; one chosen before, what the scan brought.
    const chosenNow = new Set(fresh.map(methodKey));
    const offerable = (/** @type {Trigger} */ m, /** @type {Recommendation} */ r) => !skip.has(r.setting)
      && (chosenNow.has(methodKey(m)) || brought.some((b) => b.rule === r.rule));
    const differences = offers.flatMap((o) => o.differences.filter((r) => offerable(o.method, r)));
    const offered = offers.filter((o) => o.differences.some((r) => offerable(o.method, r)));
    const hard = reasoned.filter((r) => r.strength === 'required' || r.conflict);
    // A required one or a disagreement comes first, in its own pop-up; so does every recommendation
    // when there is no reference pipeline pop-up to list them in. One pop-up per setting.
    const alone = differences.length ? hard : [...hard, ...reasoned.filter((r) => !hard.includes(r))];
    const answered = new Set();
    for (const r of alone) {
      if (answered.has(r.setting)) continue;
      answered.add(r.setting);
      show(modalFor(group(all, r)));
    }
    if (differences.length) {
      show({
        kind: 'reference',
        title: referencesTitle(offered.map((o) => o.title), offered.map((o) => o.method)),
        recommendations: rowsOf(reasoned.filter((r) => !answered.has(r.setting))),
        rows: rowsOf(differences),
      });
    }
  }

  const stopWatching = effect(() => {
    const view = session.view.value;
    const asked = session.viewEdits.value;
    if (!view) return;
    const edit = asked > seenEdits;
    seenEdits = asked;
    check(view, edit);
  });

  // After an edit, the dismissals that no longer hold go (not an edit itself: no undo step).
  let pruning = 0;
  const stopPruning = effect(() => {
    const config = session.config.value;
    session.edits.value;
    if (!config.dismissed_recommendations?.length) return;
    const id = ++pruning;
    engine.pruneDismissals(config, session.facts.peek(), session.loaded.peek()).then((pruned) => {
      if (id !== pruning || session.config.peek() !== config) return;
      if (!same(pruned.dismissed_recommendations ?? [], config.dismissed_recommendations ?? [])) {
        session.replaceConfig({ ...config, dismissed_recommendations: pruned.dismissed_recommendations ?? [] });
      }
    }, () => {});
  });

  /** @param {() => Promise<void>} fn */
  const act = async (fn) => {
    error.value = null;
    try {
      await fn();
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e);
    }
  };

  /**
   * Choose per setting: the rows given a value, settled by the engine, each applied at most once
   * per setting; the rows on Keep mine kept.
   * @param {any} start - the config
   * @param {RefRow[]} rows
   * @param {RefPick[]} picks
   * @returns {Promise<{ config: any, settled: Settled }>}
   */
  async function applyPicks(start, rows, picks) {
    const pickOf = (/** @type {number} */ i) => picks[i] ?? 'keep';
    const settled = await engine.settle(rows.flatMap((row, i) => {
      const pick = pickOf(i);
      if (pick === 'keep') return [];
      return pick === 'reference' ? row.findings : row.findings.map((f) => ({ ...f, recommended: row.alternatives[pick] }));
    }));
    let config = start;
    const applied = new Set();
    for (const [i, row] of rows.entries()) {
      const pick = pickOf(i);
      if (pick === 'keep') config = await engine.keepFindings(config, row.findings);
      else if (settled.apply.some((r) => r.rule === row.findings[0].rule) && !applied.has(row.setting)) {
        applied.add(row.setting);
        config = await engine.applyRecommendation(config, row.findings[0].rule, pick === 'reference' ? undefined : row.alternatives[pick].value);
      }
    }
    return { config, settled };
  }

  /**
   * The config after a pop-up's answer, unless the answer changed nothing: Keep mine on an authors'
   * setting records nothing, and is no edit to undo.
   * @param {any} before @param {any} config
   * @returns {boolean} whether it was an edit
   */
  const commit = (before, config) => {
    if (same(before, config)) return false;
    session.setConfig(config);
    return true;
  };

  /** Close the open pop-up; the next waiting one, if any, opens. */
  const next = () => {
    modal.value = queue.shift() ?? null;
  };

  return {
    modal,
    pending,
    /** What went wrong with the last choice, if anything (the engine's reason). */
    error,
    /**
     * Open the details of a marked setting (⚑ or ⓘ): every rule it departs from, offered ones
     * first, with whose rule each is, why and where it is from.
     * @param {string} path
     */
    async details(path) {
      const findings = (session.view.peek()?.get(path)?.recommendations ?? []).filter((r) => r.departs).sort(offeredFirst);
      if (!findings.length) return;
      const titles = await engine.referenceTitles(findings.map((f) => f.trigger));
      modal.value = detailsFor(findings, titles);
    },
    /** Close the pop-up without changing anything; open findings stay marked. */
    close: next,
    /**
     * Take a choice of a recommendation's pop-up or a setting's details.
     * @param {Choice} choice
     */
    choose(choice) {
      const open = modal.peek();
      const findings = open?.kind === 'finding' ? open.findings : [];
      next();
      return act(async () => {
        const before = session.config.peek();
        const config = choice.kind === 'apply'
          ? await engine.applyRecommendation(before, choice.rule, choice.value)
          : await engine.keepFindings(before, findings.filter((f) => f.strength !== 'required' && !f.dismissed));
        commit(before, config);
      });
    },
    /**
     * Answer a reference pipeline pop-up: `recommended` applies the recommendations and keeps every
     * authors' setting as it is, `all` applies every row, `keep` keeps every row as it is, and
     * `picks` (Choose per setting) gives each row's choice: `reference`, `keep`, or the index of
     * one of its alternatives. What is applied together is the engine's to settle.
     * @param {RefAnswer} answer
     */
    answer(answer) {
      const open = modal.peek();
      if (open?.kind !== 'reference') return Promise.resolve();
      const rows = [...open.recommendations, ...open.rows];
      next();
      return act(async () => {
        const before = session.config.peek();
        let config = before;
        /** @type {Settled | null} */
        let settled = null;
        const findings = rows.flatMap((r) => r.findings);
        if (answer === 'all') ({ config, settled } = await engine.applyAll(config, findings));
        else if (answer === 'recommended') {
          ({ config, settled } = await engine.applyAll(config, open.recommendations.flatMap((r) => r.findings)));
          config = await engine.keepFindings(config, open.rows.flatMap((r) => r.findings));
        } else if (answer === 'keep') config = await engine.keepFindings(config, findings);
        else ({ config, settled } = await applyPicks(config, rows, answer.picks));
        const left = settled && leftNotice(settled);
        if (left) session.ui.notices.value = [...session.ui.notices.peek(), left];
        // A method these choices chose is not asked again about the settings they decided. With
        // no edit there is no view to clear this, and it would hide what the next edit offers.
        if (!same(before, config)) decided = new Set(rows.map((r) => r.setting));
        commit(before, config);
      });
    },
    dispose() {
      stopWatching();
      stopPruning();
    },
  };
}

/** @typedef {ReturnType<typeof createRecommendations>} Recommendations */
