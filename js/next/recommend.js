/**
 * The pipeline rules' findings in the workspace, as QSMxT's TUI shows them (src/tui/recommend.rs):
 * every finding, reference value, marker and method's reference pipeline is the engine's
 * (`settings_view`, and qsmxt-engine's `recommendations` functions, from QSM.rs's rules); nothing
 * here knows a rule or a method.
 *
 * - **A method's reference pipeline** is offered when a change chooses a method that has one (the
 *   engine's `chosen_methods`, before and after the change) and the config differs from it
 *   (`reference_differences`): one pop-up, "MEDI's reference pipeline", with a row per setting
 *   (current → reference, why, source) and Apply all / Choose per setting / Keep mine. Nothing
 *   changes unless applied; Close (or Escape) changes nothing. Keep mine is kept in the config's
 *   `dismissed_recommendations`, so it is not offered again for the same values.
 * - **A recommendation** (a `required`, `recommended` or `discouraged` finding a change creates,
 *   not dismissed) is offered in the same pop-up, under "Recommendations", first. A required one,
 *   or two that disagree, get a pop-up of their own (Apply preselected, the other allowed values,
 *   Keep mine unless required), and the reference pipeline's follows it.
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
 * @typedef {import('./store/session.js').Session} Session
 *
 * @typedef {{ kind: 'apply', rule: string, value?: unknown, label: string } | { kind: 'ignore', label: string }} Choice
 *
 * @typedef {object} RecModal - a recommendation's pop-up, or a setting's details
 * @property {'finding'} kind
 * @property {Recommendation[]} findings - one, or equally strong ones on one setting that disagree;
 *   for the details, every rule the setting departs from
 * @property {Choice[]} choices
 * @property {number | null} preselected - the choice offered first; none when the findings disagree
 * @property {string} title
 * @property {boolean} details - a marked setting's details, not a pop-up a change opened
 * @property {string[]} [whose] - for the details, whose rule each finding is ("MEDI's reference
 *   pipeline uses PDF")
 *
 * @typedef {object} RefRow - one setting of a reference pipeline pop-up
 * @property {string} setting
 * @property {string} label
 * @property {string} current - its value now
 * @property {{ value: unknown, name: string }} reference - what the methods' rules use
 * @property {Recommendation[]} findings - each method's rule with that value (merged)
 * @property {string[]} methods - their methods ("MEDI", "PDF")
 * @property {{ value: unknown, name: string }[]} alternatives - other values the rules allow
 *
 * @typedef {object} RefModal - a chosen method's reference pipeline, and any recommendations the
 *   same change created
 * @property {'reference'} kind
 * @property {string} title
 * @property {RefRow[]} recommendations - offered first, under "Recommendations"
 * @property {RefRow[]} rows
 */
import { signal, effect } from '@preact/signals-core';

const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

/**
 * A finding offered as a change and not dismissed: what opens the pop-up or shows the ⚑ badge.
 * @param {Recommendation} r
 */
export const isOpen = (r) => r.departs && r.strength !== 'reference' && !r.dismissed;

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
  const what = !r.departs ? 'Reference'
    : r.strength === 'reference' ? 'Differs from the reference'
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

/** Offered findings first, then kept ones, then reference ones. */
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
 * Whose rule a finding is, as the details list it: "MEDI's reference pipeline uses PDF" (with the
 * engine's title for the method's reference), "NDI recommends First echo time", "MEDI requires
 * ...", "MEDI discourages Off".
 * @param {Recommendation} r
 * @param {string} referenceTitle - the engine's `reference_title` of its trigger
 */
export function whoSays(r, referenceTitle) {
  if (r.strength === 'reference') return `${referenceTitle} uses ${r.recommended.name}`;
  if (r.strength === 'discouraged') return `${r.trigger.name} discourages ${currentText(r)}`;
  return `${r.trigger.name} ${r.strength === 'required' ? 'requires' : 'recommends'} ${r.recommended.name}`;
}

/**
 * A marked setting's details: every rule it departs from (`findings`, offered ones first), whose
 * each is, and Apply (one per value, naming the methods when several use it), the other allowed
 * values and Keep mine (not when a finding is required).
 * @param {Recommendation[]} findings
 * @param {string[]} titles - the engine's reference title of each finding's trigger
 * @returns {RecModal}
 */
export function detailsFor(findings, titles) {
  /** @type {Choice[]} */
  const choices = [];
  /** @type {{ value: unknown, name: string, rule: string, methods: string[] }[]} */
  const values = [];
  for (const r of findings.filter((f) => f.strength !== 'discouraged')) {
    const v = values.find((x) => same(x.value, r.recommended.value));
    if (v) v.methods.push(r.trigger.name);
    else values.push({ ...r.recommended, rule: r.rule, methods: [r.trigger.name] });
  }
  const severalMethods = new Set(findings.map((f) => f.trigger.name)).size > 1;
  for (const v of values) {
    choices.push({ kind: 'apply', rule: v.rule, label: `Apply: ${v.name}${severalMethods ? ` (${v.methods.join(', ')})` : ''}` });
  }
  for (const r of findings) {
    for (const alt of alternatives(r)) {
      if (values.some((v) => same(v.value, alt.value)) || choices.some((c) => c.kind === 'apply' && 'value' in c && same(c.value, alt.value))) continue;
      choices.push({ kind: 'apply', rule: r.rule, value: alt.value, label: `Use ${alt.name} instead` });
    }
  }
  if (findings.every((r) => r.strength !== 'required')) choices.push({ kind: 'ignore', label: 'Keep mine' });
  const r = findings[0];
  const title = findings.length > 1 ? `${r.label}: ${findings.length} differences`
    : r.strength === 'reference' ? `Differs from ${titles[0]}` : modalFor([r]).title;
  return { kind: 'finding', findings, choices, preselected: null, title, details: true, whose: findings.map((f, i) => whoSays(f, titles[i])) };
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
      setting: r.setting, label: r.label, current: currentText(r), reference: r.recommended,
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
 *   applyAll: (config: any, findings: Recommendation[]) => Promise<any>,
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
   * After each view: what the change that asked for it created. A view that follows the scan
   * (loaded, or changed), an undo or a redo opens nothing; its findings show as markers.
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
    let offers = /** @type {{ method: Trigger, title: string, differences: Recommendation[] }[]} */ ([]);
    if (fresh.length) {
      try {
        offers = await engine.referenceOffers(config, facts, loaded, fresh);
      } catch {
        offers = [];
      }
      if (id !== checking) return;
    }
    const skip = edit ? decided : new Set();
    if (edit) decided = new Set();
    const differences = offers.flatMap((o) => o.differences).filter((r) => !skip.has(r.setting));
    const offered = offers.filter((o) => o.differences.some((r) => !skip.has(r.setting)));
    const hard = created.some((r) => r.strength === 'required' || r.conflict);
    if (created.length && (hard || !differences.length)) show(modalFor(group(all, created[0])));
    if (differences.length) {
      show({
        kind: 'reference',
        title: referencesTitle(offered.map((o) => o.title), offered.map((o) => o.method)),
        recommendations: hard ? [] : rowsOf(created),
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
        let config = session.config.peek();
        if (choice.kind === 'apply') {
          config = await engine.applyRecommendation(config, choice.rule, choice.value);
        } else {
          config = await engine.keepFindings(config, findings.filter((f) => f.strength !== 'required' && !f.dismissed));
        }
        session.setConfig(config);
      });
    },
    /**
     * Answer a reference pipeline pop-up: `all` applies every row, `keep` keeps every row as it is,
     * and `picks` (Choose per setting) gives each row's choice: `reference`, `keep`, or the index
     * of one of its alternatives.
     * @param {'all' | 'keep' | { picks: ('reference' | 'keep' | number)[] }} answer
     */
    answer(answer) {
      const open = modal.peek();
      if (open?.kind !== 'reference') return Promise.resolve();
      const rows = [...open.recommendations, ...open.rows];
      next();
      return act(async () => {
        let config = session.config.peek();
        const findings = rows.flatMap((r) => r.findings);
        if (answer === 'all') config = await engine.applyAll(config, findings);
        else if (answer === 'keep') config = await engine.keepFindings(config, findings);
        else {
          for (const [i, row] of rows.entries()) {
            const pick = answer.picks[i] ?? 'reference';
            if (pick === 'keep') config = await engine.keepFindings(config, row.findings);
            else if (pick === 'reference') config = await engine.applyRecommendation(config, row.findings[0].rule);
            else config = await engine.applyRecommendation(config, row.findings[0].rule, row.alternatives[pick].value);
          }
        }
        // A method these choices chose is not asked again about the settings they decided.
        decided = new Set(rows.map((r) => r.setting));
        session.setConfig(config);
      });
    },
    dispose() {
      stopWatching();
      stopPruning();
    },
  };
}

/** @typedef {ReturnType<typeof createRecommendations>} Recommendations */
