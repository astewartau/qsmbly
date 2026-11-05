import { createSession } from './store/session.js';
import {
  createRecommendations, marker, modalFor, detailsFor, rowsOf, referencesTitle, referenceLine, isOpen, whoSays,
  kindLabel, applyChoices, startingPicks,
} from './recommend.js';

// A tiny mocked config and engine. One setting, `x.te`, with one rule on it, `x.te-first`
// (recommended `first`, `last` allowed besides), scoped to method X. And a method choice,
// `x.method`: choosing `y` chooses Y, whose reference pipeline uses `ref` for `x.mode`. The
// engine's answers are what qsmxt-engine gives for such rules; the rules themselves are QSM.rs's
// to test. One of Y's authors' settings depends on the scan: `x.window` (0.4), on Siemens only.
const defaults = { x: { te: 'first', method: 'x', mode: 'mine', window: 0 }, dismissed_recommendations: [] };
const X = { setting: 'x.algorithm', value: 'x', name: 'X' };
const Y = { setting: 'x.method', value: 'y', name: 'Y' };
/** The strength of the mocked `x.te-first` rule. */
let teStrength = 'recommended';
afterEach(() => { teStrength = 'recommended'; windowDismissed = false; });

const rule = (over = {}) => ({
  rule: 'x.te-first', kind: over.strength === 'reference' ? 'reference' : 'reasoned', strength: 'recommended', basis: 'measured', setting: 'x.te', label: 'Echo Time',
  trigger: X,
  current: [{ value: 'sum', name: 'Sum of echo times' }], recommended: { value: 'first', name: 'First echo time' },
  allowed: { test: 'one_of', values: [{ value: 'first', name: 'First echo time' }, { value: 'last', name: 'Last echo time' }] },
  reason: 'the first echo did best', source: 'the paper', departs: true, ...over,
});
const kept = (config, r, value) => config.dismissed_recommendations.some((d) => d.rule === r && d.chosen === value);
const yRule = (config) => rule({
  rule: 'y.mode-ref', strength: 'reference', setting: 'x.mode', label: 'Mode', trigger: Y,
  current: [{ value: config.x.mode, name: config.x.mode }], recommended: { value: 'ref', name: 'Ref' },
  allowed: { test: 'one_of', values: [{ value: 'ref', name: 'Ref' }, { value: 'other', name: 'Other' }] },
  reason: "Y's authors use it", source: 'the Y paper', departs: config.x.mode !== 'ref',
});

const windowRule = (config) => rule({
  rule: 'y.window-ref', strength: 'reference', setting: 'x.window', label: 'Window', trigger: Y, needs_scan: true,
  current: [{ value: config.x.window, name: String(config.x.window) }], recommended: { value: 0.4, name: '0.4' },
  allowed: { test: 'one_of', values: [{ value: 0.4, name: '0.4' }] },
  reason: "Y's authors window Siemens scans", source: 'the Y code', departs: config.x.window !== 0.4,
  dismissed: windowDismissed,
});
/** A saved dismissal of the mocked window rule. */
let windowDismissed = false;
const siemens = (facts) => facts?.manufacturer === 'Siemens';

/** What the mocked engine says of `config` for a scan with `facts`. */
function viewOf(config, facts = null) {
  const te = config.x.te;
  const r = rule({ strength: teStrength, current: [{ value: te, name: te }], departs: te === 'sum', dismissed: kept(config, 'x.te-first', te) });
  const rows = [{ path: 'x.te', visible: true, reference: [r], recommendations: r.departs ? [r] : [] }];
  if (r.departs) rows[0].marker = { actionable: !r.dismissed, text: 'X: First echo time' };
  if (config.x.method === 'y') {
    const y = yRule(config);
    rows.push({ path: 'x.mode', visible: true, reference: [y], recommendations: y.departs ? [y] : [] });
    if (y.departs) rows[1].marker = { actionable: false, text: 'Y: Ref' };
    if (siemens(facts)) {
      const w = windowRule(config);
      rows.push({ path: 'x.window', visible: true, reference: [w], recommendations: w.departs ? [w] : [], ...(w.departs ? { marker: { actionable: false, text: 'Y: 0.4' } } : {}) });
    }
  }
  return rows;
}

const keepAll = (c, findings) => ({
  ...c,
  dismissed_recommendations: [
    ...c.dismissed_recommendations.filter((d) => !findings.some((f) => f.rule === d.rule)),
    // As the engine's `keep`: a required finding cannot be kept, and an authors' setting records nothing.
    ...findings.filter((f) => f.strength === 'recommended' || f.strength === 'discouraged').map((f) => ({ rule: f.rule, chosen: f.current[0].value })),
  ],
});
const applyOne = (c, r, pick) => {
  if (r === 'y.window-ref') return { ...c, x: { ...c.x, window: pick ?? 0.4 } };
  return r === 'y.mode-ref'
    ? { ...c, x: { ...c.x, mode: pick ?? 'ref' } }
    : { ...c, x: { ...c.x, te: pick ?? 'first' } };
};
const engine = {
  applyRecommendation: async (c, r, pick) => applyOne(c, r, pick),
  dismissRecommendation: async (c, r) => keepAll(c, viewOf(c).flatMap((v) => v.recommendations).filter((f) => f.rule === r)),
  pruneDismissals: async (c) => ({
    ...c,
    dismissed_recommendations: c.dismissed_recommendations.filter((d) => d.chosen === c.x.te),
  }),
  chosenMethods: async (c) => (c.x.method === 'y' ? [X, Y] : [X]),
  referenceOffers: async (c, f, _s, methods) => methods.map((m) => ({
    method: m, title: `${m.name}'s reference pipeline`,
    differences: m.name === 'Y' ? [yRule(c), ...(siemens(f) ? [windowRule(c)] : [])].filter((r) => r.departs && !r.dismissed) : [],
  })),
  referenceTitles: async (methods) => methods.map((m) => `${m.name}'s reference pipeline`),
  keepFindings: async (c, findings) => keepAll(c, findings),
  applyAll: async (c, findings) => findings.reduce((acc, f) => applyOne(acc, f.rule), c),
};

function harness() {
  const pending = [];
  const session = createSession({ defaults, planner: () => ({}), settingsView: (c, f) => viewOf(c, f), schedule: (run) => pending.push(run) });
  const recs = createRecommendations({ session, engine });
  const flush = async () => {
    for (let i = 0; i < 6; i++) {
      while (pending.length) await pending.shift()();
      await new Promise((r) => setTimeout(r, 0));
    }
  };
  return { session, recs, flush };
}

describe('a recommendation a change creates', () => {
  test('opens the pop-up, with Apply preselected; Apply resolves it', async () => {
    const { session, recs, flush } = harness();
    await flush();
    expect(recs.modal.value).toBeNull();
    session.set('x.te', 'sum');
    await flush();
    const modal = recs.modal.value;
    expect(modal?.kind).toBe('finding');
    expect(modal?.title).toBe('X recommends a change');
    expect(modal?.choices.map((c) => c.label)).toEqual(['Apply: First echo time', 'Use Last echo time instead', 'Keep mine']);
    expect(modal?.preselected).toBe(0);
    await recs.choose(modal.choices[0]);
    await flush();
    expect(session.config.value.x.te).toBe('first');
    expect(recs.modal.value).toBeNull();
  });

  test('another allowed value can be chosen instead', async () => {
    const { session, recs, flush } = harness();
    await flush();
    session.set('x.te', 'sum');
    await flush();
    await recs.choose(recs.modal.value.choices[1]);
    await flush();
    expect(session.config.value.x.te).toBe('last');
  });

  test('Close leaves the finding open, marked ⚑; the marker opens its details', async () => {
    const { session, recs, flush } = harness();
    await flush();
    session.set('x.te', 'sum');
    await flush();
    recs.close();
    expect(marker(session.view.value.get('x.te'))).toBe('open');
    session.set('x.te', 'sum'); // no change: no new finding, no pop-up
    await flush();
    expect(recs.modal.value).toBeNull();
    await recs.details('x.te');
    expect(recs.modal.value?.details).toBe(true);
    expect(recs.modal.value?.findings[0].rule).toBe('x.te-first');
    await recs.details('nothing.here');
  });

  test('Keep mine keeps the value; changing the setting drops it', async () => {
    const { session, recs, flush } = harness();
    await flush();
    session.set('x.te', 'sum');
    await flush();
    await recs.choose(recs.modal.value.choices[2]);
    await flush();
    expect(session.config.value.dismissed_recommendations).toEqual([{ rule: 'x.te-first', chosen: 'sum' }]);
    expect(marker(session.view.value.get('x.te'))).toBe('info');
    session.set('x.te', 'last');
    await flush();
    expect(session.config.value.dismissed_recommendations).toEqual([]);
    session.set('x.te', 'sum');
    await flush();
    expect(recs.modal.value?.findings[0].rule).toBe('x.te-first');
  });

  test('a view that follows the scan, not an edit, opens nothing', async () => {
    const { session, recs, flush } = harness();
    session.replaceConfig({ ...defaults, x: { ...defaults.x, te: 'sum', method: 'y' } });
    await flush();
    expect(recs.modal.value).toBeNull();
    expect(marker(session.view.value.get('x.te'))).toBe('open');
  });
});

describe('a method chosen with a reference pipeline', () => {
  test('one pop-up offers where the config differs from it; Close changes nothing', async () => {
    const { session, recs, flush } = harness();
    await flush();
    session.set('x.method', 'y');
    await flush();
    const modal = recs.modal.value;
    expect(modal?.kind).toBe('reference');
    expect(modal?.title).toBe("Y's reference pipeline");
    expect(modal.rows.map((r) => [r.label, r.current, r.reference.name])).toEqual([['Mode', 'mine', 'Ref']]);
    recs.close();
    await flush();
    expect(session.config.value.x.mode).toBe('mine');
    expect(marker(session.view.value.get('x.mode'))).toBe('info');
  });

  test('only authors\' settings: Apply the authors\' settings, not preselected, applies them', async () => {
    const { session, recs, flush } = harness();
    await flush();
    session.set('x.method', 'y');
    await flush();
    expect(applyChoices(recs.modal.value)).toEqual([{ answer: 'all', label: "Apply the authors' settings", preselected: false }]);
    await recs.answer('all');
    await flush();
    expect(session.config.value.x.mode).toBe('ref');
    expect(recs.modal.value).toBeNull();
  });

  test('Keep mine on an authors\' setting records nothing; it is offered again when the method is chosen again', async () => {
    const { session, recs, flush } = harness();
    await flush();
    session.set('x.method', 'y');
    await flush();
    await recs.answer('keep');
    await flush();
    expect(session.config.value.x.mode).toBe('mine');
    expect(session.config.value.dismissed_recommendations).toEqual([]);
    session.set('x.te', 'last'); // another change: the method is still chosen, nothing to offer
    await flush();
    expect(recs.modal.value).toBeNull();
    session.set('x.method', 'x');
    await flush();
    session.set('x.method', 'y');
    await flush();
    expect(recs.modal.value?.kind).toBe('reference');
  });

  test('Choose per setting: an authors\' setting starts on keep; each row takes its own value', async () => {
    const { session, recs, flush } = harness();
    await flush();
    session.set('x.method', 'y');
    await flush();
    expect(startingPicks(recs.modal.value)).toEqual(['keep']);
    await recs.answer({ picks: [0] }); // the first alternative, Other
    await flush();
    expect(session.config.value.x.mode).toBe('other');
  });

  /** One change that creates a recommendation and chooses Y: the pop-up with both kinds. */
  async function both() {
    const h = harness();
    await h.flush();
    h.session.setAll([['x.method', 'y'], ['x.te', 'sum']]);
    await h.flush();
    return h;
  }

  test('recommendations the same change creates are listed first, and only they start on their value', async () => {
    const { recs } = await both();
    const modal = recs.modal.value;
    expect(modal?.kind).toBe('reference');
    expect(modal.recommendations.map((r) => [r.setting, r.kind])).toEqual([['x.te', 'reasoned']]);
    expect(modal.rows.map((r) => [r.setting, r.kind])).toEqual([['x.mode', 'reference']]);
    expect(startingPicks(modal)).toEqual(['reference', 'keep']);
    expect(applyChoices(modal)).toEqual([
      { answer: 'recommended', label: 'Apply recommended', preselected: true },
      { answer: 'all', label: "Apply all, with the authors' settings", preselected: false },
    ]);
  });

  test('Apply recommended applies them and keeps every authors\' setting', async () => {
    const { session, recs, flush } = await both();
    await recs.answer('recommended');
    await flush();
    expect(session.config.value.x).toEqual({ te: 'first', method: 'y', mode: 'mine', window: 0 });
    expect(session.config.value.dismissed_recommendations).toEqual([]);
    expect(recs.modal.value).toBeNull();
  });

  test('Apply all, with the authors\' settings, applies both kinds', async () => {
    const { session, recs, flush } = await both();
    await recs.answer('all');
    await flush();
    expect(session.config.value.x).toEqual({ te: 'first', method: 'y', mode: 'ref', window: 0 });
  });

  test('a required recommendation gets its own pop-up first; the reference pipeline follows', async () => {
    teStrength = 'required';
    const { session, recs, flush } = harness();
    await flush();
    session.setAll([['x.method', 'y'], ['x.te', 'sum']]);
    await flush();
    expect(recs.modal.value?.kind).toBe('finding');
    expect(recs.modal.value?.title).toBe('X requires a change');
    expect(recs.modal.value?.choices.some((c) => c.kind === 'ignore')).toBe(false);
    recs.close();
    expect(recs.modal.value?.kind).toBe('reference');
    recs.close();
    expect(recs.modal.value).toBeNull();
  });
});

describe('a scan that brings authors\' settings of a chosen method', () => {
  /** Load a scan: files, then the facts the engine reads from them. Each file or sidecar is a load. */
  const load = async ({ session, flush }, manufacturer, files = [{ id: 1, name: 'phase.nii' }]) => {
    session.files.value = /** @type {any} */ (files);
    session.facts.value = /** @type {any} */ ({ echoes: 1, manufacturer });
    await flush();
  };
  /** Choose Y, closing what that offers: the method is chosen before the scan. */
  const chooseY = async ({ session, recs, flush }) => {
    session.set('x.method', 'y');
    await flush();
    recs.close();
    await flush();
  };
  const rules = (modal) => modal?.rows.map((r) => r.findings[0].rule);

  test('loaded after the method: the same pop-up, offering what it brought, starting on keep', async () => {
    const h = harness();
    await h.flush();
    await chooseY(h);
    await load(h, 'Siemens');
    const modal = h.recs.modal.value;
    expect(modal?.kind).toBe('reference');
    expect(modal.title).toBe("Y's reference pipeline");
    expect(rules(modal)).toEqual(['y.window-ref']);
    expect(applyChoices(modal).map((c) => c.preselected)).toEqual([false]);
    expect(startingPicks(modal)).toEqual(['keep']);
    expect(h.session.config.value.x.window).toBe(0);
  });

  test('loaded before the method: choosing it offers the same difference', async () => {
    const h = harness();
    await h.flush();
    await load(h, 'Siemens');
    expect(h.recs.modal.value).toBeNull();
    h.session.set('x.method', 'y');
    await h.flush();
    expect(rules(h.recs.modal.value)).toContain('y.window-ref');
  });

  test('nothing changes unless chosen; Apply takes it, and it is not offered again', async () => {
    const h = harness();
    await h.flush();
    await chooseY(h);
    await load(h, 'Siemens');
    await h.recs.answer('all');
    await h.flush();
    expect(h.session.config.value.x).toMatchObject({ window: 0.4, mode: 'mine' });
    expect(h.recs.modal.value).toBeNull();
  });

  test('a scan that adds no difference opens nothing', async () => {
    for (const manufacturer of ['Philips', null]) {
      const h = harness();
      await h.flush();
      await chooseY(h);
      await load(h, manufacturer);
      expect(h.recs.modal.value).toBeNull();
    }
  });

  test('facts with no scan loaded (the example preview) open nothing', async () => {
    const h = harness();
    await h.flush();
    await chooseY(h);
    h.session.facts.value = /** @type {any} */ ({ echoes: 5, manufacturer: 'Siemens' });
    await h.flush();
    expect(h.recs.modal.value).toBeNull();
  });

  test('differences there before the scan are not offered by it', async () => {
    const h = harness();
    await h.flush();
    await chooseY(h); // x.mode already differs, and stays a marker
    await load(h, 'Philips');
    expect(h.recs.modal.value).toBeNull();
    expect(marker(h.session.view.value.get('x.mode'))).toBe('info');
  });

  test('a further file or sidecar does not re-open what was offered, however it was answered', async () => {
    const more = [{ id: 1, name: 'phase.nii' }, { id: 2, name: 'phase.json' }];
    for (const answer of [(recs) => recs.close(), (recs) => recs.answer('keep')]) {
      const h = harness();
      await h.flush();
      await chooseY(h);
      await load(h, 'Siemens');
      await answer(h.recs);
      await h.flush();
      await load(h, 'Siemens', more);
      expect(h.recs.modal.value).toBeNull();
      expect(marker(h.session.view.value.get('x.window'))).toBe('info');
    }
  });

  test('offered once even if the manufacturer goes and comes back; a scan loaded afresh offers it again', async () => {
    const h = harness();
    await h.flush();
    await chooseY(h);
    await load(h, 'Siemens');
    h.recs.close();
    await load(h, null);
    await load(h, 'Siemens');
    expect(h.recs.modal.value).toBeNull();
    // Cleared: back to the preview, then another scan.
    h.session.files.value = [];
    h.session.facts.value = /** @type {any} */ ({ echoes: 5, manufacturer: null });
    await h.flush();
    await load(h, 'Siemens');
    expect(rules(h.recs.modal.value)).toEqual(['y.window-ref']);
  });

  test('what is dismissed for this value is not offered', async () => {
    windowDismissed = true;
    const h = harness();
    await h.flush();
    await chooseY(h);
    await load(h, 'Siemens');
    expect(h.recs.modal.value).toBeNull();
  });

  test('an undo after the scan offers nothing', async () => {
    const h = harness();
    await h.flush();
    await load(h, 'Siemens');
    h.session.set('x.method', 'y');
    await h.flush();
    h.recs.close();
    h.session.undo();
    await h.flush();
    h.session.redo();
    await h.flush();
    expect(h.recs.modal.value).toBeNull();
  });
});

describe('what a finding shows', () => {
  test('an authors\' setting is information only, and labelled as theirs', () => {
    const r = rule({ strength: 'reference' });
    expect(isOpen(r)).toBe(false);
    expect(marker({ path: 'x.te', visible: true, recommendations: [r], marker: { actionable: false, text: 'X: First echo time' } })).toBe('info');
    expect(marker({ path: 'x.te', visible: true })).toBeNull();
    expect(kindLabel(r)).toBe("Original authors' setting (not a recommendation)");
    expect(referenceLine(r).what).toBe("Differs from the authors' setting");
    expect(referenceLine({ ...r, departs: false }).what).toBe("Authors' setting");
  });

  test('a reasoned finding is labelled by its strength', () => {
    expect(isOpen(rule())).toBe(true);
    expect(['recommended', 'required', 'discouraged'].map((strength) => kindLabel(rule({ strength })))).toEqual(['Recommended', 'Required', 'Discouraged']);
    expect(['recommended', 'required', 'discouraged'].map((strength) => referenceLine(rule({ strength })).what)).toEqual(['Recommended', 'Required', 'Discouraged']);
    expect(referenceLine(rule({ departs: false })).what).toBe('Recommended');
    expect(referenceLine(rule({ dismissed: true })).what).toBe('Kept');
  });

  test('the details say whose each finding is', () => {
    expect(whoSays(rule({ strength: 'reference' }), "X's reference pipeline")).toBe("X's reference pipeline uses First echo time");
    expect(whoSays(rule(), '')).toBe('X recommends First echo time');
    expect(whoSays(rule({ strength: 'discouraged' }), '')).toBe('X discourages Sum of echo times');
  });

  test('the details of an authors\' setting preselect nothing, and its button says it is theirs', () => {
    const one = detailsFor([rule({ strength: 'reference' })], ["X's reference pipeline"]);
    expect(one.title).toBe("Original authors' setting: X's reference pipeline");
    expect(one.preselected).toBeNull();
    expect(one.choices.map((c) => c.label)).toEqual(["Use the authors' setting: First echo time", 'Use Last echo time instead', 'Keep mine']);
    const two = detailsFor([rule({ strength: 'reference' }), rule({ strength: 'reference', rule: 'z', trigger: { ...X, name: 'Z' } })], ['a', 'b']);
    expect(two.title).toBe("Original authors' settings: Echo Time of X, Z");
    expect(two.preselected).toBeNull();
    expect(two.choices[0].label).toBe("Use the authors' setting: First echo time (X, Z)");
  });

  test('the details of a reasoned finding on offer preselect Apply; not once it is kept, or when findings disagree', () => {
    for (const strength of ['recommended', 'required', 'discouraged']) {
      const details = detailsFor([rule({ strength })], ['']);
      expect(details.preselected).toBe(0);
      expect(details.choices[0].label).toBe('Apply: First echo time');
    }
    expect(detailsFor([rule({ dismissed: true })], ['']).preselected).toBeNull();
    const other = rule({ rule: 'y.te-last', trigger: Y, recommended: { value: 'last', name: 'Last echo time' }, conflict: true });
    expect(detailsFor([rule({ conflict: true }), other], ['', '']).preselected).toBeNull();
  });

  test('details with both kinds tell the values apart, a value a recommendation shares being Apply', () => {
    const authors = rule({ strength: 'reference', rule: 'y.te-ref', trigger: Y, recommended: { value: 'last', name: 'Last echo time' } });
    const apart = detailsFor([rule(), authors], ['', "Y's reference pipeline"]);
    expect(apart.preselected).toBe(0);
    expect(apart.choices.map((c) => c.label)).toEqual(['Apply: First echo time (X)', "Use the authors' setting: Last echo time (Y)", 'Keep mine']);
    const shared = detailsFor([rule(), { ...authors, recommended: rule().recommended }], ['', "Y's reference pipeline"]);
    expect(shared.choices[0].label).toBe('Apply: First echo time (X, Y)');
  });

  test('rows merge the methods that use the same value', () => {
    const a = rule({ strength: 'reference' });
    const b = rule({ strength: 'reference', rule: 'z', trigger: { ...X, name: 'Z' }, source: 'Z code' });
    const rows = rowsOf([a, b]);
    expect(rows).toHaveLength(1);
    expect(rows[0].methods).toEqual(['X', 'Z']);
    expect(referencesTitle(["MEDI's reference pipeline", "PDF's reference pipeline"], [{ ...X, name: 'MEDI' }, { ...X, name: 'PDF' }]))
      .toBe("MEDI's and PDF's reference pipelines");
  });

  test('disagreeing recommendations preselect nothing', () => {
    const other = rule({ rule: 'y.te-last', trigger: Y, recommended: { value: 'last', name: 'Last echo time' }, conflict: true });
    const both = modalFor([rule({ conflict: true }), other]);
    expect(both.preselected).toBeNull();
    expect(both.title).toBe('Recommendations disagree');
  });
});
