/**
 * Tests for the mask-section string handed to qsmxt (`--mask <section>`).
 */

import { maskSectionString } from './ConfigBridge.js';

describe('maskSectionString', () => {
  test('returns empty for no ops', () => {
    expect(maskSectionString([], 'combined')).toBe('');
    expect(maskSectionString(null, 'combined')).toBe('');
  });

  test('maps the mask source to the qsmxt input name', () => {
    expect(maskSectionString(['threshold:otsu'], 'combined')).toBe('magnitude,threshold:otsu');
    expect(maskSectionString(['threshold:otsu'], 'phase_quality')).toBe('phase-quality,threshold:otsu');
    expect(maskSectionString(['bet:0.50'], 'first_echo')).toBe('magnitude-first,bet:0.50');
  });

  test('passes HD-BET through with its patch size and step', () => {
    // The browser runs qsm-core's low-memory patch; qsmxt parses the whole op back to exactly
    // those HdBetParams — step included since qsmxt-config v9.19.1 — so the printed command
    // reproduces what the UI just did.
    expect(maskSectionString(['hd-bet:128x128x64:step=0.5'], 'combined'))
      .toBe('magnitude,hd-bet:128x128x64:step=0.5');
    expect(maskSectionString(['hd-bet:128x128x64:step=0.75', 'signal-erode', 'erode:2'], 'combined'))
      .toBe('magnitude,hd-bet:128x128x64:step=0.75,signal-erode,erode:2');
  });

  test('passes signal-gated erosion through in order', () => {
    // qsmxt parses a bare `signal-erode` as qsm-core's defaults (the QSM-CI setting).
    expect(maskSectionString(['bet:0.50', 'signal-erode'], 'combined'))
      .toBe('magnitude,bet:0.50,signal-erode');
    expect(maskSectionString(['threshold:otsu', 'fill-holes:0', 'signal-erode', 'erode:2'], 'combined'))
      .toBe('magnitude,threshold:otsu,fill-holes:0,signal-erode,erode:2');
  });
});
