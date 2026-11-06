import { itemMeta, learnItems, setItemLookup, itemsVersion } from './items.js';

describe('item meta', () => {
  afterEach(() => setItemLookup(null));

  test('a plan\'s items name themselves, with their echo', () => {
    learnItems([{ key: 'AcquiredPhase(2)', label: 'Phase (as acquired)', units: null, kind: 'phase', echo: 2, coil: null }]);
    expect(itemMeta('AcquiredPhase(2)')).toMatchObject({ label: 'Phase (as acquired), echo 2', kind: 'phase' });
  });

  test('a key not seen yet stands in for itself, and is asked of the engine once', async () => {
    const asked = [];
    setItemLookup(async (key) => { asked.push(key); return { key, label: 'Susceptibility', units: 'ppm', kind: 'chi' }; });
    const before = itemsVersion.peek();
    expect(itemMeta('Chi').label).toBe('Chi');
    itemMeta('Chi');
    await new Promise((r) => setTimeout(r, 0));
    expect(asked).toEqual(['Chi']);
    expect(itemsVersion.peek()).toBeGreaterThan(before);
    expect(itemMeta('Chi')).toMatchObject({ label: 'Susceptibility', units: 'ppm' });
  });
});
