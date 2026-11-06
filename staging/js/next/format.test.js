import { formatBytes, formatMs } from './format.js';

test('formatBytes', () => {
  expect(formatBytes(0)).toBe('0 B');
  expect(formatBytes(1023)).toBe('1023 B');
  expect(formatBytes(4 * 1024 ** 3)).toBe('4.0 GB');
  expect(formatBytes(20901826)).toBe('19.9 MB');
  expect(formatBytes(1967364505)).toBe('1.8 GB');
});

test('formatMs', () => {
  expect([0.005, 0.011, 0.0115].map(formatMs)).toEqual(['5', '11', '11.5']);
});
