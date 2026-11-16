// The export bundle's ZIP writer: what it writes, unzip (tests/e2e/unzip.js) and `unzip -l` read back.
import { zip, crc32 } from './zip.js';
import { unzip } from '../../tests/e2e/unzip.js';

const bytesOf = async (blob) => new Uint8Array(await blob.arrayBuffer());

describe('zip', () => {
  test('crc32 is the ZIP checksum', () => {
    expect(crc32(new TextEncoder().encode('123456789'))).toBe(0xcbf43926);
    expect(crc32(new Uint8Array(0))).toBe(0);
  });

  test('files come back as they went in, compressed or stored', async () => {
    const big = new Uint8Array(100_000).map((_, i) => i % 7);
    const files = [
      { path: 'a/command.txt', bytes: 'qsmxt run <bids_dir>\n' },
      { path: 'a/derivatives/qsmxt/sub-1/anat/sub-1_Chimap.nii', bytes: big },
      { path: 'a/ünïcode.md', bytes: '# Methods\n' },
    ];
    for (const compress of [true, false]) {
      const archive = await bytesOf(await zip(files, { compress }));
      const back = await unzip(archive);
      expect([...back.keys()]).toEqual(files.map((f) => f.path));
      expect(new TextDecoder().decode(back.get('a/command.txt'))).toBe('qsmxt run <bids_dir>\n');
      expect(back.get('a/derivatives/qsmxt/sub-1/anat/sub-1_Chimap.nii')).toEqual(big);
      if (compress) expect(archive.length).toBeLessThan(big.length / 4);
      else expect(archive.length).toBeGreaterThan(big.length);
    }
  });

  test('an empty archive is a valid ZIP', async () => {
    const archive = await bytesOf(await zip([]));
    expect(archive.length).toBe(22);
    expect((await unzip(archive)).size).toBe(0);
  });
});
