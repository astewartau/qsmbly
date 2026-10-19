/**
 * Reading back the export bundle in the tests.
 */
/**
 * The files in a ZIP archive js/next/zip.js wrote (stored or deflated), for the jest and e2e
 * tests: `{ path: bytes }`. Deflated files are inflated with `DecompressionStream`.
 * @param {Uint8Array} archive
 * @returns {Promise<Map<string, Uint8Array>>}
 */
export async function unzip(archive) {
  const view = new DataView(archive.buffer, archive.byteOffset, archive.byteLength);
  let end = archive.length - 22;
  while (end >= 0 && view.getUint32(end, true) !== 0x06054b50) end--;
  if (end < 0) throw new Error('not a ZIP archive');
  const count = view.getUint16(end + 10, true);
  let at = view.getUint32(end + 16, true);
  const decoder = new TextDecoder();
  /** @type {Map<string, Uint8Array>} */
  const out = new Map();
  for (let i = 0; i < count; i++) {
    const method = view.getUint16(at + 10, true);
    const packedSize = view.getUint32(at + 20, true);
    const nameLength = view.getUint16(at + 28, true);
    const extra = view.getUint16(at + 30, true) + view.getUint16(at + 32, true);
    const local = view.getUint32(at + 42, true);
    const name = decoder.decode(archive.subarray(at + 46, at + 46 + nameLength));
    const start = local + 30 + view.getUint16(local + 26, true) + view.getUint16(local + 28, true);
    const packed = archive.subarray(start, start + packedSize);
    if (method === 0) out.set(name, packed.slice());
    else if (method === 8) {
      const stream = new Blob([/** @type {BlobPart} */ (packed)]).stream().pipeThrough(new DecompressionStream(/** @type {any} */ ('deflate-raw')));
      out.set(name, new Uint8Array(await new Response(stream).arrayBuffer()));
    } else throw new Error(`${name}: compression method ${method} is not supported`);
    at += 46 + nameLength + extra;
  }
  return out;
}
