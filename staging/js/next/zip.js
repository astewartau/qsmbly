/**
 * A small ZIP writer for the export bundle: files in, one `.zip` out, with no dependency. Each
 * file is deflated where the browser can (`CompressionStream('deflate-raw')`) and it makes the
 * file smaller, else stored. No ZIP64: the bundle stays well under 4 GB.
 *
 * @typedef {{ path: string, bytes: Uint8Array | string }} ZipEntry - `path` uses `/`, no leading `/`
 */

/** @type {Uint32Array | null} */
let crcTable = null;

/**
 * The CRC-32 a ZIP records for each file.
 * @param {Uint8Array} bytes
 */
export function crc32(bytes) {
  if (!crcTable) {
    crcTable = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      crcTable[n] = c >>> 0;
    }
  }
  let crc = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) crc = crcTable[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

/**
 * `bytes` deflated (raw, as a ZIP holds it), or null where the browser cannot.
 * @param {Uint8Array} bytes
 * @returns {Promise<Uint8Array | null>}
 */
async function deflate(bytes) {
  if (typeof CompressionStream === 'undefined') return null;
  try {
    const stream = new Blob([/** @type {BlobPart} */ (bytes)]).stream().pipeThrough(new CompressionStream(/** @type {any} */ ('deflate-raw')));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  } catch {
    return null;
  }
}

/** DOS time and date of `when`, as ZIP headers hold them. @param {Date} when */
function dosTime(when) {
  const time = (when.getHours() << 11) | (when.getMinutes() << 5) | Math.floor(when.getSeconds() / 2);
  const date = ((Math.max(when.getFullYear(), 1980) - 1980) << 9) | ((when.getMonth() + 1) << 5) | when.getDate();
  return { time, date };
}

/**
 * A ZIP archive of `entries`, in their order.
 * @param {ZipEntry[]} entries
 * @param {{ compress?: boolean, when?: Date }} [options] - `compress` (default true): deflate where it helps
 * @returns {Promise<Blob>}
 */
export async function zip(entries, { compress = true, when = new Date() } = {}) {
  const encoder = new TextEncoder();
  const { time, date } = dosTime(when);
  /** @type {BlobPart[]} */
  const parts = [];
  /** @type {Uint8Array[]} */
  const central = [];
  let offset = 0;
  for (const entry of entries) {
    const name = encoder.encode(entry.path);
    const raw = typeof entry.bytes === 'string' ? encoder.encode(entry.bytes) : entry.bytes;
    const deflated = compress ? await deflate(raw) : null;
    const packed = deflated && deflated.length < raw.length ? deflated : raw;
    const method = packed === raw ? 0 : 8;
    const crc = crc32(raw);
    const local = new DataView(new ArrayBuffer(30));
    local.setUint32(0, 0x04034b50, true);
    local.setUint16(4, 20, true);
    local.setUint16(6, 0x0800, true); // UTF-8 names
    local.setUint16(8, method, true);
    local.setUint16(10, time, true);
    local.setUint16(12, date, true);
    local.setUint32(14, crc, true);
    local.setUint32(18, packed.length, true);
    local.setUint32(22, raw.length, true);
    local.setUint16(26, name.length, true);
    local.setUint16(28, 0, true);
    parts.push(local.buffer, /** @type {BlobPart} */ (name), /** @type {BlobPart} */ (packed));

    const head = new DataView(new ArrayBuffer(46));
    head.setUint32(0, 0x02014b50, true);
    head.setUint16(4, 20, true);
    head.setUint16(6, 20, true);
    head.setUint16(8, 0x0800, true);
    head.setUint16(10, method, true);
    head.setUint16(12, time, true);
    head.setUint16(14, date, true);
    head.setUint32(16, crc, true);
    head.setUint32(20, packed.length, true);
    head.setUint32(24, raw.length, true);
    head.setUint16(28, name.length, true);
    head.setUint32(42, offset, true);
    const record = new Uint8Array(46 + name.length);
    record.set(new Uint8Array(head.buffer), 0);
    record.set(name, 46);
    central.push(record);
    offset += 30 + name.length + packed.length;
  }
  const size = central.reduce((n, c) => n + c.length, 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true);
  end.setUint16(8, entries.length, true);
  end.setUint16(10, entries.length, true);
  end.setUint32(12, size, true);
  end.setUint32(16, offset, true);
  return new Blob([...parts, .../** @type {BlobPart[]} */ (central), end.buffer], { type: 'application/zip' });
}
