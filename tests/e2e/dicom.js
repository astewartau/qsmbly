// A small two-echo GRE as classic MR DICOM files, written here so the browser tests need no
// scan data: a magnitude series and a phase series, one file per slice and echo, as a Siemens
// scanner exports them. Only the attributes dcm2niix reads to build images and sidecars are set.

const SOP_CLASS = '1.2.840.10008.5.1.4.1.1.4'; // MR Image Storage
const EXPLICIT_LE = '1.2.840.10008.1.2.1';
const ROOT = '1.2.826.0.1.3680043.10.1234';

/** An even-length value, padded as DICOM pads it. */
function pad(buf, vr) {
  if (buf.length % 2 === 0) return buf;
  return Buffer.concat([buf, Buffer.from([vr === 'UI' || vr === 'OB' || vr === 'OW' ? 0 : 0x20])]);
}

/** One explicit-VR little-endian element. */
function element(group, el, vr, value) {
  let data;
  if (Buffer.isBuffer(value)) data = value;
  else if (vr === 'US') data = Buffer.from(Uint16Array.from([value]).buffer);
  else if (vr === 'UL') data = Buffer.from(Uint32Array.from([value]).buffer);
  else data = Buffer.from(String(value), 'latin1');
  data = pad(data, vr);
  const long = ['OB', 'OW', 'SQ', 'UN', 'UT'].includes(vr);
  const head = Buffer.alloc(long ? 12 : 8);
  head.writeUInt16LE(group, 0);
  head.writeUInt16LE(el, 2);
  head.write(vr, 4, 'latin1');
  if (long) head.writeUInt32LE(data.length, 8);
  else head.writeUInt16LE(data.length, 6);
  return Buffer.concat([head, data]);
}

/** A DICOM Part 10 file from its dataset elements (in tag order). */
function file(sopInstance, elements) {
  const meta = Buffer.concat([
    element(0x0002, 0x0001, 'OB', Buffer.from([0, 1])),
    element(0x0002, 0x0002, 'UI', SOP_CLASS),
    element(0x0002, 0x0003, 'UI', sopInstance),
    element(0x0002, 0x0010, 'UI', EXPLICIT_LE),
    element(0x0002, 0x0012, 'UI', `${ROOT}.1`),
  ]);
  return Buffer.concat([
    Buffer.alloc(128), Buffer.from('DICM', 'latin1'),
    element(0x0002, 0x0000, 'UL', meta.length), meta,
    ...elements,
  ]);
}

/**
 * The files: `{ name, mimeType, buffer }` for Playwright's setInputFiles.
 * @param {{ size?: number, slices?: number, echoTimes?: number[] }} [options] - echo times in ms
 */
export function dicomSeries({ size = 32, slices = 12, echoTimes = [5, 10] } = {}) {
  const files = [];
  let instance = 0;
  for (const [series, part] of [[2, 'M'], [3, 'P']]) {
    const seriesUid = `${ROOT}.2.${series}`;
    echoTimes.forEach((te, e) => {
      for (let z = 0; z < slices; z++) {
        const pixels = new Uint16Array(size * size);
        for (let y = 0; y < size; y++) {
          for (let x = 0; x < size; x++) {
            const r = Math.hypot(x - size / 2, y - size / 2, (z - slices / 2) * 2) / (size * 0.4);
            const inside = r < 1;
            if (part === 'M') pixels[y * size + x] = inside ? Math.round(1500 * Math.exp(-te / 40)) : 20;
            else {
              // Phase as Siemens stores it: 0..4095 for -π..π.
              const rad = inside ? 0.004 * te * (x - size / 2) * (y - size / 2) / 8 : 0;
              const wrapped = Math.atan2(Math.sin(rad), Math.cos(rad));
              pixels[y * size + x] = Math.round(((wrapped + Math.PI) / (2 * Math.PI)) * 4095);
            }
          }
        }
        instance++;
        const sop = `${ROOT}.3.${instance}`;
        const buffer = file(sop, [
          element(0x0008, 0x0008, 'CS', `ORIGINAL\\PRIMARY\\${part}\\ND`),
          element(0x0008, 0x0016, 'UI', SOP_CLASS),
          element(0x0008, 0x0018, 'UI', sop),
          element(0x0008, 0x0020, 'DA', '20260101'),
          element(0x0008, 0x0030, 'TM', '120000'),
          element(0x0008, 0x0060, 'CS', 'MR'),
          element(0x0008, 0x0070, 'LO', 'SIEMENS'),
          element(0x0008, 0x103e, 'LO', part === 'M' ? 'gre_qsm' : 'gre_qsm_phase'),
          element(0x0010, 0x0010, 'PN', 'Phantom'),
          element(0x0010, 0x0020, 'LO', 'phantom'),
          element(0x0018, 0x0020, 'CS', 'GR'),
          element(0x0018, 0x0023, 'CS', '3D'),
          element(0x0018, 0x0050, 'DS', '2'),
          element(0x0018, 0x0080, 'DS', '30'),
          element(0x0018, 0x0081, 'DS', String(te)),
          element(0x0018, 0x0086, 'IS', String(e + 1)),
          element(0x0018, 0x0087, 'DS', '3'),
          element(0x0018, 0x1314, 'DS', '15'),
          element(0x0020, 0x000d, 'UI', `${ROOT}.4`),
          element(0x0020, 0x000e, 'UI', seriesUid),
          element(0x0020, 0x0011, 'IS', String(series)),
          element(0x0020, 0x0013, 'IS', String(e * slices + z + 1)),
          element(0x0020, 0x0032, 'DS', `-32\\-32\\${(z * 2 - slices).toFixed(1)}`),
          element(0x0020, 0x0037, 'DS', '1\\0\\0\\0\\1\\0'),
          element(0x0020, 0x0052, 'UI', `${ROOT}.5`),
          element(0x0028, 0x0002, 'US', 1),
          element(0x0028, 0x0004, 'CS', 'MONOCHROME2'),
          element(0x0028, 0x0010, 'US', size),
          element(0x0028, 0x0011, 'US', size),
          element(0x0028, 0x0030, 'DS', '2\\2'),
          element(0x0028, 0x0100, 'US', 16),
          element(0x0028, 0x0101, 'US', 12),
          element(0x0028, 0x0102, 'US', 11),
          element(0x0028, 0x0103, 'US', 0),
          ...(part === 'P' ? [element(0x0028, 0x1052, 'DS', '-4096'), element(0x0028, 0x1053, 'DS', '2')] : []),
          element(0x7fe0, 0x0010, 'OW', Buffer.from(pixels.buffer)),
        ]);
        files.push({ name: `IM${String(instance).padStart(4, '0')}.dcm`, mimeType: 'application/dicom', buffer });
      }
    });
  }
  return files;
}
