# Vendored dcm2niix

`dcm2niix.js` and `dcm2niix.wasm` are the standard, non-JPEG build from
[`@niivue/dcm2niix` 1.3.20260724](https://www.npmjs.com/package/@niivue/dcm2niix/v/1.3.20260724).
The converter reports dcm2niix v1.0.20260724. The existing `index.js` and
`worker.js` adapters remain local to this application.

Source archive: https://registry.npmjs.org/@niivue/dcm2niix/-/dcm2niix-1.3.20260724.tgz

To update, replace both `dist/dcm2niix.js` and `dist/dcm2niix.wasm` together.
Run the unit suite and the optional local scan regression:

```sh
npm test -- --runInBand
node scripts/verify-troubleshoot-import.mjs /path/to/troubleshoot
```

The July 2026 converter fixes enhanced multi-echo DICOM handling. The older
May 2025 build produced one 4D volume per component for the Bruker sample,
with only the first echo time. QSMbly requires separate images and timing
metadata for all six echoes. The local regression compares voxel values and
spatial headers against the offline conversion without committing scan data.
