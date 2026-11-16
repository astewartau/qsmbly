// A per-structure statistics table as the engine writes it (qsmxt-engine stages::analysis::HEADER), for tests.
const HEADER = 'map\tunit\tindex\tname\tn_voxels\tn_valid\tvolume_mm3\tmean\tsd\tmedian\tmin\tmax\tp5\tp95\treference\treference_offset_ppm\n';
/** @type {string} */
export const STATS = `${HEADER}Chimap\tppm\t10\tleft thalamus\t120\t120\t\t0.01\t0.002\t0.01\t0\t0.02\t0.005\t0.015\tthalamus\t0.012500\n`
  + 'Chimap\tppm\t49\tright thalamus\t118\t118\t\t0.012\t0.002\t0.012\t0\t0.02\t0.005\t0.015\tthalamus\t0.012500\n'
  + 'R2starmap\ts-1\t10\tleft thalamus\t120\t120\t\t30\t4\t30\t20\t40\t24\t36\t\t\n';

