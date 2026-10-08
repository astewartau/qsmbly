/**
 * TGV parameter resolution
 *
 * The exported config reports the user's TGV settings, so the run must use them too.
 * qsm-core's defaults fill in only what the user left unset.
 */

const isPositive = (x) => Number.isFinite(x) && x > 0;

/**
 * Resolve the alphas and iteration count a TGV run should use.
 *
 * @param {Object} tgvSettings - { regularization, iterations, alpha0, alpha1 } (any may be unset)
 * @param {Function} defaultAlpha - (regularization) => [alpha0, alpha1], qsm-core's preset levels
 * @param {Function} defaultIterations - () => number, qsm-core's voxel-size-adaptive count
 * @returns {{ alpha0: number, alpha1: number, iterations: number }}
 */
export function resolveTgvParams(tgvSettings, defaultAlpha, defaultIterations) {
  const s = tgvSettings || {};

  let alpha0 = s.alpha0;
  let alpha1 = s.alpha1;
  if (!isPositive(alpha0) || !isPositive(alpha1)) {
    [alpha0, alpha1] = defaultAlpha(s.regularization ?? 2);
  }

  const iterations = Number.isInteger(s.iterations) && s.iterations > 0
    ? s.iterations
    : defaultIterations();

  return { alpha0, alpha1, iterations };
}
