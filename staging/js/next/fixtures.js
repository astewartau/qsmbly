/**
 * The scan the workspace plans for until one is loaded: the example dataset's, as the engine
 * reads it from its sidecars.
 */

/**
 * The example dataset (js/app/config.js EXAMPLE_DATA): 184×256×128 at 1 mm, five echoes, its
 * sidecars' EchoTime and MagneticFieldStrength. The engine turns these into its facts
 * (`scan_facts_wasm`); the sidecars carry no B0 direction, which for the example is axial.
 */
export const EXAMPLE_SCAN = {
  dims: /** @type {[number, number, number]} */ ([184, 256, 128]),
  sidecars: [0.005, 0.011, 0.017, 0.023, 0.029].map((EchoTime) => ({ EchoTime, MagneticFieldStrength: 2.8946 })),
  hasMagnitude: true,
  b0Dir: /** @type {[number, number, number]} */ ([0, 0, 1]),
};
