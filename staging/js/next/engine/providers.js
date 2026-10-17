/**
 * The Pipeline and Command tabs' providers (the `Providers` of js/next/exports.js, qsmbly #167)
 * on the live engine: the graph (`pipeline_graph` with its Mermaid text), the methods text
 * sentence by sentence (`methods_structured`), the `qsmxt run`
 * command, the settings file (`to_toml_selected`) and reading one back (`from_toml_migrating`,
 * then the browser profile). Wiring the tabs live is one line in main.js:
 * `const providers = engineProviders(engine);` in place of the recorded ones.
 *
 * @typedef {import('./client.js').EngineClient} EngineClient
 * @typedef {Record<string, any>} PipelineConfig
 * @typedef {import('../protocol.js').ScanFacts} ScanFacts
 */

/**
 * @param {EngineClient} engine
 */
export function engineProviders(engine) {
  return {
    /** @param {PipelineConfig} config @param {ScanFacts} facts */
    graph: async (config, facts) => {
      const { mermaid, ...graph } = await engine.graph(config, facts);
      return { graph, mermaid };
    },
    /** @param {PipelineConfig} config @param {ScanFacts} facts */
    methods: async (config, facts) => ({ methods: await engine.methods(config, facts) }),
    /** @param {PipelineConfig} config */
    command: async (config) => ({ command: await engine.command(config) }),
    /** @param {PipelineConfig} config */
    toml: async (config) => ({ toml: await engine.toToml(config, true) }),
    /** @param {string} text */
    importToml: (text) => engine.fromToml(text),
  };
}
