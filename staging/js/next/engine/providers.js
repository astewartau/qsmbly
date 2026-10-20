/**
 * The Pipeline and Command tabs' providers (the `Providers` of js/next/exports.js, qsmbly #167)
 * on the live engine: the graph (`pipeline_graph` with its Mermaid text; after a run, given its
 * record, `pipeline_graph_after`), the methods text sentence by sentence (`methods_structured`;
 * after a run, `methods_for_run`), the `qsmxt run`
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
    /** @param {PipelineConfig} config @param {ScanFacts} facts @param {string | null} [record] */
    graph: async (config, facts, record) => {
      if (record) {
        const { mermaid, differs, ...graph } = await engine.graphAfter(config, facts, record);
        return { graph, mermaid, differs };
      }
      const { mermaid, ...graph } = await engine.graph(config, facts);
      return { graph, mermaid };
    },
    /** @param {PipelineConfig} config @param {ScanFacts} facts @param {string | null} [record] */
    methods: async (config, facts, record) => ({
      methods: await (record ? engine.methodsAfter(config, facts, record) : engine.methods(config, facts)),
    }),
    /** @param {PipelineConfig} config */
    command: async (config) => ({ command: await engine.command(config) }),
    /**
     * @param {PipelineConfig} config
     * @param {import('../exports.js').TomlForm} [form]
     */
    toml: async (config, form = 'selected') => ({
      toml: await (form === 'changed' ? engine.toTomlChanged(config) : form === 'full' ? engine.toTomlFull(config) : engine.toToml(config, true)),
    }),
    /** @param {string} text */
    importToml: (text) => engine.fromToml(text),
  };
}
