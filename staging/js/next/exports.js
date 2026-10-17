/**
 * The seams for the Pipeline and Command tabs: the run's graph (qsmxt-engine
 * `pipeline_graph::pipeline_graph` and `PipelineGraph::to_mermaid`), its methods text sentence by
 * sentence (`methods::methods_structured`), its `qsmxt run` command
 * (`generate_command`), its settings file (`PipelineConfig::to_toml_selected`) and reading a
 * settings file back (`PipelineConfig::from_toml_migrating`).
 *
 * The tabs only ever call a {@link Providers}: engine/providers.js answers on the live engine,
 * engine/recorded.js on recorded engine answers for the jest tests.
 *
 * @typedef {Record<string, any>} PipelineConfig
 * @typedef {import('./protocol.js').ScanFacts} ScanFacts
 *
 * @typedef {'main' | 'reliable'} Branch
 *
 * @typedef {object} GraphNode - qsmxt-engine `pipeline_graph::Node`
 * @property {string} id - the step's name (`mask`, `invert-reliable`)
 * @property {string} label
 * @property {string | null} algorithm
 * @property {string | null} algorithm_label
 * @property {string[]} settings - in the methods text's words
 * @property {Branch | null} branch - a two-pass run's per-pass steps
 * @property {string | null} enabled_by - the setting that asked for the step (an optional step)
 * @property {string | null} only_when - when the step does anything, if that is not known yet
 * @property {string[]} models
 * @property {string[]} reads - item keys
 * @property {string[]} writes
 *
 * @typedef {{ from: string, to: string, items: string[] }} GraphEdge
 *
 * @typedef {object} GraphItem - qsmxt-engine `pipeline_graph::ItemInfo`
 * @property {string} key
 * @property {string} label
 * @property {string | null} units
 * @property {string} kind
 * @property {string | null} bids_name
 * @property {boolean} published
 *
 * @typedef {object} PipelineGraph - qsmxt-engine `pipeline_graph::PipelineGraph`
 * @property {GraphNode[]} nodes - in run order
 * @property {GraphEdge[]} edges
 * @property {GraphItem[]} items
 * @property {boolean} two_pass
 *
 * @typedef {import('./protocol.js').Methods} Methods
 *
 * @typedef {{ graph: PipelineGraph, mermaid: string }} GraphResult
 * @typedef {{ methods: Methods }} MethodsResult
 * @typedef {{ command: string }} CommandResult
 * @typedef {{ toml: string }} TomlResult
 * @typedef {{ config: PipelineConfig, warnings: string[] }} ImportResult
 *
 * @typedef {object} Providers
 * @property {(config: PipelineConfig, facts: ScanFacts) => Promise<GraphResult>} graph
 * @property {(config: PipelineConfig, facts: ScanFacts) => Promise<MethodsResult>} methods - the
 *   methods text of a run of `config` on the scan, each sentence with the graph's boxes it describes
 * @property {(config: PipelineConfig) => Promise<CommandResult>} command - `qsmxt run <bids_dir> …`
 * @property {(config: PipelineConfig) => Promise<TomlResult>} toml - the settings file
 * @property {(text: string) => Promise<ImportResult>} importToml - rejects with the reason a file
 *   cannot be read

 */

/**
 * Split a `qsmxt run` command into its words and its flags: `[['qsmxt', 'run', '<bids_dir>'],
 * ['--qsm-algorithm', 'tgv'], …]`. Quoted values stay whole.
 * @param {string} command
 * @returns {string[][]}
 */
export function commandParts(command) {
  const words = command.match(/'[^']*'|"[^"]*"|\S+/g) ?? [];
  /** @type {string[][]} */
  const parts = [[]];
  for (const w of words) {
    if (w.startsWith('--')) parts.push([w]);
    else parts[parts.length - 1].push(w);
  }
  return parts.filter((p) => p.length);
}

/**
 * The command as shown and copied: `<output_dir>` after `<bids_dir>`, and one flag per line.
 * @param {string} command - as `generate_command` writes it
 * @returns {string}
 */
export function formatCommand(command) {
  const [head, ...flags] = commandParts(command);
  if (!head) return '';
  const first = head.includes('<output_dir>') ? head : head.flatMap((w) => (w === '<bids_dir>' ? [w, '<output_dir>'] : [w]));
  return [first, ...flags].map((p) => p.join(' ')).join(' \\\n  ');
}
