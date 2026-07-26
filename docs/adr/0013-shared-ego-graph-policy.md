# ADR 0013: batch ego-graphs inside one shared SAC module

- Status: Accepted
- Date: 2026-07-26

## Context

Phase 10 must prove that many homogeneous logical agents can use graph
observations without allocating one policy or one set of weights per agent. It
must keep the existing RLlib environment runner, authoritative episode replay,
module-specific materialized view, SAC learner, and checkpoint contracts.

RLlib and Gymnasium require a static observation space at the environment
boundary, while the replay and learner requirement is a batch of genuinely
variable-size ego-graphs. A single centralized graph forward over every agent
in an environment would require a different packed runner and is explicitly
outside `v0.1`.

Stock SAC also owns separate actor, critic, twin-critic, and target networks.
Sharing in this phase means that all logical agents map to one RLModule state;
it does not collapse SAC's distinct actor and critic networks into one encoder.

## Decision

`EgoGraphCoordinationEnv` exposes four simultaneous logical agents. Every agent
maps to the single `shared_graph` module and receives an ego-centric graph with
between one and four visible nodes. The finite example uses a shared
`Discrete(3)` SAC policy.

The environment transports graphs through one bounded Gymnasium `Dict` space:

- node and edge arrays are padded to the configured maximum;
- `node_count` and `edge_count` identify the live prefixes;
- `controlled_node` identifies the node governed by the logical agent.

Padding is not replay semantics. `GraphEpisodeCodec` validates each graph,
removes the padded suffixes, and stores a codec-distinct
`GraphEpisodePayload`. It reuses the sparse multi-module payload layout and
therefore also reuses authoritative commit, FIFO retention, snapshot/delta,
checkpoint, and uniform module-index semantics.

Node, edge, and optional action-mask feature dimensions are immutable codec
parameters. They are recorded both in `codec_id` and the payload, so an
incompatible producer or restore is rejected before mixed graphs reach a
learner sample.

`MultiModuleEpisodeRunner` now copies nested observation trees instead of
assuming one flat NumPy array. RLlib's `MultiAgentEnvRunner` still owns action
selection, agent batching, episode construction, and weight-sequence metadata.

`GraphBatchCollator` groups sampled transitions by module and packs current and
next observations independently:

- node features are concatenated;
- edge indices are shifted by each graph's node offset;
- `graph_ptr` delimits non-empty graphs;
- controlled-node indices become global packed indices;
- optional edge features and action masks are aligned and stacked.

The example does not use action masks. The collator preserves a supplied binary
mask, but Phase 10 does not define masked-SAC action semantics.

`SharedGraphSACCatalog` is the official RLlib customization point. It replaces
the stock MLP observation encoders with `TorchGraphEncoder` while leaving
RLlib's SAC module, learner, losses, optimizers, target updates, and
serialization in ownership of RLlib.

`TorchGraphEncoder` uses pure PyTorch:

1. project node features;
2. aggregate directed neighbor messages with `index_add_`;
3. apply mean aggregation for two configurable message-passing layers;
4. concatenate the controlled-node embedding with graph mean pooling;
5. produce one fixed-size embedding per graph.

It accepts both fixed-space padded rollout batches and packed replay batches.
No PyTorch Geometric dependency is added.

## Consequences

- Every logical agent uses the same `shared_graph` module state and version.
- A forward call handles the full module batch rather than constructing a
  model per agent.
- Replay contains true variable-size graph payloads, not padded maximum-size
  observations.
- Empty edge sets, one-node graphs, mixed graph sizes, optional graph leaves,
  gradient flow, delta synchronization, and checkpoint restore are covered.
- SAC remains responsible for the algorithm and for actor/critic separation.
- The padded-to-packed conversion is correctness-first Python/PyTorch code and
  remains a Phase 11 profiling target.
- Continuous graph SAC, semantic action masking, PyTorch Geometric, centralized
  environment-wide graph forwards, and generic graph-policy registration are
  outside Phase 10.

## Rejected alternatives

### Install PyTorch Geometric

Rejected because `index_add_` is sufficient for the required mean-aggregation
example, and a new compiled dependency adds operational cost without improving
the Phase 10 correctness gate.

### Flatten every graph into one fixed vector

Rejected because it would hide the variable-size collation boundary and would
not test graph message passing.

### Build one module per logical agent

Rejected because it violates the shared-policy requirement and scales weights,
optimizer state, and learner work with agent count.

### Centralize one graph over the whole environment

Rejected because it changes runner and policy semantics. A centralized packed
runner is a separate future phase, not an implication of this ego-graph
example.
