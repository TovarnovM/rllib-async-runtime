# Architecture boundary

The project is a thin Ray-native runtime around RLlib components, not a new
implementation of SAC.

The central design is:

> Authoritative episode store + learner-specific materialized replay views.

Phase 0 contains no replay actor, async execution loop, hierarchy, or graph
encoder. It verifies only the RLlib seams that later phases depend on:

- a single-environment `SingleAgentEnvRunner` can return one complete episode;
- a local SAC `LearnerGroup` can update from replay-sampled episodes;
- inference weights can be installed between complete episodes;
- module, target-network, optimizer, and SAC temperature state can be restored;
- a standard RLlib `Algorithm` eagerly owns a control plane that the custom
  runtime must instead compose explicitly.

See [ADR 0001](adr/0001-runtime-boundary.md) for the orchestration decision and
[the implementation plan](IMPLEMENTATION_PLAN.md) for phase gates.
