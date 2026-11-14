# ADR 0015: Rainbow heads, local PER and mixed-team population

## Context

Stage 2 supplies raw shared replay, recurrent windows and mixed DQN/SAC teams.
Stage 3 adds prioritized sampling, Noisy DQN and population training without
creating another replay payload store, learner loss or controller event loop.
Ray remains pinned to 2.56.1. Its existing extension points are sufficient,
with two narrow DQN module corrections described below.

## Decision

### DQN heads and the C51 dueling correction

Keep RLlib's DQN loss, distributional projection, optimizers and target update
logic. Extend the shared action-masked module so vector, graph and recurrent
DQN use the same head semantics.

In pinned RLlib, `DQNCatalog` creates a scalar dueling value head, including
when `num_atoms > 1`. The default module centers distributional advantages
over atoms, then adds that scalar to every atom. Both operations add a
constant across the atom dimension and therefore cancel under softmax. The
value head cannot affect the resulting distribution.

For C51 with dueling enabled, create one value logit per atom and compute
`logits[a, z] = value[z] + advantage[a, z] - mean_actions(advantage[:, z])`.
Softmax remains over atoms; expected Q is the support-weighted distribution.
Nondueling C51 and ordinary scalar DQN retain their existing module path.
This changes the defective module construction, not the learner objective.

The pinned source boundaries are
[DQNCatalog](https://github.com/ray-project/ray/blob/ray-2.56.1/rllib/algorithms/dqn/dqn_catalog.py),
[DefaultDQNTorchRLModule](https://github.com/ray-project/ray/blob/ray-2.56.1/rllib/algorithms/dqn/torch/default_dqn_torch_rl_module.py)
and [DQNTorchLearner](https://github.com/ray-project/ray/blob/ray-2.56.1/rllib/algorithms/dqn/torch/dqn_torch_learner.py).
Numerical checks distinguish the corrected C51 dueling formula, nondueling
stock-module parity and stock-learner loss parity on identical corrected
modules. Preserving the broken stock dueling distribution is not a parity
requirement.

### Explicit NoisyNet lifetime

Reuse RLlib's installed
[NoisyLayer](https://github.com/ray-project/ray/blob/ray-2.56.1/rllib/models/torch/modules/noisy_layer.py)
parameter layout and factorized Gaussian transform. Its unmodified forward
samples on every call, including evaluation, and its weight-scale
initialization does not follow `sigma0`. A small subclass supplies controlled
noise, learned weight/bias scales and factorized NoisyNet initialization.
Only the existing advantage/value head linear layers are replaced; vector,
graph and recurrent observation encoders keep their existing ownership.

One training forward resets online and target noise once. Current and next
online Q calculations share that realization; target noise is independent.
One exploration forward resets online noise once. Inference uses learned
means only and consumes no noise RNG draws. Noise tensors are replaced,
rather than modified in place, so another forward cannot corrupt tensors
retained by an earlier autograd graph. Cached head lists avoid traversing
all model parameters in the hot path.

`DQNConfig.training(noisy=True, sigma0=...)` is authoritative. Contradictory
module settings fail at configuration time. The configured epsilon schedule
still applies to exploration; use `epsilon=0` for parameter-noise-only
exploration. Noisy mode does not silently discard epsilon.

Publication and checkpoints include learned means and scales, but exclude
ephemeral noise realizations. Existing CPU/CUDA Torch RNG checkpoint state
determines the next realization after restore. Target updates transfer model
parameters; target noise is drawn independently on the next training forward.

### Consumer-local prioritized sequence starts

Whole episodes remain the authoritative commit, deduplication and eviction
unit. A population has one authoritative replay actor, and each learner host
has one local payload replica shared by its module consumers. PER adds only
consumer-local coordinates, priorities and RLlib sum/min segment trees.
Different modules or population members can assign different priorities to
the same episode without modifying or retransmitting its payload.

The eligible population consists of learning-window starts, including short
tails. A handle identifies replay generation, episode, module, agent, local
start and sequence/horizon configuration. Sampling retains the existing
episode read lease. An unchanged replay cursor requires no index rebuild;
eviction removes corresponding local metadata, and stale feedback is ignored.
New starts receive the consumer's maximum priority. Repeated feedback for one
start is combined by maximum absolute error, then a positive epsilon is added.

For priority `p_i`, sampling probability is proportional to `p_i ** alpha`.
Importance weight is `(N * P(i)) ** (-beta)`, normalized by the maximum over
eligible starts. Repeat that window weight over its actual valid loss rows.
Prefix, padding and bootstrap continuation do not receive priority feedback.
`alpha=0` samples uniform eligible starts. The host waits for enough budget
for the configured PER window instead of changing its handle semantics.

The objective remains RLlib's existing per-row reduction and
algorithm-specific use of importance weights. This is a window-start
sampling contract, not a claim of an unbiased uniform-transition objective
for overlapping or unequal-length windows. Actor and temperature objectives
are not rewritten to introduce additional weighting.

Consume the learner's existing per-row priority signal once inside the local
adapter: absolute TD error for scalar DQN, unweighted distributional cross
entropy for C51, and mean absolute critic TD error for SAC. C51's signal must
not be mislabeled as scalar absolute TD error. Reduce valid learning rows to
one maximum absolute signal per sampled window. Remove the dense signal from
ordinary reports even when PER is disabled; no priority RPC, controller-side
tensor materialization or new episode transfer is introduced.

Exact resume saves priorities, slot ordering, maximum priority, replay
frontier, consumer configuration and sampler RNG alongside the existing
learner/optimizer/target and process RNG states. Saving values without slot
ordering would not preserve the next sampled batch.

### Whole-team population and PBT

Extend the existing population/member machinery to construct
`SingleMemberAsyncTeam` members. Each team retains one learner host, separate
module learners and clocks, one local replay replica and one coherent
inference publication. Resource reservations belong to the enclosing actors;
local module learners do not each reserve a Ray GPU.

PBT fitness is a team return. Mutations use module-qualified learning-rate
keys and cannot target frozen modules. Exploit transfers every module's
online/target model state, including frozen modules and SAC temperature.
It does not transplant optimizers, sampler/process RNG, priority tables,
training budgets or counters. The recipient starts a new member generation
with fresh local training state and reuses the authoritative shared replay.

The complete donor is validated before a healthy recipient is stopped:
module IDs, trainable set, sequence configuration, algorithm/model contracts,
space and parameter structure, and environment/routing/action semantics must
match. Learning rates are excluded from the model-transfer contract because
they are the supported mutations. The replacement publishes a complete team
bundle; independent actor/critic or manager/worker replacement is outside this
contract. Replacement failures use the existing population failure path.

Exact checkpoint resume restores the complete population training state;
model-only warm start and PBT intentionally start new training state. These
operations must not be presented as interchangeable forms of resume.

### Localized checkpoint compatibility

The C51 dueling correction changes the value-head shape from one output to
one output per atom. Such DQN checkpoints require
`c51_dueling_head_version=2`; old scalar-value C51 dueling checkpoints are
rejected before importing live learner state. There is no silent conversion
that claims to preserve their optimizer or next-update semantics.

Ordinary DQN and nondueling C51 do not acquire that incompatibility. Older
non-noisy DQN contracts missing the new `noisy`/`sigma0` fields are normalized
as non-noisy before existing strict validation. They cannot become Noisy
checkpoints through that compatibility path. Stage-2 mixed checkpoints may
omit PER metadata only when the current team also has no PER consumers.

The existing `PBTModelState` is extended with algorithm and configuration
identity. Its old five-field SAC serialization remains readable as legacy
SAC. DQN and mixed-team transfers require complete contracts, and parameter
fingerprints reject incompatible head structures. No parallel PBT checkpoint
framework or broad invalidation of earlier SAC checkpoints is introduced.

## Validation and consequences

Focused regressions cover the independent C51 formula and value gradients,
weighted stock-loss parity, Noisy train/exploration/inference behavior,
priority feedback and restored next-sample/next-update equivalence. Combined
graph Rainbow and recurrent DQN cases exercise n-step, masks, PER and model
restore. Population checks cover complete-team transfer, contract rejection,
shared replay, resume and fresh recipient optimizers.

These checks define acceptance; this ADR does not claim an executed CI, GPU,
learning-quality or performance result. Execution evidence and remaining
hardware gates belong in the stage status and GPU acceptance documents.
No new dependency, Ray upgrade, distributed priority service or replacement
learner loss is required by this design.
