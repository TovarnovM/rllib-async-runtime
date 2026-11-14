from __future__ import annotations

from dataclasses import asdict, replace

import pytest
import ray

from rllib_async.protocols.episodes import RawMultiModuleEpisodeCodec
from rllib_async.replay.reference import EpisodeStore
from rllib_async.runtime import (
    PopulationAsyncSAC,
    PopulationMemberSpec,
    PopulationTrainable,
    RuntimeCheckpointState,
    RuntimeState,
)
from rllib_async.runtime.checkpoint import write_population_checkpoint
from rllib_async.runtime.population import _member_hparams, _team_contract
from tests.mixed_population_helpers import mixed_pbt, mixed_specs


def test_team_contract_allows_only_rates_and_seeds() -> None:
    first = mixed_specs()[0]
    changed = mixed_specs(seed=900, manager_lr=8e-4)[0]
    assert _team_contract(first) == _team_contract(changed)
    # Equal tensor shapes do not make different value supports compatible.
    altered = dict(first.module_configs)
    altered["manager"] = altered["manager"].copy(copy_frozen=False).training(v_max=99)
    other = PopulationMemberSpec.for_team(
        first.sac_config, altered, first.runtime_config,
        per_configs=first.per_configs,
    )
    assert _team_contract(other) != _team_contract(first)
    changed_rollout = first.sac_config.copy(copy_frozen=False)
    changed_rollout.normalize_actions = not first.sac_config.normalize_actions
    other = PopulationMemberSpec.for_team(
        changed_rollout, first.module_configs, first.runtime_config,
        per_configs=first.per_configs,
    )
    assert _team_contract(other) != _team_contract(first)


def test_mixed_population_rejects_semantics_and_loss_fitness_before_actors(
    monkeypatch,
) -> None:
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    specs = mixed_specs()
    changed_rollout = specs[1].sac_config.copy(copy_frozen=False)
    changed_rollout.environment(env_config={"episode_length": 11})
    incompatible = PopulationMemberSpec.for_team(
        changed_rollout, specs[1].module_configs, specs[1].runtime_config,
        per_configs=specs[1].per_configs,
    )
    with pytest.raises(ValueError, match="complete team contract"):
        PopulationAsyncSAC((specs[0], incompatible))
    with pytest.raises(ValueError, match="team return metric"):
        PopulationAsyncSAC(
            specs, pbt_config=replace(mixed_pbt(), metric_key="learner/loss")
        )
    with pytest.raises(ValueError, match="trainable team modules"):
        PopulationMemberSpec.for_team(
            specs[0].sac_config, specs[0].module_configs, specs[0].runtime_config,
            trainable_modules=("manager",), per_configs=specs[0].per_configs,
        )


def test_mixed_pbt_mutates_one_qualified_rate_reproducibly() -> None:
    original = _member_hparams(mixed_specs()[0])
    args = {"config": mixed_pbt(), "exploit_count": 3}
    result = PopulationAsyncSAC._mutate_hparams(original, **args)
    assert result == PopulationAsyncSAC._mutate_hparams(original, **args)
    updated, name, *_ = result
    assert name == "manager/lr"
    assert updated[name] == pytest.approx(original[name] * 0.8)
    assert {key: value for key, value in updated.items() if key != name} == {
        key: value for key, value in original.items() if key != name
    }


def test_invalid_donor_does_not_stop_or_mutate_recipient(monkeypatch) -> None:
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    population = PopulationAsyncSAC(mixed_specs(), pbt_config=mixed_pbt())

    class Recipient:
        state = RuntimeState.RUNNING
        stops = 0

        def validate_pbt_state(self, state):
            raise ValueError("incompatible donor")

        def stop(self):
            self.stops += 1

    target = Recipient()
    population._members["team-0"] = target
    before = population.runtime_member_ids
    with pytest.raises(ValueError, match="incompatible donor"):
        population._replace_target(
            "team-0",
            new_hparams=population._current_hparams["team-0"],
            donor_state=object(),
        )
    assert target.stops == 0
    assert target.state is RuntimeState.RUNNING
    assert population.runtime_member_ids == before
    assert population._generations == {"team-0": 0, "team-1": 0}
    assert population._restarting_slot is None


def test_mixed_metadata_resume_and_warm_start_keep_different_contracts(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    specs = mixed_specs()
    source = PopulationAsyncSAC(specs, run_id="source", pbt_config=mixed_pbt())
    source._report_index = 7
    settings = specs[0].runtime_config
    store = EpisodeStore(
        RawMultiModuleEpisodeCodec(),
        capacity_transitions=settings.replay_capacity_transitions,
        capacity_bytes=settings.replay_capacity_bytes,
        journal_capacity=settings.replay_journal_capacity,
    )
    states = {
        member.runtime_config.member_id: RuntimeCheckpointState(
            state_version=1,
            member_id=member.runtime_config.member_id,
            runtime_config=asdict(member.runtime_config),
            replay_file="replay.snapshot",
            replay_cursor=store.cursor,
            learner=b"only metadata is read in this test",
            rollout={}, evaluation=None, controller={},
        )
        for member in source._member_specs.values()
    }
    directory = tmp_path / "population"
    directory.mkdir()
    metadata = source._checkpoint_metadata().to_mapping()
    assert metadata["schema_version"] == 3
    write_population_checkpoint(
        directory, replay_state=store.export_state(), members=states,
        pbt_metadata=metadata,
    )
    resumed = PopulationAsyncSAC.from_checkpoint(
        specs, directory, pbt_config=mixed_pbt()
    )
    assert resumed.runtime_member_ids == source.runtime_member_ids
    assert resumed._report_index == 7
    assert resumed._current_hparams == source._current_hparams
    warm = PopulationAsyncSAC.from_warm_start_checkpoint(
        mixed_specs(seed=900, manager_lr=5e-4), directory,
        run_id="warm", pbt_config=mixed_pbt(),
    )
    assert warm.runtime_member_ids != source.runtime_member_ids
    assert warm._report_index == 0
    assert warm._current_hparams["team-0"]["manager/lr"] == 5e-4
    assert warm._base_module_seeds != source._base_module_seeds


def test_mixed_population_reserves_one_gpu_per_whole_team() -> None:
    placement = PopulationTrainable.default_resource_request(
        {"members": mixed_specs(num_gpus=1), "pbt": mixed_pbt()}
    )
    assert sum(bundle.get("GPU", 0) for bundle in placement.bundles) == 2
