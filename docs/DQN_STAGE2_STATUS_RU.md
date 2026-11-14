# Этап 2: общий опыт, смешанная иерархия и рекуррентность

Рабочая ветка: `feature/dqn-hierarchy-gnn`. Основание: раздел 22.2
[технического задания](DQN_HIERARCHY_GNN_SPEC_RU.md). Baseline этапа:
`b0ba209ca066d882046a986f14f5b437ff1edd81`, включая локальную GPU-приёмку этапа 1.
Зависимости не изменены: Ray/RLlib 2.56.1, PyTorch 2.7.0, Python 3.11.

## Что реализовано

| Возможность | Реализация и граница |
| --- | --- |
| Смешанный основной runtime | `SingleMemberAsyncTeam`; DQN и SAC в одной особи, существующий event loop |
| Общий опыт | Один authoritative replay actor и один `FastReplay` в learner-host |
| Независимые learners | Локальный adapter на модуль, общий host и одно резервирование GPU |
| Частичные обновления | Независимые readiness, intensity, learned positions, optimizer/target clocks |
| Замороженные исполнители | `trainable_modules=("manager",)`; worker state не обновляется |
| Raw schema 2 | Физическое время, durations, исходные награды по ticks, reset/local task boundaries |
| Причинная иерархия | Менеджер выбирает цель до первого действия worker; routing не двигает мир |
| Sequence replay | Локальные окна одной agent history, uniform starts, short tails и reader leases |
| n-step / SMDP | Горизонт по решениям, discount по сумме физических durations |
| Векторные и графовые GRU | Явные `RecurrentDQNTorchRLModule` и `RecurrentSACTorchRLModule` |
| Независимые actor / Q1 / Q2 | Отдельные параметры, размеры памяти, GNN specs и target representations |
| Full prefix | Восстановление состояния текущими параметрами каждого энкодера |
| Bounded burn-in | Явно приближённый режим, включая `burn_in=0` |
| Публикация и evaluation | Полный inference-only bundle с общим epoch; evaluation не пишет replay |
| Checkpoint | Learners, optimizers, targets, RNG, replay cursor, sampler frontier, budgets и clocks |

Loss, оптимизаторы и target updates остаются в RLlib. Собственная функция
Bellman loss не добавлялась. Новые recurrent forward возвращают плоский
набор только валидных обучаемых позиций; ragged histories позволяют не
добавлять padding в loss. Prefix и target-only continuation не расходуют
бюджет как обученные позиции.

У continuous SAC critic сначала рассчитывается память по observations,
затем добавляется проверяемое действие. Поэтому новое действие actor
не изменяет историю critic. Online и target сети восстанавливают состояния
по отдельности, своими весами.

Collector сохраняет физическое `executed_actions` отдельно. Учебное действие
получается обратным штатным преобразованием RLlib из фактически исполненного
действия, включая clipping; это сохраняет правильный action domain для SAC
при границах Box, отличных от `[-1, 1]`.

## Public API

```python
from rllib_async.examples.temporal_hierarchy import build_temporal_hierarchy_configs
from rllib_async.replay.sequences import SequenceConfig
from rllib_async.runtime import AsyncRuntimeConfig, SingleMemberAsyncTeam

rollout, modules, spaces = build_temporal_hierarchy_configs(
    recurrent=True,
    vector_worker=True,
)
settings = AsyncRuntimeConfig(
    runner_count=1,
    max_episode_steps=16,
    batch_queue_capacity=0,
)
# Ray должен быть инициализирован вызывающим кодом.
runtime = SingleMemberAsyncTeam(
    rollout,
    modules,
    settings,
    sequence_configs={
        module_id: SequenceConfig(learning_length=4, burn_in=None)
        for module_id in modules
    },
)
try:
    runtime.start()
    report = runtime.run_for(10)
finally:
    runtime.stop()
```

`rollout_config` задаёт среду, policy mapping и явные spaces. Каждый обычный
`DQNConfig`/`SACConfig` в `module_configs` задаёт свой алгоритм, модель,
`gamma`, `n_step`, batch size, warm-up и intensity. Конфигурации входных
объектов не изменяются. `runtime_config.batch_size` не переопределяет
индивидуальные `train_batch_size_per_learner`.
Существующие per-module overrides предварительно разрешаются в эффективную
конфигурацию: collator и learner используют одинаковые discount и horizon.
При restore дополнительно проверяется контракт среды, policy mapping и
преобразования действий до создания actors.

Для mixed path `training_intensity` означает число валидных обученных позиций
на новое совместимое решение; `None` соответствует 1. Положительное явное
значение задаёт повторное использование опыта. Бюджет приходит при успешной
синхронизации уникальных данных, а не из числа повторных RPC или текущей
ёмкости snapshot. Warm-up не накапливает долг за весь период ожидания.

`n_step` в конфигурации — число последовательных решений агента. В подготовленном
batch одноимённая колонка RLlib содержит сумму физических durations, чтобы
штатный loss использовал правильную степень `gamma`. Исходные rewards
сохраняются без discount и пригодны для другого `gamma`.

`validate_startup=True` включает разовую структурную проверку. Неудачная
проверка не считается успешно завершённой. Постоянные проверки временных
границ, протокола и совместимости checkpoint сохраняются.

## Сеть и лишняя работа

- Эпизод остаётся единицей commit/eviction; окна не передаются по сети.
- Один local replay обслуживает всех learners и critics этой особи.
- Padding графового observation space удаляется при упаковке эпизода.
- Stateless consumers не строят полный prefix или recurrent history tensors.
- Пустые delta не декодируют опыт и не перестраивают индексы.
- В controller приходят summaries и ссылки, а не полные episode payload.
- Счётчики решений модулей передаются существующим rollout RPC.
- Публикация полного inference bundle выполняется с заданным интервалом.
- Дорогие структурные проверки не выполняются на каждом event-loop tick.

Это описание поведения, а не измеренный процент снижения сетевого трафика.
Сравнительные network/memory/throughput measurements относятся к этапу 3.

## Время и окончания

Temporal environment предоставляет целый `world_t`. Один `env.step()`
продвигает мир на ноль ticks для routing или на один физический tick.
Положительная duration обязательна для завершённого учебного решения.
Финальное наблюдение должно реально поступить от среды.

Новый пример отличается от старого Phase 9: цель менеджера доступна worker
до первого действия. Каждая worker invocation — явно конечная локальная
подзадача с отдельным agent ID. Её окончание не завершает задачу менеджера.
Time-limit truncation сохраняет bootstrap; граница выбранного окна не
становится terminal.

Epsilon использует реальные решения конкретного модуля. На границах эпизодов
runner получает накопленные member-wide counts завершённых эпизодов, затем
продолжает локальный счётчик. Это задержанная оценка общей шкалы, без RPC
на действие. Target clock относится к совместимым данным своего consumer.

## Демонстраторы

Команды выполняются внутри devcontainer после установки закреплённого
CPU или CUDA окружения.

```bash
uv run --locked --extra cpu --group dev python examples/mixed_hierarchy.py hierarchy --mode joint --vector-worker --validate-startup
uv run --locked --extra cpu --group dev python examples/mixed_hierarchy.py hierarchy --mode frozen --memory full-prefix
uv run --locked --extra cpu --group dev python examples/mixed_hierarchy.py memory --algorithm dqn --observation graph --memory full-prefix
uv run --locked --extra cpu --group dev python examples/mixed_hierarchy.py memory --algorithm sac --observation vector --memory burn-in --burn-in 2
uv run --locked --extra cpu --group dev python examples/mixed_hierarchy.py memory --memory stateless
```

Для GPU заменить `--extra cpu` на `--extra cu118` и добавить `--num-gpus 1`.
CLI поддерживает `--seed`, `--steps`, `--updates`, `--timeout-s`, `--n-step`,
`--learning-length`, `--checkpoint`, `--restore` и `--output`.

Пороги CLI мягкие: завершаются находящиеся в работе эпизоды, проверяется
обновление каждого обучаемого модуля и evaluation новой публикации. Отчёт
показывает фактические counts и превышение порога. Такой smoke не доказывает
сравнение качества на строго одинаковом объёме данных.

## Проверки

Новые focused tests расположены в `test_replay_sequences.py`,
`test_recurrent_algorithms.py`, `test_temporal_hierarchy.py`,
`test_temporal_collector.py`, `test_team_config.py`, `test_team_transport.py`,
`test_mixed_learner.py` и `test_single_member_async_team.py`.
Проверки охватывают reference parity,
raw reward provenance от настоящего collector, causal order, физические
n-step targets, independent state/gradients, partial updates, publication,
checkpoint/resume и фактическое размещение learner parameters.

Полный CPU gate:

```bash
uv run --locked --extra cpu --group dev ruff check .
uv run --locked --extra cpu --group dev ruff format --check .
uv run --locked --extra cpu --group dev pytest --durations=20 -m "not gpu and not cluster and not stress"
```

Реальный GPU gate для локального агента: одна видимая GPU, пять новых cases
и шесть регрессионных cases этапа 1. Выполнять в проектном devcontainer.

```bash
export CUDA_VISIBLE_DEVICES=0
uv sync --locked --extra cu118 --group dev
uv run --locked --extra cu118 --group dev \
  pytest --durations=20 -m "gpu and not cluster and not stress" \
  tests/unit/test_recurrent_algorithms.py \
  tests/integration/test_single_member_async_team.py \
  tests/unit/test_graph_algorithms.py \
  tests/unit/test_dqn_adapter.py \
  tests/integration/test_single_member_async_dqn.py \
  --junitxml=stage2-gpu-junit.xml
```

Ожидается **11 passed, 69 deselected, 0 skipped**. GPU skip не считается
подтверждением поддержки. Проверки устройств выполняются до и после restore;
mixed runtime дополнительно проверяет все три learners внутри Ray actor и
единственное назначение GPU. Сохранить проверенный SHA,
команды, exit codes, counts, duration и parameter devices внутри Ray actor.
Численные тесты и короткий smoke не подменяют learning-quality кампанию.

## Ограничения

- Новая команда использует direct batching: `batch_queue_capacity=0`.
- Старые flat checkpoints обслуживаются прежними entry points. Sparse schema 1
  без durations/raw rewards не превращается в schema 2 посредством догадки.
- Recurrent core — agent-level GRU после vector/GNN encoder. Stock флаги
  `use_lstm`/`use_attention` не включают этот контракт.
- Ограниченный burn-in не гарантирует точного состояния из далёкого прошлого.
- Состояние среды внутри эпизода не сохраняется; после restore меняются runner
  generation и seed, как в прежнем runtime. Следующий learner update и
  воспроизведение будущей асинхронной траектории — разные гарантии.
- Mixed Tune population/PBT, PER и Noisy DQN остаются этапом 3.
- Физические streaming commits не добавлены: для них нет измеренного основания.

Подробности владения и выбранных расширений приведены в
[ADR 0014](adr/0014-mixed-teams-and-local-sequences.md).
