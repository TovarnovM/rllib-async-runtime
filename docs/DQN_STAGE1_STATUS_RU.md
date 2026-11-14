# DQN, графовые энкодеры и маски: состояние этапа 1

Рабочая ветка: `feature/dqn-hierarchy-gnn`.
Основание: [ТЗ, раздел 22.1](DQN_HIERARCHY_GNN_SPEC_RU.md#221-этап-1-алгоритмическая-и-encoder-совместимость).
Версии зависимостей не меняются: Ray/RLlib 2.56.1, PyTorch 2.7.0.

## Объём изменения

Этот этап добавляет DQN в существующий одноагентный асинхронный runtime
и расширяет проверяемые сочетания алгоритмов и графовых энкодеров.
Loss, optimizer и обновление target networks остаются в RLlib.
Собственная реализация Bellman loss не добавляется.

Это ещё не реализация всего ТЗ. Смешанная DQN/SAC-команда в одном runtime,
иерархические интервалы действий, последовательности, рекуррентность,
n-step, PER и DQN-PBT требуют следующих этапов.

| Возможность | Граница реализации этапа 1 |
| --- | --- |
| Векторный DQN | `DQNLearnerAdapter`, `SingleMemberAsyncDQN`, `AsyncDQNTrainable` |
| Double / Dueling DQN | Штатные настройки и loss RLlib |
| C51 | Компонентная совместимость; сочетание с n-step/PER ещё не реализовано |
| Packed graph DQN | Модуль и каталог; отдельные forward для текущих и следующих графов |
| Discrete graph SAC | Графовый модуль, semantic action masks, штатный SAC learner |
| Continuous graph SAC | Штатный action-conditioned critic через multi-stream encoder |
| Actor / Q1 / Q2 | Независимые экземпляры энкодеров и отдельные конфигурации |
| Replay | Прежний authoritative store и learner-local view; uniform one-step sampling |
| Checkpoint / resume DQN | Веса, targets, optimizer, RNG learner, счётчики и курсор target update |
| Recurrent / Noisy DQN | Явный отказ для неподдерживаемых режимов |
| Mixed hierarchy / sequence replay / PER / DQN-PBT | Следующие этапы; не объявляются работающими |

Полный основной runtime этого этапа проверяется на **векторных наблюдениях
одной политики**. Графовые сочетания проверяются на уровне codec, packed
batch, RLModule и learner. Это не означает, что single-member controller уже
поддерживает произвольные Dict-наблюдения или смешанную графовую иерархию.

## DQN и существующий runtime

`SingleMemberAsyncDQN` использует существующий event pump, replay actor,
learner host, rollout group и evaluation. `AsyncDQNTrainable` использует
те же Tune resource/checkpoint механизмы. Настройки runtime доступны через
`AsyncRuntimeConfig`; это alias существующего `AsyncSACRuntimeConfig`,
сохраняющий старую сериализацию и импорты.

Алгоритмическая конфигурация задаётся через RLlib `DQNConfig`.
В Tune ключ называется `dqn_config`. Существующий SAC-ключ `sac_config`
не меняется. DQN принимает `Discrete(n)` с нулевым началом, а replay
передаёт целочисленные действия.

Сроки обновления target задаёт `DQNConfig.target_network_update_freq`.
Сериализуется также курсор последнего обновления: сохранения только весов
target недостаточно для корректного продолжения расписания.
Checkpoint различает DQN и SAC. Старые runtime checkpoints без метки
алгоритма трактуются как SAC; загрузка их в DQN запрещена.

Сохраняется Torch CPU/CUDA RNG. В текущей CI точное продолжение learner на
фиксированном batch проверено на CPU. Rollout actors при восстановлении получают следующую
generation и соответствующий seed; состояние среды посреди эпизода не
сохраняется. Побитовое повторение будущих траекторий не обещается.

### Счётчик epsilon

Для одноагентного этапа выбран **общий накопленный sampled-env-step счётчик
особи**, передаваемый runner вместе с очередным существующим запросом
сбора эпизода. Это явный выбор из вариантов раздела 15.4 ТЗ, а не
индивидуальное расписание каждого runner. Внутри runner счётчик монотонен.
Повторное создание runner и checkpoint/resume не начинают epsilon с нуля.

Передаётся один scalar, отдельный RPC не добавляется. Из-за асинхронного
завершения эпизодов runner видит задержанный общий счётчик. При изменении
числа runner расписание относится к совокупному объёму опыта, а не к
фиксированному числу эпизодов каждого runner. Evaluation остаётся
`explore=False` и не сдвигает обучающее расписание.

### Что остаётся прежним в replay

Эпизод — единица commit, синхронизации, eviction и смены behavior weights.
Переход — единица равномерного sampling. Один физический replay не
означает, что всем алгоритмам нужен одинаковый учебный batch.
Чанки и временные последовательности здесь ещё не вводятся.

В replay сохраняются исходные наблюдения, а графовые представления
пересчитываются энкодерами при обучении. Постоянного кэша обучаемых
embedding или actor hidden states в качестве истинного learner state нет.

## Графовые энкодеры

Используются `GraphDQNCatalog` / `GraphDQNTorchRLModule` и
`SharedGraphSACCatalog` / `GraphSACTorchRLModule`.
У DQN и SAC разные алгоритмические головы при общем графовом encoder seam.
Опциональный `model_config["graph_encoder"]` задаёт общие параметры:

```python
{
    "hidden_dim": 32,
    "message_layers": 2,
    "output_dim": 32,
    "readout": "controlled_node_and_mean",
}
```

Для SAC `model_config["graph_encoders"]` переопределяет настройки по ролям
`actor`, `q1`, `q2`. Это независимые trainable модули, а не три ссылки на
одни параметры. Target critics имеют собственные копии состояния.
Continuous critic получает и представление наблюдения, и действие;
градиент по действию доходит до actor.

`readout="mean"` позволяет самому энкодеру работать без controlled-node
readout. Контракт существующих `GraphEpisodeCodec` и `GraphBatchCollator`
по-прежнему требует поле `controlled_node`; на этом этапе оно остаётся
частью транспортной схемы.

Packed batch хранит графы без padding с `graph_ptr` и сдвинутыми индексами
рёбер. Следующее наблюдение может иметь другое число узлов и рёбер.
Поэтому Double DQN не склеивает `obs` и `next_obs` обычным `torch.cat`:
раздельные forward сохраняют правильные индексы и границы графов.

## Маски допустимых действий

Маска задаёт допустимость, а не численное значение Q. Поэтому действие с
Q=0 участвует в epsilon exploration, если оно разрешено.

- Greedy / exploration DQN выбирают только допустимые действия.
- Bellman target маскирует выбор следующего действия.
- Для SAC недопустимые действия имеют нулевую вероятность; энтропийный
  вклад задаётся конечным значением согласно `0 log(0) = 0`.
- Пустая текущая маска запрещена даже у перехода, заканчивающего эпизод.
- Пустая следующая маска допустима только при `terminated=True`.
- `truncated=True` само по себе не отменяет bootstrap и не разрешает
  пустую следующую маску.
- Для masked discrete SAC нужен явно заданный `target_entropy`;
  автоматическое значение по полному размеру action space не используется.
- Discrete graph SAC требует `twin_q=True` из-за контракта loss закреплённой
  версии RLlib. Continuous SAC сохраняет свой штатный режим.

Структурная валидация остаётся на границе приёма опыта и в опциональных
первых batch checks (`validate_startup=True`). Проверки допустимости маски
в выборе действия сохраняются: без них sampling и максимум могут быть
математически не определены. Это не повторный обход всего replay.

## Проверка и ограничения результатов

До изменений в алгоритмах на этой ветке выполнен полный CPU baseline:
**261 passed, 4 deselected**, 208.05 с на GitHub-hosted runner.
[Baseline CI](https://github.com/TovarnovM/rllib-async-runtime/actions/runs/43588663965).
Это время одной конкретной машины и запуска, не переносимая гарантия.

Новая реализация прошла полный CPU CI на commit
`2ff408c4f6a344437f87872848fc7ea26563ffea`:
**335 passed, 8 deselected**, 255.80 с; `ruff check` и `ruff format --check`
также успешны. [CI с результатами и длительностями](https://github.com/TovarnovM/rllib-async-runtime/actions/runs/43590957346).
Сценарий DQN runtime с publication, evaluation и checkpoint/resume занял
11.70 с. 74 дополнительных CPU-кейса проходят вместе с прежними проверками.
Следующий commit меняет только этот отчёт и не меняет проверенный код.

GPU-приёмка требует отдельного запуска в проектном devcontainer на целевой
машине. Старые CUDA-проверки graph SAC не доказывают CUDA-корректность
нового graph DQN, нового continuous graph SAC или их конфигураций.

| Проверка | Файл |
| --- | --- |
| Паритет со штатным DQN learner, target clock, optimizer/RNG resume | `tests/unit/test_dqn_adapter.py` |
| EnvRunner → replay → DQN learner → новые weights | `tests/compatibility/test_dqn_components.py` |
| Runtime, publication, evaluation, checkpoint/resume | `tests/integration/test_single_member_async_dqn.py` |
| Packed graph indices, actor/critic gradients, четыре learner updates | `tests/unit/test_graph_algorithms.py` |
| Greedy/epsilon, Q=0, Bellman mask, конечность SAC entropy | `tests/unit/test_action_masks.py` |
| Terminal / truncated mask на границе codec и batch | `tests/unit/test_shared_gnn.py` |
| Сохранение clocks при задержанном счётчике и пересоздании runner | `tests/unit/test_episode_runner.py`, `tests/unit/test_rollout_scheduler.py` |
| Старые SAC checkpoints и семантика module specs | `tests/unit/test_sac_adapter.py` |

Команды локальной проверки выполняются внутри devcontainer:

```bash
uv sync --locked --extra cu118 --group dev
uv run --locked --extra cu118 --group dev ruff check .
uv run --locked --extra cu118 --group dev ruff format --check .
uv run --locked --extra cu118 --group dev \
  pytest --durations=20 -m "not gpu and not cluster and not stress"
```

Новый полный runtime пример со стандартным CartPole:

```bash
uv run --locked --extra cu118 --group dev \
  python examples/dqn_single_member.py \
  --checkpoint-dir /tmp/dqn-stage1-checkpoint --stop-env-steps 512
```

Каталог checkpoint должен быть новым или пустым. Пример обучается,
сохраняет checkpoint, останавливает runtime, восстанавливается и продолжает
обучение с evaluation. Это проверка связности runtime, не бенчмарк качества.

Отдельно запустить четыре новые проверки на CUDA:

```bash
uv run --locked --extra cu118 --group dev \
  pytest -q -m gpu \
  tests/unit/test_graph_algorithms.py::test_graph_learner_update_has_finite_loss_and_updates_encoders
```

Ожидается **4 passed**. `skipped` означает отсутствие доступной CUDA,
а не успешную GPU-приёмку. Эти четыре теста проверяют learner updates;
полный DQN checkpoint/resume на CUDA отдельно ещё не подтверждён.

Для оценки ускорения нужны парные прогоны на одном оборудовании.
Эта работа не заявляет прироста throughput, сокращения реальных сетевых
байтов или улучшения reward без соответствующих измерений.
