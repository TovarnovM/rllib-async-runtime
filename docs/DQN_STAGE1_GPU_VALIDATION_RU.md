# DQN/GNN stage1: локальная CUDA-приёмка

Дата: 14 ноября 2026 года. Ветка: `feature/dqn-hierarchy-gnn`.

**Итог: PASS в заявленных границах stage1.** Полный CPU-набор, Ruff и все шесть целевых CUDA cases прошли; GPU skips отсутствуют. Production-исправления не потребовались.

## Ревизии и границы

- Исходная ветка fast-forward до `f76f351341f70d56773560862f1a6c7c73ea32be`.
- Предыдущий CPU CI: `2ff408c4f6a344437f87872848fc7ea26563ffea`, 335 passed / 8 deselected; это внешний результат из [предыдущего отчёта](<DQN_STAGE1_STATUS_RU.md>), не подмена текущей проверки.
- Добавленные тесты: `0927aa02945ea1dbc2b405bd08161d7b05e66fc9`; на нём выполнен пилот.
- Финальная кодовая ревизия: `f5666416c6eeb01ba1fe6cfba4c90f471588f343`. Отличие от пилота — только перенос одного `assert` по выводу Ruff.
- Основной checkout и `main` сохранены на `b4eaec642d2cfd8651e53b974bbd3860fc4851fd`; работа ведётся в отдельном worktree. Этот отчёт добавлен отдельным documentation-only commit; проверенный исполняемый код соответствует `f566641`.

Проверяется ограниченный stage1: нерекуррентный uniform one-step DQN, single-member vector runtime, graph DQN/C51 и два graph SAC learner cases. Смешанная иерархия и stage2 не реализовывались.

## Окружение

Все `uv`, Python, Ruff и pytest выполняются в проектном devcontainer, собранном Dev Containers CLI из неизменённого repository-owned образа и запущенном существующим Slurm Docker runner. Host Python/venv не использовались.

| Параметр | Значение |
| --- | --- |
| GPU | workpc411/local, NVIDIA GeForce RTX 3090, 24 GiB, compute capability 8.6 |
| Видимость GPU | Только физическое устройство 0, явно согласованная подготовка контейнера; Torch видит ровно 1 CUDA device |
| Driver / CUDA build | 550.144.03 / 11.8 |
| Python / Torch | 3.11.13 / 2.7.0+cu118 |
| Ray/RLlib / NumPy | 2.56.1 / 2.4.6 |
| uv / pytest | 0.11.29 / 8.4.1 |
| CPU GPU-host | AMD Ryzen 9 5950X, 32 logical CPU |
| CPU gate | pc-411-01, Intel Xeon E5-2697 v4 @ 2.30 GHz, 72 logical CPU; CUDA в контейнер не предоставляется |
| Логические ресурсы job | 8 CPU, 24 GiB; GPU job — 1 gpu_slot; одна compute-job одновременно |
| Потоки / shared memory | OMP, MKL, OpenBLAS, NumExpr = 1; shm = 16 GiB |

Использован образ:

```text
workpc411:5000/rllib-verification@sha256:914bfee34f4d92d1323fc6e88a3c5de8963f4a155bd57115868a5d7404a97726
```

Lockfile, версии зависимостей, драйвер, общая Ray fixture и основной devcontainer config не менялись. Cached базовые слои сохранены; добавлено 830028 uncompressed bytes. `UV_OFFLINE=1` разрешает использовать уже имеющийся locked cache. В каждой job проверяются SHA, чистота checkout, реальный путь импортированного пакета и editable install: текущий frozen checkout, не seed-копия из образа.

## Команды и результаты

Команды ниже запускаются внутри контейнера. К pytest добавлен только `--junit-xml=<отдельный путь>` для сохранения доказательств. Полные stdout/stderr, exit code, фактическая командная строка, JUnit и внешний wall time сохранены отдельно.

```bash
uv sync --locked --extra cu118 --group dev
uv run --locked --extra cu118 --group dev \
  pytest -q -m gpu \
  tests/unit/test_graph_algorithms.py::test_graph_learner_update_has_finite_loss_and_updates_encoders

# Короткая проверка обоих существующих тестов: CPU + GPU, всего 4 cases.
uv run --locked --extra cu118 --group dev pytest -q \
  tests/unit/test_dqn_adapter.py::test_full_restore_preserves_optimizer_rng_and_target_update_schedule \
  tests/integration/test_single_member_async_dqn.py::test_dqn_runtime_publishes_evaluates_and_resumes

uv run --locked --extra cu118 --group dev ruff check .
uv run --locked --extra cu118 --group dev ruff format --check .
uv run --locked --extra cu118 --group dev \
  pytest --durations=20 -m "not gpu and not cluster and not stress"
uv run --locked --extra cu118 --group dev \
  pytest -q --durations=20 -m gpu \
  tests/unit/test_graph_algorithms.py \
  tests/unit/test_dqn_adapter.py \
  tests/integration/test_single_member_async_dqn.py
```

| Gate | SHA | Результат | Время pytest / внешнее, с |
| --- | --- | --- | --- |
| Исходные 4 graph GPU cases, усиленные device assertion | `0927aa0` | 4 passed, 4 deselected; 0 skipped | 10.57 / 14.54 |
| Adapter + runtime, CPU/GPU development cases | `0927aa0` | 4 passed; 0 skipped | 66.15 / 68.53 |
| Ruff check / format, финальные | `f566641` | PASS / PASS, 92 files formatted | — / 0.115 и 0.115 |
| Полный CPU-набор | `f566641` | 335 passed, 10 deselected; 0 skipped, 18 warnings | 690.64 / 694.07 |
| Финальные 6 GPU cases | `f566641` | 6 passed, 46 deselected; 0 skipped | 39.88 / 42.24 |

Полный CPU-набор выполнен **один раз**. Число CPU cases осталось 335; дополнительные 2 deselected относительно прежнего CI — новые GPU параметры существующих restore-тестов. Deselected GPU/stress cases не считаются успешными CPU-проверками; GPU skips не принимаются за PASS.

## Что именно доказано тестами

### Реальное размещение и graph updates

В [существующем graph-тесте](<../tests/unit/test_graph_algorithms.py>) сохранены четыре алгоритмических case: DQN, C51, continuous graph SAC и masked discrete graph SAC. Каждый выполняет update с finite loss и изменением ожидаемых encoder parameters.

[Один переиспользуемый test helper](<../tests/helpers.py>) вызывает штатный `LearnerGroup.foreach_learner` и читает `named_parameters()` живых модулей, включая targets. Для GPU все параметры должны иметь тип устройства `cuda`; карты `parameter → device` записаны в JUnit. Проверки наличия CUDA только у driver недостаточно и ею приёмка не ограничивается.

У runtime inspection выполняется **внутри learner actor** через существующий Ray `__ray_call__`, вне training loop. У original и restored learner проверяются реальные параметры и единственный accelerator ID. И в пилоте, и в финальном прогоне карты содержат `cuda:0`, а Ray resource inventory — `GPU=1.0`. Нового production RPC или проверки на каждом update нет.

### Adapter checkpoint / продолжение

[Существующий adapter restore test](<../tests/unit/test_dqn_adapter.py>) параметризован CPU/GPU; CPU case сохранён. Проверяются:

- первый настоящий update после learning-start действительно меняет online weights и выдаёт finite losses;
- полное состояние online/target modules, optimizer, update/sample/publication counters и опубликованные веса;
- точное восстановление CPU и CUDA Torch RNG state, затем повторение следующих случайных draws;
- original и restored выполняют следующий update на одном `fixed_batch` с NumPy seed 17; совпадают module/optimizer/target state, counters и publication;
- границы target schedule: 15 → cursor 0, 16 → 16, 31 → 16, 32 → 32; на 15/31 target weights неизменны, на 16/32 обновляются. Это проверяет restore **ненулевого** cursor, а не только начального нуля;
- прежние отказы при несовместимых config/model и недопустимом target clock сохранены.

Существующий float32-допуск `rtol=1e-6`, `atol=1e-7` не ослаблялся. Сравнивается продолжение на одном backend, а не bitwise CPU↔GPU parity. RNG states — byte tensors: их проверка не допускает изменения байтов.

### Runtime checkpoint / resume

[Существующий runtime test](<../tests/integration/test_single_member_async_dqn.py>) также параметризован CPU/GPU, без копирования тела. DQNConfig и AsyncRuntimeConfig согласованно запрашивают одну GPU.

Проверяются сбор и commit опыта, learner updates, publication и evaluation. Checkpoint создаётся на quiescent cut: authoritative replay, FastReplay и материализованный active cursor согласованы с checkpoint. CUDA payload остаётся opaque в управляющем процессе; значения для сравнения читаются у владельца через существующую диагностику.

Original runtime останавливается **до** создания restored runtime. После restore сохраняются learner/sample counters, publication count/version, rollout counters; затем обучение, публикации и сбор опыта продолжаются. Queue остаётся без prefetch, как в исходном case.

Epsilon использует накопленный member-wide sampled-env-step clock. Тест проверяет его сохранение и монотонное продолжение; поколения восстановленных runners увеличиваются на 1. В финальном CUDA case: clock 156 → 192, epsilon последнего действия 0.985275 → 0.981855. Это не требование одинаковых будущих environment trajectories: restored runners имеют следующее generation.

Новые диагностические RPC имеют timeout 10 с, training waits — 20 с, actor-death/GPU-return waits — конечные 5 с. Оба runtime освобождаются в `finally`; проверены завершение созданных replay/learner/rollout/evaluation actors и возврат GPU. Дополнительно весь процесс ограничен Slurm walltime; проверяется отсутствие Ray processes после pytest и контейнеров с точным job label после завершения job.

## Диагностика и минимальность изменений

1. Первая сборка завершилась до тестов: DNS не позволил `uv` обновить метаданные `hatchling==1.27.0`. Исходный лог сохранён. Повтор с `UV_OFFLINE=1` успешно использовал cached зависимости, без их обновления и без изменения сети.
2. Первый `ruff check` прошёл; format-check потребовал только перенос одного нового `assert`. Его точный formatter diff применён в `f566641`; CPU pytest до этого не запускался.
3. Продуктовых CUDA/restore ошибок ни в пилоте, ни в финальных gates не обнаружено. Изменён только тестовый код; production, RLlib losses, serializers, shared fixtures, зависимости и main не изменялись. Нового test framework нет: два существующих тела параметризованы, helper имеет три конкретных использования.

## Артефакты и cleanup

Корень игнорируемых локальных артефактов относительно основного checkout:

```text
artifacts/verification/20261114T131341Z-dqn-stage1-gpu/
```

Структура артефактов:

```text
PLAN.md, campaign.json, SUMMARY.md
preflight/, configs/, logs/, budget/
results/<task>/<command>/{stdout.log,stderr.log,exit.txt,command.txt,result.json,junit.xml}
```

JUnit существует у pytest-команд; JSON metadata содержит counts, длительности и parameter-device properties. Immutable source bundles и Slurm inputs также сохранены под `/cluster/campaigns/dqn-stage1-gpu-20261114T131341Z/`.

| Slurm job | Task | Итог | Allocation, с |
| --- | --- | --- | --- |
| 13638 | pilot-01 | PASS: graph GPU + CPU/GPU restore development cases | 88 |
| 13639 | cpu-01 | Format-check failure, pytest не запускался | 8 |
| 13640 | cpu-02 | PASS: Ruff + единственный полный CPU-набор | 702 |
| 13641 | gpu-01 | PASS: финальные 6 GPU cases | 48 |

Всего GPU-node allocations **136/1800 с**, CPU-node allocations **710/3600 с**, 4/10 разрешённых jobs. Все четыре terminal; контейнеров с их точными labels нет. В успешных pytest-командах нет оставшихся Ray processes после bounded grace. Посторонние процессы/контейнеры не останавливались. Локальные артефакты около 3.9 MB, shared inputs/results около 1.8 MB; большие логи, образы и checkpoint-файлы в Git не добавлены.

## Ограничения и незакрытые границы stage2

Эта приёмка не является измерением качества обучения, скорости или memory scaling. Не проверялись multi-GPU/multi-node, отказ узла, перенос checkpoint между разными CUDA device counts, все комбинации C51/masks или полноценный graph runtime checkpoint. Старые GPU population/SAC tests вне выбранных трёх файлов не входят в шесть целевых cases; CPU regressions проверяются полным набором.

Согласно [спецификации, §22.2](<DQN_HIERARCHY_GNN_SPEC_RU.md#222-этап-2-общий-опыт-иерархия-и-последовательности>), по-прежнему не закрыты:

- общий heterogeneous episode contract с durations/raw rewards и смешанный DQN/SAC consumer runtime;
- sequence views/reference oracle, n-step/SMDP target preparation;
- partial module readiness/updates, независимые clocks и несколько разноалгоритмических local learners в одном member/host при общем replay;
- causal manager/worker protocol, variable duration и goal boundaries;
- recurrent DQN/SAC, full-prefix oracle/burn-in, соответствующий checkpoint/next-update proof;
- DEMO-02/03 и их CPU/GPU acceptance matrix.

PER, Noisy/full Rainbow, mixed population/PBT и quality/performance campaigns относятся к следующим этапам и этой проверкой не реализованы. Закрытие stage1 CUDA gate не означает готовность смешанной иерархии или разрешение начать stage2 в рамках текущей задачи.
