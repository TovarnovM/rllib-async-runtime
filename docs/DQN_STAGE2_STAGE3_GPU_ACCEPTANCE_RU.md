# Совместная приёмка DQN: этапы 2 и 3

## Статус и смысл кампании

GPU-приёмка этапа 2 по согласованию с пользователем объединена с проверками
этапа 3. Этот документ и исполняемый runner задают порядок локальному агенту;
они **не являются результатом GPU-прогона**. Прежняя GPU-приёмка этапа 1
не доказывает корректность добавленных recurrent/PER/Noisy/PBT путей.

Рабочая ветка: `feature/dqn-hierarchy-gnn`. Проверяется один чистый commit
с закреплённым `uv.lock`. Все Python-команды выполняются в devcontainer.
Планируемая машина: два RTX 3090; достаточно одного GPU на learner-host.
DDP, MPI и обучение одной модели на двух GPU здесь не используются.

## Порядок исполнения

Runner [`benchmarks/validate_dqn_gpu.py`](../benchmarks/validate_dqn_gpu.py)
выполняет блоки строго последовательно, каждый в новом процессе:

| Блок | Видимые GPU | Требуемый результат |
| --- | --- | --- |
| CPU prerequisite | Нет | Полный `not gpu and not cluster and not stress`, без skips |
| Stage 2 compatibility | Первый выделенный GPU | 11 passed, без skips |
| Stage 3 Rainbow/PER | Тот же выделенный GPU | 2 passed, без skips |
| Stage 3 mixed population | Два выделенных GPU | 1 passed, без skips |

Runner сохраняет существующее выделение `CUDA_VISIBLE_DEVICES`, включая UUID
и Slurm allocation. Для первого GPU берётся первый элемент выделения,
для популяции — первые два. Физические устройства вне выделения не используются.
Если двух GPU нет, общая кампания завершается ошибкой до тестов. Для отдельной
диагностики можно выполнить pytest вручную, но это не полная приёмка.

Перед запуском внутри devcontainer:

```bash
set -euo pipefail
git switch feature/dqn-hierarchy-gnn
git status --short
uv sync --locked --extra cu118 --group dev
uv run --locked --extra cu118 --group dev ruff check .
uv run --locked --extra cu118 --group dev ruff format --check --diff .

run_id="dqn-stage23-$(git rev-parse --short HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"
artifact_dir="artifacts/validation/$run_id"

uv run --locked --extra cu118 --group dev \
  python -m benchmarks.validate_dqn_gpu \
  --output "$artifact_dir" \
  --block-timeout-s 1800
```

Каталог `--output` должен быть новым. Runner сам проверяет чистоту дерева,
снимает SHA, hash `uv.lock`, версии Python/uv/Ray/Torch/CUDA, GPU inventory.
Никакие generated artifacts не коммитить в ветку.

Отдельный `--collect-only` фиксирует ожидаемые node IDs непосредственно перед
каждым блоком. Затем проверяются exit code, точное множество JUnit cases,
отсутствие failures/errors/skips и CUDA device properties каждого GPU-case.
Исчезнувший тест или скрытый CPU fallback не считается успешной приёмкой.
При первой ошибке следующие блоки не запускаются; manifest получает `failed`.

## Что именно проверяется

### Stage 2: один GPU, 11 cases

| Тест | Cases | Обязательное доказательство |
| --- | ---: | --- |
| `tests/unit/test_graph_algorithms.py::test_graph_learner_update_has_finite_loss_and_updates_encoders` | 4 | DQN, C51, continuous/discrete SAC: finite loss и обновлённые encoder weights на CUDA |
| `tests/unit/test_dqn_adapter.py::test_full_restore_preserves_optimizer_rng_and_target_update_schedule` | 1 | Optimizers, RNG, target clock и следующий update после restore |
| `tests/integration/test_single_member_async_dqn.py::test_dqn_runtime_publishes_evaluates_and_resumes` | 1 | Полный runtime, publication, evaluation, checkpoint/resume |
| `tests/unit/test_recurrent_algorithms.py::test_recurrent_learner_update_and_next_update_after_restore` | 4 | Vector/graph DQN и discrete/continuous SAC; независимые памяти, GRU updates, следующий update |
| `tests/integration/test_single_member_async_team.py::test_mixed_recurrent_runtime_publishes_evaluates_and_resumes` | 1 | Смешанная команда, один host/GPU, общий replay, отдельные clocks, согласованный restore |

Количество deselected cases может меняться с добавлением CPU-тестов.
Критичны выбранные 11 node IDs, положительные assertions и ноль skips.
JUnit содержит устройства исходных и восстановленных моделей. Runtime-case
дополнительно проверяет фактическое Ray-назначение GPU и завершение своих акторов.

### Stage 3: алгоритмы, один GPU, 2 cases

Файл: `tests/unit/test_rainbow.py`.
Тест: `test_rainbow_per_next_sample_and_update_after_restore`.

- `rainbow-graph`: graph DQN + Double + Dueling + C51 + n-step + Noisy + PER
  и action masks.
- `recurrent-graph`: recurrent graph DQN + Double + Dueling + n-step + Noisy
  + PER и action masks; expectation-Q вариант.

Проверяются реальные CUDA-параметры, неизменность raw payload, следующий PER
sample с handles/probabilities/IS weights, RNG Noisy и следующий update
после восстановления моделей и optimizer state. Это не заменяется проверкой
одного только совпадения online weights.

### Stage 3: популяция, два GPU, 1 case

`tests/integration/test_mixed_population.py::test_mixed_population_replacement_checkpoint_and_resume[two-gpu]`.

Два learner-host получают разные выделенные GPU. Внутри каждого host все
локальные DQN/SAC learners используют его единственный GPU. Проверяются
общий authoritative replay, согласованная замена команды, fresh optimizers
при exploit, checkpoint и resume. JUnit сохраняет пары
`original_parameter_devices` / `restored_parameter_devices` и
`original_accelerator_ids` / `restored_accelerator_ids`.

Обязательный результат — разные назначения двум host, а не просто строка
`cuda:0` в каждом процессе: Ray перенумеровывает выделенные устройства внутри
актора. Нельзя интерпретировать две локальные строки `cuda:0` как один GPU.

## Отдельные измерительные и исследовательские запуски

Они не входят в обычный CPU CI и не должны увеличивать время каждого PR.
Используется [`benchmarks/team_acceptance.py`](../benchmarks/team_acceptance.py).
Это **offline comparison на фиксированном raw dataset**, а не измерение
онлайн-скорости среды или доказательство качества на прикладной задаче.

Один dataset собирается настоящим collector через среду, затем используется
всеми сравниваемыми вариантами данной training seed. World steps считаются
один раз при сборе. Повторные выборки не увеличивают число взаимодействий.
Каждому модулю выдаётся точный бюджет valid learning positions; фактическое
число optimizer updates записывается отдельно и может различаться.
В offline benchmark `target_network_update_freq=0`: targets обновляются при
каждом optimizer update. Production sampling clock не подделывается новыми
world steps; при фиксированном dataset он естественно перестаёт расти.

PER измеряется с `--learning-length 1`: production sampler сохраняет размер
окна, поэтому при multirow windows остаток бюджета меньше окна ожидает
новых данных. Однопозиционное окно позволяет этому offline harness потратить
ровно заданный бюджет. Full-prefix/burn-in и n-step targets сохраняются;
multirow PER проверяется функциональными тестами. Для сравнения uniform/PER
использовать `--learning-length 1` в обоих запусках.

Короткая проверка измерительного пути на CPU:

```bash
uv run --locked --extra cu118 --group dev \
  python -m benchmarks.team_acceptance components \
  --sampling per --dqn-profile rainbow --learning-length 1 \
  --world-steps 32 --learned-positions 128 \
  --output "$artifact_dir/components-cpu.json"
```

Измерение смешанной команды на одном GPU, пять повторов с указанными seeds:

```bash
uv run --locked --extra cu118 --group dev \
  python -m benchmarks.team_acceptance components \
  --num-gpus 1 --sampling per --dqn-profile rainbow --learning-length 1 \
  --memories stateless full-prefix burn-in \
  --seeds 20261114 20261115 20261116 20261117 20261118 \
  --world-steps 256 --learned-positions 4096 \
  --output "$artifact_dir/components-gpu.json"
```

Счётчики `/proc/net/dev` — фактические counters сетевого namespace, включая
фоновые пакеты. Они не являются атрибуцией трафика конкретной модели.
Локальный Ray может передавать payload через shared memory без сетевого
интерфейса. Такой результат не доказывает величину трафика между узлами.
Для межузлового вывода требуется отдельный запуск на целевой topology
с внешним сбором NIC counters на каждом узле. Не суммировать одинаковые
namespace counters, снятые несколькими процессами на одном узле.

Отдельно сохраняются реальные process RSS/peak RSS, CUDA allocator peaks,
estimated retained payload capacity, actual module devices, decode/collate
wall time, общий синхронизированный CUDA training wall time и стоимость
финальной публикации. Estimated payload bytes не выдаются за RSS или трафик.
CUDA allocator peaks не включают всю память драйвера и Ray object store.

Проверки harness подтверждают:

- разные локальные sampling views удерживают исходные payload references;
- training, включая local PER, работает при запрещённом replay RPC;
- пустые delta не вызывают перестроение индекса;
- payload hash остаётся прежним после всех encoder updates;
- каждый модуль потратил ровно заявленный valid-position budget.

Actor/Q1/Q2 получают один выбранный batch. Это свойство learner path
дополнительно проверяется numerical/ownership tests; benchmark не создаёт
отдельный replay или выборку на critic.

Профилирование запускать отдельно, с меньшим бюджетом:

```bash
uv run --locked --extra cu118 --group dev \
  python -m benchmarks.team_acceptance components \
  --sampling per --learning-length 1 --learned-positions 128 \
  --profile-dir "$artifact_dir/profiles" \
  --output "$artifact_dir/components-profiled.json"
```

`cProfile` здесь запускается внутри learner actor и включает CPU вызовы
sampling/collation/encoder/learner, а не только ожидание driver. Его timings
нельзя смешивать с чистыми timing runs. Для разделения CUDA prefix/forward/
backward kernels использовать отдельный Torch/Nsight profile на целевой машине;
обычный production loop такого сбора не делает.

## Заранее заданное сравнение памяти

Training seeds: `20261114, 20261115, 20261116, 20261117, 20261118`.
Evaluation seed streams: `910001, 910002, 910003, 910004, 910005`; каждый даёт
20 отдельных evaluation episodes. Они одинаковы для методов и отделены от
training. Длина эпизода — 5, сбор — 320 world ticks на training seed,
обучение — 8192 valid positions на вариант. Stateful и memoryless actor
получают одинаковые наблюдения без скрытой подсказки в поздних observations.

```bash
timeout --signal=TERM --kill-after=30s 1800s \
  uv run --locked --extra cu118 --group dev \
  python -m benchmarks.team_acceptance quality \
  --memories stateless full-prefix --require-quality \
  --output "$artifact_dir/memory-vector-dqn-quality.json"
```

До получения результатов зафиксирован критерий: средний return full-prefix
не ниже `0.6`, mean return stateless по модулю не выше `0.3`, средняя парная
разность не ниже `0.4`, положительная разность минимум в четырёх seeds.
В этой задаче return равен `+1/-1`, поэтому `0.6` соответствует 80% успеха;
идеальная memoryless policy без информации о cue имеет ожидаемый return 0.
Это экспериментальный acceptance criterion, не уже измеренный результат.

`--require-quality` завершает команду ошибкой при незакрытом критерии,
сохраняя все raw returns. Не менять seeds/порог после неудачи ради PASS:
дальнейший бюджет или гиперпараметры оформить новым экспериментом и сохранить
первый результат. Одиночный smoke seed не является quality evidence.

Исходный CPU run с бюджетом 8192 не прошёл quality gate (full-prefix mean
0.20, stateless -0.024; задачу решил один seed). Его данные сохранены в
[отчёте этапа 3](DQN_STAGE3_STATUS_RU.md). Следующий отдельно зафиксированный
эксперимент меняет только learning budget на 32768, сохраняя те же seeds,
dataset budget и пороги:

```bash
uv run --locked --extra cu118 --group dev \
  python -m benchmarks.team_acceptance quality --require-quality \
  --learned-positions 32768 \
  --output "$artifact_dir/memory-vector-dqn-quality-32768.json"
```

Дополнительные точки выполняются тем же entry point:

| Сравнение/точка | Аргументы сверх предыдущей команды |
| --- | --- |
| Graph вместо vector | `--graph` |
| Continuous recurrent SAC | `--algorithm sac` |
| Приближённый burn-in | `--memories stateless full-prefix burn-in --burn-in 2` |
| C51 относительно expectation-Q | `--dqn-profile c51` |
| Rainbow + локальный PER | `--dqn-profile rainbow --sampling per --learning-length 1` |

Каждая точка получает отдельный output filename. DQN и continuous SAC имеют
разные action semantics и сравниваются как разные задачи; это не discrete
DQN-versus-SAC ablation. Такое сравнение потребует отдельной discrete-SAC
конфигурации с теми же действиями.

## Что вернуть после кампании

Локальный агент возвращает архив каталога evidence, его SHA-256 и короткий
отчёт: checked SHA/lock hash, hardware, CPU counts, результаты трёх GPU-блоков,
время и exit codes, фактические device assignments, причины любых failures,
путь к JUnit/logs/JSON/profiles. После изменения проверяемого кода затронутый
блок повторить на новом SHA; старые logs не переименовывать в новый результат.

Нормальное завершение закрывает собственные Ray actors/runtime. Runner
ограничивает каждый test subprocess таймаутом и останавливает его process group;
после timeout кампания считается failed. Каталоги `*-ray` сохраняются для
диагностики. При остаточных Ray processes завершать только процессы этой
кампании; не выполнять глобальный `ray stop` на общей рабочей машине.

GPU, quality и performance выводы записываются раздельно. Зелёные численные
тесты не означают выигрыша reward или throughput; быстрый benchmark не
заменяет checkpoint/device correctness.
