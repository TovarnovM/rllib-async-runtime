# MARL: MPE2, графовые SAC/DQN и PBT на кластере

## Назначение и границы

Рабочая ветка: `feature/dqn-hierarchy-gnn`. Два режима одного стенда:

- `continuous`: все агенты используют общую continuous SAC policy;
- `mixed`: первая половина агентов использует общую DQN policy, остальные —
  continuous SAC. Дискретный SAC здесь не применяется.

Одна особь PBT — вся команда. В каждой кампании один общий raw replay для всех
особей и алгоритмов, один learner-host на особь, отдельные DQN/SAC learners
внутри host. Энкодеры пересчитываются при обучении; латентные представления
в replay не сохраняются. SAC использует существующие actor/Q энкодеры проекта.
Централизованный критик совместных действий в этот стенд не входит.

Симулятор — закреплённый `mpe2==1.1.1`, Simple Spread. Физика и награды исходные.
Граф содержит агента, других агентов и ориентиры; связи двунаправленные от ego,
признаки рёбер — относительные смещения. Никаких скрытых скоростей других
агентов из `world` адаптер не читает. Пять действий DQN отображаются в native
continuous action через one-hot; соответствие исходной дискретной физике
проверяет тест. Завершение `max_cycles` остаётся truncation с bootstrap.

Начинать с 8 агентов, 100 шагов/эпизод, GNN hidden=64, batch=256. Вариант с
16 агентами запускать отдельной кампанией. PBT меняет только learning rates;
число агентов, графовая архитектура и пространства действий в популяции общие.

## Подготовка локальным агентом

Все команды ниже выполняются в проектном контейнере, не в Python окружении
хоста. Не устанавливать зависимости на вычислительные узлы во время замера.
На узлах без интернета заранее подготовить один образ и кеш зависимостей,
включая `marl` extra. Использовать существующие Slurm/Docker и локальный registry;
лаунчер не создаёт кластер, не управляет Slurm и не останавливает чужие Ray jobs.

```bash
git switch feature/dqn-hierarchy-gnn
git pull --ff-only
uv sync --locked --extra cu118 --extra marl --group dev
uv run --locked --extra cu118 --extra marl --group dev ruff check .
uv run --locked --extra cu118 --extra marl --group dev ruff format --check .
uv run --locked --extra cu118 --extra marl --group dev pytest \
  tests/unit/test_dqn_runtime.py tests/unit/test_stage3_evidence.py
uv run --locked --extra cu118 --extra marl --group dev pytest \
  -m 'not gpu and not cluster and not stress' --durations=20
uv run --locked --extra cu118 --extra marl --group dev pytest \
  tests/integration/test_marl_spread.py -m gpu \
  --junitxml=artifacts/marl-fractional-gpu.xml
```

Последняя команда должна действительно выполнить случай `shared-gpu-dqn-sac`,
а не пропустить его. Он проверяет CUDA-параметры обоих алгоритмов, размещение
двух learner-host на одной физической GPU, PBT и checkpoint/resume. Полный
набор GPU-проверок этапов 2–3 описан отдельно в существующем acceptance-документе.

Зафиксировать SHA, образ/digest, `uv.lock`, версии драйвера, Ray и PyTorch,
Slurm allocation, CPU affinity, выделенную RAM и `/dev/shm` каждого узла.
Рабочая директория и данные checkpoint должны быть доступны нужным Ray workers.
Не возвращать общий абсолютный `UV_PROJECT_ENVIRONMENT`: исправление `bd3812e`
использует локальную `.venv` каждого рабочего каталога Ray.

Перед масштабированием проверить фактическую скорость интерфейсов и доступные
ресурсы. 4×64/4×128/8×64/8×128 означают 256/512/512/1024 rollout-процессов,
плюс отдельные evaluation-процессы, learners, replay и управление. Стартовый
rollout budget — один CPU на процесс; не маскировать нехватку CPU нулевыми
ресурсными требованиями. В runtime env лаунчер задаёт OMP/MKL/OpenBLAS=1.

## Размещение и preflight

На двух RTX 3090: 4 особи по 0.5 GPU либо 8 по 0.25 GPU. Это совместное
использование карт, без разделения VRAM и без гарантии доли вычислительной
мощности. Все особи активны. DDP/MPI не используются.

Существующая Tune placement group размещает общий replay и все learner-host
в одном core bundle на GPU-узле. По умолчанию: 2 CPU на learner-host, 1 CPU
на replay, отдельный CPU на controller. Rollout/evaluation имеют отдельные
bundles. Проверить фактическое размещение по `node_id` в отчётах.

```bash
uv run --locked --extra cu118 --extra marl --group dev \
  python -m benchmarks.marl_cluster --mode continuous --address auto \
  --members 4 --workers 64 --num-gpus-per-member 0.5 \
  --preflight-only --output /results/marl/preflight-continuous-4x64
```

Preflight сохраняет потребность, доступные ресурсы и состав узлов. Проверяется
суммарная доступность и возможность уместить core bundle на одном узле.
Это снимок ресурсов, а не резервирование: свободное место на конкретном узле
может измениться. Для кампании нужна согласованная allocation; не конкурировать
с чужими работами. Путь `--output` должен быть новым для каждого запуска.

## Порядок запусков

Сначала малая конфигурация: 2 особи × 2–4 rollout, оба режима, GPU sharing,
не менее одного PBT exploit и восстановление. Затем 4×16 → 4×32 → 4×64 → 4×128.
После проверки ресурсов — 8×32 → 8×64 → 8×128. Каждая ступень — отдельный запуск;
не переходить выше, пока есть ошибки, растущие очереди или деградация обучения.

Целевой continuous запуск:

```bash
uv run --locked --extra cu118 --extra marl --group dev \
  python -m benchmarks.marl_cluster --mode continuous --address auto \
  --members 4 --workers 64 --num-gpus-per-member 0.5 \
  --reports 120 --report-interval-s 10 --pbt-interval-reports 6 \
  --output /results/marl/continuous-4x64-seed1
```

Целевой смешанный запуск:

```bash
uv run --locked --extra cu118 --extra marl --group dev \
  python -m benchmarks.marl_cluster --mode mixed --address auto \
  --members 8 --workers 64 --num-gpus-per-member 0.25 \
  --reports 120 --report-interval-s 10 --pbt-interval-reports 6 \
  --output /results/marl/mixed-8x64-seed1
```

Для 128 workers изменить только `--workers`, сохранить остальные параметры
для сравнения. Для первых малых прогонов поставить `--members 2 --workers 2`.
`--validate-startup` включает существующие одноразовые проверки контрактов;
не включать дорогую диагностику на каждом шаге.

Два режима запускать последовательно на одной allocation, иначе измеряется
их взаимная конкуренция. Конфигурация задаёт ограниченные очереди `high=2W`,
`low=W`; это стартовая настройка, а не обещание оптимальности. Отдельно отслеживать
сумму replay replicas, object store, journal и монотонные dedup metadata:
ограничение retained replay не означает ограниченную память всего процесса.

При resume сохранить параметры среды, популяции, ресурсов и seed, добавить
`--resume /path/to/checkpoint` и новый `--output`. Путь checkpoint печатается
после успешного запуска. Изменение размера популяции/архитектуры через exact
resume не поддерживается. Checkpoint выполняет согласованную паузу всех особей;
её учитывать отдельно и в полной длительности запуска.

## Мониторинг и результаты

Лаунчер создаёт по одному малому monitor actor на узел. Он запускает отдельный
процесс `benchmarks.cluster_monitor`: `/proc` и `nvidia-smi` читаются примерно
раз в секунду, вне event loop и вне rollout. Логи сжимаются локально, передаются
в итоговый каталог после обучения. Docker должен использовать host networking;
monitor должен видеть Ray-процессы в своём PID namespace. Если это не так,
запустить такой же sampler отдельно в корректном namespace и сохранить лог.

Сохраняются:

- CPU busy/iowait по каждому ядру и узлу, load average, доступная RAM;
- CPU/RSS по PID Ray-процессов, CPU в процентах одного ядра (может быть >100);
- RX/TX физических и виртуальных интерфейсов, speed, errors/drops, TCP counters;
- utilization, VRAM, power, temperature каждой GPU и VRAM по PID;
- Tune JSON/TensorBoard с прогрессом каждой политики, batch-building latency,
  learner update latency, policy lag, pending samples/commits и backpressure;
- поколения PBT, награды evaluation и время замен;
- checkpoint, конфигурация и preflight.

GPU utilization относится ко всей карте. Он не доказывает, что каждая особь
обучается: проверять `module_counters.optimizer_updates` и learned positions
отдельно. Legacy `data_wait_fraction=0` у mixed host не является измерением
отсутствия простоя; смотреть CPU/GPU, batch latency, частоту обновлений и очереди.
CPU сборка батча и передача на GPU пока не перекрываются отдельной prefetch
очередью. Решение о таком усложнении принимать по замерам этого стенда.

```bash
uv run --locked --extra cu118 --extra marl --group dev \
  python -m benchmarks.marl_report /results/marl/mixed-8x64-seed1
```

`summary.json` содержит распределения загрузки по узлам/интерфейсам/GPU и
последние метрики особей. Окно начинается с первого Tune report, поэтому
инициализация акторов не смешивается с основной нагрузкой. Исходные логи
сохраняют и startup, и финальный checkpoint. Не складывать RX+TX как один
поток и не суммировать физический интерфейс с его bridge/veth.

## Критерии приёмки локальной кампании

1. Ни один запрошенный GPU тест не пропущен; нет ошибок API, действий или графов.
2. Все особи и обе политики mixed-команды собирают опыт и обновляют параметры.
3. PBT меняет целую команду; остальные продолжают прогресс во время ожиданий.
4. После checkpoint/resume продолжаются обучение и PBT, сохраняются поколения.
5. Мониторинг всех узлов полный; `monitor-errors.json` пуст, GPU telemetry
   действительно присутствует на GPU-узле, физические интерфейсы идентифицированы.
6. Нет постоянно растущих очередей, OOM, actor failures или скрытого spilling.
   Зафиксировать также Ray object-store/spilling статистику и логи.
7. Для выводов о качестве нужны более длинные прогоны минимум на 3–5 seed и
   сравнение evaluation с начальной/случайной политикой. Короткий тест обновлений
   или высокий utilization не считается доказательством обучения задаче.
8. Сравнить полезные env/agent steps и learned positions за секунду, награду
   за время, простой и PBT downtime. Если увеличение workers не даёт выигрыша,
   указать измеренное ограничение: CPU симуляции/inference, replay, batch-building,
   GPU или NIC. Не раздувать данные и не ослаблять backpressure ради 100% сети.

Передать SHA, конфиги, JUnit, Tune logs/checkpoint, `node-*.jsonl.gz`, preflight,
summary и короткий вывод по каждой ступени. GPU/кластерные результаты здесь
не объявляются пройденными до запуска на целевом оборудовании.
