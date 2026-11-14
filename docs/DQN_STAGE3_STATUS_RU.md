# Этап 3: Rainbow, локальный PER и смешанная популяция

Рабочая ветка: `feature/dqn-hierarchy-gnn`. Основание —
[ТЗ, раздел 22.3](DQN_HIERARCHY_GNN_SPEC_RU.md#223-этап-3-rainbow-популяция-и-измеренная-эффективность).
Реализация и приёмка разделены: добавленный код не означает автоматически
подтверждённое качество, скорость или успешный GPU-прогон.

## Реализованное поведение

### Один replay, независимые приоритеты

`PrioritizedSequenceSampler` использует существующий `FastReplay` и штатные
sum/min segment trees RLlib. Он хранит локальные sample handles, priorities,
индексы и selection frontier, не создавая второй буфер payload. Каждый learner
может иметь собственное распределение выборки на одном и том же raw опыте.
Sampling и обновление priorities не обращаются к replay actor.

Контракт окна фиксирован для sampler: module, learning length, burn-in/prefix
и n-step. Бюджет обучения не меняет этот контракт. Короткие хвосты оплачиваются
по фактическому числу valid learning positions. `alpha=0` сохраняет uniform
распределение и RNG следующей выборки.

Importance weight выбранного окна применяется ко всем его valid positions.
Используется штатный reduction learner по valid rows; это не утверждение
о равномерном распределении по всем индивидуальным transitions. Priority
последовательности — максимум локальных сигналов по valid positions.
Padding не участвует; дубликаты объединяются; stale handles после eviction
или смены generation не применяются к новому sample.

Checkpoint сохраняет priorities, расположение slots и RNG следующей выборки.
Неактивный sampler не удерживает устаревший полный read view; выбранное окно
сохраняет собственный reader lease до завершения обработки.

### Rainbow и штатные learner losses

Поддерживается совместное сочетание graph DQN + Double + Dueling + C51 +
n-step + NoisyNet + PER. Отдельно покрыт recurrent graph DQN + n-step +
NoisyNet + PER. Action masks сохраняют прежний контракт допустимых действий.

NoisyNet использует factorized noise на устройстве learner, отдельный шум
online/target сетей и сброс перед forward. Evaluation использует средние
параметры, не расходуя RNG. Шумовые buffers не публикуются и не сохраняются
как параметры модели. Epsilon остаётся самостоятельным механизмом; полный
Noisy-профиль явно задаёт `epsilon=0`.

В закреплённой версии Ray исправлена локальная композиция C51 с dueling:
value-head выдаёт по значению на атом, advantage центрируется по действиям.
Скалярный value-head прежнего варианта сокращался при softmax. Старые полные
checkpoint именно C51+dueling отклоняются до изменения живой модели;
совместимые обычные DQN/SAC checkpoint сохраняют прежний путь восстановления.

Losses берутся из штатных DQN/SAC learners. Для SAC IS weights затрагивают
critic loss согласно контракту RLlib, а не actor/temperature losses.
Сигнал C51 для PER — distributional cross-entropy, не абсолютный scalar TD
error. Массивы этих сигналов извлекаются локально один раз и исключаются
из публикуемой сводки метрик.

### Популяция, PBT и восстановление

`PopulationMemberSpec.for_team` подключает смешанную команду к существующим
`PopulationAsyncSAC` / `PopulationTrainable`. Все особи активны, authoritative
replay общий. Второй population event loop не вводится.

Exploit переносит всю совместимую команду: online/target models и SAC
temperature. Проверка algorithms, spaces, модели и семантики команды проходит
до остановки получателя. Получатель получает свежие optimizers, PER/samplers,
RNG, generation и нулевой исторический training debt. Разрешены scalar LR
mutations с module prefix, например `manager/lr` или `worker_1/actor_lr`.
Мутации graph topology и произвольное смешивание несовместимых команд не входят
в контракт.

Exact checkpoint, в отличие от exploit, восстанавливает optimizer/sampler/RNG
state. Shared replay записывается один раз. Metadata смешанной популяции имеет
schema 3; существующая SAC schema 2 сохраняется. Warm start использует прежнюю
семантику нового run с перенесёнными моделями и опытом.

Согласованный checkpoint сначала завершает pending rollout commits всех
команд, затем повторно синхронизирует learners без обучения. Все member cursors
должны совпасть с общим replay snapshot. Это устраняет найденную гонку, когда
последний commit второй команды ещё не входил в сохранённые priorities первой.
Дополнительная синхронизация выполняется только при checkpoint.

DEMO-04: [`examples/mixed_population.py`](../examples/mixed_population.py).
Интеграционная проверка проводит обучение двух команд с Rainbow-менеджером,
публикацию новых весов, evaluation, replacement, checkpoint, exact resume
и warm start с контролем завершения акторов.

## Проверки и доказательства

Базовый CPU gate этапа 2: commit `6d22868bc1a9d04754409aabc4b0c9192648ca58`,
427 passed, 15 deselected, 0 skipped. Полный зелёный CPU CI этапа 3:
commit `ef3a51f06e7670a130b32cd059180bb612ebd32b`,
[run 43625394145](https://github.com/TovarnovM/rllib-async-runtime/actions/runs/43625394145).
**Ruff check PASS; Ruff format PASS; 479 passed, 18 deselected, 0 skipped,
24 warnings, 294.31 s.** Три новых GPU cases входят в deselected; это не GPU PASS.
Самый дорогой новый integration case — mixed population, 48.02 s с PBT,
exact resume и warm start. Результаты разных hosted runner запусков не
используются как доказательство ускорения тестового набора.

Новые проверки покрывают PER probabilities/IS weights, tail aggregation,
eviction и reader leases, точный следующий sample/update после restore,
численную C51/dueling семантику, Noisy RNG, штатные weighted losses,
whole-team preflight и lifecycle популяции. Обычный CI сохраняет весь прежний
CPU набор; GPU/cluster/stress не входят в ежедневный запуск.

## Результаты первого CPU-эксперимента

Исходный код: `5ae907e8d7728008b9b4a174cbedfaf023fe7c52`.
[Workflow и журналы](https://github.com/TovarnovM/rllib-async-runtime/actions/runs/43624747327),
[архив 34 исходных файлов](https://github.com/TovarnovM/rllib-async-runtime/actions/runs/43624747327/artifacts/14125034496).
SHA-256 архива:
`05c6d910e98483644ff3b050ae7847bbe1891dbbdc30f79fa3e21b4de13abbac`.
Артефакт Actions хранится 14 дней; для долговременной приёмки его нужно архивировать.

Компонентный recurrent Rainbow/PER case прошёл все gates: 384 valid positions
по трём модулям, 48 optimizer updates, 1.322 s обучения; sample/decode/collate
заняли 0.0391 s. Это одна короткая CPU-точка, не steady-state GPU throughput.
Подтверждены общий payload, его неизменность, ноль replay RPC при обучении,
отсутствие rebuild на пустых delta и реальные CPU parameter devices.

Flat SAC сравнивался с baseline `6d22868bc1a9d04754409aabc4b0c9192648ca58`
в пяти чередующихся парах на одном runner и одном lock. Медиана отношения
candidate/baseline: **0.98477 по времени на world step**, **0.98479 по времени
на learner update**. Размах парных отношений относительно медианы — 6.36%
и 6.02%; объём обучения сопоставим. Порог регрессии 10% пройден. Это не
доказательство переносимого ускорения; aggregate peak memory не измерена.

Первый memory quality experiment: 320 world ticks на seed, 8192 valid positions
на вариант, прежние пять training seeds и отдельные evaluation streams.

| Training seed | Stateless return | Full-prefix return |
| --- | ---: | ---: |
| 20261114 | -0.04 | -0.04 |
| 20261115 | -0.04 | 1.00 |
| 20261116 | 0.04 | 0.04 |
| 20261117 | -0.04 | 0.04 |
| 20261118 | -0.04 | -0.04 |
| Среднее | **-0.024** | **0.200** |

**Заранее заданный критерий не выполнен.** Задачу решил один из пяти seeds;
небольшая разница двух постоянных действий не считается освоением памяти.
Исходный отрицательный результат сохраняется. Workflow завершился failure
из-за quality gate, несмотря на успешные component и flat regression gates.

До следующего запуска фиксируется отдельный эксперимент: те же данные,
пять seeds, evaluation, архитектуры, target schedule и пороги; единственное
изменение — **32768 valid positions** на вариант вместо 8192. Это увеличение
learning budget в четыре раза, не переоценка исходного результата. В окнах,
начинающихся после cue, prefix восстанавливает состояние без градиента;
только часть выбранных окон обучает перенос cue от начала до terminal loss.
Поэтому проверяется достаточность исходного бюджета, без изменения sampler
или recurrence semantics ради прохождения порога.

## Результат второго quality-эксперимента

Commit с предварительно зафиксированным новым бюджетом:
`b0599ee9231b92ef10f9b7aa5477a9458e2d1a26`.
[Успешный workflow](https://github.com/TovarnovM/rllib-async-runtime/actions/runs/43626039733),
[исходный JSON в архиве](https://github.com/TovarnovM/rllib-async-runtime/actions/runs/43626039733/artifacts/14126361092).
SHA-256 архива:
`99093a2c4db7279c3777998a8b0e69f83a1bf92bac2a5a66e6d82a2d091ad37d`.

При **32768 valid positions** все пять full-prefix моделей получили return
**1.00**, среднее и стандартное отклонение между training seeds — **1.00 / 0.00**.
Stateless результаты остались `[-0.04, -0.04, 0.04, -0.04, -0.04]`, среднее
**-0.024**. Средняя парная разность **1.024**, положительна во всех пяти парах.
Все заранее сохранённые пороги пройдены. На одну модель приходится 100
evaluation episodes из пяти отдельных streams; одинаковые streams используются
обоими методами, они не являются новыми training interactions.

CPU training wall times: stateless **4.01–4.24 s**, full-prefix **10.64–12.22 s**
на seed; это время learner budget, отдельно от создания акторов и evaluation.
Стоимость recurrence явно выше. Результат подтверждает обучение памяти
в этой небольшой vector-DQN задаче на фиксированном raw dataset. Он не
доказывает превосходства Rainbow, GNN или иерархии на прикладных задачах,
качество на GPU либо достаточность первоначального бюджета 8192.

## Измерительные инструменты

[`benchmarks/team_acceptance.py`](../benchmarks/team_acceptance.py) собирает
фиксированный raw dataset настоящим collector и обучает варианты на одинаковом
valid-position budget. Это offline comparison; repeated sampling не считается
новыми world steps. Сохраняются wall times, process RSS, CUDA allocator peaks,
actual devices, payload digest и namespace NIC counters. NIC counters не
выдаются за атрибуцию трафика конкретному learner.

Memory quality experiment заранее фиксирует пять training seeds и отдельные
evaluation streams. Порог и команды приведены в
[объединённом плане приёмки](DQN_STAGE2_STAGE3_GPU_ACCEPTANCE_RU.md).
Неудачный критерий сохраняется как незакрытый; порог не меняется ради PASS.

[`benchmarks/compare_flat_revisions.py`](../benchmarks/compare_flat_revisions.py)
сравнивает неизменённый flat SAC benchmark с baseline этапа 2 в двух isolated
worktrees на одной машине и одном lock. Пять пар чередуют порядок исполнения;
проверяется сопоставимое число learner updates на world step. Порог времени
проверяется и на world step, и на learner update, чтобы меньший объём обучения
не создавал ложный PASS. Шум масштаба 10% даёт `inconclusive`, а не выбор
лучшего прогона. Измеренные отдельно driver
RSS и largest-child RSS не являются aggregate peak memory всей системы.

Workflow `Stage 3 CPU experiments` запускается вручную или commit marker
`[stage3-validation]`; `[stage3-quality]` выполняет только продолжение quality
с бюджетом 32768, без повторения неизменённых component/flat измерений.
Он работает отдельно от обычного CI и сохраняет исходные JSON/logs
и результаты критериев; диагностическое профилирование отделено от timing runs.

Для PER-варианта benchmark с точным valid-position budget используется
`--learning-length 1`: фиксированное более длинное окно может оставить
нерасходуемый остаток кредита. Full-prefix recurrence и n-step при этом
сохраняются; multirow PER проверяется отдельными функциональными тестами.

## Открытая приёмка

- GPU этапов 2 и 3: **не запускалась в этой среде**. Единый runner требует
  сначала CPU prerequisite, затем 11 GPU cases этапа 2, 2 алгоритмических
  cases этапа 3 и 1 двух-GPU lifecycle case. Skips не принимаются за PASS.
- CPU memory quality на пяти seeds и парный flat-SAC time regression gate
  закрыты приведёнными экспериментами. Дополнительные graph/SAC/burn-in и
  Rainbow ablations остаются отдельными исследовательскими точками плана;
  результаты vector-DQN не переносятся на них автоматически.
- Aggregate peak memory всей системы и межузловой traffic требуют измерений
  на целевой машине/topology. Локальные process/NIC counters не закрывают
  эти пункты автоматически.
- Перенос GPU-приёмки согласован пользователем; незапущенные проверки не
  обозначаются как успешные. Подробный исполняемый план:
  [DQN_STAGE2_STAGE3_GPU_ACCEPTANCE_RU.md](DQN_STAGE2_STAGE3_GPU_ACCEPTANCE_RU.md).

Архитектурные основания и compatibility boundary:
[ADR 0015](adr/0015-stage3-rainbow-per-mixed-population.md).
