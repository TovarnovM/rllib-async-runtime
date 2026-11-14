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

DEMO-04: [`examples/mixed_population.py`](../examples/mixed_population.py).
Интеграционная проверка проводит обучение двух команд с Rainbow-менеджером,
публикацию новых весов, evaluation, replacement, checkpoint, exact resume
и warm start с контролем завершения акторов.

## Проверки и доказательства

Базовый CPU gate этапа 2: commit `6d22868bc1a9d04754409aabc4b0c9192648ca58`,
427 passed, 15 deselected, 0 skipped. Результаты этапа 3 будут зафиксированы
здесь после завершения CI и отдельных экспериментов.

Новые проверки покрывают PER probabilities/IS weights, tail aggregation,
eviction и reader leases, точный следующий sample/update после restore,
численную C51/dueling семантику, Noisy RNG, штатные weighted losses,
whole-team preflight и lifecycle популяции. Обычный CI сохраняет весь прежний
CPU набор; GPU/cluster/stress не входят в ежедневный запуск.

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
`[stage3-validation]`, отдельно от обычного CI. Он сохраняет исходные JSON/logs
и результаты критериев; диагностическое профилирование отделено от timing runs.

Для PER-варианта benchmark с точным valid-position budget используется
`--learning-length 1`: фиксированное более длинное окно может оставить
нерасходуемый остаток кредита. Full-prefix recurrence и n-step при этом
сохраняются; multirow PER проверяется отдельными функциональными тестами.

## Открытая приёмка

- GPU этапов 2 и 3: **не запускалась в этой среде**. Единый runner требует
  сначала CPU prerequisite, затем 11 GPU cases этапа 2, 2 алгоритмических
  cases этапа 3 и 1 двух-GPU lifecycle case. Skips не принимаются за PASS.
- Полный closure требует результатов quality и performance экспериментов,
  а не только численных unit/integration tests.
- Aggregate peak memory всей системы и межузловой traffic требуют измерений
  на целевой машине/topology. Локальные process/NIC counters не закрывают
  эти пункты автоматически.
- Перенос GPU-приёмки согласован пользователем; незапущенные проверки не
  обозначаются как успешные. Подробный исполняемый план:
  [DQN_STAGE2_STAGE3_GPU_ACCEPTANCE_RU.md](DQN_STAGE2_STAGE3_GPU_ACCEPTANCE_RU.md).

Архитектурные основания и compatibility boundary:
[ADR 0015](adr/0015-stage3-rainbow-per-mixed-population.md).
