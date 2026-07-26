# Technical debt

## Phase 8 — target two-GPU gate

**Status:** open

The only unresolved Phase 8 acceptance criterion is the real two-GPU gate on
the target workstation:

```bash
uv run --locked --extra cu118 --group dev \
  pytest -m gpu tests/gpu/test_two_gpu_population.py
```

This debt is closed only when the command passes on the target two-GPU machine.
Until then:

- PR #11 must remain a draft;
- Phase 8 must remain unchecked in `docs/IMPLEMENTATION_PLAN.md`.
