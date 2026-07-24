from __future__ import annotations

from collections.abc import Iterator

import pytest
import ray


@pytest.fixture
def ray_runtime() -> Iterator[None]:
    """Start Ray only for tests that actually exercise remote scheduling."""
    ray.init(
        num_cpus=2,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
    )
    yield
    ray.shutdown()
