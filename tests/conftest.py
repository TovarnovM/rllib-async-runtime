from __future__ import annotations

from collections.abc import Iterator

import pytest
import ray


@pytest.fixture(scope="session")
def ray_runtime() -> Iterator[None]:
    """Share Ray startup; each test still owns and stops its own actors."""
    ray.init(
        num_cpus=2,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
    )
    try:
        yield
    finally:
        ray.shutdown()
