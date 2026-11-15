from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

from benchmarks.validate_dqn_gpu import run_command


@pytest.mark.integration
def test_ray_working_directory_cleanup_preserves_driver_installation(
    tmp_path: Path,
) -> None:
    # Use the same interpreter across phases, without a repairing `uv sync`.
    probe = (
        "import json,sys,rllib_async; "
        "from pathlib import Path; "
        "print(json.dumps([sys.prefix, str(Path(rllib_async.__file__).resolve())]))"
    )
    before = subprocess.check_output([sys.executable, "-c", probe], text=True)
    worker_result = tmp_path / "worker.json"
    driver = textwrap.dedent(
        """\
        import json
        import sys
        from pathlib import Path

        import ray

        ray.init(
            address="local",
            num_cpus=1,
            num_gpus=0,
            include_dashboard=False,
            log_to_driver=False,
            runtime_env={"working_dir": str(Path.cwd())},
        )
        try:
            @ray.remote
            def installation():
                import rllib_async
                return [sys.prefix, str(Path(rllib_async.__file__).resolve())]

            result = ray.get(installation.remote(), timeout=600)
            Path(sys.argv[1]).write_text(json.dumps(result))
        finally:
            ray.shutdown()
        """
    )
    # Short sockets, and a separate cluster even if ray_runtime is already active.
    with tempfile.TemporaryDirectory(prefix="r23-", dir="/tmp") as ray_tmp:
        log_path = tmp_path / "driver.log"
        result = run_command(
            [sys.executable, "-c", driver, str(worker_result)],
            environment={
                **os.environ,
                "RAY_TMPDIR": ray_tmp,
                "CUDA_VISIBLE_DEVICES": "",
            },
            log_path=log_path,
            timeout_s=900,
        )
        assert result["exit_code"] == 0, log_path.read_text()
    assert not Path(ray_tmp).exists()
    after = subprocess.check_output([sys.executable, "-c", probe], text=True)
    assert json.loads(after) == json.loads(before)
    worker_prefix, worker_package = json.loads(worker_result.read_text())
    # Prove this exercised packaged code and an isolated worker environment.
    assert Path(worker_prefix).is_relative_to(ray_tmp)
    assert Path(worker_package).is_relative_to(ray_tmp)
