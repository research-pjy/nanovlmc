"""Exercise snapshot/log setup using a fake scheduler, never a real Slurm job."""
import os
from pathlib import Path
import shutil
import subprocess


def test_submission_freezes_committed_source(tmp_path):
    repo = tmp_path / "checkout"
    repo.mkdir()
    for name in ("dgx", "configs"):
        shutil.copytree(name, repo / name)
    shutil.copyfile(".gitignore", repo / ".gitignore")
    def git(*args):
        return subprocess.check_output(["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", *args], cwd=repo, stderr=subprocess.STDOUT, text=True)
    git("-c", "init.templateDir=", "init")
    git("add", ".")
    git("-c", "user.name=Test Fixture", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
    assets = tmp_path / "assets"; assets.mkdir()
    outputs = tmp_path / "outputs"
    # Shell-quote paths; pytest's temporary paths may contain spaces.
    import shlex
    values = {"OUTPUT_ROOT": str(outputs), "DATA_DIR": str(assets), "IMAGES_ROOT": str(assets),
              "PARTITION": "fixture", "GPU_GRES": "gpu:fixture:1", "CONDA_SH": "/unused/conda.sh", "CONDA_ENV": "fixture"}
    (repo / "dgx/cluster.local.sh").write_text("\n".join(f"export {k}={shlex.quote(v)}" for k, v in values.items()) + "\n")
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    stub = bin_dir / "sbatch"
    stub.write_text('#!/bin/bash\nset -eu\ntest -d "$OUTPUT_ROOT/scheduler-logs"\nprintf "%s\\n" "$@" > "$OUTPUT_ROOT/received-arguments.txt"\nprintf "12345\\n"\n')
    stub.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
    result = subprocess.run(["bash", "dgx/submit.sh", "smoke", "configs/smoke_mini_image_conv.yaml"], cwd=repo, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    snapshots = list((outputs / "releases").iterdir())
    assert len(snapshots) == 1
    frozen = snapshots[0] / "configs/smoke_mini_image_conv.yaml"
    original = frozen.read_text()
    (repo / "configs/smoke_mini_image_conv.yaml").write_text("changed after submission")
    assert frozen.read_text() == original
    assert not (snapshots[0] / "dgx/cluster.local.sh").exists()
    assert (snapshots[0] / "COMMIT").read_text().strip() == git("rev-parse", "HEAD").strip()
    assert "12345" in (outputs / "submissions.tsv").read_text()
