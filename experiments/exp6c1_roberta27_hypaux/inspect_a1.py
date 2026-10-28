"""Standard-library-only A1 installation and read-only foundation inventory."""
import json
import os
import platform
import subprocess
from pathlib import Path

from a1lib import A0_RUN, A0_SOURCE, EXP6B, HERE, REPO, UPSTREAM, check_sources, read, require, sha


def main():
    require(platform.node() == "D3020-A3", "Expected Ubuntu D3020-A3")
    source = check_sources()
    print("A1_PACKAGE_AND_FROZEN_A0_SOURCE_VERIFIED", len(source["a1"]), len(source["a0"]))
    require((A0_RUN / "SUCCESS.json").is_file(), "A0 success proof missing")
    require((A0_RUN / "train_exit_code.txt").read_text().strip() == "0", "A0 exit code is not zero")
    print("A0_FROZEN_RUN", A0_RUN)
    print("A0_SUCCESS_SHA256", sha(A0_RUN / "SUCCESS.json"))
    print("A0_RECORDED_TEST_METRICS", json.dumps(read(A0_RUN / "test_metrics.json")))
    print("A0_SNAPSHOT", read(A0_RUN / "plan.json")["snapshot"])
    require(UPSTREAM.is_dir() and EXP6B.is_dir(), "Required validated HypEmo source/Exp6B artifacts missing")
    print("READONLY_HYPEMO_SOURCE", UPSTREAM)
    print("READONLY_FORMULAA_REFERENCE", EXP6B)
    py = REPO / ".venv-bert/bin/python"
    require(py.is_file(), "A0 Python environment missing")
    query = "import importlib.metadata as m,json,sys; print(json.dumps({'python':sys.version,'executable':sys.executable,'packages':{p:m.version(p) for p in ['torch','transformers','numpy','scikit-learn','tokenizers','safetensors']}}))"
    subprocess.run([str(py), "-c", query], check=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1"))
    subprocess.run(["git", "-C", str(REPO), "status", "--short"], check=True)
    subprocess.run(["nvidia-smi"], check=True)
    for parent in [Path.home(), Path.home() / "Documents", REPO, REPO / "experiments", HERE]:
        p = parent / "AGENTS.md"
        if p.is_file():
            print("REMOTE_INSTRUCTIONS_REVIEW_BEFORE_SMOKE", p)
            print(p.read_text())
    print("EXP6C1_INSPECTION_COMPLETE_NO_TRAINING")


if __name__ == "__main__":
    main()
