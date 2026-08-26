"""Cross-platform test runner for zeromodels.

Usage:
    zeromodels-test <command>

Commands:
    all                  Full test suite (torch, excludes slow/link tests)
    backend-torch        Backend tests on torch
    backend-jax          Backend tests on jax
    backend-tf           Backend tests on tensorflow
    sas-torch            Serialization + saving on torch
    sas-tf               Serialization + saving on tensorflow
    sas-jax              Serialization + saving on jax
    df-torch             Data format tests on torch
    df-tf                Data format tests on tensorflow (GPU auto-skip)
    df-jax               Data format tests on jax
    gpu                  GPU-marked tests only
    gpu-all              Full test suite on GPU (torch + tf)
    help                 Show this message
"""

import os
import subprocess
import sys

PYTEST = [sys.executable, "-m", "pytest"]


def _run(backend, *pytest_args):
    """Run pytest with the given backend and arguments."""
    env = os.environ.copy()
    if backend:
        env["KERAS_BACKEND"] = backend
    cmd = PYTEST + list(pytest_args)
    print(f"\n{'=' * 60}")
    print(f"  KERAS_BACKEND={backend or '(default)'}  {' '.join(pytest_args)}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(cmd, env=env)
    return result.returncode


COMMANDS = {}


def command(name):
    def decorator(fn):
        COMMANDS[name] = fn
        return fn

    return decorator


@command("all")
def test_all():
    return _run(
        "torch",
        "tests/",
        "-v",
        "--durations=20",
        "-m",
        "not slow and not gpu",
    )


@command("backend-torch")
def test_backend_torch():
    return _run("torch", "tests/integration/test_backend_compatibility.py", "-v")


@command("backend-jax")
def test_backend_jax():
    return _run("jax", "tests/integration/test_backend_compatibility.py", "-v")


@command("backend-tf")
def test_backend_tf():
    return _run("tensorflow", "tests/integration/test_backend_compatibility.py", "-v")


SAS_FILES = [
    "tests/integration/test_serialization.py",
    "tests/integration/test_model_saving.py",
]


@command("sas-torch")
def test_sas_torch():
    return _run("torch", *SAS_FILES, "-v")


@command("sas-tf")
def test_sas_tf():
    return _run("tensorflow", *SAS_FILES, "-v")


@command("sas-jax")
def test_sas_jax():
    return _run("jax", *SAS_FILES, "-v")


DF_FILE = "tests/integration/test_data_formats.py"


@command("df-torch")
def test_df_torch():
    return _run("torch", DF_FILE, "-v")


@command("df-tf")
def test_df_tf():
    return _run("tensorflow", DF_FILE, "-v")


@command("df-jax")
def test_df_jax():
    return _run("jax", DF_FILE, "-v")


@command("gpu")
def test_gpu():
    rc1 = _run("torch", "tests/", "-v", "-m", "gpu")
    rc2 = _run(
        "tensorflow",
        "tests/integration/test_data_formats.py",
        "-v",
        "-k",
        "channels_first",
    )
    return rc1 or rc2


@command("gpu-all")
def test_gpu_all():
    rc1 = _run(
        "torch",
        "tests/",
        "-v",
        "--durations=20",
        "-m",
        "not slow and not link_validation",
    )
    rc2 = _run(
        "tensorflow",
        "tests/",
        "-v",
        "--durations=20",
        "-m",
        "not slow and not link_validation",
    )
    return rc1 or rc2


@command("help")
def show_help():
    print(__doc__)
    return 0


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        show_help()
        if len(sys.argv) >= 2 and sys.argv[1] not in COMMANDS:
            print(f"\nError: Unknown command '{sys.argv[1]}'")
            return 1
        return 0

    return COMMANDS[sys.argv[1]]()


if __name__ == "__main__":
    sys.exit(main())
