# Tests

## Setup

```bash
pip install -e ".[test]"
```

## Commands

**Windows:** `zeromodels-test <command>` &nbsp;|&nbsp; **Linux/macOS:** `make test-<command>`

| Command | What it tests | Backend |
|---------|--------------|---------|
| `all` | Full suite (excludes slow/link/gpu) | torch |
| `backend-torch` | Forward pass, 43 models | torch |
| `backend-jax` | Forward pass, 43 models | jax |
| `backend-tf` | Forward pass, 43 models | tensorflow |
| `sas-torch` | Serialization + saving | torch |
| `sas-tf` | Serialization + saving | tensorflow |
| `sas-jax` | Serialization + saving | jax |
| `df-torch` | Data format (channels_first/last) | torch |
| `df-tf` | Data format (channels_last + channels_first auto-skips without GPU) | tensorflow |
| `df-jax` | Data format (channels_first/last) | jax |
| `gpu` | GPU-marked tests only | torch + tf |
| `gpu-all` | **Full test suite on GPU** (torch + tf) | torch, tf |

## CI/CD (automatic, CPU)

| Job | Backends | Trigger |
|-----|----------|---------|
| Lint & Format |: | PR, release |
| Test the code | torch, tf, jax | PR, release |

## Local GPU (manual)

```bash
zeromodels-test data-format-gpu    # TF channels_first (needs cuDNN)
zeromodels-test gpu                # All GPU-marked tests
```

## Adding a new model

Add one entry to `tests/base/model_test_registry.py`: all integration tests pick it up automatically.
