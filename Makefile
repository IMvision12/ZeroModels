.DEFAULT_GOAL := help

PYTEST := python -m pytest

.PHONY: help
help:
	@echo ""
	@echo " zeromodels test suite"
	@echo " ────────────────────────────────────────────────"
	@echo " make test-all                Full suite (torch)"
	@echo " make test-backend-torch      Backend on torch"
	@echo " make test-backend-jax        Backend on jax"
	@echo " make test-backend-tf         Backend on tensorflow"
	@echo " make test-serialization      Serialization roundtrip"
	@echo " make test-saving             Model save/load"
	@echo " make test-data-format        channels_first/last"
	@echo " make test-data-format-gpu    channels_first on TF GPU"
	@echo " make test-links              Link validation (slow)"
	@echo " make test-gpu                All GPU-only tests"
	@echo ""

.PHONY: test-all
test-all:
	KERAS_BACKEND=torch $(PYTEST) tests/ -v --durations=20 \
		-m "not slow and not gpu"

.PHONY: test-backend-torch
test-backend-torch:
	KERAS_BACKEND=torch $(PYTEST) tests/integration/test_backend_compatibility.py -v

.PHONY: test-backend-jax
test-backend-jax:
	KERAS_BACKEND=jax $(PYTEST) tests/integration/test_backend_compatibility.py -v

.PHONY: test-backend-tf
test-backend-tf:
	KERAS_BACKEND=tensorflow $(PYTEST) tests/integration/test_backend_compatibility.py -v

.PHONY: test-serialization
test-serialization:
	KERAS_BACKEND=torch $(PYTEST) tests/integration/test_serialization.py -v

.PHONY: test-saving
test-saving:
	KERAS_BACKEND=torch $(PYTEST) tests/integration/test_model_saving.py -v

.PHONY: test-data-format
test-data-format:
	KERAS_BACKEND=torch $(PYTEST) tests/integration/test_data_formats.py -v

.PHONY: test-data-format-gpu
test-data-format-gpu:
	KERAS_BACKEND=tensorflow $(PYTEST) tests/integration/test_data_formats.py -v \
		-k "channels_first"

.PHONY: test-gpu
test-gpu:
	KERAS_BACKEND=torch $(PYTEST) tests/ -v -m gpu
	KERAS_BACKEND=tensorflow $(PYTEST) tests/integration/test_data_formats.py -v \
		-k "channels_first"
