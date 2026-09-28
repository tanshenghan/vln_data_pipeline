PYTHON ?= .venv/bin/python
LEROBOT_PYTHON ?= data_pipeline/.venv/lerobot/bin/python

.PHONY: test test-platform test-gt test-web

test: test-platform test-gt test-web

test-platform:
	PYTHONPATH=src:. $(PYTHON) -m pytest tests catalog/tests -q

test-gt:
	PYTHONPATH=src:. HF_HUB_OFFLINE=1 $(LEROBOT_PYTHON) -m pytest data_pipeline/tests --ignore=data_pipeline/tests/test_rollout_lightnav.py -q

test-web:
	node --check apps/web/app.js
	node --check apps/web/objectnav.js
	node --test catalog/tests/test_*.js
