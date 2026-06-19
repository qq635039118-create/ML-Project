PYTHON ?= python3
APP_DIR := overlap_asr_llm
PORT ?= 7861
HOST ?= 127.0.0.1
UI ?= web
CONFIG ?= all_pipelines
RESULTS ?= outputs/all_pipelines/results.json
DEVICE ?= auto
BATCH_SIZE ?= 1

.PHONY: help install deps mock smoke test pytest exp experiment eval evaluate app front frontend package remote

help:
	@printf '%s\n' \
		'Short commands for this repository:' \
		'' \
		'  make mock                  Run the lightweight mock experiment' \
		'  make smoke                 Run the stdlib smoke test' \
		'  make test                  Run unit tests with unittest' \
		'  make exp                   Run configs/all_pipelines.json incrementally' \
		'  make exp CONFIG=direct_asr Run one experiment config' \
		'  make eval                  Evaluate outputs/all_pipelines/results.json' \
		'  make app                   Open the web frontend on 127.0.0.1:7861' \
		'  make app PORT=7862         Open the frontend on another port' \
		'  make package               Build the submission zip' \
		'  make remote                Show git branch, status, and remotes' \
		'' \
		'Setup helpers:' \
		'  make install               pip install -e overlap_asr_llm' \
		'  make deps                  pip install requirements + editable package'

install:
	cd $(APP_DIR) && $(PYTHON) -m pip install -e .

deps:
	cd $(APP_DIR) && $(PYTHON) -m pip install -r requirements.txt
	cd $(APP_DIR) && $(PYTHON) -m pip install -e .

mock:
	cd $(APP_DIR) && PYTHONPATH=src $(PYTHON) -m overlap_asr_llm.cli run --config configs/mock.json --mock

smoke:
	cd $(APP_DIR) && PYTHONPATH=src $(PYTHON) scripts/smoke_test.py

test:
	cd $(APP_DIR) && PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -q

pytest:
	cd $(APP_DIR) && PYTHONPATH=src $(PYTHON) -m pytest -q

exp experiment:
	@config="$(CONFIG)"; \
	if [ "$$config" = "all" ]; then config="all_pipelines"; fi; \
	cd $(APP_DIR) && PYTHONPATH=src $(PYTHON) -m overlap_asr_llm.cli run --config "configs/$$config.json" --incremental

eval evaluate:
	@config="$(CONFIG)"; \
	if [ "$$config" = "all" ]; then config="all_pipelines"; fi; \
	cd $(APP_DIR) && PYTHONPATH=src $(PYTHON) -m overlap_asr_llm.cli evaluate --config "configs/$$config.json" --results "$(RESULTS)" --device "$(DEVICE)" --batch-size "$(BATCH_SIZE)"

app front frontend:
	cd $(APP_DIR) && PYTHONPATH=src $(PYTHON) scripts/launch_speaker_app.py --host "$(HOST)" --port "$(PORT)" --ui "$(UI)" $(if $(SHARE),--share,)

package:
	cd $(APP_DIR) && PYTHONPATH=src $(PYTHON) scripts/package_submission.py

remote:
	git status -sb
	git remote -v
