# Overlapping Speech ASR + LLM Project

This project turns the final-project plan into a reproducible experiment suite for
overlapping speech, speaker diarization, speech separation, ASR, and LLM/RAG-based
transcript refinement.

## Project Question

When two or more speakers overlap, which pipeline gives the most useful
transcript?

- Direct ASR on mixed audio
- Mixed audio plus speaker diarization
- Speech separation followed by ASR
- LLM/RAG refinement of ASR outputs

## Quick Start

Run the full pipeline in mock mode. This requires only Python standard library
modules and is useful for checking the project structure before installing GPU
dependencies.

Chinese team quick start: `docs/QUICK_START_FOR_TEAM_CN.md`

```bash
cd project
PYTHONPATH=src python3 -m overlap_asr_llm.cli run --config configs/experiment.json --mock
```

The command writes:

- `outputs/results.json`
- `outputs/results.csv`
- `outputs/run_summary.md`
- `outputs/qualitative_review_template.csv`

## Real Model Mode

Install optional dependencies in a GPU environment before running real models.

```bash
pip install -r requirements.txt
pip install -e .
```

Then edit `configs/experiment.json` so each sample points to a real audio file.
Run:

```bash
overlap-asr-llm run --config configs/experiment.json
```

## Smoke Test

Run the stdlib-only smoke test:

```bash
python3 scripts/smoke_test.py
```

Supported provider values:

- ASR: `mock`, `whisper`, `whisper:<model-name>`, `funasr`
- Diarization: `mock`
- Separation: `mock`, `sepformer`, `sepformer:<huggingface-model-id>`
- LLM/RAG refinement: `mock`

The current implementation supports graceful failure. If a requested real model
cannot be imported, the error is recorded for that pipeline instead of crashing
the whole experiment.

## Repository Layout

```text
configs/experiment.json         Experiment definition
data/samples/                   Small sample placeholders or metadata
docs/VIDEO_SCRIPT.md            English video outline
docs/EXPERIMENT_DESIGN.md       One-page experiment design
src/overlap_asr_llm/            Experiment code
tests/                          Lightweight tests
CONTRIBUTIONS.md                Team task/contribution template
REPOSITORY.md                   GitHub repository URL template
```

## Four Pipelines

1. `direct_asr`: transcribe mixed audio directly.
2. `diarization_asr`: transcribe mixed audio and assign speaker labels.
3. `separation_asr`: separate speakers, then transcribe each stream.
4. `llm_rag_refine`: use project glossary/context to clean up pipeline outputs.

## Metrics

If a reference transcript is available, the runner computes:

- CER: character error rate
- WER: word error rate
- Runtime in seconds

If no reference is available, metrics are left empty and the output can still be
used for qualitative analysis.

The qualitative review template is for manual readability scores and failure
cases such as missing words, speaker mixing, separation artifacts, and LLM
hallucination risk.

## Submission Checklist

- English video, at least 10 minutes
- Source code
- `CONTRIBUTIONS.md`
- `REPOSITORY.md`
- Experiment outputs and figures
- Zip file uploaded to Google Drive
- Private WeChat or email message containing only the Google Drive sharing link
