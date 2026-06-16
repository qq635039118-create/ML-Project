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

Chinese team quick start:
`overlap_asr_llm/docs/QUICK_START_FOR_TEAM_CN.md`

```bash
cd overlap_asr_llm
python -m overlap_asr_llm.cli run --config configs/mock.json --mock
```

The command writes:

- `outputs/mock/results.json`
- `outputs/mock/results.csv`
- `outputs/mock/run_summary.md`

## Real Model Mode

Install real-model dependencies before running real experiments:

```bash
pip install -r requirements.txt
pip install -e .
```

The current real experiment configs are:

- `configs/direct_asr.json`
- `configs/diarization_asr.json`
- `configs/separation_asr.json`
- `configs/all_pipelines.json`

Each experiment config has its own `models` block, so models can be changed
directly in that file. Shared sample paths, prompt, and references are in
`configs/base.json`.

Run the final comparison:

```bash
python -m overlap_asr_llm.cli run --config configs/all_pipelines.json --incremental
```

## Smoke Test

Run the smoke test inside the project environment:

```bash
python scripts/smoke_test.py
```

Supported provider values:

- ASR: `mock`, `whisper`, `whisper:<model-name>`,
  `faster-whisper`, `faster-whisper:<model-name>`, `funasr`
- Diarization: `mock`, `pyannote`, `pyannote:<huggingface-model-id>`
- Separation: `mock`, `sepformer`, `sepformer:<huggingface-model-id>`,
  `clearvoice`, `clearvoice:<model-name>`
- LLM/RAG refinement: `mock`

The current implementation supports graceful failure. If a requested real model
cannot be imported, the error is recorded for that pipeline instead of crashing
the whole experiment.

## Repository Layout

```text
overlap_asr_llm/configs/base.json               Shared sample2 data, prompt, and references
overlap_asr_llm/configs/direct_asr.json         Direct ASR experiment
overlap_asr_llm/configs/diarization_asr.json    Diarization + ASR experiment
overlap_asr_llm/configs/separation_asr.json     Separation + ASR experiment
overlap_asr_llm/configs/all_pipelines.json      Final comparison experiment
overlap_asr_llm/configs/mock.json               Mock smoke-test config
overlap_asr_llm/data/samples2/                  Current five-condition Mandarin audio set
overlap_asr_llm/docs/VIDEO_SCRIPT.md            English video outline
overlap_asr_llm/docs/EXPERIMENT_DESIGN.md       One-page experiment design
overlap_asr_llm/src/overlap_asr_llm/            Experiment code
overlap_asr_llm/tests/                          Lightweight tests
overlap_asr_llm/CONTRIBUTIONS.md                Team task/contribution template
overlap_asr_llm/REPOSITORY.md                   GitHub repository URL template
```

## Four Pipelines

1. `direct_asr`: transcribe mixed audio directly.
2. `diarization_asr`: transcribe mixed audio and assign speaker labels.
3. `separation_asr`: separate speakers, then transcribe each stream.
4. `llm_rag_refine`: use project glossary/context to clean up pipeline outputs.

## Metrics

If a reference transcript is available, the runner computes:

- Primary CER/WER: selected by `score_basis`
- Flat CER/WER: plain flattened transcript scoring
- Timeline CER/WER: segment text sorted by time
- Speaker-block CER/WER: speaker-attributed scoring with best speaker mapping
- Runtime in seconds

If no reference is available, metrics are left empty and the output can still be
used for qualitative analysis.

## Submission Checklist

- English video, at least 10 minutes
- Source code
- `CONTRIBUTIONS.md`
- `REPOSITORY.md`
- Experiment outputs and figures
- Zip file uploaded to Google Drive
- Private WeChat or email message containing only the Google Drive sharing link
