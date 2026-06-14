# Team Contributions

This file records the planned team split and the current evidence that can be
checked in the repository. Replace the member placeholders with real names
before final submission.

## Current Project Status

The current repository has moved beyond the mock prototype stage. The main
sample2 experiment has been run with five overlap conditions and three real
pipelines:

- Direct ASR: `faster-whisper:large-v3`
- Diarization ASR: `pyannote/speaker-diarization-community-1` plus
  `faster-whisper:large-v3`
- Separation ASR: `clearvoice:MossFormer2_SS_16K` plus
  `faster-whisper:large-v3`

The current result set is stored under:

```text
outputs/all_pipelines/
```

The main summary file is:

```text
outputs/all_pipelines/run_summary.md
```

Core metrics have also been refactored. In addition to the primary `cer` and
`wer`, the output now records `flat_*`, `timeline_*`, and `speaker_block_*`
scores, plus `score_basis` and `best_speaker_mapping`.

## Planned Responsibility Split

| Member | Main Responsibility | Evidence | Contribution |
| --- | --- | --- | --- |
| Member 1 | Literature review and research question | `docs/EXPERIMENT_DESIGN.md`, related work notes, presentation slides | 16.7% |
| Member 2 | Dataset construction and reference annotation | `data/samples2/`, `configs/base.json`, speaker references | 16.7% |
| Member 3 | Direct ASR baseline and prompt tuning | `providers.py`, direct ASR results in `run_summary.md` | 16.7% |
| Member 4 | Diarization and separation experiments | `providers.py`, `pipelines.py`, diarization and separation segment outputs | 16.7% |
| Member 5 | Metrics, scoring, and LLM/RAG refinement path | `metrics.py`, `pipelines.py`, `tests/test_metrics.py`; LLM/RAG still needs real-model completion | 16.7% |
| Member 6 | Documentation, video, final packaging | `docs/`, `CONTRIBUTIONS.md`, `REPOSITORY.md`, final archive or release | 16.5% |

## Current Completion Notes

Completed:

- Five controlled overlap conditions are available: none, light, medium, heavy,
  and opposite-order overlap.
- Three real non-LLM pipelines have been run and compared.
- Speaker-aware scoring has been added for diarization and separation outputs.
- Result writers export summary tables and segment-level CSV/JSON files.
- The unit-test suite passes with `unittest`.

Still to finish:

- Replace member placeholders with real names and commit or PR evidence.
- Run or implement a real LLM/RAG refinement provider. The current provider is
  still mock-only.
- Add a formal overlap-aware pipeline selector instead of relying only on manual
  analysis.
- Update manual readability and hallucination notes for the current sample2
  three-pipeline run.

## Individual Step 1 Evidence

Each member should record evidence that they used an AI tool to star and fork:

`https://github.com/zhangqi444/open-forge`

| Member | Star/Fork Evidence | Notes |
| --- | --- | --- |
| Member 1 | TODO | TODO |
| Member 2 | TODO | TODO |
| Member 3 | TODO | TODO |
| Member 4 | TODO | TODO |
| Member 5 | TODO | TODO |
| Member 6 | TODO | TODO |
