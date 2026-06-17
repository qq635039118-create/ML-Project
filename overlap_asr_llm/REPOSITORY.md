# Repository Information

GitHub repository URL:

```text
TODO: https://github.com/<org-or-user>/<repo>
```

## Current Repository State

The project is an experiment suite for overlapping Mandarin speech recognition.
It compares direct ASR, two diarization-ASR orders, speech separation plus ASR,
and an LLM/RAG refinement path.

Current implemented status:

- Mock pipeline framework is complete.
- Real direct ASR is supported through Whisper and faster-whisper providers; the
  current main config uses `whisper:large-v3` for the server result set.
- Real speaker diarization is supported through pyannote.
- Real speech separation is supported through ClearVoice and SpeechBrain
  SepFormer providers.
- Real LLM/RAG refinement is supported through an OpenAI-compatible API refiner.
- The main current experiment compares the four planned pipeline families plus a
  diarization-order ablation on five sample2 overlap conditions.

Current main experiment:

```text
configs/all_pipelines.json
outputs/all_pipelines/run_summary.md
outputs/all_pipelines/readability_summary.md
```

The latest result table includes primary CER/WER, speaker-block CER/WER,
timeline CER/WER, runtime, and the score basis used for each row.
The readability summary adds BERTScore F2, text-only TRS, and speaker-aware TRS.

The GitHub repository intentionally keeps the selected server outputs needed for
review: the `all_pipelines` CSV/JSON/Markdown summaries, generated separated
audio, and the single-pipeline result directories for `direct_asr`,
`diarization_asr`, `separation_asr`, and `speaker_llm_pipeline`. It does not
track local model caches, unrelated reference materials, temporary output
directories, or `outputs/all_pipelines/run_summary.html`.

Latest server-run headline results:

- `diarization_asr` has the best average primary CER/WER and average TRS text:
  CER `0.1532`, WER `0.1563`, TRS text `85.8221`.
- `direct_asr` is the fastest path, averaging `6.72s`.
- `separation_asr` is poor on most standard overlap conditions but is clearly
  best on `sample2_opposite_overlap`.
- `llm_rag_refine` improves several sample-level TRS text scores, but the
  current result should be interpreted as constrained readability cleanup, not
  evidence that the LLM recovers speech that ASR missed.

## Important Paths

```text
src/overlap_asr_llm/config.py      Configuration loading
src/overlap_asr_llm/providers.py   ASR, diarization, separation, and LLM wrappers
src/overlap_asr_llm/pipelines.py   Experiment pipeline orchestration
src/overlap_asr_llm/metrics.py     CER, WER, and speaker-block scoring
src/overlap_asr_llm/io.py          CSV, JSON, Markdown, and segment writers
configs/                           Experiment configurations
data/samples2/                     Current five-condition Mandarin audio set
outputs/                           Selected experiment results, excluding caches
docs/                              Project design and reporting notes
tests/                             Unit tests
```

## Verification

Basic unit tests can be run with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -q
```

If `pytest` is installed, the same tests can also be run through `pytest`.

## Collaboration Workflow

The instructor can inspect the commit history to verify how each team member
contributed. Recommended workflow:

- Each member works on a named branch.
- Each task is merged through a pull request.
- Commit messages mention the experiment, model, dataset, or document changed.
- Final submission uses a tagged release or clearly named final commit.

Before final submission:

- Replace the GitHub URL placeholder above.
- Make sure all team member names and contribution evidence are filled in
  `CONTRIBUTIONS.md`.
- Confirm the final result directory and summary file are referenced in the
  report or video.
- Package only the intended code, docs, configs, selected outputs, and sample
  audio. Do not add local model caches, unrelated PDFs/PPTs, or temporary
  experiment folders.
