# Experiment Design

## Current Progress

The core non-LLM experiment is now implemented and has produced a complete
sample2 result set. The current completion level is about 70-75% of the full
plan:

- Completed: controlled sample set, direct ASR, diarization ASR, separation ASR,
  reference-based scoring, segment exports, runtime measurement, and unit tests.
- Partially complete: speaker-label consistency analysis and qualitative
  failure-case notes.
- Not yet complete: real LLM/RAG refinement, automatic overlap-aware pipeline
  selection, and a full manual readability review for the current sample2 run.

The main current output is:

```text
outputs/all_pipelines/run_summary.md
```

## Research Question

How do diarization, speech separation, and LLM/RAG refinement change transcript
quality when speech overlap becomes more severe?

The project is not only a reproduction of existing ASR pipelines. It uses a
controlled debate-audio setting to ask when each component helps, when it fails,
and what a more robust overlap-aware pipeline should do next.

## Related Topics

- Overlapping speech recognition: ASR often misses or merges words when two
  speakers talk at the same time.
- Speaker diarization: useful for assigning speaker turns, but it does not
  reconstruct masked speech.
- Speech separation: can split speakers before ASR, but separation artifacts may
  lower recognition quality.
- LLM/RAG post-correction: can improve readability and domain terms, but must be
  constrained to avoid hallucinating debate content.

## Conditions

Use at least five audio conditions in the final experiment. This requirement is
complete in `data/samples2/`:

- No overlap
- Light overlap
- Medium overlap
- Heavy overlap
- Opposite-order overlap from the same debate material

## Pipelines

- Direct ASR on mixed audio: implemented and run with faster-whisper.
- Direct ASR plus speaker diarization: implemented and run with pyannote plus
  faster-whisper.
- Speech separation followed by ASR on each separated stream: implemented and
  run with ClearVoice plus faster-whisper.
- LLM/RAG refinement using project glossary and previous pipeline outputs:
  framework exists, but the current provider is still mock-only.

## Proposed Improvement

Use an overlap-aware pipeline selector:

- Low overlap: direct ASR first because it is faster and avoids separation
  artifacts.
- Medium or heavy standard overlap: current results still favor direct ASR for
  the sample2 set, so separation should be used selectively rather than as the
  default.
- Opposite-order overlap: speech separation before ASR is currently the best
  observed strategy.
- Final refinement: LLM/RAG may format, punctuate, and normalize debate terms,
  but it must not add unsupported claims or missing words.

## Measurements

- CER and WER when reference transcripts are available: complete.
- Runtime for each pipeline: complete.
- Speaker-label consistency: partially complete through speaker-block scoring
  and best speaker mapping.
- Manual readability score from 1 to 5: partially complete in qualitative
  notes, but needs to be refreshed for the current sample2 result set.
- Failure cases and hallucination cases: partially complete. LLM hallucination
  still requires a real LLM/RAG run.

The result schema now includes several scoring views:

- `flat_cer` and `flat_wer`: direct text comparison after flattening the output.
- `timeline_cer` and `timeline_wer`: comparison after sorting segments by time.
- `speaker_block_cer` and `speaker_block_wer`: speaker-attributed comparison
  using the best speaker-label mapping.
- `cer` and `wer`: the primary score selected by `score_basis`.

## Expected Story

Direct ASR is expected to be fast but fragile under overlap. Separation may
recover speaker-specific content but can introduce audio artifacts. Diarization
can make transcripts easier to read but depends on reliable segmentation. LLM/RAG
refinement can improve formatting and domain terms, but every correction must be
checked against the audio to avoid hallucination.

Current sample2 results refine that story:

- Direct ASR is the fastest pipeline and performs best for none, light, medium,
  and heavy overlap in the current run.
- Diarization ASR improves speaker readability and is competitive when the
  segmentation is clean, but it is slower than direct ASR.
- Separation ASR is weak on most standard overlap conditions because separation
  artifacts hurt transcription quality, but it is clearly strongest for the
  opposite-order overlap case.

## Outputs

The runner writes `results.csv`, `results.json`, and `run_summary.md` for
comparing model text, runtime, CER, WER, score basis, and errors. Pipelines that
produce segments also write `diarization_segments.*` and
`separation_segments.*`.
