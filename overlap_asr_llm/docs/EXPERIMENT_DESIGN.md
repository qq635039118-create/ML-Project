# Experiment Design

## Current Progress

The core experiment is implemented and has produced a complete sample2 result
set with the four planned pipeline families plus a diarization-order ablation.
The current completion level is about 90-95% of the full plan:

- Completed: controlled sample set, direct ASR, two diarization-ASR orders,
  separation ASR, API LLM/RAG refinement, reference-based scoring, segment
  exports, runtime measurement, and unit tests.
- Partially complete: speaker-label consistency analysis, qualitative
  failure-case notes, and overlap-aware routing rules.
- Not yet complete: final team metadata, repository URL, and any optional
  production implementation of an automatic overlap-aware selector.

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
- Full-audio ASR plus speaker diarization alignment: implemented and run with
  pyannote plus faster-whisper. This is the default `diarization_asr` path.
- Turn-level diarization followed by ASR on each speaker turn: implemented as
  `diarization_turn_asr` for comparing the original "diarization first, then
  ASR" order against the more stable full-audio ASR alignment path.
- Speech separation followed by ASR on each separated stream: implemented and
  run with ClearVoice plus faster-whisper.
- LLM/RAG refinement using project glossary and previous pipeline outputs:
  implemented and run with an OpenAI-compatible API refiner. The refiner is
  constrained to formatting and terminology cleanup and must not invent missing
  transcript words.

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
- Speaker-label consistency: complete for the current evaluation through
  speaker-block scoring and best speaker mapping.
- Manual readability score from 1 to 5: summarized in qualitative notes.
- Failure cases and hallucination cases: documented for direct, diarization,
  separation, and constrained LLM/RAG outputs.

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
  and heavy overlap in the current run, except for a small LLM/RAG gain on light
  overlap.
- Full-audio diarization ASR improves speaker readability and has the best
  average CER/WER, but it is slower than direct ASR.
- Turn-level diarization ASR is useful as an ablation, but short speaker-turn
  excerpts are more prone to ASR hallucination and should not be the default
  path.
- Separation ASR is weak on most standard overlap conditions because separation
  artifacts hurt transcription quality, but it is clearly strongest for the
  opposite-order overlap case.
- LLM/RAG refinement improves formatting and slightly helps light overlap, but
  the constrained refiner intentionally does not recover missing speech content.

## Outputs

The runner writes `results.csv`, `results.json`, and `run_summary.md` for
comparing model text, runtime, CER, WER, score basis, and errors. Pipelines that
produce segments also write `diarization_segments.*`, `separation_segments.*`,
or `llm_rag_source_segments.*` depending on the selected experiment.
