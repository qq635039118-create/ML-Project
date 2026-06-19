# Experiment Design

## Current Progress

The core experiment is implemented and has produced a complete sample2 result
set with the four planned pipeline families plus a diarization-order ablation.
The current completion level is about 90-95% of the full plan:

- Completed: controlled sample set, direct ASR, two diarization-ASR orders,
  separation ASR, API LLM/RAG refinement, reference-based scoring, segment
  exports, runtime measurement, unit tests, repository URL documentation,
  short Makefile commands, and safe submission packaging.
- Partially complete: speaker-label consistency analysis, qualitative
  failure-case notes, and overlap-aware routing rules.
- Not yet complete: final team metadata and any optional production
  implementation of an automatic overlap-aware selector.

The main current output is:

```text
outputs/all_pipelines/run_summary.md
outputs/all_pipelines/readability_summary.md
```

The selected main output is the server result set for the current
`configs/all_pipelines.json` comparison, using `whisper:large-v3`,
`pyannote/speaker-diarization-community-1`, `clearvoice:MossFormer2_SS_16K`,
and the API LLM refiner.

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

- Direct ASR on mixed audio: implemented and run with Whisper large-v3.
- Full-audio ASR plus speaker diarization alignment: implemented and run with
  pyannote plus Whisper large-v3. This is the default `diarization_asr` path.
- Turn-level diarization followed by ASR on each speaker turn: implemented as
  `diarization_turn_asr` for comparing the original "diarization first, then
  ASR" order against the more stable full-audio ASR alignment path.
- Speech separation followed by ASR on each separated stream: implemented and
  run with ClearVoice plus Whisper large-v3.
- LLM/RAG refinement using project glossary and previous pipeline outputs:
  implemented and run with an OpenAI-compatible API refiner. The refiner is
  constrained to formatting and terminology cleanup and must not invent missing
  transcript words.

## Proposed Improvement

Use an overlap-aware pipeline selector:

- Low overlap: direct ASR first because it is faster and avoids separation
  artifacts.
- Medium or heavy standard overlap: current results favor direct ASR for speed
  and simple deployment, while constrained LLM/RAG gives the best TRS text; use
  separation selectively rather than as the default.
- Opposite-order overlap: speech separation before ASR is currently the best
  observed strategy.
- Final refinement: LLM/RAG may format, punctuate, and normalize debate terms,
  but it must not add unsupported claims or missing words.

## Measurements

- CER and WER when reference transcripts are available: complete.
- Runtime for each pipeline: complete.
- Speaker-label consistency: complete for the current evaluation through
  speaker-block scoring and best speaker mapping.
- Readability post-evaluation: complete for the current output set with
  BERTScore F2, text-only TRS, and speaker-aware TRS.
- Manual readability score from 1 to 5: retained only as qualitative notes from
  earlier inspection.
- Failure cases and hallucination cases: documented for direct, diarization,
  separation, and constrained LLM/RAG outputs.

The result schema now includes several scoring views:

- `flat_cer` and `flat_wer`: direct text comparison after flattening the output.
- `timeline_cer` and `timeline_wer`: comparison after sorting segments by time.
- `speaker_block_cer` and `speaker_block_wer`: speaker-attributed comparison
  using the best speaker-label mapping. This groups all text from each predicted
  speaker into a block, then tries label assignments such as `SPEAKER_00` to
  `speaker_1` or `speaker_2` and keeps the lowest error. It avoids unfairly
  penalizing diarization outputs when the speaker label names are swapped.
- `cer` and `wer`: the primary score selected by `score_basis`.

## Expected Story

Direct ASR is expected to be fast but fragile under overlap. Separation may
recover speaker-specific content but can introduce audio artifacts. Diarization
can make transcripts easier to read but depends on reliable segmentation. LLM/RAG
refinement can improve formatting and domain terms, but every correction must be
checked against the audio to avoid hallucination.

Current sample2 results refine that story:

- Full-audio diarization ASR has the best average primary CER/WER and average
  TRS text in the current server run: CER `0.1532`, WER `0.1563`, TRS text
  `85.8221`.
- Direct ASR is the fastest pipeline, averaging `6.72s`, and remains a strong
  baseline for standard non-opposite overlap cases.
- Turn-level diarization ASR is useful as an ablation, but short speaker-turn
  excerpts are more prone to ASR hallucination and should not be the default
  path.
- Separation ASR is weak on most standard overlap conditions because separation
  artifacts hurt transcription quality, but it is clearly strongest for the
  opposite-order overlap case: CER `0.0632`, WER `0.0769`, TRS text `89.0998`.
- LLM/RAG refinement wins the TRS text comparison for light, medium, and heavy
  overlap, but the constrained refiner should be described as readability
  cleanup rather than reliable recovery of missing speech content.

## Outputs

The runner writes `results.csv`, `results.json`, and `run_summary.md` for
comparing model text, runtime, CER, WER, score basis, and errors. Pipelines that
produce segments also write `diarization_segments.*`, `separation_segments.*`,
or `llm_rag_source_segments.*` depending on the selected experiment.

The readability evaluator writes `readability_results.csv`,
`readability_results.json`, and `readability_summary.md` for BERTScore F2,
TRS text, TRS speaker, and overlap-ratio reporting.

For GitHub, the repository keeps the current selected `all_pipelines` CSV/JSON
and Markdown outputs, the generated separated audio, and the single-pipeline
result directories. Local model caches, temporary experiment folders, unrelated
reference materials, frontend uploads, local secrets, virtual environments, zip
archives, and HTML summaries are excluded from commits and submission packages.
