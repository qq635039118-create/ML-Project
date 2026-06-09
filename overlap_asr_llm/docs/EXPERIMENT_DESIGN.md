# Experiment Design

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

Use at least five audio conditions in the final experiment:

- No overlap
- Light overlap
- Medium overlap
- Heavy overlap
- Opposite-order overlap from the same debate material

## Pipelines

- Direct ASR on mixed audio
- Direct ASR plus speaker diarization
- Speech separation followed by ASR on each separated stream
- LLM/RAG refinement using project glossary and previous pipeline outputs

## Proposed Improvement

Use an overlap-aware pipeline selector:

- Low overlap: direct ASR first, because it is faster and avoids separation
  artifacts.
- Medium or heavy overlap: speech separation before ASR, then compare separated
  streams against the mixed-audio transcript.
- Final refinement: LLM/RAG may format, punctuate, and normalize debate terms,
  but it must not add unsupported claims or missing words.

## Measurements

- CER and WER when reference transcripts are available
- Runtime for each pipeline
- Speaker-label consistency
- Manual readability score from 1 to 5
- Failure cases and hallucination cases

## Expected Story

Direct ASR is expected to be fast but fragile under overlap. Separation may
recover speaker-specific content but can introduce audio artifacts. Diarization
can make transcripts easier to read but depends on reliable segmentation. LLM/RAG
refinement can improve formatting and domain terms, but every correction must be
checked against the audio to avoid hallucination.

## Outputs

The runner writes `outputs/results.csv`, `outputs/results.json`, and
`outputs/run_summary.md` for comparing model text, runtime, CER, and WER.
