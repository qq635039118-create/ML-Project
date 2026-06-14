# Qualitative Findings

These notes preserve manual observations that should not be lost when generated
files under `outputs/` are refreshed.

## Current Sample2 Three-Pipeline Run

Current result directory:

```text
outputs/all_pipelines/
```

Metric-based summary:

- Direct ASR has the best average primary CER/WER among the three current real
  pipelines and is also the fastest.
- Diarization ASR is useful for speaker-attributed subtitles, but its recognition
  quality drops as overlap becomes stronger.
- Separation ASR performs poorly on most standard overlap cases because the
  separated audio introduces artifacts, but it performs best on the
  opposite-order overlap sample.
- The biggest scoring change after the metric refactor is that diarization and
  separation outputs can now be evaluated with `speaker_block` scores, which are
  fairer when speaker labels are swapped or streams are not in timeline order.

| Sample | Best Current Pipeline | Readability | Failure Type | Observation | Hallucination Risk |
| --- | --- | ---: | --- | --- | --- |
| sample2_no_overlap | diarization_asr | 5 | minor_asr_errors | Diarization gives a clean speaker-attributed transcript and slightly beats direct ASR on primary CER/WER. | low |
| sample2_light_overlap | direct_asr | 4 | minor_missing_words | Direct ASR remains readable and has the best score; diarization helps structure but adds recognition errors. | low |
| sample2_mid_overlap | direct_asr | 3 | speaker_mixing | Direct ASR is still the strongest scored output, but overlap starts to merge or mask speaker content. | low |
| sample2_heavy_overlap | direct_asr | 3 | missing_words | Direct ASR remains better than the diarization and separation outputs in the current run, but trust drops. | medium |
| sample2_opposite_overlap | separation_asr | 4 | separation_artifact | Separation recovers the two speaker streams much better than timeline-based ASR, although some words are distorted. | medium |

The current run does not include a real LLM/RAG refinement result, so
hallucination risk for LLM post-correction still needs to be evaluated later.

## Earlier Manual Notes

| Sample | Pipeline | Readability | Failure Type | Observation | Hallucination Risk |
| --- | --- | ---: | --- | --- | --- |
| no_overlap | direct_asr | 5 | none | Transcript is readable and speaker overlap is minimal. | low |
| light_overlap | direct_asr | 4 | missing_words | Some short words are missing when speakers start close to each other. | low |
| heavy_overlap | direct_asr | 2 | speaker_mixing | Two speakers are merged into one confusing transcript. | medium |
| heavy_overlap | separation_asr | 3 | separation_artifact | Separated audio is easier to follow but introduces distorted words. | medium |
| heavy_overlap | llm_rag_refine | 4 | hallucination | Text is cleaner but may add unsupported debate terms. | high |

## Real FunASR Direct ASR Run

| Sample | Pipeline | Readability | Failure Type | Observation | Hallucination Risk |
| --- | --- | ---: | --- | --- | --- |
| no_overlap | direct_asr | 4 | poor_punctuation | Readable overall but contains repeated words and some punctuation/segmentation noise. | low |
| light_overlap | direct_asr | 3 | missing_words | Main argument is still visible but short words and transitions start to drop or merge. | low |
| mid_overlap | direct_asr | 3 | speaker_mixing | Transcript remains partially readable but speaker turns begin to merge and wording becomes unstable. | medium |
| heavy_overlap | direct_asr | 2 | speaker_mixing | Heavy overlap causes missing phrases and merged speaker content; transcript is hard to trust. | medium |
| opposite_overlap | direct_asr | 2 | unreadable | Opposite-order overlap produces fragmented wording and distorted sentence flow. | medium |
