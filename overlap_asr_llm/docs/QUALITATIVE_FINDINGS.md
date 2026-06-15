# Qualitative Findings

These notes preserve manual observations that should not be lost when generated
files under `outputs/` are refreshed.

## Current Sample2 Four-Pipeline Run

Current result directory:

```text
outputs/all_pipelines/
```

Metric-based summary:

- Diarization ASR has the best average primary CER/WER in the current four
  pipeline run and gives speaker-attributed subtitles.
- Direct ASR is the fastest pipeline and remains competitive for none, medium,
  and heavy overlap.
- Separation ASR performs poorly on most standard overlap cases because the
  separated audio introduces artifacts, but it performs best on the
  opposite-order overlap sample.
- LLM/RAG refinement is constrained to formatting and terminology cleanup. It
  slightly improves light overlap but does not recover words that ASR missed.
- The biggest scoring change after the metric refactor is that diarization and
  separation outputs can now be evaluated with `speaker_block` scores, which are
  fairer when speaker labels are swapped or streams are not in timeline order.

| Sample | Best Current Pipeline | Readability | Failure Type | Observation | Hallucination Risk |
| --- | --- | ---: | --- | --- | --- |
| sample2_no_overlap | direct_asr / diarization_asr / llm_rag_refine | 5 | minor_asr_errors | The three non-separation paths are effectively tied; diarization adds speaker labels with small runtime cost. | low |
| sample2_light_overlap | llm_rag_refine | 4 | minor_missing_words | LLM/RAG gives the best score after constrained cleanup, but the gain over direct ASR is small. | low |
| sample2_mid_overlap | direct_asr / diarization_asr / llm_rag_refine | 3 | speaker_mixing | The three non-separation paths are tied; overlap starts to merge or mask speaker content. | low |
| sample2_heavy_overlap | direct_asr / diarization_asr / llm_rag_refine | 3 | missing_words | Direct-style transcription remains better than separation, but trust drops as overlap grows. | medium |
| sample2_opposite_overlap | separation_asr | 4 | separation_artifact | Separation recovers the two speaker streams much better than timeline-based ASR, although some words are distorted. | medium |

The current run includes real API LLM/RAG refinement. The prompt constrains the
model not to add unsupported words, so the observed LLM risk is mostly failure
to improve missing content rather than free-form hallucination.

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
