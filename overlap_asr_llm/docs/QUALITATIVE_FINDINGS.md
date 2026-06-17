# Qualitative Findings

These notes preserve manual observations that should not be lost when generated
files under `outputs/` are refreshed. The repository keeps selected result
tables, summaries, and separated audio, while local model caches and unrelated
reference materials stay out of Git.

## Current Sample2 Server Run

Current result directory:

```text
outputs/all_pipelines/
```

Metric-based summary:

- The current selected result set was produced on the server for the
  `configs/all_pipelines.json` comparison with Whisper large-v3, pyannote,
  ClearVoice, and API LLM/RAG refinement.
- Diarization ASR has the best average primary CER/WER and average TRS text in
  the current run: CER `0.1532`, WER `0.1563`, TRS text `85.8221`. It also gives
  speaker-attributed subtitles.
- Direct ASR is the fastest pipeline and remains competitive for none, medium,
  and heavy overlap, with avg runtime `6.72s`.
- Separation ASR performs poorly on most standard overlap cases because the
  separated audio introduces artifacts, but it performs best on the
  opposite-order overlap sample: CER `0.0632`, WER `0.0769`, TRS text `89.0998`.
- LLM/RAG refinement is constrained to formatting and terminology cleanup. It
  wins TRS text on light, medium, and heavy overlap, but does not reliably
  recover words that ASR missed.
- The biggest scoring change after the metric refactor is that diarization and
  separation outputs can now be evaluated with `speaker_block` scores, which are
  fairer when speaker labels are swapped or streams are not in timeline order.

| Sample | Best Current Pipeline | TRS Text | Failure Type | Observation | Hallucination Risk |
| --- | --- | ---: | --- | --- | --- |
| sample2_no_overlap | diarization_turn_asr | 97.9449 | minor_asr_errors | All non-separation paths are strong; turn-level diarization is slightly best by TRS/CER but slower than direct ASR. | low |
| sample2_light_overlap | llm_rag_refine | 95.2357 | minor_missing_words | LLM/RAG gives the best readability score after constrained cleanup, with only a small CER gain over direct ASR. | low |
| sample2_mid_overlap | llm_rag_refine | 89.2127 | speaker_mixing | LLM/RAG improves readability metrics, while direct/diarization remain close by CER. | low |
| sample2_heavy_overlap | llm_rag_refine | 86.5547 | missing_words | LLM/RAG and direct-style outputs remain much better than separation, but trust drops as overlap grows. | medium |
| sample2_opposite_overlap | separation_asr | 89.0998 | separation_artifact | Separation recovers the two speaker streams much better than timeline-based ASR, although some words are distorted. | medium |

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
