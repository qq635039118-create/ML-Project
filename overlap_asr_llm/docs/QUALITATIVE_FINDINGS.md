# Qualitative Findings

These notes preserve manual observations that should not be lost when generated
files under `outputs/` are refreshed.

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
