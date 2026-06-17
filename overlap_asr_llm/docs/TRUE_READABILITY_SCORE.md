# True Readability Evaluation for ASR Transcripts

This note defines the readability evaluation used by this project after an ASR
experiment has already produced `results.json`. It is a post-run evaluation
layer, not a replacement for the existing CER/WER pipeline metrics.

The implementation lives in `src/overlap_asr_llm/readability.py`. It writes the
following files next to `results.json`:

- `readability_results.json`
- `readability_results.csv`
- `readability_summary.md`

## Motivation

WER and CER measure surface edit distance. They are still necessary, but they do
not fully answer whether a transcript is useful to a reader under overlapping
speech. In this project, a transcript can look acceptable by CER/WER while still
missing a speaker's clause, mixing speakers, or adding unsupported text.

The readability evaluation therefore reports:

- audio overlap severity,
- surface accuracy,
- semantic faithfulness and coverage,
- hallucination risk,
- speaker consistency,
- a compact reporting index called TRS.

TRS is a project reporting index. It is not presented as a new published metric.

## Audio Overlap Severity

Before comparing systems, each sample should have an overlap severity estimate.
This project uses the overlap ratio definition from the LibriCSS paper:

$$
\mathrm{OVR}
=
\frac{L_{\mathrm{ovl}}}{L_{\mathrm{all}}}
$$

where:

- `L_ovl` is total overlapped speech duration,
- `L_all` is total speech duration.

In project terms:

$$
L_{\mathrm{ovl}}
=
\text{duration where at least two speakers are active}
$$

$$
L_{\mathrm{all}}
=
\text{duration where at least one speaker is active}
$$

Interpretation:

$$
\mathrm{OVR}=0
$$

means no overlap, while:

$$
\mathrm{OVR}=0.30
$$

means 30 percent of speech time contains simultaneous speakers.

The current implementation estimates OVR from timestamped `segments` in
`results.json`. It prefers segments from `diarization_asr`, then
`diarization_turn_asr`, then `separation_asr`. Direct ASR rows reuse the
sample-level OVR estimate because direct ASR has no speaker-time segments.

The output field `ovr_source` is currently:

- `config`: read from sample metadata in the experiment config,
- `estimated`: computed from available result segments,
- `unavailable`: no usable timestamped segments were found.

If future synthetic data exports oracle speaker activity, that can be added as
`ovr_source=oracle`.

## Surface Accuracy

WER is defined as:

$$
\mathrm{WER}
=
\frac{S + D + I}{N}
$$

where:

- `S` is substitutions,
- `D` is deletions,
- `I` is insertions,
- `N` is the number of reference words.

For Mandarin ASR, CER is usually more stable because character-level evaluation
avoids word segmentation ambiguity:

$$
\mathrm{CER}
=
\frac{S_c + D_c + I_c}{N_c}
$$

where the counts are computed at the character level.

The existing experiment runner already computes CER/WER. The readability
evaluation reads those values from `results.json`.

## Semantic Faithfulness With BERTScore

BERTScore compares contextual token embeddings between a hypothesis text `H`
and a reference text `R`.

Precision:

$$
P_{\mathrm{BERT}}
=
\frac{1}{|H|}
\sum_{h_i \in H}
\max_{r_j \in R}
\mathrm{sim}(h_i,r_j)
$$

Recall:

$$
R_{\mathrm{BERT}}
=
\frac{1}{|R|}
\sum_{r_j \in R}
\max_{h_i \in H}
\mathrm{sim}(r_j,h_i)
$$

F1:

$$
F_{\mathrm{BERT}}
=
\frac{
2P_{\mathrm{BERT}}R_{\mathrm{BERT}}
}{
P_{\mathrm{BERT}} + R_{\mathrm{BERT}}
}
$$

Project interpretation:

- `BERT precision`: how much hypothesis content is supported by the reference.
  Lower values indicate hallucination or unsupported insertions.
- `BERT recall`: how much reference content is covered by the hypothesis.
  Lower values indicate missed clauses.
- `BERT F1`: balanced semantic faithfulness.

The current reports use `bert-base-chinese` for BERTScore.

## Recall-Weighted Coverage

Overlap speech often causes missing phrases. To emphasize coverage, the project
also reports the standard F-beta aggregation over BERTScore precision and
recall:

$$
F_{\beta,\mathrm{BERT}}
=
\frac{
(1+\beta^2)P_{\mathrm{BERT}}R_{\mathrm{BERT}}
}{
\beta^2P_{\mathrm{BERT}} + R_{\mathrm{BERT}}
}
$$

The implemented readability report uses:

$$
\beta = 2
$$

so the reported `bert_f2` is:

$$
F_{2,\mathrm{BERT}}
=
\frac{
5P_{\mathrm{BERT}}R_{\mathrm{BERT}}
}{
4P_{\mathrm{BERT}} + R_{\mathrm{BERT}}
}
$$

This is useful for heavy overlap because missed reference content should count
more strongly than in balanced F1.

## Speaker Consistency

For speaker-attributed outputs, the existing project speaker-block score is
converted into a higher-is-better consistency value. Speaker-block scoring first
groups hypothesis text by predicted speaker, then tries the best mapping between
predicted labels and reference speakers before computing CER/WER. This matters
because diarization systems often use arbitrary labels: a predicted `SPEAKER_00`
may correspond to reference `speaker_2`, not `speaker_1`.

$$
\mathrm{SpeakerConsistency}
=
1 - \min(\mathrm{SpeakerBlockCER},1)
$$

If a pipeline has no speaker-attributed output, speaker consistency is reported
as unavailable rather than zero:

$$
\mathrm{SpeakerConsistency}
=
\mathrm{N/A}
$$

This keeps direct ASR comparable on text quality without unfairly penalizing it
for not producing speaker labels.

## Formatting Readability

Punctuation and segmentation affect human readability. If a reliable
punctuation reference is available, punctuation error rate can be reported:

$$
\mathrm{PER}
=
\frac{S_p + D_p + I_p}{N_p}
$$

and:

$$
\mathrm{FormattingReadability}
=
1 - \min(\mathrm{PER},1)
$$

PER is not currently included in the implemented TRS because the project does
not yet maintain punctuation as a separate reliable annotation layer.

## True Readability Score

TRS is only a compact reporting index for tables. Component metrics should
always be shown nearby.

The current implementation intentionally uses a plain geometric mean, not a
weighted geometric mean. This avoids hand-tuned weights in the first version and
keeps the score easy to explain. A weak dimension still lowers the total score,
which matches the project concern that a transcript can be unusable because of
one severe failure mode.

Text-only readability:

$$
\mathrm{TRS}_{text}
=
100
\times
\sqrt{
\left(1-\min(\mathrm{CER},1)\right)
\times
F_{2,\mathrm{BERT}}
}
$$

Speaker-aware readability:

$$
\mathrm{TRS}_{speaker}
=
100
\times
\sqrt[3]{
\left(1-\min(\mathrm{CER},1)\right)
\times
F_{2,\mathrm{BERT}}
\times
\left(1-\min(\mathrm{SpeakerBlockCER},1)\right)
}
$$

If `SpeakerBlockCER` is unavailable, `TRS_speaker` is unavailable. The
implementation does not treat missing speaker metrics as zero.

## Output Fields

`readability_results.csv` and `readability_results.json` contain:

| Field | Meaning |
| --- | --- |
| `sample_id` | Sample identifier |
| `overlap_level` | Existing config label such as `none`, `light`, `medium`, `heavy` |
| `ovr` | Overlap ratio from config metadata or estimated segments |
| `ovr_source` | `config`, `estimated`, or `unavailable` |
| `pipeline` | Pipeline name |
| `model` | Model label from original result |
| `cer` | Existing primary CER |
| `wer` | Existing primary WER |
| `bert_precision` | BERTScore precision |
| `bert_recall` | BERTScore recall |
| `bert_f1` | BERTScore F1 |
| `bert_f2` | Recall-weighted BERTScore coverage |
| `speaker_block_cer` | Existing speaker-block CER when available |
| `speaker_consistency` | Higher-is-better speaker consistency |
| `trs_text` | Text-only readability score |
| `trs_speaker` | Speaker-aware readability score when available |
| `runtime_seconds` | Original pipeline runtime |
| `error` | Original pipeline error field |

## Recommended Report Interpretation

Do not choose a single winner only by CER/WER. Report winners by dimension:

```text
Best by CER/WER: ...
Best by semantic coverage: ...
Best under high OVR: ...
Best speaker-attributed transcript: ...
Failure notes: missed clauses / hallucinated insertions / speaker confusion
```

For low OVR, direct ASR may be sufficient if CER and BERTScore are both strong.
For medium and high OVR, compare whether diarization or separation improves
`bert_recall`, `bert_f2`, and `TRS_speaker` without causing hallucinated
insertions.

Current server-run interpretation:

- Best average text readability: `diarization_asr`, avg TRS text `85.8221`.
- Best average speaker-aware readability: `diarization_asr`, avg TRS speaker
  `85.4050`.
- Best by TRS text per sample: `diarization_turn_asr` for no overlap,
  `llm_rag_refine` for light/medium/heavy overlap, and `separation_asr` for the
  opposite-order overlap sample.
- High-overlap review: for `sample2_heavy_overlap`, `llm_rag_refine` is the
  best TRS-text candidate; for `sample2_opposite_overlap`, `separation_asr` is
  the best TRS-text and BERT F2 candidate.

## References

- Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, Yoav Artzi.
  "BERTScore: Evaluating Text Generation with BERT." arXiv:1904.09675.
  https://arxiv.org/abs/1904.09675
- Zhuo Chen, Takuya Yoshioka, Liang Lu, et al. "Continuous Speech Separation:
  Dataset and Analysis." arXiv:2001.11482.
  https://arxiv.org/abs/2001.11482
- Fan Yu, Shiliang Zhang, Yihui Fu, et al. "M2MeT: The ICASSP 2022 Multi-channel
  Multi-party Meeting Transcription Challenge." arXiv:2110.07393.
  https://arxiv.org/abs/2110.07393
- Jimmy Tobin, Qisheng Li, Subhashini Venugopalan, Katie Seaver, Richard Cave,
  Katrin Tomanek. "Assessing ASR Model Quality on Disordered Speech using
  BERTScore." arXiv:2209.10591.
  https://arxiv.org/abs/2209.10591
- Somnath Roy. "Semantic-WER: A Unified Metric for the Evaluation of ASR
  Transcript for End Usability." arXiv:2106.02016.
  https://arxiv.org/abs/2106.02016
- Junwei Liao, Sefik Emre Eskimez, Liyang Lu, Yu Shi, Ming Gong, Linjun Shou,
  Hong Qu, Michael Zeng. "Improving Readability for Automatic Speech
  Recognition Transcription." arXiv:2004.04438.
  https://arxiv.org/abs/2004.04438
