# English Video Script Outline

## 0:00-1:00 Motivation

Introduce overlapping speech as a real-world ASR failure mode in meetings,
interviews, debates, and classrooms. State the main question: when two speakers
overlap, should we use direct ASR, diarization, separation, or LLM/RAG
refinement?

## 1:00-2:30 Paper and Related Work

Summarize the paper in `project/xutong_paper.pdf`, then explain the gap this
project explores: comparing direct ASR, diarization, separation, and LLM/RAG
refinement under controlled overlap levels.

Briefly cover four related topics: overlapping speech recognition, speaker
diarization, speech separation, and LLM/RAG post-correction.

## 2:30-5:00 Controlled Experiment

Show the five debate-audio samples. Explain that they use the same debate
material but change speaker timing to create no, light, medium, heavy, and
opposite-order overlap.

## 5:00-7:30 Pipeline Comparison

Compare direct ASR, diarization ASR, turn-level diarization ASR, separation ASR,
and LLM/RAG refinement. Show runtime, readable examples, CER/WER, and the
post-run readability metrics from `readability_summary.md`: BERTScore F2, TRS
text, and TRS speaker.

## 7:30-9:30 Failure Cases

Compare examples where separation helps, where it hurts, where diarization
confuses speakers, and where LLM refinement improves readability or hallucinates.
Make the limitations explicit: diarization labels speakers but cannot recover
masked words; separation can introduce artifacts; LLM/RAG must not invent
content.

## 9:30-10:30 Conclusion

Propose an overlap-aware strategy: low overlap uses direct ASR, heavier overlap
uses separation plus ASR, and LLM/RAG is restricted to formatting and terminology
correction. Summarize engineering trade-offs, team contributions, and future
work.
