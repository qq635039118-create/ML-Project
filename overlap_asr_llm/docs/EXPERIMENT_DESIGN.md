# Experiment Design

## Research Question

How do diarization, speech separation, and LLM/RAG refinement change transcript
quality when speech overlap becomes more severe?

## Conditions

Use at least five audio conditions in the final experiment:

- No overlap
- Light overlap
- Medium overlap
- Heavy overlap
- Mixed-language or domain-specific speech

## Pipelines

- Direct ASR on mixed audio
- Direct ASR plus speaker diarization
- Speech separation followed by ASR on each separated stream
- LLM/RAG refinement using project glossary and previous pipeline outputs

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
