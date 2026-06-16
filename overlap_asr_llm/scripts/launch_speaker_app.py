"""Launch a FunClip-style speaker transcript demo.

The app uploads one audio file, runs ASR + diarization, optionally compares
PyAnnote and SpeechBrain, and can send the selected transcript through a
lightweight RAG + OpenAI-compatible LLM refiner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from overlap_asr_llm.providers import make_asr, make_diarizer, make_llm_refiner
from overlap_asr_llm.rag import retrieve_rag_context, tags_for_sample


def _format_timestamp(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    minutes, second_value = divmod(seconds, 60)
    hours, minute_value = divmod(int(minutes), 60)
    return f"{hours:02d}:{minute_value:02d}:{second_value:06.3f}"


def _subtitle_text(segments: list[dict[str, object]]) -> str:
    lines = []
    for segment in sorted(segments, key=lambda item: float(item.get("start", 0.0))):
        start = _format_timestamp(float(segment.get("start", 0.0)))
        end = _format_timestamp(float(segment.get("end", 0.0)))
        speaker = str(segment.get("speaker", "UNKNOWN"))
        text = str(segment.get("text", "")).strip()
        lines.append(f"{start} --> {end} [{speaker}] {text}")
    return "\n".join(lines)


def _speaker_text(segments: list[dict[str, object]]) -> str:
    by_speaker: dict[str, list[str]] = {}
    for segment in sorted(segments, key=lambda item: float(item.get("start", 0.0))):
        speaker = str(segment.get("speaker", "UNKNOWN"))
        start = _format_timestamp(float(segment.get("start", 0.0)))
        end = _format_timestamp(float(segment.get("end", 0.0)))
        text = str(segment.get("text", "")).strip()
        by_speaker.setdefault(speaker, []).append(f"[{start} - {end}] {text}")

    blocks = []
    for speaker, lines in by_speaker.items():
        blocks.append(f"## {speaker}\n" + "\n".join(lines))
    return "\n\n".join(blocks)


def _best_speaker(
    segment: dict[str, object],
    speaker_turns: list[dict[str, object]],
) -> str:
    if not speaker_turns:
        return "UNKNOWN"

    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", start))
    if end <= start:
        end = start + 0.01

    best_label = "UNKNOWN"
    best_overlap = 0.0
    for turn in speaker_turns:
        turn_start = float(turn.get("start", 0.0))
        turn_end = float(turn.get("end", turn_start))
        overlap = max(0.0, min(end, turn_end) - max(start, turn_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = str(turn.get("speaker", "UNKNOWN"))
    return best_label


def _attach_text_to_turns(
    transcript_segments: list[dict[str, object]],
    speaker_turns: list[dict[str, object]],
) -> list[dict[str, object]]:
    turns = [dict(turn) for turn in speaker_turns]
    for turn in turns:
        turn["text"] = ""

    for segment in transcript_segments:
        speaker = _best_speaker(segment, turns)
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        best_turn = None
        best_overlap = 0.0
        for turn in turns:
            if str(turn.get("speaker")) != speaker:
                continue
            turn_start = float(turn.get("start", 0.0))
            turn_end = float(turn.get("end", turn_start))
            overlap = max(0.0, min(end, turn_end) - max(start, turn_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_turn = turn
        if best_turn is None:
            turns.append(
                {
                    "start": start,
                    "end": end,
                    "speaker": speaker,
                    "text": text,
                }
            )
        else:
            current = str(best_turn.get("text", "")).strip()
            best_turn["text"] = f"{current} {text}".strip()
    return turns


def _run_one(
    audio_path: Path,
    asr_model: str,
    diarizer_model: str,
    speakers: int,
    language: str,
    prompt: str | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    asr = make_asr(asr_model)
    transcript = asr.transcribe(audio_path, language, prompt=prompt)
    if hasattr(asr, "release_gpu"):
        asr.release_gpu()

    diarizer = make_diarizer(diarizer_model)
    speaker_turns = diarizer.diarize(audio_path, speakers)
    if hasattr(diarizer, "release_gpu"):
        diarizer.release_gpu()

    segments = _attach_text_to_turns(transcript.segments, speaker_turns)
    speaker_labels = ",".join(
        dict.fromkeys(str(segment.get("speaker", "UNKNOWN")) for segment in segments)
    )
    return {
        "model": f"{asr_model}+{diarizer_model}",
        "runtime_seconds": round(time.perf_counter() - started, 4),
        "speaker_labels": speaker_labels,
        "speaker_count": len([item for item in speaker_labels.split(",") if item]),
        "raw_asr_text": transcript.text,
        "segments": segments,
        "subtitle_text": _subtitle_text(segments),
        "speaker_text": _speaker_text(segments),
    }


def process_audio(
    audio_file: str,
    asr_model: str,
    diarizer_choice: str,
    speakers: int,
    language: str,
    overlap_level: str,
    hotwords: str,
    use_llm: bool,
    llm_model: str,
) -> tuple[str, str, str, str]:
    if not audio_file:
        return "Please upload an audio file.", "", "", ""

    audio_path = Path(audio_file)
    prompt = hotwords.strip() or None
    diarizers = (
        ["pyannote", "speechbrain"]
        if diarizer_choice == "both_compare"
        else [diarizer_choice]
    )

    results = []
    errors = []
    for diarizer in diarizers:
        try:
            results.append(
                _run_one(
                    audio_path=audio_path,
                    asr_model=asr_model,
                    diarizer_model=diarizer,
                    speakers=int(speakers),
                    language=language,
                    prompt=prompt,
                )
            )
        except Exception as exc:
            errors.append(f"{diarizer}: {type(exc).__name__}: {exc}")

    if not results:
        return "\n".join(errors), "", "", ""

    compare_rows = [
        {
            "model": result["model"],
            "runtime_seconds": result["runtime_seconds"],
            "speaker_count": result["speaker_count"],
            "speaker_labels": result["speaker_labels"],
        }
        for result in results
    ]
    compare_text = json.dumps(compare_rows, ensure_ascii=False, indent=2)
    if errors:
        compare_text += "\n\nErrors:\n" + "\n".join(errors)

    selected = results[0]
    speaker_text = selected["speaker_text"]
    subtitle_text = selected["subtitle_text"]
    llm_text = ""

    if use_llm:
        try:
            tags = tags_for_sample(overlap_level, "speaker_transcript")
            context = retrieve_rag_context(tags, base_context=[])
            source_text = (
                "Retrieved context:\n"
                + "\n".join(f"- {item}" for item in context)
                + "\n\nSpeaker transcript:\n"
                + subtitle_text
            )
            refiner_kind = f"api:{llm_model}" if llm_model.strip() else "api"
            llm_text = make_llm_refiner(refiner_kind).refine(source_text, context)
        except Exception as exc:
            llm_text = f"LLM refinement error: {type(exc).__name__}: {exc}"

    return compare_text, speaker_text, subtitle_text, llm_text


def main() -> int:
    import gradio as gr

    if not hasattr(gr, "Blocks"):
        raise RuntimeError(
            "This app requires gradio>=4.0.0. Run: pip install -U gradio"
        )

    parser = argparse.ArgumentParser(description="Launch speaker transcript app.")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    with gr.Blocks(title="Overlap ASR Speaker Transcript") as app:
        gr.Markdown("# Overlap ASR Speaker Transcript")
        with gr.Row():
            audio_file = gr.Audio(label="Audio", type="filepath")
            with gr.Column():
                asr_model = gr.Textbox(
                    label="ASR model",
                    value="faster-whisper:large-v3",
                )
                diarizer_choice = gr.Dropdown(
                    label="Diarization",
                    choices=["pyannote", "speechbrain", "both_compare"],
                    value="both_compare",
                )
                speakers = gr.Number(label="Expected speakers", value=2, precision=0)
                language = gr.Textbox(label="Language", value="zh")
                overlap_level = gr.Dropdown(
                    label="Overlap level",
                    choices=["none", "light", "medium", "heavy", "opposite", "unknown"],
                    value="unknown",
                )
                hotwords = gr.Textbox(label="ASR prompt / hotwords", value="")
                use_llm = gr.Checkbox(label="Use LLM/RAG refinement", value=False)
                llm_model = gr.Textbox(label="LLM model", value="")
                run_button = gr.Button("Run", variant="primary")

        compare_output = gr.Textbox(label="Model comparison", lines=8)
        speaker_output = gr.Textbox(label="Speaker grouped transcript", lines=14)
        subtitle_output = gr.Textbox(label="Subtitle transcript", lines=14)
        llm_output = gr.Textbox(label="LLM/RAG refinement", lines=14)

        run_button.click(
            process_audio,
            inputs=[
                audio_file,
                asr_model,
                diarizer_choice,
                speakers,
                language,
                overlap_level,
                hotwords,
                use_llm,
                llm_model,
            ],
            outputs=[compare_output, speaker_output, subtitle_output, llm_output],
        )

    app.launch(server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
