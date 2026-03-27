"""
app.py
======
Doctor-Patient Medical Assistant — Voice-Powered Web App

A Gradio web application that lets you (or your patient) speak directly into
the device microphone.  The recording is:

  1. Transcribed to text via Google Web Speech API (``SpeechRecognition``).
  2. Parsed to extract known symptoms and risk-history keywords.
  3. Passed through ``ClinicalBlindspotDetector`` to surface commonly missed
     diagnoses with prioritised alerts.

Run
---
    pip install -r requirements.txt
    python app.py

The app opens automatically in your default browser at http://127.0.0.1:7860
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr

from clinical_blindspot import (
    BlindspotAlert,
    ClinicalBlindspotDetector,
    PatientProfile,
    BLINDSPOT_DATABASE,
)
from speech_to_text import SpeechTranscriber, _SR_AVAILABLE

# ---------------------------------------------------------------------------
# Symptom / history vocabulary — derived dynamically from BLINDSPOT_DATABASE
# ---------------------------------------------------------------------------

# Build vocabulary sets directly from the knowledge base so that they stay
# in sync with any future additions to BLINDSPOT_DATABASE without manual
# updates here.

_ALL_SYMPTOMS: List[str] = sorted(
    {
        s
        for entry in BLINDSPOT_DATABASE
        for s in list(entry["key_symptoms"]) + list(entry.get("supporting_symptoms", set()))
    }
)

# Sort longest first so multi-word phrases like "shortness of breath" are
# matched (and blanked) before shorter tokens that share words with them.
_ALL_SYMPTOMS_SORTED: List[str] = sorted(_ALL_SYMPTOMS, key=len, reverse=True)

_ALL_HISTORY_KEYWORDS: List[str] = sorted(
    {
        kw
        for entry in BLINDSPOT_DATABASE
        for kw in entry.get("risk_history", set())
    }
)

_ALL_HISTORY_SORTED: List[str] = sorted(
    _ALL_HISTORY_KEYWORDS, key=len, reverse=True
)

# ---------------------------------------------------------------------------
# NLP helpers
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Lower-case and collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def extract_symptoms(text: str) -> List[str]:
    """
    Return all known symptoms found in *text* as a list of lowercase strings.

    Strategy: longest-first scan with regex word-boundary replacement so that
    multi-word phrases like ``"shortness of breath"`` are matched and masked
    before shorter tokens that overlap with them, preventing double-counting.
    """
    normalised = _normalise(text)
    found: List[str] = []
    remaining = normalised
    for symptom in _ALL_SYMPTOMS_SORTED:
        pattern = r"\b" + re.escape(symptom) + r"\b"
        if re.search(pattern, remaining):
            found.append(symptom)
            # Mask the matched span to prevent shorter overlapping terms from
            # matching within the same span.
            remaining = re.sub(pattern, " ", remaining)
    return sorted(set(found))


def extract_history(text: str) -> List[str]:
    """
    Return all known medical/family history keywords found in *text*.

    Uses the same longest-first + word-boundary replacement strategy as
    ``extract_symptoms`` to avoid double-counting overlapping keyword spans.
    """
    normalised = _normalise(text)
    found: List[str] = []
    remaining = normalised
    for keyword in _ALL_HISTORY_SORTED:
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, remaining):
            found.append(keyword)
            remaining = re.sub(pattern, " ", remaining)
    return sorted(set(found))


# ---------------------------------------------------------------------------
# Speech-to-text helper
# ---------------------------------------------------------------------------


def transcribe_audio(audio_path: Optional[str]) -> Tuple[str, str]:
    """
    Transcribe an audio file recorded by the Gradio microphone widget.

    Parameters
    ----------
    audio_path : str or None
        Path to the temporary audio file provided by Gradio.

    Returns
    -------
    (transcript, error_message) : (str, str)
        On success ``error_message`` is empty; on failure ``transcript`` is
        empty and ``error_message`` describes the problem.
    """
    if not audio_path:
        return "", "No audio recorded.  Click the microphone button and speak."

    if not _SR_AVAILABLE:
        return "", (
            "SpeechRecognition is not installed.  Run:\n"
            "  pip install SpeechRecognition pyaudio"
        )

    transcriber = SpeechTranscriber(backend="google")
    result = transcriber.transcribe_file(audio_path)

    if result.success:
        return result.text, ""
    return "", result.error


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

_PRIORITY_EMOJI = {"high": "🔴", "medium": "🟡", "low": "🟢"}


def format_alerts(alerts: List[BlindspotAlert]) -> str:
    """Render a list of ``BlindspotAlert`` objects as Markdown."""
    if not alerts:
        return (
            "✅ **No clinical blind spots flagged** for the provided profile.\n\n"
            "_This may mean the symptom set does not match any tracked conditions "
            "at the configured threshold, or that the transcription missed some "
            "symptoms.  Review the transcript above and adjust if needed._"
        )

    lines = ["## 🩺 Clinical Blind-Spot Analysis\n"]
    for alert in alerts:
        emoji = _PRIORITY_EMOJI.get(alert.priority, "⚪")
        lines.append(f"### {emoji} {alert.condition} &nbsp;·&nbsp; `{alert.priority.upper()} priority`\n")
        lines.append(f"**Match score:** {alert.match_score:.0%}  \n")
        if alert.matched_symptoms:
            lines.append(
                f"**Matched symptoms:** {', '.join(alert.matched_symptoms)}  \n"
            )
        lines.append(f"**Why it's often missed:** {alert.description}  \n")
        lines.append(
            f"**Recommended tests:** {', '.join(alert.recommended_tests)}  \n"
        )
        lines.append("---\n")

    lines.append(
        "\n> ⚠️ **Disclaimer:** This tool is for clinical decision support only. "
        "It does not replace professional medical judgment."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def run_analysis(
    audio_path: Optional[str],
    age: int,
    sex: str,
    extra_symptoms_text: str,
    extra_history_text: str,
    min_match_score: float,
) -> Tuple[str, str, str]:
    """
    Full pipeline: audio → transcript → symptom extraction → blindspot detection.

    Returns
    -------
    (transcript, symptoms_found, results_markdown) : (str, str, str)
    """
    # --- Transcribe mic audio ---
    transcript, error = transcribe_audio(audio_path)

    if error:
        return "", "", f"❌ **Transcription error:** {error}"

    # Combine transcribed speech with any manually typed additions
    combined_symptoms_text = transcript
    if extra_symptoms_text.strip():
        combined_symptoms_text += " " + extra_symptoms_text.strip()

    combined_history_text = transcript
    if extra_history_text.strip():
        combined_history_text += " " + extra_history_text.strip()

    # --- Extract symptoms & history ---
    symptoms = extract_symptoms(combined_symptoms_text)
    history = extract_history(combined_history_text)

    if not symptoms:
        symptoms_display = "_No known symptoms detected in the transcript._"
        return (
            transcript,
            symptoms_display,
            (
                "⚠️ **No recognizable symptoms found.**\n\n"
                "Try describing your symptoms more explicitly, for example:\n"
                '> _"I have fatigue, weight gain, cold intolerance, and dry skin."_\n\n'
                "You can also type additional symptoms in the **Extra symptoms** box."
            ),
        )

    symptoms_display = ", ".join(symptoms)

    # --- Build patient profile ---
    patient = PatientProfile(
        age=age,
        sex=sex,
        symptoms=symptoms,
        medical_history=history,
        family_history=[],
    )

    # --- Run detector ---
    detector = ClinicalBlindspotDetector(min_match_score=min_match_score)
    alerts = detector.analyze(patient)

    return transcript, symptoms_display, format_alerts(alerts)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""

    with gr.Blocks(
        title="Doctor-Patient Medical Assistant",
    ) as demo:
        gr.Markdown(
            """
# 🏥 Doctor-Patient Medical Assistant
### Voice-Powered Clinical Blind-Spot Detector

**How to use:**
1. Fill in **age** and **sex** below.
2. Click **🎙 Record** and describe your symptoms out loud  
   _(e.g. "I have fatigue, weight gain, cold intolerance, and dry skin")_.
3. Click **Analyse** — the app transcribes your speech and checks for
   commonly missed conditions.

> _You can also type extra symptoms or history directly in the text boxes._
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                # --- Patient demographics ---
                gr.Markdown("### Patient Demographics")
                age_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=40,
                    step=1,
                    label="Age (years)",
                )
                sex_radio = gr.Radio(
                    choices=["female", "male", "other"],
                    value="female",
                    label="Sex",
                )

                # --- Voice input ---
                gr.Markdown("### 🎙 Voice Input")
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Record symptoms (speak clearly after clicking the mic icon)",
                )

                # --- Optional manual supplements ---
                with gr.Accordion("➕ Add extra symptoms or history (optional)", open=False):
                    extra_symptoms = gr.Textbox(
                        label="Extra symptoms (comma-separated)",
                        placeholder="e.g. fatigue, weight gain, dry skin",
                        lines=2,
                    )
                    extra_history = gr.Textbox(
                        label="Medical / family history (comma-separated)",
                        placeholder="e.g. hypertension, family history of thyroid",
                        lines=2,
                    )

                # --- Sensitivity ---
                with gr.Accordion("⚙️ Detection sensitivity", open=False):
                    min_match = gr.Slider(
                        minimum=0.10,
                        maximum=0.80,
                        value=0.30,
                        step=0.05,
                        label="Minimum symptom match score (lower = more sensitive)",
                    )

                analyse_btn = gr.Button("🔍 Analyse", variant="primary", size="lg")

            with gr.Column(scale=2):
                # --- Outputs ---
                gr.Markdown("### Results")
                transcript_box = gr.Textbox(
                    label="📝 Transcribed speech",
                    lines=3,
                    interactive=False,
                    placeholder="Your spoken words will appear here after analysis…",
                )
                symptoms_box = gr.Textbox(
                    label="🔎 Symptoms detected",
                    lines=2,
                    interactive=False,
                    placeholder="Recognised symptoms will be listed here…",
                )
                results_box = gr.Markdown(
                    value="_Results will appear here after you click Analyse._",
                    label="Clinical Analysis",
                )

        # --- Wire up the button ---
        analyse_btn.click(
            fn=run_analysis,
            inputs=[
                audio_input,
                age_slider,
                sex_radio,
                extra_symptoms,
                extra_history,
                min_match,
            ],
            outputs=[transcript_box, symptoms_box, results_box],
        )

        gr.Markdown(
            """
---
> ⚠️ **Medical Disclaimer:** This application is a clinical decision-support
> prototype.  It does not replace professional medical advice, diagnosis, or
> treatment.  Always consult a qualified healthcare provider.
"""
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )
