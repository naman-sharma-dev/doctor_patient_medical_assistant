"""
speech_to_text.py
=================
A speech-to-text module for the Doctor-Patient Medical Assistant.

Provides real-time microphone capture and audio-file transcription using the
``SpeechRecognition`` library.  Supports two recognition back-ends:

* **Google Web Speech API** (online, default) — high accuracy, no API key
  required for standard use.
* **CMU Sphinx** (offline fallback) — works without internet access; requires
  the optional ``pocketsphinx`` package.

Installation
------------
    pip install SpeechRecognition pyaudio
    # Optional — for offline/Sphinx support:
    pip install pocketsphinx

Quick start
-----------
    from speech_to_text import SpeechTranscriber

    transcriber = SpeechTranscriber()

    # From microphone (5-second window)
    result = transcriber.transcribe_microphone(duration=5)
    if result.success:
        print("Heard:", result.text)

    # From an audio file
    result = transcriber.transcribe_file("recording.wav")
    print(result.text)

Medical workflow
----------------
    from speech_to_text import SpeechTranscriber

    transcriber = SpeechTranscriber()
    notes = transcriber.capture_patient_symptoms()
    for field, value in notes.items():
        print(f"{field}: {value}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------

try:
    import speech_recognition as sr  # type: ignore

    _SR_AVAILABLE = True
except ImportError:
    _SR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TranscriptionResult:
    """
    Result of a single speech-to-text transcription attempt.

    Attributes
    ----------
    text : str
        The transcribed text.  Empty string on failure.
    success : bool
        ``True`` when transcription succeeded.
    backend : str
        The recognition back-end that produced the result
        (``"google"`` or ``"sphinx"``).
    duration_seconds : float
        Wall-clock time taken for the recognition call.
    error : str
        Human-readable error description when ``success`` is ``False``.
    """

    text: str = ""
    success: bool = False
    backend: str = "google"
    duration_seconds: float = 0.0
    error: str = ""

    def __str__(self) -> str:  # pragma: no cover
        if self.success:
            return (
                f"[OK | {self.backend} | {self.duration_seconds:.2f}s] {self.text}"
            )
        return f"[FAIL | {self.backend}] {self.error}"


# ---------------------------------------------------------------------------
# Core transcriber
# ---------------------------------------------------------------------------


class SpeechTranscriber:
    """
    Wraps the ``SpeechRecognition`` library with a simple, medical-assistant-
    friendly API.

    Parameters
    ----------
    backend : str
        Primary recognition back-end: ``"google"`` (default) or ``"sphinx"``.
    language : str
        BCP-47 language tag used by the Google back-end (default ``"en-US"``).
    energy_threshold : int
        Microphone energy threshold for detecting speech vs. silence
        (default ``300``; lower values are more sensitive).
    pause_threshold : float
        Seconds of silence before the end of a phrase is detected
        (default ``0.8``).
    """

    SUPPORTED_BACKENDS = ("google", "sphinx")

    def __init__(
        self,
        backend: str = "google",
        language: str = "en-US",
        energy_threshold: int = 300,
        pause_threshold: float = 0.8,
    ) -> None:
        if not _SR_AVAILABLE:
            raise ImportError(
                "The 'SpeechRecognition' package is required. "
                "Install it with:  pip install SpeechRecognition pyaudio"
            )
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{backend}'. "
                f"Choose from: {self.SUPPORTED_BACKENDS}"
            )

        self.backend = backend
        self.language = language

        self._recognizer = sr.Recognizer()
        self._recognizer.energy_threshold = energy_threshold
        self._recognizer.pause_threshold = pause_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe_microphone(
        self,
        duration: Optional[float] = None,
        prompt: str = "",
    ) -> TranscriptionResult:
        """
        Record from the default microphone and transcribe.

        Parameters
        ----------
        duration : float, optional
            Maximum recording time in seconds.  When ``None`` (default) the
            recognizer listens until a pause is detected.
        prompt : str
            Optional on-screen prompt displayed before listening begins.

        Returns
        -------
        TranscriptionResult
        """
        if prompt:
            print(prompt)

        try:
            with sr.Microphone() as source:
                print("Adjusting for ambient noise… (please wait)")
                self._recognizer.adjust_for_ambient_noise(source, duration=1)
                print(
                    "Listening"
                    + (f" (up to {duration}s)…" if duration else "…")
                    + "  Speak now."
                )
                audio = self._recognizer.listen(source, phrase_time_limit=duration)
        except OSError as exc:
            return TranscriptionResult(
                backend=self.backend,
                error=f"Microphone unavailable: {exc}",
            )

        return self._recognize(audio)

    def transcribe_file(self, file_path: str | Path) -> TranscriptionResult:
        """
        Transcribe speech from a WAV, AIFF, or FLAC audio file.

        Parameters
        ----------
        file_path : str or Path
            Path to the audio file.

        Returns
        -------
        TranscriptionResult
        """
        path = Path(file_path)
        if not path.exists():
            return TranscriptionResult(
                backend=self.backend,
                error=f"File not found: {file_path}",
            )

        try:
            with sr.AudioFile(str(path)) as source:
                audio = self._recognizer.record(source)
        except (ValueError, OSError) as exc:
            return TranscriptionResult(
                backend=self.backend,
                error=f"Could not read audio file: {exc}",
            )

        return self._recognize(audio)

    def capture_patient_symptoms(
        self,
        fields: Optional[List[str]] = None,
        duration_per_field: float = 10.0,
    ) -> Dict[str, str]:
        """
        Guided voice-capture workflow for collecting patient information.

        Iterates through a list of clinical fields, prompts the patient (or
        clinician) to speak their response, and returns a dictionary of
        field → transcribed text pairs.

        Parameters
        ----------
        fields : list[str], optional
            Clinical fields to capture.  Defaults to a standard set of
            symptom-intake fields.
        duration_per_field : float
            Maximum recording time per field in seconds (default ``10.0``).

        Returns
        -------
        dict[str, str]
            Mapping of field name → transcribed response.
            Failed transcriptions are stored as empty strings.
        """
        if fields is None:
            fields = [
                "Chief complaint",
                "Duration of symptoms",
                "Severity (1-10)",
                "Associated symptoms",
                "Relevant medical history",
                "Current medications",
                "Allergies",
            ]

        captured: Dict[str, str] = {}
        print("\n" + "=" * 55)
        print(" Patient Symptom Intake — Voice Capture")
        print("=" * 55)
        print("Speak clearly after each prompt.  Say 'skip' to skip.\n")

        for i, field_name in enumerate(fields, start=1):
            prompt = f"[{i}/{len(fields)}]  {field_name}: "
            result = self.transcribe_microphone(
                duration=duration_per_field,
                prompt=prompt,
            )
            if result.success:
                text = result.text.strip()
                if text.lower() == "skip":
                    captured[field_name] = ""
                    print(f"  ↳ Skipped.\n")
                else:
                    captured[field_name] = text
                    print(f"  ↳ Captured: {text}\n")
            else:
                captured[field_name] = ""
                print(f"  ↳ Could not transcribe ({result.error}).  Left blank.\n")

        print("=" * 55)
        print("Voice capture complete.")
        print("=" * 55 + "\n")
        return captured

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _recognize(self, audio: "sr.AudioData") -> TranscriptionResult:
        """Run the chosen back-end and return a ``TranscriptionResult``."""
        start = time.perf_counter()
        try:
            if self.backend == "google":
                text = self._recognizer.recognize_google(
                    audio, language=self.language
                )
            else:
                text = self._recognizer.recognize_sphinx(audio)

            elapsed = time.perf_counter() - start
            return TranscriptionResult(
                text=text,
                success=True,
                backend=self.backend,
                duration_seconds=elapsed,
            )

        except sr.UnknownValueError:
            elapsed = time.perf_counter() - start
            return TranscriptionResult(
                backend=self.backend,
                duration_seconds=elapsed,
                error="Speech was unintelligible or no speech detected.",
            )
        except sr.RequestError as exc:
            elapsed = time.perf_counter() - start
            return TranscriptionResult(
                backend=self.backend,
                duration_seconds=elapsed,
                error=f"Recognition service error: {exc}",
            )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def transcribe(
    source: str | Path | None = None,
    backend: str = "google",
    language: str = "en-US",
    duration: Optional[float] = None,
) -> TranscriptionResult:
    """
    One-call convenience wrapper.

    Parameters
    ----------
    source : str or Path or None
        Path to an audio file, or ``None`` to use the microphone (default).
    backend : str
        ``"google"`` (default) or ``"sphinx"``.
    language : str
        BCP-47 language tag (Google back-end only).
    duration : float, optional
        Maximum recording/transcription time in seconds.

    Returns
    -------
    TranscriptionResult
    """
    transcriber = SpeechTranscriber(backend=backend, language=language)
    if source is None:
        return transcriber.transcribe_microphone(duration=duration)
    return transcriber.transcribe_file(source)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_cli() -> None:  # pragma: no cover
    """Interactive command-line interface for the speech-to-text module."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="speech_to_text",
        description="Doctor-Patient Medical Assistant — Speech-to-Text",
    )
    sub = parser.add_subparsers(dest="command")

    # --- mic subcommand ---
    mic_parser = sub.add_parser("mic", help="Transcribe from microphone")
    mic_parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=None,
        metavar="SECS",
        help="Max recording time in seconds (default: until pause)",
    )
    mic_parser.add_argument(
        "-b",
        "--backend",
        choices=SpeechTranscriber.SUPPORTED_BACKENDS,
        default="google",
        help="Recognition back-end (default: google)",
    )
    mic_parser.add_argument(
        "-l",
        "--language",
        default="en-US",
        help="BCP-47 language code for Google back-end (default: en-US)",
    )

    # --- file subcommand ---
    file_parser = sub.add_parser("file", help="Transcribe an audio file")
    file_parser.add_argument("path", help="Path to WAV / AIFF / FLAC file")
    file_parser.add_argument(
        "-b",
        "--backend",
        choices=SpeechTranscriber.SUPPORTED_BACKENDS,
        default="google",
    )
    file_parser.add_argument("-l", "--language", default="en-US")

    # --- intake subcommand ---
    sub.add_parser("intake", help="Guided patient symptom intake via voice")

    args = parser.parse_args()

    if not _SR_AVAILABLE:
        print(
            "Error: SpeechRecognition is not installed.\n"
            "Run:  pip install SpeechRecognition pyaudio"
        )
        raise SystemExit(1)

    if args.command == "mic":
        t = SpeechTranscriber(backend=args.backend, language=args.language)
        result = t.transcribe_microphone(duration=args.duration)
        print(result)

    elif args.command == "file":
        t = SpeechTranscriber(backend=args.backend, language=args.language)
        result = t.transcribe_file(args.path)
        print(result)

    elif args.command == "intake":
        t = SpeechTranscriber()
        notes = t.capture_patient_symptoms()
        print("\n--- Captured Patient Notes ---")
        for key, value in notes.items():
            print(f"  {key}: {value or '(not captured)'}")

    else:
        parser.print_help()


if __name__ == "__main__":
    _run_cli()
