from __future__ import annotations

import os
import re
import time
import json
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any

# whisper is the open-source OpenAI Whisper package
# import whisper


SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".mp4", ".webm", ".ogg", ".flac", ".aac"}


@dataclass
class TranscriptionResult:
    ok: bool
    text: str
    language: str
    duration_sec: float
    model_name: str
    audio_path: str
    segments: Optional[list] = None
    error: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class SpeechToTextWhisper:
    """
    Production-oriented Whisper wrapper:
    - Validates input file (exists, ext)
    - Caches model (singleton per instance)
    - Optional file-based caching by audio hash
    - Returns unified TranscriptionResult with ok/error
    """

    def __init__(
        self,
        model_name: str = "medium",
        language: str = "ar",
        device: Optional[str] = None,
        cache_dir: str = "models/whisper/cache",
        enable_result_cache: bool = True,
        verbose: bool = False,
    ) -> None:
        self.model_name = model_name
        self.language = language
        self.device = device  # None -> whisper chooses
        self.verbose = verbose

        self.enable_result_cache = enable_result_cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load model once
        # self._model = whisper.load_model(self.model_name, device=self.device)

    @staticmethod
    def _validate_audio_path(audio_path: str) -> Path:
        p = Path(audio_path)
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {p}")
        if not p.is_file():
            raise ValueError(f"Audio path is not a file: {p}")
        if p.suffix.lower() not in SUPPORTED_AUDIO_EXTS:
            raise ValueError(
                f"Unsupported audio extension: {p.suffix}. Supported: {sorted(SUPPORTED_AUDIO_EXTS)}"
            )
        return p

    @staticmethod
    def _compute_file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _normalize_text_ar(text: str) -> str:
        """
        Minimal, safe normalization for Arabic text:
        - strip extra spaces
        - normalize some letters
        - remove repeated punctuation
        """
        if not text:
            return ""

        text = text.strip()
        # Normalize Arabic letter variants (light)
        text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
        text = text.replace("ة", "ه")  # optional: helps matching in simple rules
        # Collapse spaces
        text = re.sub(r"\s+", " ", text)
        # Remove repeated punctuation
        text = re.sub(r"([!؟?.،])\1+", r"\1", text)
        return text

    def _cache_path_for_audio(self, audio_hash: str) -> Path:
        return self.cache_dir / f"{audio_hash}_{self.model_name}_{self.language}.json"

    def transcribe(
        self,
        audio_path: str,
        return_segments: bool = False,
        normalize_text: bool = True,
        temperature: float = 0.0,
        beam_size: int = 5,
        best_of: int = 5,
    ) -> TranscriptionResult:
        """
        Transcribe Arabic speech from audio file to text.

        Args:
            audio_path: path to audio file
            return_segments: include segments in result (useful for timestamps)
            normalize_text: light normalization for downstream NLP/rules
            temperature: decoding temperature (0.0 more deterministic)
            beam_size/best_of: decoding params; higher may improve quality but slower

        Returns:
            TranscriptionResult
        """
        started = time.time()

        try:
            p = self._validate_audio_path(audio_path)

            audio_hash = None
            cache_file = None
            if self.enable_result_cache:
                audio_hash = self._compute_file_hash(p)
                cache_file = self._cache_path_for_audio(audio_hash)
                if cache_file.exists():
                    cached = json.loads(cache_file.read_text(encoding="utf-8"))
                    # Reconstruct dataclass
                    res = TranscriptionResult(**cached)
                    return res

            # Whisper transcribe
            # fp16=False is safer on CPU/Windows; Whisper will manage internally but we force stability
            result = self._model.transcribe(
                str(p),
                language=self.language,
                task="transcribe",
                fp16=False,
                verbose=self.verbose,
                temperature=temperature,
                beam_size=beam_size,
                best_of=best_of,
            )

            text = result.get("text", "") or ""
            detected_lang = result.get("language", self.language) or self.language

            if normalize_text:
                text = self._normalize_text_ar(text)

            duration_sec = time.time() - started

            res = TranscriptionResult(
                ok=True,
                text=text,
                language=detected_lang,
                duration_sec=duration_sec,
                model_name=self.model_name,
                audio_path=str(p),
                segments=result.get("segments") if return_segments else None,
                error=None,
                meta={
                    "audio_hash": audio_hash,
                    "cached": False,
                },
            )

            if self.enable_result_cache and cache_file is not None:
                cache_file.write_text(json.dumps(asdict(res), ensure_ascii=False, indent=2), encoding="utf-8")

            return res

        except Exception as e:
            duration_sec = time.time() - started
            return TranscriptionResult(
                ok=False,
                text="",
                language=self.language,
                duration_sec=duration_sec,
                model_name=self.model_name,
                audio_path=audio_path,
                segments=None,
                error=f"{type(e).__name__}: {str(e)}",
                meta={"cached": False},
            )


if __name__ == "__main__":
    # Quick local test:
    # Put a test file at: data/raw/audio/test.wav
    stt = SpeechToTextWhisper(model_name="medium", language="ar", enable_result_cache=True, verbose=False)
    out = stt.transcribe("data/raw/audio/test.wav", return_segments=False)
    print(json.dumps(asdict(out), ensure_ascii=False, indent=2))
