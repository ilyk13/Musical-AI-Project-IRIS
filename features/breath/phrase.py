"""Live phrase tracking from BreathCNN + voicing cues."""

from __future__ import annotations

import math

import numpy as np

from features.pitch.accuracy import phrase_pitch_score_from_deviations
from features.pitch.gesture import (
    GESTURE_GLISSANDO,
    GESTURE_STEADY,
    GESTURE_VIBRATO,
    is_slide_gesture,
)

BREATH_WINDOW = 480
BREATH_THRESHOLD = 0.6
MIN_BREATH_FRAMES = 6      # 60 ms sustained breath prob (less eager to end)
MIN_SILENCE_FRAMES = 45    # 450 ms unvoiced gap ends a confirmed phrase
MIN_SILENCE_ABORT = 45     # 450 ms silence cancels an unconfirmed voice blip
MIN_PHRASE_FRAMES = 45     # 450 ms voiced before a phrase is confirmed
MIN_PITCH_FRAMES = 3       # need ≥3 steady frames to judge pitch (matches /analyze)


def _pitch_score_from_deviation(mean_dev_cents: float) -> float:
    """Gentle curve — ~12¢ avg ≈ 88, ~25¢ ≈ 75, ~40¢ ≈ 62."""
    t = min(max(mean_dev_cents, 0.0) / 52.0, 1.0)
    return float(50.0 + 48.0 * (1.0 - t ** 0.75))


def _breath_score_from_cpp(cpp_mean: float) -> float:
    """Breath support score with a supportive floor for audible singing."""
    return float(np.clip(
        np.interp(cpp_mean, [0.0, 1.5, 4.0, 7.0, 10.0, 16.0], [48.0, 58.0, 68.0, 78.0, 88.0, 96.0]),
        48.0, 96.0,
    ))


def _vib_score_from_params(
    rate_hz: float,
    depth_cents: float,
    consistency: float,
) -> float:
    """Continuous vibrato quality — avoids harsh bucket jumps."""
    depth_amp = depth_cents / 2.0
    rate_score = float(np.clip(100.0 - abs(rate_hz - 6.0) * 14.0, 52.0, 98.0))
    if 25.0 <= depth_amp <= 75.0:
        depth_score = 90.0
    else:
        depth_score = float(np.clip(90.0 - min(abs(depth_amp - 50.0), 50.0) * 0.9, 52.0, 90.0))
    even_score = float(np.clip(55.0 + consistency * 180.0, 52.0, 95.0))
    return float(np.clip(rate_score * 0.40 + depth_score * 0.35 + even_score * 0.25, 52.0, 96.0))


def _score_label(score: int | None) -> str:
    if score is None:
        return "Complete"
    if score >= 90:
        return "Excellent"
    if score >= 78:
        return "Great"
    if score >= 65:
        return "Good"
    if score >= 52:
        return "Solid"
    return "Keep refining"


def _phrase_headline(score_label: str, phrase_score: int | None, *, has_tip: bool) -> str:
    if has_tip:
        return score_label
    if phrase_score is None:
        return "Phrase complete"
    return f"{score_label} phrase"


def _format_score_breakdown(
    pitch: float | None,
    breath: float | None,
    vibrato: float | None,
) -> str | None:
    parts: list[str] = []
    if pitch is not None:
        parts.append(f"Pitch {int(round(pitch))}")
    if vibrato is not None:
        parts.append(f"Vibrato {int(round(vibrato))}")
    if breath is not None:
        parts.append(f"Breath {int(round(breath))}")
    return " · ".join(parts) if parts else None


def breath_window_from_roll(
    buf: np.ndarray,
    samples_rx: int,
    frame_idx: int,
    *,
    window_size: int = BREATH_WINDOW,
    hop: int = 160,
    buf_index,
) -> np.ndarray:
    """480-sample window centred on NanoPitch frame (matches preprocess)."""
    half = window_size // 2
    centre = frame_idx * hop + hop // 2
    s_abs = centre - half
    seg = np.zeros(window_size, dtype=np.float32)
    for i in range(window_size):
        abs_i = s_abs + i
        if abs_i < 0 or abs_i >= samples_rx:
            continue
        pos = buf_index(len(buf), samples_rx, abs_i)
        if 0 <= pos < len(buf):
            seg[i] = buf[pos]
    return seg


class LivePhraseMetrics:
    """Accumulates per-phrase stats for live coaching at breath boundaries."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.start_t = 0.0
        self.end_t = 0.0
        self.deviations: list[float] = []
        self.slide_steps: list[float] = []
        self._last_slide_f0: float | None = None
        self.cpp_samples: list[float] = []
        self.vibrato_frames = 0
        self.glissando_frames = 0
        self.voiced_frames = 0
        self.pitch_frames = 0
        self.steady_frames = 0

    def add_frame(
        self,
        t: float,
        *,
        voiced: bool,
        scored: bool,
        et_dev_cents: float,
        gesture: int,
        cpp: float,
        f0_hz: float = 0.0,
    ) -> None:
        self.end_t = t
        if voiced:
            self.voiced_frames += 1
        if scored:
            self.pitch_frames += 1
            self.deviations.append(float(et_dev_cents))
        if is_slide_gesture(gesture) and f0_hz > 0:
            if self._last_slide_f0 is not None and self._last_slide_f0 > 0:
                step = abs(
                    1200.0 * math.log2(f0_hz / self._last_slide_f0 + 1e-12)
                )
                self.slide_steps.append(step)
            self._last_slide_f0 = float(f0_hz)
        elif f0_hz > 0:
            self._last_slide_f0 = None
        if gesture == GESTURE_STEADY:
            self.steady_frames += 1
        if gesture == GESTURE_VIBRATO:
            self.vibrato_frames += 1
        if gesture == GESTURE_GLISSANDO:
            self.glissando_frames += 1
        if cpp > 1.5:
            self.cpp_samples.append(float(cpp))

    def finish(
        self,
        phrase_id: int,
        vib_rate_hz: float,
        vib_depth_cents: float,
        vib_consistency: float,
    ) -> dict:
        duration = max(0.0, self.end_t - self.start_t)
        pitch_tips: list[str] = []
        vib_tips: list[str] = []
        breath_tips: list[str] = []
        positives: list[str] = []
        pitch_score: float | None = None
        breath_score: float | None = None
        vib_score: float | None = None
        headline = "Phrase complete"
        rating = "good"
        detail = f"{round(duration, 1)}s line"

        vib_frac = self.vibrato_frames / max(self.voiced_frames, 1)
        gliss_frac = self.glissando_frames / max(self.voiced_frames, 1)
        has_vib = (
            vib_frac >= 0.10
            and not math.isnan(vib_rate_hz)
            and not math.isnan(vib_depth_cents)
            and vib_depth_cents >= 12.0
            and vib_consistency >= 0.04
            and 3.5 <= vib_rate_hz <= 9.5
        )
        has_gliss = gliss_frac >= 0.08

        if self.pitch_frames >= MIN_PITCH_FRAMES and self.deviations:
            pitch_score = phrase_pitch_score_from_deviations(
                self.deviations,
                self.slide_steps or None,
            )
            if not has_vib:
                if pitch_score >= 88:
                    positives.append("Pitch stayed very centered")
                elif pitch_score >= 74:
                    positives.append("Pitch mostly held steady")
                elif pitch_score >= 60:
                    pitch_tips.append("Aim for one steady center on the next line")
                else:
                    pitch_tips.append("Try holding one note a little longer next time")
        elif self.voiced_frames >= 20 and self.pitch_frames < MIN_PITCH_FRAMES and not has_vib:
            pitch_tips.append("Sustain the line a bit longer for pitch feedback")

        if has_gliss:
            positives.append("Nice slide between notes")

        if self.cpp_samples:
            cpp_mean = float(np.mean(self.cpp_samples))
            breath_score = _breath_score_from_cpp(cpp_mean)
            if cpp_mean > 10:
                positives.append("Breath support stayed strong")
            elif cpp_mean > 5:
                breath_tips.append("Push breath support through the whole line")
            elif cpp_mean > 2:
                breath_tips.append("Engage support from the start of the phrase")

        if has_vib:
            vib_score = _vib_score_from_params(
                vib_rate_hz, vib_depth_cents, vib_consistency,
            )
            depth_amp = vib_depth_cents / 2.0
            if 5.0 <= vib_rate_hz <= 7.0 and 25.0 <= depth_amp <= 75.0:
                positives.append(f"Vibrato settled ({vib_rate_hz:.1f} Hz)")
            elif vib_rate_hz < 5.0:
                vib_tips.append(f"Vibrato a little slow ({vib_rate_hz:.1f} Hz) — aim for 5–7 Hz")
            elif vib_rate_hz > 7.0:
                vib_tips.append(f"Vibrato a little fast ({vib_rate_hz:.1f} Hz) — ease the pulse")
            elif vib_consistency < 0.15:
                vib_tips.append("Let the vibrato settle into an even pulse")
            elif depth_amp < 20 or depth_amp > 90:
                vib_tips.append("Even out the vibrato depth a touch")
            else:
                positives.append("Vibrato present on that line")

        parts: list[tuple[float, float]] = []
        if pitch_score is not None:
            parts.append((pitch_score, 0.45 if not has_vib else 0.28))
        if vib_score is not None:
            parts.append((vib_score, 0.27))
        if breath_score is not None:
            parts.append((breath_score, 0.28 if pitch_score is not None else 0.45))
        weight_sum = sum(w for _, w in parts)
        phrase_score = (
            int(round(sum(s * w for s, w in parts) / weight_sum))
            if parts else None
        )
        score_label = _score_label(phrase_score)
        breakdown = _format_score_breakdown(pitch_score, breath_score, vib_score)

        actionable = breath_tips + vib_tips + pitch_tips
        bullets = (actionable + positives)[:3]
        has_tip = bool(actionable)

        if pitch_tips:
            headline = f"{score_label} phrase" if phrase_score else "Pitch"
            rating = "warn"
            detail = pitch_tips[0]
        elif vib_tips:
            headline = f"{score_label} phrase" if phrase_score else "Vibrato"
            rating = "warn"
            detail = vib_tips[0]
        elif breath_tips:
            headline = f"{score_label} phrase" if phrase_score else "Breath"
            rating = "warn"
            detail = breath_tips[0]
        else:
            headline = _phrase_headline(score_label, phrase_score, has_tip=False)
            rating = "good" if (phrase_score or 0) >= 52 else "warn"
            detail = positives[0] if positives else breakdown or f"{round(duration, 1)}s line"

        if len(bullets) > 1 and not actionable:
            detail = f"{bullets[0]} · {bullets[1]}"
        elif len(actionable) > 1:
            detail = f"{actionable[0]} · {actionable[1]}"

        return {
            "phrase_id": phrase_id,
            "duration_s": round(duration, 1),
            "pitch_score": int(round(pitch_score)) if pitch_score is not None else None,
            "breath_score": int(round(breath_score)) if breath_score is not None else None,
            "vib_score": int(round(vib_score)) if vib_score is not None else None,
            "phrase_score": phrase_score,
            "score_label": score_label,
            "score_breakdown": breakdown,
            "headline": headline,
            "detail": detail,
            "bullets": bullets[:3],
            "rating": rating,
        }


class PhraseTracker:
    """Stateful phrase boundaries from breath probability + voicing."""

    def __init__(self) -> None:
        self.phrase_id = 0
        self.phrase_start_t = 0.0
        self.in_phrase = False
        self.pending_phrase = False
        self.breath_run = 0
        self.silence_run = 0
        self.phrase_voiced_frames = 0
        self.last_breath_prob = 0.0
        self.metrics = LivePhraseMetrics()
        self.last_feedback: dict | None = None

    def _abort_pending(self) -> None:
        self.pending_phrase = False
        self.phrase_voiced_frames = 0
        self.breath_run = 0
        self.silence_run = 0
        self.metrics.reset()

    def _confirm_phrase(self) -> str:
        self.pending_phrase = False
        self.in_phrase = True
        self.phrase_id += 1
        self.phrase_start_t = self.metrics.start_t
        return "start"

    def update(
        self,
        breath_prob: float,
        voiced: bool,
        t: float,
        *,
        scored: bool = False,
        et_dev_cents: float = 0.0,
        gesture: int = 0,
        cpp: float = 0.0,
        f0_hz: float = 0.0,
        vib_rate_hz: float = float("nan"),
        vib_depth_cents: float = float("nan"),
        vib_consistency: float = 0.0,
    ) -> dict:
        boundary = None
        self.last_breath_prob = float(breath_prob)
        self._last_t = t

        if breath_prob >= BREATH_THRESHOLD:
            self.breath_run += 1
        else:
            self.breath_run = 0

        if voiced:
            self.silence_run = 0
            if not self.in_phrase:
                if not self.pending_phrase:
                    self.pending_phrase = True
                    self.phrase_voiced_frames = 0
                    self.metrics.reset()
                    self.metrics.start_t = t
                self.phrase_voiced_frames += 1
                self.metrics.add_frame(
                    t,
                    voiced=True,
                    scored=scored,
                    et_dev_cents=et_dev_cents,
                    gesture=gesture,
                    cpp=cpp,
                    f0_hz=f0_hz,
                )
                if (
                    self.pending_phrase
                    and self.phrase_voiced_frames >= MIN_PHRASE_FRAMES
                ):
                    boundary = self._confirm_phrase()
            else:
                self.phrase_voiced_frames += 1
                self.metrics.add_frame(
                    t,
                    voiced=True,
                    scored=scored,
                    et_dev_cents=et_dev_cents,
                    gesture=gesture,
                    cpp=cpp,
                    f0_hz=f0_hz,
                )
        else:
            self.silence_run += 1
            if self.pending_phrase and not self.in_phrase:
                if self.silence_run >= MIN_SILENCE_ABORT:
                    self._abort_pending()
            elif self.in_phrase and self.phrase_voiced_frames >= MIN_PHRASE_FRAMES:
                if (
                    self.breath_run >= MIN_BREATH_FRAMES
                    or self.silence_run >= MIN_SILENCE_FRAMES
                ):
                    completed = self.phrase_id
                    feedback = self.metrics.finish(
                        completed, vib_rate_hz, vib_depth_cents, vib_consistency,
                    )
                    self.last_feedback = feedback
                    self.in_phrase = False
                    self.pending_phrase = False
                    self.breath_run = 0
                    self.phrase_voiced_frames = 0
                    snap = self.snapshot("end", completed_id=completed)
                    snap["phrase_feedback"] = feedback
                    return snap

        snap = self.snapshot(boundary)
        if self.in_phrase:
            snap["phrase_feedback"] = None
        elif self.last_feedback:
            snap["phrase_feedback"] = self.last_feedback
        return snap

    def snapshot(self, boundary: str | None = None, completed_id: int | None = None) -> dict:
        if boundary == "end" and completed_id is not None:
            base = {
                "phrase_id": completed_id,
                "phrase_start_t": round(self.phrase_start_t, 2),
                "phrase_elapsed_s": round(max(0.0, self._last_t - self.phrase_start_t), 2),
                "phrase_boundary": "end",
                "breath_prob": round(self.last_breath_prob, 3),
                "in_phrase": False,
                "pending_phrase": False,
            }
            if self.last_feedback:
                base["phrase_feedback"] = self.last_feedback
            return base
        elapsed = (
            round(max(0.0, self._last_t - self.phrase_start_t), 2)
            if self.in_phrase
            else 0.0
        )
        return {
            "phrase_id": self.phrase_id if self.in_phrase else 0,
            "phrase_start_t": round(self.phrase_start_t, 2) if self.in_phrase else 0.0,
            "phrase_elapsed_s": elapsed,
            "phrase_boundary": boundary,
            "breath_prob": round(self.last_breath_prob, 3),
            "in_phrase": self.in_phrase,
            "pending_phrase": self.pending_phrase,
            "phrase_feedback": self.last_feedback if not self.in_phrase else None,
        }

    @property
    def last_t(self) -> float:
        return getattr(self, "_last_t", 0.0)
