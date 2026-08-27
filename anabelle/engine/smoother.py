"""Temporal hysteresis so avatar emotion does not flicker every chunk."""

from __future__ import annotations

from collections import Counter, deque


class EmotionSmoother:
    """Require a label to appear in a sliding window before switching."""

    def __init__(self, window: int = 3) -> None:
        self.window = max(1, window)
        self._history: deque[str] = deque(maxlen=self.window)
        self._current = "NEUTRAL"

    @property
    def current(self) -> str:
        return self._current

    def update(self, emotion: str) -> str:
        self._history.append(emotion)
        counts = Counter(self._history)
        top_emotion, top_count = counts.most_common(1)[0]
        # Switch only when the new label dominates the recent window.
        if top_count >= (self.window + 1) // 2:
            self._current = top_emotion
        return self._current
