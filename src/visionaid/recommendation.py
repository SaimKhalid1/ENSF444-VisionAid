"""Eyewear recommendation rules derived from predicted face shape."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Recommendation:
    face_shape: str
    recommended_frames: tuple[str, ...]
    rationale: str


RECOMMENDATIONS: dict[str, Recommendation] = {
    "heart": Recommendation(
        face_shape="heart",
        recommended_frames=("round", "aviator", "cat-eye"),
        rationale="Heart-shaped faces often balance well with softer or bottom-heavy frames.",
    ),
    "oblong": Recommendation(
        face_shape="oblong",
        recommended_frames=("square", "aviator", "cat-eye"),
        rationale="Longer faces usually benefit from frames that add width and visual depth.",
    ),
    "oval": Recommendation(
        face_shape="oval",
        recommended_frames=("rectangular", "square", "aviator"),
        rationale="Oval faces are versatile, so structured frames usually preserve facial balance.",
    ),
    "round": Recommendation(
        face_shape="round",
        recommended_frames=("rectangular", "square", "cat-eye"),
        rationale="Angular frames usually add contrast to rounder facial contours.",
    ),
    "square": Recommendation(
        face_shape="square",
        recommended_frames=("round", "aviator", "cat-eye"),
        rationale="Softer curves can offset a stronger jawline and broad forehead.",
    ),
}


def get_recommendation(face_shape: str) -> Recommendation:
    """Return the frame recommendation bundle for a predicted face shape."""
    return RECOMMENDATIONS[face_shape]

