"""Eyewear frame recommendation rules derived from predicted face shape.

Overview
--------
After the ML classifier predicts one of five face-shape categories, this
module converts that prediction into a short list of eyewear frame styles.

The mapping is based on widely cited optician guidelines:

* **Round / oval faces** benefit from angular frames that add definition.
* **Square / heart faces** benefit from softer, curved frames that offset
  a strong jawline or wide forehead.
* **Oblong faces** benefit from wider frames that add visual breadth.

Usage
-----
Import :func:`get_recommendation` and pass the predicted face-shape label::

    from src.visionaid.recommendation import get_recommendation

    rec = get_recommendation("round")
    print(rec.recommended_frames)   # ('rectangular', 'square', 'cat-eye')
    print(rec.rationale)

The returned :class:`Recommendation` object is a frozen dataclass, so all
attributes are read-only.

Supported face-shape labels
----------------------------
``"heart"``, ``"oblong"``, ``"oval"``, ``"round"``, ``"square"``

These labels correspond exactly to the five classes in the faceshape dataset
and to the ``label`` column in ``data/processed/landmark_features.csv``.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Recommendation:
    """A bundle of eyewear frame recommendations for one face shape.

    Attributes
    ----------
    face_shape:
        The predicted face-shape label this recommendation was built for
        (e.g. ``"round"``, ``"square"``).
    recommended_frames:
        An ordered tuple of up to three eyewear frame style names.
        The first item is the primary recommendation.
    rationale:
        A human-readable sentence explaining why these frames were chosen
        for this face shape.  Suitable for display in a UI or report.

    Notes
    -----
    The dataclass is frozen (immutable) so that the recommendation catalogue
    ``RECOMMENDATIONS`` can be safely treated as a module-level constant.
    """

    face_shape: str
    recommended_frames: tuple[str, ...]
    rationale: str


# ---------------------------------------------------------------------------
# Recommendation catalogue
# ---------------------------------------------------------------------------

# Each entry maps a face-shape label to the three most suitable frame styles.
# The rationale string is written to be user-facing — clear, concise, and
# suitable for inclusion in a presentation or product UI.

RECOMMENDATIONS: dict[str, Recommendation] = {
    # Heart-shaped faces are widest at the forehead and taper to a narrow chin.
    # Frames that draw attention downward (round, aviator, cat-eye) balance
    # the wider upper face.
    "heart": Recommendation(
        face_shape="heart",
        recommended_frames=("round", "aviator", "cat-eye"),
        rationale=(
            "Heart-shaped faces are widest at the forehead and narrow at the chin. "
            "Softer or bottom-heavy frames such as round, aviator, and cat-eye styles "
            "draw attention downward and create a more balanced appearance."
        ),
    ),
    # Oblong faces are significantly longer than they are wide.
    # Frames that add width and visual depth — square, aviator, cat-eye —
    # help make the face appear shorter and fuller.
    "oblong": Recommendation(
        face_shape="oblong",
        recommended_frames=("square", "aviator", "cat-eye"),
        rationale=(
            "Oblong faces are longer than they are wide. Frames with strong horizontal "
            "lines such as square and aviator styles add perceived width and help "
            "shorten the visual length of the face."
        ),
    ),
    # Oval faces are considered the most versatile because their proportions
    # are well balanced.  Structured frames work well and preserve the balance.
    "oval": Recommendation(
        face_shape="oval",
        recommended_frames=("rectangular", "square", "aviator"),
        rationale=(
            "Oval faces have balanced proportions and work well with most frame styles. "
            "Structured rectangular and square frames maintain the natural symmetry "
            "without adding or subtracting from facial width."
        ),
    ),
    # Round faces have similar width and height with soft, curved contours.
    # Angular frames add contrast and make the face appear slimmer.
    "round": Recommendation(
        face_shape="round",
        recommended_frames=("rectangular", "square", "cat-eye"),
        rationale=(
            "Round faces benefit from angular frames that add definition and contrast. "
            "Rectangular and square styles elongate the face visually, while "
            "cat-eye frames lift the upper half."
        ),
    ),
    # Square faces have a strong, wide jawline and broad forehead.
    # Softer, rounder frames offset the angular structure.
    "square": Recommendation(
        face_shape="square",
        recommended_frames=("round", "aviator", "cat-eye"),
        rationale=(
            "Square faces have a pronounced jawline and broad forehead. "
            "Rounded and curved frames such as oval, aviator, and cat-eye styles "
            "soften the strong angular features."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_recommendation(face_shape: str) -> Recommendation:
    """Return the eyewear frame recommendation bundle for a predicted face shape.

    Parameters
    ----------
    face_shape:
        One of the five face-shape labels produced by the classifier:
        ``"heart"``, ``"oblong"``, ``"oval"``, ``"round"``, or ``"square"``.

    Returns
    -------
    Recommendation
        A frozen dataclass with ``face_shape``, ``recommended_frames``, and
        ``rationale`` attributes.

    Raises
    ------
    KeyError
        If ``face_shape`` is not one of the five supported labels.

    Examples
    --------
    >>> from src.visionaid.recommendation import get_recommendation
    >>> rec = get_recommendation("oval")
    >>> rec.recommended_frames
    ('rectangular', 'square', 'aviator')
    >>> print(rec.rationale)
    Oval faces have balanced proportions and work well with most frame styles. ...
    """
    # Look up the pre-built recommendation for this face shape.
    # Raises KeyError automatically if the label is not in the catalogue,
    # which is the desired behaviour — the caller should validate the label
    # before calling this function.
    return RECOMMENDATIONS[face_shape]


def list_supported_shapes() -> list[str]:
    """Return the sorted list of face-shape labels with a recommendation entry.

    Useful for validation before calling :func:`get_recommendation`.

    Returns
    -------
    list[str]
        Alphabetically sorted face-shape names.

    Examples
    --------
    >>> from src.visionaid.recommendation import list_supported_shapes
    >>> list_supported_shapes()
    ['heart', 'oblong', 'oval', 'round', 'square']
    """
    return sorted(RECOMMENDATIONS.keys())
