from __future__ import annotations

from confetti.errors import CONFETTIConfigurationError

_THEMES = {
    "dark": {
        "bg": "#1a1d23",
        "panel": "#22262e",
        "grid": "#2e323b",
        "text": "#c8cdd5",
        "text_dim": "#7a8190",
        "original": "#02c4a1",
        "counterfactual": "#ff6b6b",
        "cam": "#ffb347",
        "channel_colors": ["#02c4a1", "#6c8dfa", "#c084fc", "#fb923c", "#f472b6", "#38bdf8"],
        "heatmap_cmap": "inferno",
    },
    "light": {
        "bg": "#ffffff",
        "panel": "#f8f9fb",
        "grid": "#e5e7eb",
        "text": "#1e293b",
        "text_dim": "#64748b",
        "original": "#02c4a1",
        "counterfactual": "#ef4444",
        "cam": "#d97706",
        "channel_colors": ["#059585", "#4f46e5", "#7c3aed", "#ea580c", "#db2777", "#0284c7"],
        "heatmap_cmap": "YlOrRd",
    },
}

VALID_THEMES = frozenset(_THEMES.keys())


def get_theme(theme: str) -> dict:
    """Return the color palette for the given theme name.

    Parameters
    ----------
    theme : str
        ``"light"`` or ``"dark"``.

    Returns
    -------
    dict
        Color palette mapping.

    Raises
    ------
    CONFETTIConfigurationError
        If ``theme`` is not a recognized value.
    """
    if theme not in _THEMES:
        raise CONFETTIConfigurationError(
            message=f"Invalid theme '{theme}'. Expected one of: {', '.join(sorted(VALID_THEMES))}.",
            param="theme",
            hint=f"Use one of: {', '.join(sorted(VALID_THEMES))}.",
        )
    return _THEMES[theme]
