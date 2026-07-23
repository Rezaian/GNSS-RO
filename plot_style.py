"""
Plot Style Module
=================
Central matplotlib font sizing for every PNG the pipelines emit.

v3.4.7 — plot fonts were reviewed on the compiled Windows build running on a
1366x768 (non-Full-HD) laptop.  The panels are rendered at 12x10 in / 150 dpi
and then scaled to fit the plot pane, so anything drawn at matplotlib's
defaults ended up too small to read.

Two knobs control everything:

    PLOT_BODY_BUMP   +2 px  -> tick labels, axis labels, legends, annotations
    PLOT_TITLE_BUMP  +4 px  -> subplot titles and figure suptitles

Set both to 0 to restore the exact v3.4.6 appearance.

`apply_plot_fonts()` is idempotent and cheap; each plotting entry point calls
it before building its figure so the settings survive the `spawn`-based
multiprocessing used by the GUI (rcParams are per-process).
"""

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

PLOT_BODY_BUMP = 2
PLOT_TITLE_BUMP = 4

# matplotlib defaults these are measured against
_BASE_BODY = 10
_BASE_AXES_TITLE = 12
_BASE_FIG_TITLE = 12


def body(base=_BASE_BODY):
    """Bumped size for ordinary plot text."""
    return base + PLOT_BODY_BUMP


def title(base=_BASE_AXES_TITLE):
    """Bumped size for titles and suptitles."""
    return base + PLOT_TITLE_BUMP


def apply_plot_fonts():
    """Push the bumped sizes into rcParams for the current process.

    Covers everything drawn WITHOUT an explicit ``fontsize=`` argument —
    principally the tick labels and the axis labels, which had no explicit
    size in v3.4.6 and were therefore stuck at matplotlib's default 10.
    """
    try:
        import matplotlib
    except ImportError:
        return

    matplotlib.rcParams.update({
        'font.size':        body(_BASE_BODY),
        'axes.labelsize':   body(_BASE_BODY),
        'xtick.labelsize':  body(_BASE_BODY),
        'ytick.labelsize':  body(_BASE_BODY),
        'legend.fontsize':  body(_BASE_BODY),
        'axes.titlesize':   title(_BASE_AXES_TITLE),
        'figure.titlesize': title(_BASE_FIG_TITLE),
    })
