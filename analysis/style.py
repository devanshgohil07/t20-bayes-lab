"""One house style for every figure: light ground, one accent, four league colours."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID, PAPER = "#14181f", "#6b7280", "#e6e8ec", "#ffffff"
ACCENT = "#b3122b"
LEAGUE = {"ipl": "#2f5d8c", "cpl": "#c1663e", "bbl": "#4f8a5b", "t20i": "#8a6bab"}
LCOL = [LEAGUE["ipl"], LEAGUE["cpl"], LEAGUE["bbl"], LEAGUE["t20i"]]
LNAME = ["IPL", "CPL", "BBL", "T20I"]

plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.axisbelow": True, "legend.frameon": False,
    "figure.dpi": 130, "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
})

def save(fig, name, figs_dir):
    import os
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figs_dir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote figs/{name}.pdf / .png")

def figtitle(fig, t, sub=None, top=0.86, x=0.008):
    """Figure-level title and subtitle, left aligned. Used for multi-panel figures so
    that a long title can never collide with an axes title."""
    fig.subplots_adjust(top=top)
    fig.text(x, 0.995, t, fontsize=11.5, fontweight="bold", color=INK,
             ha="left", va="top")
    if sub:
        fig.text(x, 0.945, sub, fontsize=8.8, color=MUTED, ha="left", va="top")


def title(ax, t, sub=None):
    y = 1.075 if sub else 1.02
    ax.text(0, y, t, transform=ax.transAxes, fontsize=11.5, fontweight="bold",
            color=INK, va="bottom")
    if sub:
        ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=8.8, color=MUTED,
                va="bottom")
