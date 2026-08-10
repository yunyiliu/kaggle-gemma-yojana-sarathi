"""Figures for the writeup card and media gallery.

Deliberately plain: one claim per figure, no decoration that is not carrying information.
Run with the system python (matplotlib is not a project dependency - the tool itself does
not draw anything, and adding a plotting library to requirements.txt for two PNGs would be
a lie about what the project needs).

    python3 scripts_figures.py
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch

# The default sans stack has no Devanagari, and the project name is in Devanagari.
_DEVA = pathlib.Path("/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc")
DEVA = FontProperties(fname=str(_DEVA)) if _DEVA.exists() else None

OUT = pathlib.Path(__file__).resolve().parent / "assets"
OUT.mkdir(exist_ok=True)

INK = "#14110f"
MUTED = "#8a8078"
ACCENT = "#c1440e"        # the shipped strategy
GRID = "#e4ded7"
PAPER = "#faf7f3"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "figure.facecolor": PAPER, "axes.facecolor": PAPER,
})

K = list(range(0, 9))
CURVES = {
    "benefit  (ships)": ([0, 52, 52, 52, 52, 59, 68, 69, 70], ACCENT, 2.6, "-"),
    "coverage (ablation)": ([0, 0, 52, 52, 52, 52, 64, 69, 69], "#3d5a80", 1.8, "-"),
    "fixed    (a paper form)": ([0, 0, 0, 0, 52, 62, 62, 62, 64], "#6b705c", 1.6, "--"),
    "random": ([0, 4, 7, 10, 13, 15, 19, 22, 27], MUTED, 1.4, ":"),
}


def fig_curve() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=200)
    for label, (ys, colour, lw, ls) in CURVES.items():
        ax.plot(K, ys, color=colour, lw=lw, ls=ls, label=label,
                marker="o" if colour == ACCENT else None, ms=4.5, zorder=3)

    ax.annotate("52% of what she is owed,\nafter ONE question",
                xy=(1, 52), xytext=(2.15, 30), color=ACCENT, fontsize=11.5, weight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.6,
                                connectionstyle="arc3,rad=-0.25"), zorder=5)
    ax.annotate("a paper form, asking the same fields\nin form order, has established"
                "\nnothing until question four",
                xy=(2.6, 0), xytext=(3.3, 16), color="#6b705c", fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#6b705c", lw=1.2,
                                connectionstyle="arc3,rad=0.2"), zorder=5)

    ax.set_xlim(0, 8.3); ax.set_ylim(-2, 78)
    ax.set_xticks(K)
    ax.set_yticks([0, 25, 50, 75]); ax.set_yticklabels(["0", "25%", "50%", "75%"])
    ax.set_xlabel("questions asked", fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Interviews get abandoned. What have you secured when it stops?",
                 fontsize=14.5, weight="bold", loc="left", pad=16)
    ax.text(0, 1.015, "fraction of a household's true entitlement established, "
                      "mean over 8 households · no model involved",
            transform=ax.transAxes, fontsize=10, color=MUTED)
    leg = ax.legend(frameon=False, fontsize=10.5, loc="lower right")
    for t in leg.get_texts():
        t.set_family("monospace")
    fig.tight_layout()
    fig.savefig(OUT / "interview_curve.png", facecolor=PAPER)
    plt.close(fig)


def fig_card() -> None:
    """Card / thumbnail. Kaggle crops these, so nothing critical near the edges."""
    fig = plt.figure(figsize=(12, 6.75), dpi=200)
    fig.patch.set_facecolor("#14110f")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")

    ax.text(0.06, 0.80, "योजना सारथी", fontsize=32, color="#f0ebe4",
            fontproperties=DEVA)
    ax.text(0.06, 0.665, "Yojana Sarathi", fontsize=17, color="#c9bfb4",
            family="monospace")

    ax.text(0.06, 0.50, "The scheme is not the hard part.", fontsize=31,
            color="#f0ebe4", weight="bold")
    ax.text(0.06, 0.375, "The interview is.", fontsize=31, color=ACCENT, weight="bold")

    ax.text(0.06, 0.235,
            "An offline welfare-entitlement navigator on Gemma 4 E4B.",
            fontsize=13.5, color="#c9bfb4")
    ax.text(0.06, 0.155,
            "52% of a household's entitlement established after one question.",
            fontsize=13.5, color="#c9bfb4")

    for i, (x, label) in enumerate([(0.06, "100% recall"), (0.245, "0% invention"),
                                    (0.44, "49 tests"), (0.585, "no cloud")]):
        box = FancyBboxPatch((x, 0.045), 0.155 if i < 2 else 0.115, 0.062,
                             boxstyle="round,pad=0.008", linewidth=1,
                             edgecolor="#4a4038", facecolor="#1e1a17", zorder=2)
        ax.add_patch(box)
        ax.text(x + (0.0775 if i < 2 else 0.0575), 0.076, label, fontsize=11.5,
                color="#c9bfb4", ha="center", va="center", family="monospace", zorder=3)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.savefig(OUT / "card.png", facecolor="#14110f")
    plt.close(fig)


def fig_models() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=200)
    rows = [
        ("Gemma 4 E2B   before the prompt fix", 7.1, ACCENT),
        ("Gemma 4 E2B   after it", 0.0, "#3d5a80"),
        ("Gemma 4 E4B   ships", 0.0, "#3d5a80"),
    ]
    ys = range(len(rows))
    ax.barh(list(ys), [r[1] for r in rows], color=[r[2] for r in rows],
            height=0.5, zorder=3)
    for y, (label, v, _c) in zip(ys, rows):
        ax.text(0.12, y, f"  {v:.1f}%" if v else "  0%", va="center", fontsize=11.5,
                color=INK if v else MUTED, weight="bold" if v else "normal")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], family="monospace", fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 9); ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_xticklabels(["0", "2%", "4%", "6%", "8%"])
    ax.set_xlabel("invention rate — fields filled in that nobody stated", fontsize=11)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title("The smaller model leaves less slack for a defective prompt",
                 fontsize=14, weight="bold", loc="left", pad=14)
    ax.text(0, 1.03, 'recall was 100% in all three. the rule E2B ignored was buried at the '
                     'tail of an unrelated bullet — E4B applied it anyway',
            transform=ax.transAxes, fontsize=9.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "model_choice.png", facecolor=PAPER)
    plt.close(fig)


def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.0), dpi=200)
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    stages = [
        (0.005, 0.225, "she speaks", "Hindi · Hinglish · English\nbighas · lakhs",
         "#1e1a17", "#c9bfb4", True),
        (0.255, 0.19, "Gemma 4 E4B", "speech into 21 declared fields\n100% recall / 0% invention",
         ACCENT, "#ffffff", False),
        (0.475, 0.22, "engine.py + schemes.yaml", "eligible / ineligible / UNCERTAIN\n"
         "no model on this line", "#3d5a80", "#ffffff", False),
        (0.725, 0.27, "Gemma 4 E4B", "message · checklist · audit trail\nin her language",
         ACCENT, "#ffffff", False),
    ]
    for x, w, title, sub, fc, tc, dark in stages:
        ax.add_patch(FancyBboxPatch((x, 0.30), w, 0.42, boxstyle="round,pad=0.012",
                                    facecolor=fc, edgecolor="none", zorder=2))
        ax.text(x + w / 2, 0.60, title, ha="center", fontsize=12, weight="bold",
                color=tc, zorder=3)
        ax.text(x + w / 2, 0.435, sub, ha="center", fontsize=9, color=tc, zorder=3,
                linespacing=1.5)
    for x in (0.238, 0.452, 0.702):
        ax.annotate("", xy=(x + 0.017, 0.51), xytext=(x, 0.51),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6))

    # the loop: the engine names the next question, and it goes back to her
    ax.annotate("", xy=(0.115, 0.285), xytext=(0.585, 0.285),
                arrowprops=dict(arrowstyle="-|>", color="#3d5a80", lw=1.5,
                                connectionstyle="arc3,rad=0.42"))
    ax.text(0.35, 0.035, "what to ask next — the one field worth the most money",
            ha="center", fontsize=10.5, color="#3d5a80", style="italic")

    ax.text(0, 0.90, "Gemma fills in the form and writes the answer up.\n"
                     "It never decides who qualifies.",
            fontsize=12.5, color=INK, weight="bold", linespacing=1.45)
    fig.tight_layout()
    fig.savefig(OUT / "pipeline.png", facecolor=PAPER)
    plt.close(fig)


if __name__ == "__main__":
    fig_card()
    fig_curve()
    fig_models()
    fig_pipeline()
    for f in sorted(OUT.glob("*.png")):
        print(f"{f.name:24s} {f.stat().st_size/1024:6.0f} KB")
