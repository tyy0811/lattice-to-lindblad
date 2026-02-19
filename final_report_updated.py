#!/usr/bin/env python3
"""
Research Highlight PDF — 3-page deliverable
Page 1: Motivation + Approach
Page 2: Schwinger String-Breaking Centerpiece
Page 3: OQS Sequential Suppression + Continuum Mass Gap + Synthesis

Usage:
    python3 build_highlight.py

Output:
    ./research_highlight.pdf  (also copied to a writable outputs directory)

Dependencies:
    pip install reportlab

Image files expected in the same directory (extracted from the report PDFs):
    string_breaking_charge.png   — charge-density heatmap (3552x1008)
    excitation_echo.png          — Nex + Loschmidt echo   (3552x960)
    sequential_suppression.png   — 1S vs 2S suppression   (2199x1599)
    continuum_massgap.png        — mass-gap extrapolation  (2130x1409)
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor, white, black
from reportlab.pdfgen import canvas
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
import os, shutil

# =====================================================================
# COLOUR PALETTE  (edit these to restyle the whole document)
# =====================================================================
DARK_NAVY     = HexColor("#0d1b2a")
ACCENT_BLUE   = HexColor("#2196F3")
ACCENT_TEAL   = HexColor("#00BCD4")
ACCENT_ORANGE = HexColor("#FF9800")
TEXT_DARK      = HexColor("#1a1a2e")
TEXT_BODY      = HexColor("#2d2d44")
TEXT_LIGHT     = HexColor("#5a6070")
SUBTLE_LINE    = HexColor("#c0cad8")
TAG_BG         = HexColor("#e3f2fd")
TAG_BORDER     = HexColor("#90caf9")
BOX_BG         = HexColor("#f8fafc")

# =====================================================================
# PAGE GEOMETRY
# =====================================================================
W, H = A4                          # 595.27 x 841.89 pt
MARGIN_LEFT   = 38
MARGIN_RIGHT  = 38
MARGIN_TOP    = 36
MARGIN_BOTTOM = 32
CONTENT_W     = W - MARGIN_LEFT - MARGIN_RIGHT

# =====================================================================
# IMAGE PATHS  (change these if your PNGs live elsewhere)
# =====================================================================
IMG_DIR = "/Users/zenith/Desktop/Quantum_Simulation/figure"
IMG_CHARGE   = os.path.join(IMG_DIR, "gauge_string_breaking_row1_charge.png")
IMG_ECHO     = os.path.join(IMG_DIR, "gauge_string_breaking_row4_excitation_echo.png")
IMG_SEQ      = os.path.join(IMG_DIR, "sequential_suppression.png")
IMG_MASSGAP  = os.path.join(IMG_DIR, "dmrg_massgap_plot.png")


def resolve_export_dir():
    """Choose a writable export directory for copied artifacts."""
    env_dir = os.environ.get("REPORT_OUTPUT_DIR")
    if env_dir:
        return env_dir

    cloud_dir = "/mnt/user-data/outputs"
    cloud_parent = "/mnt/user-data"
    if os.path.isdir(cloud_parent) and os.access(cloud_parent, os.W_OK):
        return cloud_dir

    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# =====================================================================
# HELPER DRAWING FUNCTIONS
# =====================================================================

def draw_header_bar(c, y, height, text, subtitle=None):
    """Dark header bar spanning the content width.
    `y` is the BOTTOM of the bar; the bar extends upward by `height`."""
    c.setFillColor(DARK_NAVY)
    c.roundRect(MARGIN_LEFT - 4, y, CONTENT_W + 8, height, 4, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(MARGIN_LEFT + 10, y + height - 18, text)
    if subtitle:
        c.setFont("Helvetica", 7.5)
        c.setFillColor(HexColor("#90caf9"))
        c.drawString(MARGIN_LEFT + 10, y + 6, subtitle)


def draw_section_header(c, y, text, color=ACCENT_BLUE):
    """Coloured left-bar section header.  Returns y moved down past it."""
    c.setFillColor(color)
    c.roundRect(MARGIN_LEFT, y, 3, 14, 1, fill=1, stroke=0)
    c.setFillColor(TEXT_DARK)
    c.setFont("Helvetica-Bold", 10.5)
    c.drawString(MARGIN_LEFT + 10, y + 2, text)
    return y - 6                    # gap below header


def draw_body_text(c, x, y, text, width,
                   font="Helvetica", size=8.5, leading=12.5,
                   color=TEXT_BODY):
    """Draw wrapped body text using a Paragraph.  Returns new y."""
    style = ParagraphStyle(
        'body', fontName=font, fontSize=size, leading=leading,
        textColor=color, alignment=TA_JUSTIFY, spaceAfter=2,
    )
    p = Paragraph(text, style)
    _, ph = p.wrap(width, 9999)
    p.drawOn(c, x, y - ph)
    return y - ph


def draw_tag(c, x, y, text):
    """Small rounded keyword tag.  Returns the x position for the next tag."""
    c.setFont("Helvetica-Bold", 6.5)
    tw = c.stringWidth(text, "Helvetica-Bold", 6.5)
    c.setFillColor(TAG_BG)
    c.setStrokeColor(TAG_BORDER)
    c.setLineWidth(0.5)
    c.roundRect(x, y - 2, tw + 10, 13, 3, fill=1, stroke=1)
    c.setFillColor(ACCENT_BLUE)
    c.drawString(x + 5, y + 1, text)
    return x + tw + 14


def draw_footer(c, page_num):
    """Thin rule + author line + page number."""
    y = MARGIN_BOTTOM - 4
    c.setStrokeColor(SUBTLE_LINE); c.setLineWidth(0.4)
    c.line(MARGIN_LEFT, y + 10, W - MARGIN_RIGHT, y + 10)
    c.setFont("Helvetica", 6.5); c.setFillColor(TEXT_LIGHT)
    c.drawString(MARGIN_LEFT, y,
                 "J. Yeung \u2014 Research Highlight \u2014 February 2026")
    c.drawRightString(W - MARGIN_RIGHT, y, f"Page {page_num} of 3")


def draw_kpi_box(c, x, y, w, h, label, value, accent=ACCENT_BLUE):
    """Metric card: coloured accent bar on top, big number, small label."""
    c.setFillColor(BOX_BG)
    c.setStrokeColor(SUBTLE_LINE); c.setLineWidth(0.5)
    c.roundRect(x, y, w, h, 4, fill=1, stroke=1)
    c.setFillColor(accent)
    c.roundRect(x, y + h - 3, w, 3, 1, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 12); c.setFillColor(accent)
    c.drawCentredString(x + w / 2, y + h - 24, value)
    style = ParagraphStyle('kpi', fontName='Helvetica', fontSize=6.5,
                           leading=8, textColor=TEXT_LIGHT,
                           alignment=TA_CENTER)
    p = Paragraph(label, style)
    p.wrap(w - 8, 40)
    p.drawOn(c, x + 4, y + 4)


def draw_workstream_box(c, bx, y, col_w, box_h, title, accent, items):
    """Side-by-side info box used on page 1."""
    # background
    c.setFillColor(BOX_BG)
    c.setStrokeColor(accent); c.setLineWidth(0.7)
    c.roundRect(bx, y - box_h, col_w, box_h, 5, fill=1, stroke=1)
    # coloured title bar
    c.setFillColor(accent)
    c.roundRect(bx, y - 1, col_w, 20, 5, fill=1, stroke=0)
    c.setFillColor(white); c.setFont("Helvetica-Bold", 8.5)
    c.drawString(bx + 8, y + 3, title)
    # labelled paragraphs
    iy = y - 18
    for label, text in items:
        c.setFillColor(accent); c.setFont("Helvetica-Bold", 7.5)
        c.drawString(bx + 8, iy, label + ".")
        iy -= 3
        iy = draw_body_text(c, bx + 8, iy, text, col_w - 16,
                            size=7.5, leading=10.5)
        iy -= 5


# =====================================================================
# PAGE 1 — Motivation + Approach
# =====================================================================
def page1(c):
    c.saveState()
    y = H - MARGIN_TOP

    # ---- Title banner ----
    banner_h = 70
    c.setFillColor(DARK_NAVY)
    c.rect(0, y - banner_h + 10, W, banner_h + MARGIN_TOP - 10,
           fill=1, stroke=0)
    c.setFillColor(ACCENT_TEAL)
    c.rect(0, y - banner_h + 10, W, 2.5, fill=1, stroke=0)

    c.setFillColor(white); c.setFont("Helvetica-Bold", 17)
    c.drawString(MARGIN_LEFT, y - 14,
                 "Real-Time Quantum Simulation of Gauge Theories")
    c.drawString(MARGIN_LEFT, y - 34, "and Open Quantum Systems")
    c.setFont("Helvetica", 9); c.setFillColor(HexColor("#90caf9"))
    c.drawString(MARGIN_LEFT, y - 52,
        "Schwinger model string breaking  |  pNRQCD-motivated "
        "quarkonium suppression  |  Lattice + Lindblad framework")

    y -= banner_h -4

    # ---- Tags row ----
    y -= 20
    tx = MARGIN_LEFT
    for tag in ["LATTICE GAUGE THEORY", "OPEN QUANTUM SYSTEMS",
                "EXACT DIAGONALIZATION", "LINDBLAD DYNAMICS",
                "CONTINUUM LIMIT"]:
        tx = draw_tag(c, tx, y, tag)

    # ---- Motivation ----
    y -= 24
    y = draw_section_header(c, y, "Motivation")
    y -= 4
    y = draw_body_text(c, MARGIN_LEFT, y, (
        "Understanding the real-time dynamics of strongly-coupled "
        "quantum field theories is one of the central challenges in "
        "modern theoretical physics. From the string breaking that "
        "screens colour charges in QCD to the sequential suppression "
        "of quarkonium states in the quark-gluon plasma, these "
        "phenomena are inherently non-equilibrium and inaccessible to "
        "Euclidean lattice Monte Carlo. This project attacks both "
        "problems from complementary directions \u2014 unitary lattice "
        "simulation and dissipative open-system evolution \u2014 building "
        "a complete, validated computational portfolio aligned with the "
        "pNRQCD / lattice gauge theory research programme."
    ), CONTENT_W)

    # ---- Two Workstreams ----
    y -= 20
    y = draw_section_header(c, y, "Two Complementary Workstreams")
    y -= 20                        # extra breathing room before boxes

    col_w = (CONTENT_W - 12) / 2
    box_h = 172

    # Gauge simulation (left)
    draw_workstream_box(c, MARGIN_LEFT, y, col_w, box_h,
        "Gauge Simulation (Schwinger Model)", ACCENT_BLUE, [
        ("Physics",
         "1+1D QED (Schwinger model) with staggered fermions on a "
         "lattice. The gauge-eliminated spin-chain Hamiltonian "
         "preserves the full U(1) gauge structure while mapping onto "
         "qubits."),
        ("Method",
         "Exact diagonalisation in the charge-neutral sector (dim up to 3432);"
         " U(1) charge-sector symmetry projection (M=0) VQE with equivariant HVA "
         "and semi-local generator grouping (even/odd mass + spatial electric blocks, "
         "analytic adjoint gradients + warm-starting); first-order Suzuki–Trotter "
         "evolution validated against exact propagation."),
        ("Target",
         "Electric-field quench (E<sub>0</sub>=1 \u2192 0) at N=14 "
         "sites across two fermion-mass regimes, producing real-time "
         "charge-density and electric-field heatmaps that distinguish "
         "confinement from string breaking."),
    ])

    # Open quantum systems (right)
    draw_workstream_box(c, MARGIN_LEFT + col_w + 12, y, col_w, box_h,
        "Open Quantum Systems (pNRQCD)", ACCENT_ORANGE, [
        ("Physics",
         "pNRQCD-motivated singlet\u2013octet Lindblad model for "
         "heavy quarkonium in the quark-gluon plasma. The "
         "1\u2295 8 Hilbert space captures colour dissociation and "
         "recombination with detailed-balance Lindblad operators."),
        ("Method",
         "QuTiP master-equation solver (mesolve) calibrated to a "
         "total dissociation width at T<sub>ref</sub>=400 MeV. "
         "Validated against closed-form analytic solutions at "
         "ODE-solver precision (~10<super>\u22127</super>)."),
        ("Target",
         "Sequential suppression of 1S vs 2S quarkonium states, "
         "Bjorken longitudinal cooling with time-dependent Lindblad "
         "rates, and continuum-limit mass-gap extrapolation of the "
         "Schwinger model."),
    ])

    y -= box_h + 20

    # ---- Validation Chain ----
    y = draw_section_header(c, y, "Progressive Validation Chain",
                            color=ACCENT_TEAL)
    y -= 4
    y = draw_body_text(c, MARGIN_LEFT, y, (
        "Every production result is underwritten by a chain of "
        "verified baselines. The gauge workstream validates the "
        "Hamiltonian (Hermiticity, vacuum energy, static-limit "
        "spectrum), benchmarks the VQE against exact diagonalisation "
        "(error ~10<super>\u221211</super> at N=4, "
        "~9\u00d710<super>\u22124</super> at N=8), and confirms "
        "Trotter convergence with O(\u0394t) scaling before running "
        "the quench. The OQS workstream validates the Lindblad solver "
        "against closed-form analytic solutions "
        "(error ~10<super>\u22127</super>), verifies "
        "detailed-balance equilibrium across 20 temperatures, and "
        "confirms that the 9-level model reproduces the correct "
        "Boltzmann distribution with colour degeneracy before "
        "producing the sequential suppression and Bjorken-cooling "
        "figures."
    ), CONTENT_W, size=8, leading=11.5)

    # ---- KPI boxes ----
    y -= 14
    kpi_w = (CONTENT_W - 3 * 10) / 4
    kpi_h = 48
    for i, (label, value, acc) in enumerate([
        ("MC area law validated",    "0.2107", ACCENT_BLUE),
        ("VQE error (N=8, 6L)",     "9.2e-4", ACCENT_BLUE),
        ("Trotter convergence ratio","2.0",    ACCENT_TEAL),
        ("OQS solver error",        "~1e-7",  ACCENT_ORANGE),
    ]):
        draw_kpi_box(c, MARGIN_LEFT + i * (kpi_w + 10),
                     y - kpi_h, kpi_w, kpi_h, label, value, acc)

    draw_footer(c, 1)
    c.restoreState()


# =====================================================================
# PAGE 2 — Schwinger String-Breaking Centerpiece
# =====================================================================
def page2(c):
    c.saveState()
    y = H - MARGIN_TOP

    # ---- Header bar ----
    bar_h = 34
    draw_header_bar(c, y - bar_h, bar_h,
        "Schwinger Model: String Breaking via Electric-Field Quench",
        "Gauge-Simulation Workstream  |  N = 14 sites  "
        "|  Exact time evolution")
    y -= bar_h + 20

    # ---- Quench protocol ----
    y = draw_section_header(c, y, "Quench Protocol")
    y -= 10
    y = draw_body_text(c, MARGIN_LEFT, y, (
        "The gauge-eliminated staggered-fermion Schwinger Hamiltonian "
        "is constructed at N=14 sites (7 unit cells, 13 internal "
        "links) in the global-charge-neutral sector (Hilbert space "
        "dimension 3432). A uniform electric string is prepared in "
        "the ground state of H(E<sub>0</sub>=1), then at t=0 the "
        "system evolves under H(E<sub>0</sub>=0) via exact sparse "
        "matrix exponentiation (no Trotter error). Two fermion-mass "
        "regimes are compared: <b>heavy</b> (m/g=2.5, true ground "
        "state of H(E<sub>0</sub>=1)) where pair creation is "
        "exponentially suppressed, and <b>light</b> (m/g=0.1, "
        "constrained string prep) where copious Schwinger pair "
        "production screens the initial string."
    ), CONTENT_W, size=8, leading=11)

    # ---- Charge-density heatmap ----
    y -= 20
    y = draw_section_header(c, y, "Charge-Density Heatmaps (Figure 3a)")
    y -= 6
    img_w = CONTENT_W
    img_h = img_w * 1008 / 3552
    c.drawImage(IMG_CHARGE, MARGIN_LEFT, y - img_h,
                img_w, img_h, preserveAspectRatio=True)
    y -= img_h + 4
    y = draw_body_text(c, MARGIN_LEFT, y, (
        "<b>Figure 1.</b> Space\u2013time heatmaps of unit-cell "
        "charge density after the E<sub>0</sub>=1\u21920 quench. "
        "<b>Left:</b> Heavy regime (m/g=2.5) \u2014 only "
        "small-amplitude boundary oscillations (~0.03); the bulk "
        "remains charge-neutral, confirming the electric string is "
        "metastable. <b>Right:</b> Light regime (m/g=0.1) \u2014 "
        "large-amplitude charge structures (~0.2) propagate inward "
        "along an approximate Lieb\u2013Robinson light cone, "
        "progressively screening the string. By t~5/g the charge "
        "distribution spans the full chain, signaling complete "
        "string breaking."
    ), CONTENT_W, font="Helvetica", size=7.5, leading=10,
       color=TEXT_LIGHT)

    # ---- Scalar diagnostics ----
    y -= 20
    y = draw_section_header(c, y, "Scalar Diagnostics")
    y -= 6
    img_w2 = CONTENT_W
    img_h2 = img_w2 * 960 / 3552
    c.drawImage(IMG_ECHO, MARGIN_LEFT, y - img_h2,
                img_w2, img_h2, preserveAspectRatio=True)
    y -= img_h2 + 4
    y = draw_body_text(c, MARGIN_LEFT, y, (
        "<b>Figure 2.</b> <b>Left:</b> Excitation proxy "
        "N<sub>ex</sub>(t) counting particle\u2013hole pairs above "
        "the post-quench vacuum. The heavy regime maintains "
        "N<sub>ex</sub>\u22480.5 (residual fluctuations), while the "
        "light regime rapidly generates ~5 excitations. "
        "<b>Right:</b> Loschmidt echo. The heavy regime stays near "
        "unity with periodic finite-size recurrences (period ~7/g), "
        "while the light regime drops to ~0.05 within t~1/g, "
        "confirming irreversible departure from the string manifold."
    ), CONTENT_W, font="Helvetica", size=7.5, leading=10,
       color=TEXT_LIGHT)

    # ---- Physics interpretation ----
    y -= 20
    y = draw_section_header(c, y, "Physics Interpretation")
    y -= 4
    y = draw_body_text(c, MARGIN_LEFT, y, (
        "The qualitative contrast between the two regimes \u2014 "
        "static confinement vs. dynamic string breaking \u2014 is "
        "the central result of the gauge-simulation workstream. At "
        "m/g=2.5 the Schwinger pair-creation rate scales as "
        "exp(\u2212\u03c0m<super>2</super>/g<super>2</super>) "
        "\u2248 3\u00d710<super>\u22129</super>, so real pair "
        "production is exponentially suppressed and the string is "
        "metastable. At m/g=0.1 the barrier is negligible and "
        "copious pair creation screens the initial string within a "
        "few units of 1/g. The light-cone propagation pattern, "
        "wavefront collision, and late-time interference are all "
        "consistent with the expected physics of the massive "
        "Schwinger model on a finite lattice."
    ), CONTENT_W, size=8, leading=11)

    # ---- Energy excess KPIs ----
    y -= 14
    kpi_w = (CONTENT_W - 2 * 10) / 3
    kpi_h = 44
    for i, (label, value, acc) in enumerate([
        ("Heavy: excess / bandwidth", "0.1%",  ACCENT_BLUE),
        ("Light: excess / bandwidth", "2.4%",  ACCENT_ORANGE),
        ("Light: Loschmidt echo",     "~0.05", ACCENT_TEAL),
    ]):
        draw_kpi_box(c, MARGIN_LEFT + i * (kpi_w + 10),
                     y - kpi_h, kpi_w, kpi_h, label, value, acc)

    draw_footer(c, 2)
    c.restoreState()


# =====================================================================
# PAGE 3 — OQS Sequential Suppression + Continuum + Synthesis
# =====================================================================
def page3(c):
    c.saveState()
    y = H - MARGIN_TOP

    # ---- Header bar ----
    bar_h = 34
    draw_header_bar(c, y - bar_h, bar_h,
        "Open Quantum Systems: Sequential Suppression "
        "& Continuum Validation",
        "pNRQCD Lindblad Workstream  |  1\u22958 colour structure  "
        "|  Analytic benchmarks")
    y -= bar_h + 20

    # ---- Sequential suppression ----
    y = draw_section_header(c, y,
        "Sequential Quarkonium Suppression at T = 300 MeV")
    y -= 6

    fig_w  = CONTENT_W * 0.52
    desc_w = CONTENT_W * 0.44
    fig_h  = fig_w * 1599 / 2199

    c.drawImage(IMG_SEQ, MARGIN_LEFT, y - fig_h,
                fig_w, fig_h, preserveAspectRatio=True)

    # right-column description
    rx = MARGIN_LEFT + fig_w + 12
    ry = y - 2
    ry = draw_body_text(c, rx, ry, (
        "The 1\u22958 singlet\u2013octet Lindblad model with 16 "
        "collapse operators (8 dissociation + 8 recombination "
        "channels) evolves from a pure singlet initial state at "
        "fixed T=300 MeV. A state-independent per-channel rate "
        "\u03b3<sub>0</sub>=31.1 MeV is calibrated to a total "
        "dissociation width of 100 MeV at T<sub>ref</sub>=400 MeV "
        "for the 1S state."
    ), desc_w, size=7.5, leading=10.5)
    ry -= 6
    ry = draw_body_text(c, rx, ry, (
        "<b>Key results.</b> The loosely-bound 2S state "
        "(\u0394E=200 MeV) drops below 50% survival by t~1 fm/c, "
        "while the tightly-bound 1S (\u0394E=500 MeV) does not "
        "cross 50% until t~3.5 fm/c \u2014 a 3.5\u00d7 difference "
        "in half-life. Both curves approach their analytic Boltzmann "
        "equilibria: P<sub>eq</sub>(1S)=0.40, "
        "P<sub>eq</sub>(2S)=0.20. The double ratio "
        "P<super>2S</super><sub>s</sub>/"
        "P<super>1S</super><sub>s</sub> = 0.49 at "
        "\u03c4<sub>QGP</sub> provides a direct connection to the "
        "experimentally measured R<sub>AA</sub> hierarchy."
    ), desc_w, size=7.5, leading=10.5)
    ry -= 10

    kw = (desc_w - 8) / 2;  kh = 38
    draw_kpi_box(c, rx,          ry - kh, kw, kh,
                 "1S survival at QGP end", "0.40", ACCENT_BLUE)
    draw_kpi_box(c, rx + kw + 8, ry - kh, kw, kh,
                 "2S/1S double ratio",     "0.49", ACCENT_ORANGE)

    y -= fig_h + 14

    # ---- Continuum mass gap ----
    y = draw_section_header(c, y,
        "Continuum Mass-Gap Extrapolation (Schwinger Model)")
    y -= 6

    fig2_w  = CONTENT_W * 0.52
    desc2_w = CONTENT_W * 0.44
    fig2_h  = fig2_w * 887 / 2048  # aspect from dmrg_massgap_plot.png

    c.drawImage(IMG_MASSGAP, MARGIN_LEFT, y - fig2_h,
                fig2_w, fig2_h, preserveAspectRatio=True)

    rx2 = MARGIN_LEFT + fig2_w + 12
    ry2 = y - 2
    ry2 = draw_body_text(c, rx2, ry2, (
        "The massless Schwinger-model gap is computed as "
        "M<sub>gap</sub>/g = ΔE/(2√x) using exact diagonalisation "
        "for N=4–20 and extended to N=30,40 with TeNPy DMRG "
        "(χ=100, conserve=\"Sz\"). "
        "DMRG reproduces ED energies and gaps for N≤20 at "
        "∼10<super>−13</super> relative precision "
        "(see dmrg_massgap_results.csv)."
    ), desc2_w, size=7.5, leading=10.5)
    ry2 -= 6
    ry2 = draw_body_text(c, rx2, ry2, (
        "Because the electric-field term is long-ranged, a naive "
        "O(N<super>2</super>) coupling construction inflates the MPO bond "
        "dimension ∼O(N) and dominates sweep time. The DMRG "
        "implementation instead encodes the electric term via its "
        "running-sum form as a compact finite-state-machine MPO with "
        "constant bond dimension (≈5 including hopping), enabling stable "
        "N=40 runs. In the figure, the N=30,40 points tighten the 1/N trend "
        "and move visibly toward the exact Schwinger value "
        "1/√π ≈ 0.564, while the largest-N curves are nearly "
        "flat versus (ag)<super>2</super>=1/x, indicating small residual "
        "finite-size error at N=40."
    ), desc2_w, size=7.5, leading=10.5)
    ry2 -= 10

    kw2 = (desc2_w - 8) / 2;  kh2 = 38
    draw_kpi_box(c, rx2,           ry2 - kh2, kw2, kh2,
                 "Largest system (DMRG)", "N=40", ACCENT_TEAL)
    draw_kpi_box(c, rx2 + kw2 + 8, ry2 - kh2, kw2, kh2,
                 "Exact 1/√π",            "0.564", ACCENT_BLUE)

    y -= fig2_h + 14

    # ---- Synthesis ----
    y = draw_section_header(c, y, "Synthesis and Outlook",
                            color=ACCENT_TEAL)
    y -= 4
    y = draw_body_text(c, MARGIN_LEFT, y, (
        "This project delivers a complete, reproducible portfolio "
        "spanning two complementary approaches to real-time "
        "gauge-theory dynamics. The <b>gauge-simulation track</b> "
        "progresses from Hamiltonian validation through "
        "symmetry-preserving VQE to production-quality "
        "string-breaking dynamics with quantitative scalar "
        "diagnostics. The <b>open-quantum-systems track</b> builds "
        "from analytic solver validation through 9-level "
        "colour-degenerate dynamics to sequential suppression and "
        "Bjorken cooling, reproducing the phenomenologically "
        "observed R<sub>AA</sub> hierarchy. The continuum mass-gap "
        "extrapolation bridges both tracks by demonstrating that the "
        "lattice Hamiltonian pipeline recovers exact continuum QFT "
        "predictions. Together, these results constitute a "
        "self-contained research portfolio directly aligned with the "
        "pNRQCD and lattice gauge theory programme."
    ), CONTENT_W, size=8, leading=11.5)

    draw_footer(c, 3)
    c.restoreState()


# =====================================================================
# BUILD
# =====================================================================
def build():
    out = os.path.join(IMG_DIR, "research_highlight.pdf")
    c = canvas.Canvas(out, pagesize=A4)
    c.setTitle("Research Highlight \u2014 Real-Time Quantum Simulation")
    c.setAuthor("J. [Author]")

    page1(c);  c.showPage()
    page2(c);  c.showPage()
    page3(c);  c.save()

    # Copy to outputs for download
    export_dir = resolve_export_dir()
    os.makedirs(export_dir, exist_ok=True)
    export_out = os.path.join(export_dir, "research_highlight.pdf")
    shutil.copy(out, export_out)
    print(f"Done: {out}")
    print(f"Copied: {export_out}")


if __name__ == "__main__":
    build()
