import os
import re
import datetime
from fpdf import FPDF


# ── Unicode-safe character replacement ──────────────────────────────────────
def _safe(text: str) -> str:
    """Replace Unicode chars unsupported by Helvetica with ASCII equivalents."""
    if not text:
        return ""
    replacements = {
        "\u2022": "-",   "\u2023": "-",   "\u25cf": "-",   # bullets
        "\u2013": "-",   "\u2014": "-",                     # dashes
        "\u2026": "...",                                     # ellipsis
        "\u2018": "'",   "\u2019": "'",                     # single quotes
        "\u201c": '"',   "\u201d": '"',                     # double quotes
        "\u00a0": " ",                                       # non-breaking space
        "\u00b7": ".",   "\u00b0": " deg",                  # middle dot, degree
        "\u00ae": "(R)", "\u00a9": "(c)", "\u2122": "(TM)", # symbols
        "\u2192": "->",  "\u2190": "<-",  "\u2194": "<->", # arrows
        "\u2265": ">=",  "\u2264": "<=",  "\u2260": "!=",  # math
        "\u00d7": "x",   "\u00f7": "/",   "\u00b1": "+/-", # operators
        "\u2019": "'",   "\u201a": ",",                     # misc quotes
        "|": "|",        # keep pipe as-is — it IS latin-1 safe
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    # Final fallback for anything remaining
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _slugify(text: str, max_len: int = 50) -> str:
    """Convert query text to a safe filename string."""
    if not text:
        return "insighthub_report"
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)       # remove special chars
    slug = re.sub(r"[\s_-]+", "_", slug)        # spaces to underscores
    slug = slug[:max_len].rstrip("_")           # truncate
    return slug or "insighthub_report"


class InsightPDF(FPDF):
    def __init__(self, query: str = "", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query = _safe(query)

    def header(self):
        # Purple header bar
        self.set_fill_color(110, 86, 255)
        self.rect(0, 0, 210, 13, "F")
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(255, 255, 255)
        self.set_xy(8, 3.5)
        self.cell(100, 6, "InsightHub", ln=False)
        self.set_font("Helvetica", "", 8)
        self.set_xy(110, 3.5)
        self.cell(
            92, 6,
            f"Generated {datetime.date.today().strftime('%B %d, %Y')}",
            ln=False, align="R",
        )
        self.set_text_color(17, 24, 39)
        self.ln(18)

    def footer(self):
        self.set_y(-14)
        self.set_draw_color(110, 86, 255)
        self.set_line_width(0.3)
        self.line(20, self.get_y(), 190, self.get_y())
        self.set_font("Helvetica", "", 8)
        self.set_text_color(156, 163, 175)
        q_short = (self.query[:55] + "...") if len(self.query) > 55 else self.query
        label   = f"InsightHub  |  {q_short}" if q_short else "InsightHub Research Report"
        self.set_xy(20, self.get_y() + 2)
        self.cell(140, 5, _safe(label), ln=False)
        self.cell(30, 5, f"Page {self.page_no()}", align="R")


def generate_pdf(
    report: str,
    title:  str = "InsightHub",
    query:  str = "",
) -> bytes:
    """
    Generate a professional PDF report.
    Returns raw bytes for st.download_button().

    Args:
        report: Markdown report string from the Synthesizer agent
        title:  Brand name (always "InsightHub")
        query:  User query — used as PDF title and footer label
    """
    # Sanitize all input
    report = _safe(report or "")
    query  = _safe(query  or "")

    pdf = InsightPDF(query=query)
    pdf.set_margins(20, 20, 20)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # ── TITLE BLOCK (first page only, if query provided) ─────────────────
    if query.strip():
        # Small brand label
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(110, 86, 255)
        pdf.cell(0, 5, "INSIGHTHUB RESEARCH REPORT", ln=True)
        pdf.ln(3)

        # Query as main title — font size based on length
        font_size = 20 if len(query) < 40 else 16 if len(query) < 80 else 13
        pdf.set_font("Helvetica", "B", font_size)
        pdf.set_text_color(17, 24, 39)
        # Truncate if extremely long
        q_display = query if len(query) <= 120 else query[:117] + "..."
        pdf.multi_cell(0, 8, q_display, ln=True)
        pdf.ln(2)

        # Subtitle line
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(156, 163, 175)
        subtitle = f"Research Report  |  {datetime.date.today().strftime('%B %d, %Y')}"
        pdf.cell(0, 5, subtitle, ln=True)
        pdf.ln(2)

        # Purple accent divider line
        pdf.set_draw_color(110, 86, 255)
        pdf.set_line_width(0.5)
        pdf.line(20, pdf.get_y(), 190, pdf.get_y())
        pdf.ln(8)

        # Reset text color for body
        pdf.set_text_color(17, 24, 39)

    # ── REPORT BODY ───────────────────────────────────────────────────────
    for line in report.split("\n"):
        s = line.strip()

        if s.startswith("## ") or s.startswith("# "):
            heading = s.lstrip("#").strip()
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(74, 58, 255)
            pdf.ln(4)
            pdf.multi_cell(0, 7, _safe(heading), ln=True)
            pdf.ln(1)
            pdf.set_text_color(17, 24, 39)

        elif re.match(r"^\d+\.", s):
            # Numbered list
            content = re.sub(r"^\d+\.\s*", "", s)
            # Bold the part before the first colon if present
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(75, 85, 99)
            pdf.multi_cell(0, 6, _safe(f"  {content}"), ln=True)

        elif s.startswith("- ") or s.startswith("* "):
            content = s[2:]
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(75, 85, 99)
            pdf.multi_cell(0, 6, _safe(f"  - {content}"), ln=True)

        elif s == "":
            pdf.ln(3)

        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(75, 85, 99)
            pdf.multi_cell(0, 6, _safe(s), ln=True)

    return bytes(pdf.output())


def make_filename(query: str, prefix: str = "insighthub") -> str:
    """Generate a clean filename from the query."""
    slug = _slugify(query)
    return f"{prefix}_{slug}.pdf"