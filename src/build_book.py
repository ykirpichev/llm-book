#!/usr/bin/env python3
"""Build Engineering Large Language Models from the Markdown manuscript.

The parser intentionally supports a small, editorially controlled Markdown subset.
That keeps the source pleasant to edit while giving the PDF deterministic layout.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from reportlab.graphics.shapes import Circle, Drawing, Line, Path as GPath, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import inch
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfdoc import PDFString
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    ListFlowable,
    ListItem,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents
from book_figures import FIGURES, render_figure


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"
DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "engineering-large-language-models.pdf"

PAGE_SIZE = (7 * inch, 10 * inch)
PAGE_W, PAGE_H = PAGE_SIZE
MARGIN_X = 17 * mm
MARGIN_TOP = 20 * mm
MARGIN_BOTTOM = 18 * mm

INK = HexColor("#292B29")
MUTED = HexColor("#6B6D68")
CHARCOAL = HexColor("#29473F")
TEAL = HexColor("#789487")
CYAN = HexColor("#D9E4DE")
CORAL = HexColor("#B9856A")
GOLD = HexColor("#B69A62")
PAPER = HexColor("#FBFAF6")
PANEL = HexColor("#F1F0EA")
PALE_TEAL = HexColor("#EDF2EF")
PALE_CORAL = HexColor("#F6EEE9")
PALE_GOLD = HexColor("#F5F1E7")
WHITE = colors.white


def register_fonts() -> tuple[str, str, str, str]:
    """Prefer a modern embedded font, but keep the build portable."""
    candidates = [
        (
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf",
        ),
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",
        ),
    ]
    for paths in candidates:
        if all(Path(p).exists() for p in paths):
            names = ("HandbookSans", "HandbookSans-Bold", "HandbookSans-Italic", "HandbookSans-BoldItalic")
            for name, path in zip(names, paths):
                pdfmetrics.registerFont(TTFont(name, path))
            pdfmetrics.registerFontFamily(
                "HandbookSans",
                normal=names[0],
                bold=names[1],
                italic=names[2],
                boldItalic=names[3],
            )
            return names
    return ("Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique")


FONT, FONT_BOLD, FONT_ITALIC, FONT_BOLD_ITALIC = register_fonts()

MATH_FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
if Path(MATH_FONT_PATH).exists():
    MATH_FONT = "ArialUnicodeMath"
    pdfmetrics.registerFont(TTFont(MATH_FONT, MATH_FONT_PATH))
else:
    # Portable fallback. Manuscript equations avoid symbols outside this font's
    # common Unicode coverage when the macOS math-capable font is unavailable.
    MATH_FONT = FONT


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def inline_markup(text: str) -> str:
    text = esc(text.strip())
    # Stash code spans before bold/italic so "*" inside `...` cannot form markup.
    code_spans: list[str] = []

    def _code_token(index: int) -> str:
        # NUL-delimited sentinels cannot collide with ordinary manuscript text
        # and are removed before the string reaches ReportLab.
        return f"\x00CODE_SPAN_{index}\x00"

    def _stash_code(match: re.Match[str]) -> str:
        code_spans.append(match.group(1))
        return _code_token(len(code_spans) - 1)

    text = re.sub(r"`([^`]+)`", _stash_code, text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"\[([^]]+)]\(([^)]+)\)", r'<link href="\2" color="#49685D">\1</link>', text)
    for index, code in enumerate(code_spans):
        text = text.replace(_code_token(index), f'<font name="Courier">{code}</font>')
    return text


def equation_markup(text: str) -> str:
    """Convert a deliberately small equation syntax into ReportLab markup."""
    text = esc(text.strip())
    text = re.sub(r"_\{([^}]+)\}", r"<sub>\1</sub>", text)
    text = re.sub(r"\^\{([^}]+)\}", r"<super>\1</super>", text)
    return text


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:7]
    return f"{slug[:52]}-{digest}"


def make_styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=sample["BodyText"],
        fontName=FONT,
        fontSize=9.35,
        leading=13.1,
        textColor=INK,
        spaceAfter=6.2,
        allowWidows=0,
        allowOrphans=0,
    )
    return {
        "body": body,
        "lead": ParagraphStyle(
            "Lead",
            parent=body,
            fontSize=12.2,
            leading=17.4,
            textColor=MUTED,
            spaceAfter=12,
        ),
        "part": ParagraphStyle(
            "Part",
            fontName=FONT_BOLD,
            fontSize=10,
            leading=12,
            textColor=TEAL,
            uppercase=True,
            spaceAfter=8,
        ),
        "chapter": ParagraphStyle(
            "Chapter",
            fontName=FONT_BOLD,
            fontSize=28,
            leading=32,
            textColor=INK,
            spaceAfter=14,
        ),
        "h2": ParagraphStyle(
            "H2",
            fontName=FONT_BOLD,
            fontSize=16.5,
            leading=20,
            textColor=INK,
            spaceBefore=13,
            spaceAfter=7,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "H3",
            fontName=FONT_BOLD,
            fontSize=12.2,
            leading=15,
            textColor=TEAL,
            spaceBefore=9,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "h4": ParagraphStyle(
            "H4",
            fontName=FONT_BOLD,
            fontSize=10.2,
            leading=13,
            textColor=INK,
            spaceBefore=7,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=body,
            leftIndent=13,
            firstLineIndent=0,
            spaceAfter=2,
        ),
        "number": ParagraphStyle(
            "Number",
            parent=body,
            leftIndent=16,
            firstLineIndent=0,
            spaceAfter=2,
        ),
        "caption": ParagraphStyle(
            "Caption",
            fontName=FONT,
            fontSize=7.7,
            leading=10,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=9,
        ),
        "equation": ParagraphStyle(
            "Equation",
            fontName=MATH_FONT,
            fontSize=12.5,
            leading=17,
            textColor=INK,
            alignment=TA_CENTER,
            leftIndent=18,
            rightIndent=18,
            spaceBefore=8,
            spaceAfter=3,
        ),
        "code": ParagraphStyle(
            "Code",
            fontName="Courier",
            fontSize=6.75,
            leading=8.7,
            textColor=HexColor("#ECEAE4"),
            leftIndent=8,
            rightIndent=8,
            spaceBefore=5,
            spaceAfter=5,
        ),
        "quote": ParagraphStyle(
            "Quote",
            parent=body,
            fontName=FONT_ITALIC,
            fontSize=10.2,
            leading=14.5,
            textColor=MUTED,
            leftIndent=16,
            rightIndent=10,
            borderColor=TEAL,
            borderWidth=0,
            borderPadding=(2, 4, 2, 12),
            spaceBefore=4,
            spaceAfter=9,
        ),
        "toc1": ParagraphStyle(
            "TOC1",
            fontName=FONT_BOLD,
            fontSize=10.5,
            leading=14,
            textColor=INK,
            leftIndent=0,
            firstLineIndent=0,
            spaceBefore=5,
        ),
        "toc2": ParagraphStyle(
            "TOC2",
            fontName=FONT,
            fontSize=8.6,
            leading=11,
            textColor=MUTED,
            leftIndent=14,
            firstLineIndent=0,
            spaceBefore=1,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=body,
            fontSize=7.9,
            leading=10.5,
            textColor=MUTED,
        ),
    }


STYLES = make_styles()


class Rule(Flowable):
    def __init__(self, color=TEAL, width=42, thickness=3, space_before=3, space_after=10):
        super().__init__()
        self.color = color
        self.rule_width = width
        self.thickness = thickness
        self.spaceBefore = space_before
        self.spaceAfter = space_after
        self.width = 1
        self.height = thickness

    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.roundRect(0, 0, self.rule_width, self.thickness, self.thickness / 2, stroke=0, fill=1)


class Heading(Paragraph):
    def __init__(self, text: str, style: ParagraphStyle, level: int, toc: bool = True):
        self.raw_text = text
        self.level = level
        self.toc = toc
        self.anchor = slugify(text)
        super().__init__(f'<a name="{self.anchor}"/>{inline_markup(text)}', style)


class ChapterBand(Flowable):
    def __init__(self, kicker: str, title: str, number: str | None = None):
        super().__init__()
        self.kicker = kicker
        self.title = title
        self.number = number
        self.raw_text = title
        self.level = 2
        self.toc = True
        self.anchor = slugify(title)
        self.height = 140
        self.width = PAGE_W - 2 * MARGIN_X

    def draw(self):
        c = self.canv
        c.saveState()
        c.setFillColor(CHARCOAL)
        c.roundRect(0, 0, self.width, self.height, 12, fill=1, stroke=0)
        c.setFillColor(TEAL)
        c.roundRect(0, self.height - 9, self.width, 9, 4, fill=1, stroke=0)
        if self.number:
            c.setFillColor(HexColor("#527064"))
            c.setFont(FONT_BOLD, 66)
            c.drawRightString(self.width - 18, self.height - 70, self.number)
        c.setFillColor(CYAN)
        c.setFont(FONT_BOLD, 8.5)
        c.drawString(20, self.height - 34, self.kicker.upper())
        title = Paragraph(inline_markup(self.title), ParagraphStyle(
            "BandTitle", fontName=FONT_BOLD, fontSize=24, leading=27.5, textColor=WHITE,
        ))
        title.wrapOn(c, self.width - 50, 92)
        title.drawOn(c, 20, 27)
        c.restoreState()


class BoxedFlowable(Flowable):
    def __init__(self, content: Sequence[Flowable], background, accent, width: float, padding: float = 10):
        super().__init__()
        self.content = list(content)
        self.background = background
        self.accent = accent
        self.box_width = width
        self.padding = padding
        self._sizes: list[tuple[float, float]] = []

    def wrap(self, availWidth, availHeight):
        self.box_width = min(self.box_width, availWidth)
        inner = self.box_width - 2 * self.padding - 4
        self._sizes = []
        h = self.padding * 2
        for f in self.content:
            w, fh = f.wrap(inner, availHeight)
            self._sizes.append((w, fh))
            h += fh + getattr(f, "spaceBefore", 0) + getattr(f, "spaceAfter", 0)
        self.width = self.box_width
        self.height = h
        return self.width, self.height

    def draw(self):
        c = self.canv
        c.saveState()
        c.setFillColor(self.background)
        c.roundRect(0, 0, self.width, self.height, 7, stroke=0, fill=1)
        c.setFillColor(self.accent)
        c.roundRect(0, 0, 4, self.height, 2, stroke=0, fill=1)
        y = self.height - self.padding
        inner = self.width - 2 * self.padding - 4
        for f, (_, fh) in zip(self.content, self._sizes):
            y -= getattr(f, "spaceBefore", 0)
            y -= fh
            f.drawOn(c, self.padding + 4, y)
            y -= getattr(f, "spaceAfter", 0)
        c.restoreState()


class CodePanel(Flowable):
    def __init__(self, code: str, language: str, width: float, display_lines=None):
        super().__init__()
        self.code = code.rstrip()
        self.language = language or "text"
        self.box_width = width
        self.source_lines = self.code.splitlines() or [""]
        self._display_lines = display_lines
        self.lines = []
        self.font_size = 8.5
        self.line_h = 11.0
        self.header_h = 18
        self.padding = 9

    def wrap(self, availWidth, availHeight):
        self.width = min(availWidth, self.box_width)
        columns = max(1, int((self.width - 2 * self.padding - 9)
                            / pdfmetrics.stringWidth("M", "Courier", self.font_size)))
        self.lines = self._display_lines if self._display_lines is not None else [
            (line[start:start + columns], start > 0)
            for raw in self.source_lines
            for line in [raw.expandtabs(4)]
            for start in range(0, max(1, len(line)), columns)
        ]
        self.height = self.header_h + self.padding * 2 + len(self.lines) * self.line_h
        return self.width, self.height

    def split(self, availWidth, availHeight):
        max_lines = int((availHeight - self.header_h - 2 * self.padding) // self.line_h)
        if max_lines < 5 or max_lines >= len(self.lines):
            return []
        first = CodePanel("", self.language, self.box_width, self.lines[:max_lines])
        rest = CodePanel("", f"{self.language} (continued)", self.box_width, self.lines[max_lines:])
        return [first, rest]

    def draw(self):
        c = self.canv
        c.saveState()
        c.setFillColor(HexColor("#F2F1ED"))
        c.roundRect(0, 0, self.width, self.height, 7, fill=1, stroke=0)
        c.setFillColor(HexColor("#355149"))
        c.roundRect(0, self.height - self.header_h, self.width, self.header_h, 7, fill=1, stroke=0)
        c.setFillColor(CYAN)
        c.setFont(FONT_BOLD, 6.8)
        c.drawString(self.padding, self.height - 12.2, self.language.upper())
        c.setFillColor(INK)
        c.setFont("Courier", self.font_size)
        y = self.height - self.header_h - self.padding - self.font_size
        for line, continued in self.lines:
            if continued:
                c.drawString(self.padding, y, ">")
            c.drawString(self.padding + 9, y, line)
            y -= self.line_h
        c.restoreState()


def arrow(d: Drawing, x1, y1, x2, y2, color=TEAL, width=1.7):
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=width))
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 6
    p1 = (x2, y2)
    p2 = (x2 - size * math.cos(angle - 0.48), y2 - size * math.sin(angle - 0.48))
    p3 = (x2 - size * math.cos(angle + 0.48), y2 - size * math.sin(angle + 0.48))
    d.add(Polygon([*p1, *p2, *p3], fillColor=color, strokeColor=color))


def box(d: Drawing, x, y, w, h, label, fill=WHITE, stroke=TEAL, font=7.8, label_color=INK):
    d.add(Rect(x, y, w, h, rx=6, ry=6, fillColor=fill, strokeColor=stroke, strokeWidth=1.2))
    lines = label.split("\n")
    for i, line in enumerate(lines):
        d.add(String(x + w / 2, y + h / 2 + (len(lines) - 1) * 5 - i * 10 - 2,
                     line, textAnchor="middle", fontName=FONT_BOLD, fontSize=font, fillColor=label_color))


def diagram(name: str, width: float) -> Drawing:
    if name in FIGURES:
        return render_figure(name, width, (FONT, FONT_BOLD), globals())
    # Author all diagrams on one fixed canvas; scale the whole drawing into
    # the page instead of shrinking only its background around fixed nodes.
    w = 430
    h = 190
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, rx=9, ry=9, fillColor=PANEL, strokeColor=HexColor("#D8D7D1")))

    if name == "training_pipeline":
        labels = ["Raw data", "Curate", "Pretrain", "Post-train", "Evaluate", "Serve"]
        bw, bh, gap = 56, 34, 11
        total = len(labels) * bw + (len(labels) - 1) * gap
        x = (w - total) / 2
        y = 92
        for i, label in enumerate(labels):
            fill = PALE_TEAL if i in (1, 4) else WHITE
            box(d, x, y, bw, bh, label, fill=fill, font=7)
            if i < len(labels) - 1:
                arrow(d, x + bw, y + bh / 2, x + bw + gap - 2, y + bh / 2)
            x += bw + gap
        d.add(String(24, 154, "THE TRAINING SYSTEM IS A CLOSED LOOP", fontName=FONT_BOLD, fontSize=9, fillColor=INK))
        arrow(d, w - 54, 88, w - 54, 52, CORAL)
        arrow(d, w - 54, 52, 54, 52, CORAL)
        arrow(d, 54, 52, 54, 88, CORAL)
        d.add(String(w / 2, 34, "production telemetry and failure cases", textAnchor="middle", fontName=FONT, fontSize=7.5, fillColor=MUTED))
    elif name == "data_pipeline":
        # Policy controls are a boundary around the entire transformation chain,
        # not a side branch from one arbitrary stage. The only forward path ends
        # in a versioned release, whose train/evaluation/red-team splits remain
        # distinct governed outputs.
        d.add(Rect(14, 61, w - 28, 116, rx=8, ry=8,
                   fillColor=PALE_CORAL, strokeColor=CORAL, strokeWidth=1.2))
        d.add(String(w / 2, 161, "GOVERNANCE AND QUALITY CONTROLS SPAN THE PIPELINE",
                     textAnchor="middle", fontName=FONT_BOLD, fontSize=8.2, fillColor=CORAL))

        labels = ["Collect", "Normalize", "Deduplicate", "Score", "Balance", "Verify"]
        widths = [52, 62, 70, 48, 54, 48]
        gap = 8
        x = (w - sum(widths) - gap * (len(widths) - 1)) / 2
        y = 111
        for i, (label, bw) in enumerate(zip(labels, widths)):
            box(d, x, y, bw, 27, label, fill=WHITE, font=6.8)
            if i < len(labels) - 1:
                arrow(d, x + bw, y + 13.5, x + bw + gap - 2, y + 13.5, TEAL, 1.25)
            x += bw + gap

        d.add(String(w / 2, 81, "rights | privacy | safety | source quality | split leakage",
                     textAnchor="middle", fontName=FONT_BOLD, fontSize=7.2, fillColor=CORAL))

        box(d, 117, 12, 196, 34, "Versioned dataset releases\ntrain | evaluation | red-team",
            fill=PALE_TEAL, font=7.2)
        verify_center = (w - sum(widths) - gap * (len(widths) - 1)) / 2 + sum(widths[:-1]) + gap * 5 + widths[-1] / 2
        d.add(Line(verify_center, 109, verify_center, 49, strokeColor=TEAL, strokeWidth=1.4))
        arrow(d, verify_center, 49, 315, 29, TEAL, 1.4)
    elif name == "distillation":
        box(d, 24, 110, 112, 45, "Teacher\nrich distribution", fill=PALE_GOLD, stroke=GOLD)
        box(d, w - 136, 110, 112, 45, "Student\nserving budget", fill=PALE_TEAL)
        arrow(d, 139, 133, w - 139, 133, CORAL, 2)
        d.add(String(w / 2, 145, "soft targets at temperature T", textAnchor="middle", fontName=FONT_BOLD, fontSize=7.6, fillColor=CORAL))
        box(d, w / 2 - 72, 36, 144, 44, "Objective\nalpha CE + beta T^2 KL", fill=WHITE, stroke=INK)
        arrow(d, 82, 106, w / 2 - 38, 82, GOLD)
        arrow(d, w - 82, 106, w / 2 + 38, 82, TEAL)
        d.add(String(w / 2, 18, "Optimize for downstream utility, not teacher imitation alone.", textAnchor="middle", fontName=FONT, fontSize=7.5, fillColor=MUTED))
    elif name == "attention":
        box(d, 24, 122, 72, 35, "Q", fill=PALE_TEAL)
        box(d, 24, 75, 72, 35, "K", fill=PALE_TEAL)
        box(d, 24, 28, 72, 35, "V", fill=PALE_TEAL)
        box(d, 146, 99, 106, 42, "scores\nQK^T / sqrt(d)", fill=WHITE, stroke=INK)
        box(d, 286, 99, 92, 42, "mask +\nsoftmax", fill=PALE_GOLD, stroke=GOLD)
        box(d, 286, 34, 92, 42, "weighted\nsum", fill=PALE_CORAL, stroke=CORAL)
        arrow(d, 98, 140, 144, 126)
        arrow(d, 98, 92, 144, 113)
        arrow(d, 254, 120, 284, 120, GOLD)
        arrow(d, 332, 97, 332, 78, CORAL)
        arrow(d, 98, 46, 284, 51, TEAL)
        d.add(String(w / 2, 168, "ATTENTION DATAFLOW AND MATERIALIZATION POINTS", textAnchor="middle", fontName=FONT_BOLD, fontSize=9, fillColor=INK))
    elif name == "speculative_decoding":
        box(d, 22, 112, 90, 42, "Draft model\npropose gamma", fill=PALE_TEAL)
        box(d, 168, 112, 94, 42, "Target model\nverify in parallel", fill=PALE_GOLD, stroke=GOLD)
        box(d, 318, 112, 90, 42, "Commit prefix\n+ one token", fill=PALE_CORAL, stroke=CORAL)
        arrow(d, 114, 133, 166, 133)
        arrow(d, 264, 133, 316, 133)
        arrow(d, 363, 108, 363, 62, CORAL)
        arrow(d, 363, 62, 67, 62, CORAL)
        arrow(d, 67, 62, 67, 108, CORAL)
        d.add(String(w / 2, 43, "correction on rejection; target bonus if all proposals pass", textAnchor="middle", fontName=FONT_BOLD, fontSize=8, fillColor=CORAL))
        d.add(String(w / 2, 26, "continue after committed output, unless EOS or an output limit stops it", textAnchor="middle", fontName=FONT, fontSize=8, fillColor=MUTED))
        d.add(String(w / 2, 173, "EXACT SAMPLING WITH A CHEAPER PROPOSAL DISTRIBUTION", textAnchor="middle", fontName=FONT_BOLD, fontSize=9, fillColor=INK))
    elif name == "roofline":
        d.add(String(w / 2, 174, "SCHEMATIC: ILLUSTRATIVE REGIMES, NOT MEASURED POINTS", textAnchor="middle", fontName=FONT_BOLD, fontSize=8, fillColor=INK))
        x0, y0, x1, y1 = 52, 34, w - 32, 158
        d.add(Line(x0, y0, x0, y1, strokeColor=INK, strokeWidth=1.2))
        d.add(Line(x0, y0, x1, y0, strokeColor=INK, strokeWidth=1.2))
        knee_x, knee_y = x0 + (x1 - x0) * .56, y0 + (y1 - y0) * .72
        d.add(Line(x0, y0 + 8, knee_x, knee_y, strokeColor=TEAL, strokeWidth=3))
        d.add(Line(knee_x, knee_y, x1, knee_y, strokeColor=CORAL, strokeWidth=3))
        d.add(String(x0 - 28, (y0 + y1) / 2, "FLOP/s", angle=90, fontName=FONT_BOLD, fontSize=7.5, fillColor=INK))
        d.add(String((x0 + x1) / 2, 14, "arithmetic intensity (FLOP/byte)", textAnchor="middle", fontName=FONT_BOLD, fontSize=7.5, fillColor=INK))
        d.add(String(x0 + 42, y0 + 52, "bandwidth bound", fontName=FONT_BOLD, fontSize=7.5, fillColor=TEAL))
        d.add(String(knee_x + 30, knee_y + 10, "compute bound", fontName=FONT_BOLD, fontSize=7.5, fillColor=CORAL))
        d.add(Circle(x0 + 92, y0 + 36, 4, fillColor=TEAL, strokeColor=WHITE))
        d.add(String(x0 + 99, y0 + 31, "decode", fontName=FONT, fontSize=7, fillColor=MUTED))
        d.add(Circle(knee_x + 42, knee_y - 4, 4, fillColor=CORAL, strokeColor=WHITE))
        d.add(String(knee_x + 49, knee_y - 9, "prefill", fontName=FONT, fontSize=7, fillColor=MUTED))
    elif name == "continuous_batching":
        d.add(String(20, 168, "STATIC BATCH", fontName=FONT_BOLD, fontSize=8.5, fillColor=MUTED))
        d.add(String(20, 80, "CONTINUOUS BATCH", fontName=FONT_BOLD, fontSize=8.5, fillColor=TEAL))
        colors_ = [TEAL, GOLD, CORAL, HexColor("#81766D")]
        for row in range(3):
            y = 135 - row * 19
            for col in range(7 - row * 2):
                d.add(Rect(105 + col * 30, y, 25, 13, rx=2, ry=2, fillColor=colors_[row], strokeColor=None))
        # Three slots initially occupied; replacement requests enter at boundaries.
        schedules = [[0]*7+[3]*3, [1]*5+[0]*5, [2]*3+[1]*7]
        for row, schedule in enumerate(schedules):
            y = 55 - row * 18
            for col, request in enumerate(schedule):
                d.add(Rect(105 + col * 27, y, 22, 12, rx=2, ry=2, fillColor=colors_[request], strokeColor=None))
        d.add(String(105, 7, "slots refill at request completion; time ->", fontName=FONT, fontSize=7.2, fillColor=MUTED))
    elif name == "disaggregation":
        d.add(String(w / 2, 172, "PREFILL-DECODE DISAGGREGATION AND THE KV HANDOFF", textAnchor="middle", fontName=FONT_BOLD, fontSize=8.6, fillColor=INK))
        box(d, 26, 84, 104, 62, "Prefill pool\ncompute-heavy\nbig batches", fill=PALE_TEAL, stroke=TEAL, font=7.2)
        box(d, w / 2 - 55, 92, 110, 46, "KV handoff\nX bytes / BW\n+ protocol", fill=PALE_GOLD, stroke=GOLD, font=7.2)
        box(d, w - 130, 84, 104, 62, "Decode pool\ncadence-bound\npaged KV", fill=PALE_CORAL, stroke=CORAL, font=7.2)
        arrow(d, 132, 115, w / 2 - 57, 115, TEAL, 1.6)
        arrow(d, w / 2 + 57, 115, w - 132, 115, CORAL, 1.6)
        d.add(String(w / 2, 44, "wins only when phase specialization and isolation exceed", textAnchor="middle", fontName=FONT, fontSize=7.4, fillColor=MUTED))
        d.add(String(w / 2, 32, "transfer time + extra queueing + operational complexity", textAnchor="middle", fontName=FONT, fontSize=7.4, fillColor=MUTED))
    elif name == "memory_hierarchy":
        levels = [
            ("Registers: thread-local state", 188, CORAL),
            ("Shared memory / L1: SM locality", 218, GOLD),
            ("L2: device-wide cache", 248, TEAL),
            ("HBM: device memory", 278, HexColor("#81766D")),
            ("Host / remote memory: transfer boundary", 308, MUTED),
        ]
        d.add(String(w / 2, 172, "MEMORY LEVELS: SCOPE, CAPACITY, AND MOVEMENT", textAnchor="middle", fontName=FONT_BOLD, fontSize=9, fillColor=INK))
        for i, (label, ww, color) in enumerate(levels):
            x = (w - ww) / 2
            y = 140 - i * 27
            d.add(Rect(x, y, ww, 22, rx=4, ry=4, fillColor=WHITE, strokeColor=color, strokeWidth=1.2))
            d.add(String(w / 2, y + 7, label, textAnchor="middle", fontName=FONT_BOLD, fontSize=8, fillColor=INK))
        d.add(String(w / 2, 13, "schematic width indicates broader capacity; levels are not nested containers", textAnchor="middle", fontName=FONT, fontSize=7.4, fillColor=MUTED))
    elif name == "gemm_tiling":
        d.add(String(w / 2, 169, "TILED GEMM: DATA REUSE BEFORE MORE MATH", textAnchor="middle", fontName=FONT_BOLD, fontSize=9, fillColor=INK))
        for r in range(6):
            for c in range(6):
                fill = PALE_TEAL if 1 <= r <= 3 else WHITE
                d.add(Rect(30 + c * 18, 48 + r * 18, 16, 16, fillColor=fill, strokeColor=HexColor("#BBBAB4"), strokeWidth=.4))
        for r in range(6):
            for c in range(6):
                fill = PALE_GOLD if 2 <= c <= 4 else WHITE
                d.add(Rect(166 + c * 18, 48 + r * 18, 16, 16, fillColor=fill, strokeColor=HexColor("#BBBAB4"), strokeWidth=.4))
        box(d, w - 119, 78, 88, 48, "C tile\nregisters", fill=PALE_CORAL, stroke=CORAL)
        d.add(String(150, 98, "x", textAnchor="middle", fontName=FONT_BOLD, fontSize=10, fillColor=INK))
        arrow(d, 276, 101, w - 121, 101)
        d.add(String(82, 32, "A tile -> shared memory", textAnchor="middle", fontName=FONT, fontSize=7, fillColor=MUTED))
        d.add(String(218, 32, "B tile -> shared memory", textAnchor="middle", fontName=FONT, fontSize=7, fillColor=MUTED))
    elif name == "online_softmax":
        d.add(String(w / 2, 172, "ONLINE SOFTMAX: MERGE TILE STATES WITHOUT FULL SCORES", textAnchor="middle", fontName=FONT_BOLD, fontSize=8.6, fillColor=INK))
        box(d, 18, 108, 108, 42, "Tile A\nm_a, l_a, o_a", fill=PALE_TEAL)
        box(d, 18, 42, 108, 42, "Tile B\nm_b, l_b, o_b", fill=PALE_GOLD, stroke=GOLD)
        box(d, 168, 72, 96, 48, "m = max\nrescale\nadd", fill=WHITE, stroke=INK)
        box(d, 308, 72, 104, 48, "Merged\nm, l, o", fill=PALE_CORAL, stroke=CORAL)
        arrow(d, 128, 129, 166, 104)
        arrow(d, 128, 63, 166, 88)
        arrow(d, 266, 96, 306, 96, CORAL)
        d.add(String(w / 2, 22, "rescale l and o to the shared maximum; normalize o / l at the end", textAnchor="middle", fontName=FONT, fontSize=7.4, fillColor=MUTED))
    elif name == "prefill_decode_kernels":
        d.add(String(w / 2, 172, "ONE LAYER, TWO KERNEL REGIMES", textAnchor="middle", fontName=FONT_BOLD, fontSize=9, fillColor=INK))
        box(d, 18, 88, 176, 64, "Prefill\nmany Q rows x long K/V\nGEMM + FlashAttention", fill=PALE_TEAL)
        box(d, w - 194, 88, 176, 64, "Decode\n1 new Q / sequence\npaged KV + cadence", fill=PALE_CORAL, stroke=CORAL)
        box(d, w / 2 - 78, 28, 156, 36, "Same math\ndifferent schedule", fill=PALE_GOLD, stroke=GOLD, font=7.4)
        arrow(d, 106, 86, w / 2 - 40, 58, MUTED, 1.2)
        arrow(d, w - 106, 86, w / 2 + 40, 58, MUTED, 1.2)
        d.add(String(w / 2, 12, "optimize the regime you are in; do not reuse the other tile plan blindly", textAnchor="middle", fontName=FONT, fontSize=7.2, fillColor=MUTED))
    elif name == "parallelism_map":
        labels = [
            ("Data\nexample batches", 30, 118, TEAL),
            ("Tensor\nweight dimensions", 165, 118, CORAL),
            ("Pipeline\nlayer ranges", 300, 118, GOLD),
            ("Sequence / context\ntoken-axis work", 95, 51, HexColor("#81766D")),
            ("Expert\nexpert weights", 235, 51, MUTED),
        ]
        for label, x, y, color in labels:
            box(d, x, y, 100, 42, label, fill=WHITE, stroke=color)
        d.add(String(w / 2, 174, "PARALLELISM AXES SOLVE DIFFERENT BOTTLENECKS", textAnchor="middle", fontName=FONT_BOLD, fontSize=9, fillColor=INK))
        d.add(String(w / 2, 20, "compose only after estimating communication, memory, and pipeline bubbles", textAnchor="middle", fontName=FONT, fontSize=7.3, fillColor=MUTED))
    elif name == "leadership_loop":
        center = (w / 2, 94)
        nodes = [
            (center[0], 157, "Frame"),
            (center[0] + 120, 115, "Align"),
            (center[0] + 75, 40, "Decide"),
            (center[0] - 75, 40, "Commit"),
            (center[0] - 120, 115, "Learn"),
        ]
        for i, (x, y, label) in enumerate(nodes):
            d.add(Circle(x, y, 25, fillColor=WHITE, strokeColor=[TEAL, GOLD, CORAL, HexColor("#81766D"), INK][i], strokeWidth=1.6))
            d.add(String(x, y - 3, label, textAnchor="middle", fontName=FONT_BOLD, fontSize=7.6, fillColor=INK))
            nx, ny, _ = nodes[(i + 1) % len(nodes)]
            angle = math.atan2(ny - y, nx - x)
            arrow(d, x + 27 * math.cos(angle), y + 27 * math.sin(angle), nx - 27 * math.cos(angle), ny - 27 * math.sin(angle), MUTED, 1.1)
        d.add(String(center[0], center[1] + 3, "intent", textAnchor="middle", fontName=FONT_BOLD, fontSize=8, fillColor=TEAL))
        d.add(String(center[0], center[1] - 9, "+ evidence", textAnchor="middle", fontName=FONT_BOLD, fontSize=8, fillColor=TEAL))
    else:
        raise ValueError(f"Unknown diagram: {name}")
    scale = min(1.0, width / w)
    d.scale(scale, scale)
    d.width = w * scale
    d.height = h * scale
    return d


@dataclass
class Meta:
    title: str = "Engineering Large Language Models"
    subtitle: str = "Training, Inference, CUDA, Distributed Systems, and Technical Leadership"
    author: str = "Yury Kirpichev"
    edition: str = "Public Edition - September 2026"
    copyright_year: str = "2026"
    publication_date: str = "September 2026"
    keywords: str = "large language models, LLM systems, model training, inference, CUDA, distributed systems"


class HandbookDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, meta: Meta):
        self.meta = meta
        self.current_chapter = ""
        self._heading_counter = 0
        super().__init__(
            filename,
            pagesize=PAGE_SIZE,
            leftMargin=MARGIN_X,
            rightMargin=MARGIN_X,
            topMargin=MARGIN_TOP,
            bottomMargin=MARGIN_BOTTOM,
            title=meta.title,
            author=meta.author,
            subject=meta.subtitle,
            creator="Codex / ReportLab",
        )
        body_frame = Frame(
            MARGIN_X,
            MARGIN_BOTTOM,
            PAGE_W - 2 * MARGIN_X,
            PAGE_H - MARGIN_TOP - MARGIN_BOTTOM,
            id="body",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates([
            PageTemplate(id="Cover", frames=[body_frame], onPage=self._cover_page),
            PageTemplate(id="Front", frames=[body_frame], onPage=self._front_page),
            PageTemplate(id="Body", frames=[body_frame], onPage=self._body_page),
            PageTemplate(id="Part", frames=[body_frame], onPage=self._part_page),
        ])

    def beforeDocument(self):
        # multiBuild runs the story more than once to resolve the TOC. Reset
        # page-header state so a later pass cannot inherit the final chapter
        # name from the previous pass.
        self.current_chapter = ""
        super().beforeDocument()

    def _cover_page(self, canvas, doc):
        canvas.saveState()
        canvas.setKeywords(self.meta.keywords)
        canvas.showOutline()
        canvas._doc.Catalog.Lang = PDFString("en-US")
        canvas.setFillColor(CHARCOAL)
        canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        canvas.setFillColor(HexColor("#365A50"))
        for i in range(7):
            canvas.circle(PAGE_W - 24 - i * 30, PAGE_H - 38 - i * 34, 88 - i * 7, fill=0, stroke=1)
        canvas.setFillColor(TEAL)
        canvas.rect(0, PAGE_H - 18, PAGE_W, 18, fill=1, stroke=0)
        canvas.setStrokeColor(CYAN)
        canvas.setLineWidth(1.4)
        for i in range(6):
            x = 40 + i * 72
            canvas.line(x, 70, x + 132, 202)
        canvas.setFillColor(WHITE)
        canvas.setFont(FONT_BOLD, 27)
        canvas.drawString(42, PAGE_H - 160, "ENGINEERING")
        canvas.setFillColor(CYAN)
        canvas.setFont(FONT_BOLD, 31)
        canvas.drawString(42, PAGE_H - 202, "LARGE LANGUAGE")
        canvas.setFillColor(WHITE)
        canvas.setFont(FONT_BOLD, 35)
        canvas.drawString(42, PAGE_H - 244, "MODELS")
        canvas.setFillColor(HexColor("#D9DED9"))
        canvas.setFont(FONT, 11)
        subtitle = ["TRAINING  /  INFERENCE  /  CUDA", "DISTRIBUTED SYSTEMS  /  TECHNICAL LEADERSHIP"]
        canvas.drawString(44, PAGE_H - 290, subtitle[0])
        canvas.drawString(44, PAGE_H - 307, subtitle[1])
        canvas.setFillColor(WHITE)
        canvas.setFont(FONT_BOLD, 10)
        canvas.drawString(44, 58, self.meta.author.upper())
        canvas.setFillColor(CYAN)
        canvas.drawRightString(PAGE_W - 44, 58, self.meta.edition.upper())
        canvas.restoreState()

    def _front_page(self, canvas, doc):
        """Render front matter without running heads or folios."""
        canvas.saveState()
        canvas.restoreState()

    def _body_page(self, canvas, doc):
        canvas.saveState()
        page = canvas.getPageNumber()
        if page > 1:
            canvas.setStrokeColor(HexColor("#D8D7D1"))
            canvas.setLineWidth(.45)
            canvas.line(MARGIN_X, PAGE_H - 13 * mm, PAGE_W - MARGIN_X, PAGE_H - 13 * mm)
            canvas.setFont(FONT, 6.8)
            canvas.setFillColor(MUTED)
            header = self.current_chapter.upper() if self.current_chapter else self.meta.title.upper()
            canvas.drawString(MARGIN_X, PAGE_H - 10.5 * mm, header[:78])
            canvas.setFillColor(TEAL)
            canvas.setFont(FONT_BOLD, 7.2)
            canvas.drawRightString(PAGE_W - MARGIN_X, 9 * mm, str(page))
            canvas.setFillColor(MUTED)
            canvas.setFont(FONT, 6.5)
            canvas.drawString(MARGIN_X, 9 * mm, self.meta.author)
        canvas.restoreState()

    def _part_page(self, canvas, doc):
        """Render a clean part opener without inheriting the previous running header."""
        canvas.saveState()
        page = canvas.getPageNumber()
        canvas.setFillColor(TEAL)
        canvas.setFont(FONT_BOLD, 7.2)
        canvas.drawRightString(PAGE_W - MARGIN_X, 9 * mm, str(page))
        canvas.setFillColor(MUTED)
        canvas.setFont(FONT, 6.5)
        canvas.drawString(MARGIN_X, 9 * mm, self.meta.author)
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, (Heading, ChapterBand)):
            key = flowable.anchor
            self.canv.bookmarkPage(key)
            if flowable.level <= 2:
                self.canv.addOutlineEntry(flowable.raw_text, key, level=max(0, flowable.level - 1), closed=False)
            if flowable.toc and flowable.level in (1, 2):
                self.notify("TOCEntry", (flowable.level - 1, flowable.raw_text, self.page, key))
            if flowable.level == 1:
                self.current_chapter = flowable.raw_text


def parse_table(lines: Sequence[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", c or "-") for c in cells):
            continue
        rows.append(cells)
    return rows


def table_flowable(rows: list[list[str]], width: float, leading: Flowable | None = None) -> Flowable:
    if not rows:
        return Spacer(1, 1)
    cols = max(len(r) for r in rows)
    padded = [r + [""] * (cols - len(r)) for r in rows]
    header_style = ParagraphStyle(
        "TableHeader",
        parent=STYLES["small"],
        fontName=FONT_BOLD,
        textColor=WHITE,
    )
    data = [
        [Paragraph(inline_markup(cell), header_style if row_index == 0 else STYLES["small"]) for cell in row]
        for row_index, row in enumerate(padded)
    ]
    col_widths = [width / cols] * cols
    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), CHARCOAL),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), .35, HexColor("#D0CFC9")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PANEL]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return KeepTogether(([leading] if leading is not None else []) + [table, Spacer(1, 7)])


def callout_flowable(kind: str, title: str, body_lines: Sequence[str], width: float) -> Flowable:
    palette = {
        "insight": (PALE_TEAL, TEAL, "INTERVIEW INSIGHT"),
        "pitfall": (PALE_CORAL, CORAL, "COMMON PITFALL"),
        "decision": (PALE_GOLD, GOLD, "ENGINEERING DECISION"),
        "formula": (PANEL, INK, "FORMULA"),
        "check": (PALE_TEAL, TEAL, "DESIGN CHECK"),
    }
    bg, accent, default_title = palette.get(kind.lower(), (PANEL, TEAL, "NOTE"))
    heading = Paragraph(
        f'<font color="{accent.hexval()}"><b>{esc(title or default_title).upper()}</b></font>',
        ParagraphStyle("CalloutTitle", parent=STYLES["small"], fontName=FONT_BOLD, fontSize=7.6, leading=9.2, spaceAfter=4),
    )
    body_text = " ".join(line.strip() for line in body_lines if line.strip())
    body = Paragraph(inline_markup(body_text), ParagraphStyle("CalloutBody", parent=STYLES["body"], fontSize=8.7, leading=12, spaceAfter=0))
    return KeepTogether([Spacer(1, 4), BoxedFlowable([heading, body], bg, accent, width), Spacer(1, 7)])


def make_toc() -> TableOfContents:
    toc = TableOfContents()
    toc.levelStyles = [STYLES["toc1"], STYLES["toc2"]]
    toc.dotsMinLevel = 0
    return toc


def metadata_from_text(text: str) -> tuple[Meta, str]:
    meta = Meta()
    if not text.startswith("---\n"):
        return meta, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return meta, text
    front = text[4:end]
    for line in front.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip().strip('"')
        if hasattr(meta, key):
            setattr(meta, key, value)
    return meta, text[end + 5:]


def flush_paragraph(buffer: list[str], story: list[Flowable]):
    if not buffer:
        return
    text = " ".join(line.strip() for line in buffer).strip()
    if text:
        style = STYLES["lead"] if text.startswith("LEAD: ") else STYLES["body"]
        if text.startswith("LEAD: "):
            text = text[6:]
        story.append(Paragraph(inline_markup(text), style))
    buffer.clear()


def build_story(files: Sequence[Path], width: float) -> tuple[Meta, list[Flowable]]:
    combined = "\n\n".join(path.read_text(encoding="utf-8") for path in files)
    meta, content = metadata_from_text(combined)
    lines = content.splitlines()
    # The first registered template is the cover. Page two is a conventional
    # copyright/imprint page; the authored front matter begins on page three.
    story: list[Flowable] = [
        Spacer(1, PAGE_H - MARGIN_TOP - MARGIN_BOTTOM - 8),
        NextPageTemplate("Front"),
        PageBreak(),
        Spacer(1, 230),
        Paragraph(inline_markup(meta.title), STYLES["h2"]),
        Rule(TEAL, 42, 2.4),
        Spacer(1, 12),
        Paragraph(f"Copyright © {esc(meta.copyright_year)} {esc(meta.author)}", STYLES["body"]),
        Paragraph(inline_markup("Book text and original figures: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original code examples and build tools: [MIT License](https://opensource.org/license/mit). Third-party material retains its own terms."), STYLES["small"]),
        Spacer(1, 8),
        Paragraph(
            inline_markup(f"{meta.edition}."),
            STYLES["body"],
        ),
        Spacer(1, 8),
        Paragraph(
            "This book provides technical and educational information. Readers are responsible for validating designs, code, performance claims, security controls, and operational decisions in their own environments.",
            STYLES["small"],
        ),
        Paragraph(
            "Revised September 24, 2026: Part VI updated through that date; Part VII worked examples updated September 24, with its frontier snapshot dated September 23. Other parts retain the September 13 research cutoff unless explicitly dated.",
            STYLES["small"],
        ),
        Paragraph(
            "Prepared with AI-assisted writing, technical review, and editing, including Astra, Sol, and Terra review passes. This edition has not received independent human technical review or professional copy editing. CPU teaching examples are tested; accelerator code and performance have not been validated on hardware here.",
            STYLES["small"],
        ),
        NextPageTemplate("Body"),
        PageBreak(),
    ]
    buffer: list[str] = []
    chapter_no = 0
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped:
            flush_paragraph(buffer, story)
            i += 1
            continue
        if stripped.startswith("```"):
            flush_paragraph(buffer, story)
            language = stripped[3:].strip()
            code: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            gap = Spacer(1, 4)
            if (story and isinstance(story[-1], Paragraph)
                    and story[-1].getPlainText().startswith("Example status:")):
                # Keep the status with the start of the (splittable) code panel.
                # The spacer must participate in the chain as well.
                story[-1].keepWithNext = True
                gap.keepWithNext = True
            story.extend([gap, CodePanel("\n".join(code), language, width), Spacer(1, 7)])
            i += 1
            continue
        if stripped.startswith(":::callout"):
            flush_paragraph(buffer, story)
            args = stripped[len(":::callout"):].strip().split("|", 1)
            kind = args[0].strip() if args else "note"
            title = args[1].strip() if len(args) > 1 else ""
            body: list[str] = []
            i += 1
            while i < len(lines) and lines[i].strip() != ":::" :
                body.append(lines[i])
                i += 1
            story.append(callout_flowable(kind, title, body, width))
            i += 1
            continue
        if stripped.startswith(":::diagram"):
            flush_paragraph(buffer, story)
            args = stripped[len(":::diagram"):].strip().split("|", 1)
            name = args[0].strip()
            caption = args[1].strip() if len(args) > 1 else name.replace("_", " ").title()
            # A section heading immediately introducing a figure belongs to the
            # same unsplittable group; keepWithNext alone stops at KeepTogether.
            introduction = [story.pop()] if story and isinstance(story[-1], Heading) else []
            story.append(KeepTogether(introduction + [
                Spacer(1, 6), diagram(name, width),
                Paragraph(inline_markup(caption), STYLES["caption"]),
            ]))
            i += 1
            continue
        if stripped.startswith(":::equation"):
            flush_paragraph(buffer, story)
            equation_source = stripped[len(":::equation"):].strip()
            if equation_source.count("|") != 1:
                raise ValueError("equations require exactly one '|' between expression and caption")
            args = equation_source.split("|", 1)
            expression = args[0].strip()
            caption = args[1].strip() if len(args) > 1 else ""
            equation_group = [Paragraph(equation_markup(expression), STYLES["equation"])]
            if caption:
                equation_group.append(Paragraph(inline_markup(caption), STYLES["caption"]))
            story.append(KeepTogether(equation_group))
            i += 1
            continue
        if stripped == ":::toc":
            flush_paragraph(buffer, story)
            if story and not isinstance(story[-1], PageBreak):
                story.append(PageBreak())
            story.extend([Heading("Contents", STYLES["chapter"], 1, toc=False), Rule(), make_toc(), PageBreak()])
            i += 1
            continue
        if stripped == ":::pagebreak":
            flush_paragraph(buffer, story)
            story.append(PageBreak())
            i += 1
            continue
        if stripped.startswith("|") and "|" in stripped[1:]:
            flush_paragraph(buffer, story)
            table_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            # A heading's automatic keepWithNext does not reliably cross the
            # table's KeepTogether wrapper. Put both in the SAME group.
            leading = None
            if story and isinstance(story[-1], Heading):
                leading = story.pop()
            elif (story and isinstance(story[-1], Paragraph)
                  and story[-1].getPlainText().endswith(":")
                  and len(story[-1].getPlainText()) < 250):
                # A short table introduction should not be stranded on the
                # previous page while the table starts on the next one.
                leading = story.pop()
            story.append(table_flowable(parse_table(table_lines), width, leading))
            continue
        if stripped.startswith("# "):
            flush_paragraph(buffer, story)
            title = stripped[2:].strip()
            story.extend([NextPageTemplate("Part"), PageBreak(), Paragraph(inline_markup(title.upper()), STYLES["part"]), Heading(title, STYLES["chapter"], 1), Rule(CORAL, 58, 4), Spacer(1, 8)])
            i += 1
            continue
        if stripped.startswith("## "):
            flush_paragraph(buffer, story)
            title = stripped[3:].strip()
            chapter_no += 1
            if story and not isinstance(story[-1], PageBreak):
                story.extend([NextPageTemplate("Body"), PageBreak()])
            story.extend([
                ChapterBand(f"Chapter {chapter_no:02d}", title, str(chapter_no).zfill(2)),
                Spacer(1, 16),
            ])
            i += 1
            continue
        if stripped.startswith("### "):
            flush_paragraph(buffer, story)
            story.append(Heading(stripped[4:].strip(), STYLES["h2"], 3, toc=False))
            i += 1
            continue
        if stripped.startswith("#### "):
            flush_paragraph(buffer, story)
            story.append(Heading(stripped[5:].strip(), STYLES["h3"], 4, toc=False))
            i += 1
            continue
        if stripped.startswith("##### "):
            flush_paragraph(buffer, story)
            story.append(Heading(stripped[6:].strip(), STYLES["h4"], 5, toc=False))
            i += 1
            continue
        if stripped.startswith("> "):
            flush_paragraph(buffer, story)
            quotes: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("> "):
                quotes.append(lines[i].strip()[2:])
                i += 1
            story.append(Paragraph(inline_markup(" ".join(quotes)), STYLES["quote"]))
            continue
        if re.match(r"^-\s+", stripped):
            flush_paragraph(buffer, story)
            items: list[ListItem] = []
            while i < len(lines) and re.match(r"^-\s+", lines[i].strip()):
                text = re.sub(r"^-\s+", "", lines[i].strip())
                items.append(ListItem(Paragraph(inline_markup(text), STYLES["bullet"]), leftIndent=10))
                i += 1
            story.append(ListFlowable(items, bulletType="bullet", start="circle", leftIndent=16, bulletFontName=FONT_BOLD, bulletColor=TEAL, spaceAfter=6))
            continue
        if re.match(r"^\d+\.\s+", stripped):
            flush_paragraph(buffer, story)
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i].strip()):
                text = re.sub(r"^\d+\.\s+", "", lines[i].strip())
                items.append(ListItem(Paragraph(inline_markup(text), STYLES["number"]), leftIndent=10))
                i += 1
            numbered = ListFlowable(items, bulletType="1", leftIndent=20, bulletFontName=FONT_BOLD, bulletColor=TEAL, spaceAfter=6)
            # Keep short exercise sets together rather than leaving the last
            # question alone on an otherwise empty page.
            if len(items) <= 8:
                group = [numbered]
                # Nest the heading in the same group: ReportLab's automatic
                # keep-with-next handling does not reliably cross a nested
                # KeepTogether, which can strand an exercise title.
                if story and isinstance(story[-1], Heading):
                    group.insert(0, story.pop())
                story.append(KeepTogether(group))
            else:
                story.append(numbered)
            continue
        if stripped == "---":
            flush_paragraph(buffer, story)
            story.extend([Spacer(1, 6), Rule(HexColor("#CECDC7"), width, .65, 0, 7)])
            i += 1
            continue
        buffer.append(stripped)
        i += 1

    flush_paragraph(buffer, story)
    return meta, story


def build(output: Path):
    files = sorted(MANUSCRIPT.glob("*.md"))
    if not files:
        raise SystemExit(f"No manuscript files found in {MANUSCRIPT}")
    output.parent.mkdir(parents=True, exist_ok=True)
    usable_width = PAGE_W - 2 * MARGIN_X
    meta, story = build_story(files, usable_width)
    doc = HandbookDocTemplate(str(output), meta)
    doc.multiBuild(story)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
