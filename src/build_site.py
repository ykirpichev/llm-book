#!/usr/bin/env python3
"""Render the book's manuscript and original figures as a portable static reader."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET

import markdown
from reportlab.graphics import renderSVG

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_book import diagram, metadata_from_text

ROOT = Path(__file__).resolve().parents[1]
SITE_URL = 'https://ykirpichev.github.io/llm-book/'
REPO = 'https://github.com/ykirpichev/llm-book'
PDF = REPO + '/releases/download/public-2026-09-24-experiments/engineering-large-language-models.pdf'
TITLE = 'Engineering Large Language Models'


def escape(value: str) -> str:
    return html.escape(value, quote=True)


def slug(value: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', value.lower()).strip('-')


@dataclass
class Page:
    title: str
    filename: str
    body: str
    source: str
    part: str = ''
    chapter: int = 0


def split_book(manuscript: Path) -> list[Page]:
    """Split on book headings only, never headings inside fenced code."""
    pages = []
    chapter = 0
    for source in sorted(manuscript.glob('*.md')):
        _, content = metadata_from_text(source.read_text())
        if source.name.startswith('00_'):
            pages.append(Page('Introduction & reading guide', 'introduction.html', content, source.name))
            continue
        part = ''
        current = None
        lines: list[str] = []
        fenced = False
        for line in content.splitlines():
            if line.startswith('```'):
                fenced = not fenced
            if not fenced and re.match(r'^#{1,2} ', line):
                if current:
                    current.body = '\n'.join(lines)
                    pages.append(current)
                lines = []
                if line.startswith('# '):
                    part = line[2:].strip()
                    current = Page(part, f'part-{source.name[:2]}.html', '', source.name, part)
                else:
                    chapter += 1
                    title = line[3:].strip()
                    current = Page(title, f'{chapter:02d}-{slug(title)}.html', '', source.name, part, chapter)
            else:
                lines.append(line)
        if fenced:
            raise ValueError(f'Unclosed code fence in {source}')
        if current:
            current.body = '\n'.join(lines)
            pages.append(current)
    return pages


def equation_html(expression: str) -> str:
    """Preserve the PDF's Unicode equation dialect without treating it as LaTeX."""
    value = escape(expression)
    value = re.sub(r'_\{([^}]+)\}', r'<sub>\1</sub>', value)
    return re.sub(r'\^\{([^}]+)\}', r'<sup>\1</sup>', value)


def render_markdown(source: str) -> tuple[str, str]:
    engine = markdown.Markdown(extensions=['tables', 'fenced_code', 'sane_lists', 'toc'],
                               extension_configs={'toc': {'permalink': '¶'}})
    body = engine.convert(source)
    # Book sections are ### below the chapter's h1; make the web hierarchy h2/h3/h4.
    body = re.sub(r'<(/?)h([3-5])\b', lambda m: f'<{m[1]}h{int(m[2])-1}', body)
    body = body.replace('<table>', '<div class="table-scroll" tabindex="0" role="region" aria-label="Scrollable table"><table>')
    body = body.replace('</table>', '</table></div>')
    return body, engine.toc


class Renderer:
    def __init__(self, output: Path):
        self.output = output
        self.figures: set[str] = set()
        self.dimensions: dict[str, tuple[float, float]] = {}

    def figure(self, name: str) -> str:
        if not re.fullmatch(r'[a-z0-9_]+', name):
            raise ValueError(f'Invalid figure name: {name}')
        if name not in self.figures:
            drawing = diagram(name, 650)
            self.dimensions[name] = (drawing.width, drawing.height)
            svg = renderSVG.drawToString(drawing)
            # Registered PDF font names are not web fonts. Keep the artwork as vectors.
            svg = svg.replace('HandbookSans-BoldItalic', 'Arial').replace('HandbookSans-Italic', 'Arial')
            svg = svg.replace('HandbookSans-Bold', 'Arial').replace('HandbookSans', 'Arial')
            svg = svg.replace('ArialUnicodeMath', 'Arial').replace('Helvetica', 'Arial')
            ET.fromstring(svg)  # Fail the build if a drawing did not export as valid SVG.
            (self.output / 'assets' / f'{name}.svg').write_text(svg)
            self.figures.add(name)
        return f'assets/{name}.svg'

    def render(self, source: str) -> tuple[str, str]:
        lines = source.splitlines()
        result = []
        i = 0
        fenced = False
        while i < len(lines):
            line = lines[i]
            if line.startswith('```'):
                fenced = not fenced
                result.append(line)
            elif fenced:
                result.append(line)
            elif line.startswith(':::diagram '):
                name, caption = line[len(':::diagram '):].split('|', 1)
                image = self.figure(name.strip())
                width, height = self.dimensions[name.strip()]
                caption_html, _ = render_markdown(caption.strip())
                result.append(f'\n<figure><div class="figure-scroll" tabindex="0" role="region" aria-label="Scrollable diagram"><img src="{image}" width="{width:g}" height="{height:g}" alt="{escape(caption.strip())}" loading="lazy"></div><a class="figure-open" href="{image}">Open full diagram ↗</a><figcaption>{caption_html}</figcaption></figure>\n')
            elif line.startswith(':::equation '):
                expression, caption = line[len(':::equation '):].split('|', 1)
                caption_html, _ = render_markdown(caption.strip())
                result.append(f'\n<figure class="equation"><div class="equation-expression" tabindex="0">{equation_html(expression.strip())}</div><figcaption>{caption_html}</figcaption></figure>\n')
            elif line.startswith(':::callout '):
                kind, title = line[len(':::callout '):].split('|', 1)
                content = []
                i += 1
                while i < len(lines) and lines[i].strip() != ':::':
                    content.append(lines[i]); i += 1
                if i == len(lines):
                    raise ValueError('Unclosed callout')
                body, _ = render_markdown('\n'.join(content))
                result.append(f'\n<aside class="callout {escape(kind)}"><p class="callout-title">{escape(title)}</p>{body}</aside>\n')
            elif line == ':::toc':
                result.append('\n[Explore all nine parts and seventy chapters](index.html#contents).\n')
            elif line == ':::pagebreak':
                result.append('')
            elif line.startswith(':::'):
                raise ValueError(f'Unknown manuscript directive: {line}')
            elif line.startswith('LEAD: '):
                lead, _ = render_markdown(line[6:])
                result.append(f'\n<div class="lead">{lead}</div>\n')
            else:
                result.append(line)
            i += 1
        if fenced:
            raise ValueError('Unclosed code fence')
        return render_markdown('\n'.join(result))


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts = []
    def handle_data(self, data):
        self.parts.append(data)


def plain_text(body: str) -> str:
    parser = TextExtractor(); parser.feed(body)
    return ' '.join(' '.join(parser.parts).split())


def contents(pages: list[Page], current: Page | None = None) -> str:
    parts = [p for p in pages if p.part and not p.chapter]
    result = ['<nav aria-label="Book chapters"><a class="nav-top" href="index.html">Book overview</a>',
              '<a class="nav-top" href="introduction.html">Introduction &amp; reading guide</a>']
    for part in parts:
        opened = ' open' if current and current.part == part.part else ''
        active = ' aria-current="page"' if current and part.filename == current.filename else ''
        label = part.title.replace(' - ', ' · ', 1)
        result.append(f'<details{opened}><summary>{escape(label)}</summary><a class="part-intro" href="{part.filename}"{active}>Part introduction</a>')
        for p in pages:
            if p.part == part.part and p.chapter:
                active = ' aria-current="page"' if current and p.filename == current.filename else ''
                result.append(f'<a href="{p.filename}"{active}><span>{p.chapter:02d}</span>{escape(p.title)}</a>')
        result.append('</details>')
    result.append('</nav>')
    return ''.join(result)


def chapter_links(pages: list[Page], part: str) -> str:
    return '<ol class="chapter-list">' + ''.join(
        f'<li><a href="{p.filename}"><span>{p.chapter:02d}</span>{escape(p.title)}<span aria-hidden="true">↗</span></a></li>'
        for p in pages if p.chapter and p.part == part) + '</ol>'


def shell(title: str, filename: str, body: str, pages: list[Page], current: Page | None = None) -> str:
    description = f'{title}. Read the free systems engineering book by Yury Kirpichev, with worked examples, diagrams, and runnable code.'
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title) if title == TITLE else escape(title) + ' · ' + TITLE}</title><meta name="description" content="{escape(description)}">
<link rel="canonical" href="{SITE_URL}{'' if filename == 'index.html' else filename}">
<meta property="og:title" content="{escape(title)}"><meta property="og:description" content="{escape(description)}">
<meta property="og:type" content="article"><meta name="theme-color" content="#29473f">
<link rel="stylesheet" href="assets/site.css"><script src="assets/site.js" defer></script></head>
<body><a class="skip" href="#main">Skip to content</a>
<header class="topbar"><a class="brand" href="index.html"><span class="brand-mark">E<span>LLM</span></span><span class="brand-name">Engineering Large<br>Language Models</span></a>
<form action="search.html" role="search"><label class="sr-only" for="book-search">Search the book</label><input id="book-search" name="q" type="search" placeholder="Search the book…" required><button type="submit" aria-label="Search">⌕</button></form>
<a class="header-pdf" href="{PDF}">Download PDF <span aria-hidden="true">↓</span></a><button class="menu-toggle" aria-expanded="false" aria-controls="booknav">Contents</button></header>
<aside class="sidebar" id="booknav"><div class="sidebar-label">THE BOOK</div>{contents(pages, current)}<div class="sidebar-foot"><a href="{REPO}">Source &amp; contributions ↗</a><span>Free to read. Free to build on.</span></div></aside>
<main id="main" tabindex="-1">{body}<footer><p>Engineering Large Language Models · Yury Kirpichev</p><p><a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> for text and figures · <a href="{REPO}/blob/main/LICENSE-CODE">MIT</a> for code</p><p><a href="{REPO}/issues">Report a correction</a> · <a href="{REPO}/blob/main/docs/release-readiness.md">Review &amp; validation</a></p></footer></main>
</body></html>'''


def build(output: Path, manuscript: Path = ROOT / 'manuscript') -> dict:
    output.mkdir(parents=True, exist_ok=True)
    assets = output / 'assets'; assets.mkdir(exist_ok=True)
    for name in ('site.css', 'site.js'):
        shutil.copyfile(ROOT / 'src' / 'web' / name, assets / name)
    pages = split_book(manuscript)
    renderer = Renderer(output)
    index = []
    for i, page in enumerate(pages):
        rendered, toc = renderer.render(page.body)
        index.append({'title': page.title, 'url': page.filename, 'part': page.part, 'text': plain_text(rendered)})
        label = f'Chapter {page.chapter:02d}' if page.chapter else ('Reading guide' if not page.part else 'Part introduction')
        breadcrumb = f'<a href="index.html#contents">All chapters</a><span>/</span>{escape(page.part or "Start here")}'
        body = f'<div class="breadcrumb">{breadcrumb}</div><article><header class="chapter-header"><p class="eyebrow">{label}</p><h1>{escape(page.title)}</h1></header>'
        if '<li>' in toc:
            body += f'<details class="page-toc"><summary>On this page</summary>{toc}</details>'
        body += f'<div class="prose">{rendered}</div>'
        if page.part and not page.chapter:
            body += '<h2>Chapters in this part</h2>' + chapter_links(pages, page.part)
        body += f'<p class="source-link"><a href="{REPO}/blob/main/manuscript/{page.source}">View chapter source ↗</a></p></article><nav class="page-turn" aria-label="Reading navigation">'
        if i:
            prev = pages[i-1]
            body += f'<a href="{prev.filename}"><small>← Previous</small>{escape(prev.title)}</a>'
        else:
            body += '<a href="index.html"><small>← Back</small>Book overview</a>'
        if i+1 < len(pages):
            nxt = pages[i+1]
            body += f'<a href="{nxt.filename}"><small>Next →</small>{escape(nxt.title)}</a>'
        body += '</nav>'
        (output / page.filename).write_text(shell(page.title, page.filename, body, pages, page))
    chapters = [p for p in pages if p.chapter]
    parts = [p for p in pages if p.part and not p.chapter]
    hero = f'''<section class="hero"><p class="eyebrow">A FREE, OPEN TECHNICAL BOOK</p><h1>Engineering<br>Large Language<br><em>Models.</em></h1><p class="hero-subtitle">From tokens to production systems.</p><p class="hero-description">A systems guide to training, inference, accelerators, distributed infrastructure, and the decisions that connect them.</p><p class="byline">By Yury Kirpichev <span>Public edition · September 2026</span></p><div class="hero-actions"><a class="button" href="introduction.html">Start reading <span>→</span></a><a class="text-button" href="{PDF}">Download PDF ↓</a></div><div class="book-stats"><span><strong>{len(chapters)}</strong> chapters</span><span><strong>{len(parts)}</strong> parts</span><span><strong>{len(renderer.figures)}</strong> original diagrams</span></div></section>
<section class="reading-note"><span class="eyebrow">YOUR READING PATH</span><p>Start with foundations, training, and inference. Continue into accelerators and distributed systems, or jump to production design. Worked examples and a capstone connect the ideas.</p></section><section id="contents"><div class="section-heading"><p class="eyebrow">EXPLORE THE BOOK</p><h2>One system. Nine perspectives.</h2></div>'''
    for part in parts:
        roman, title = part.title.removeprefix('Part ').split(' - ', 1)
        hero += f'<section class="part-card"><header><span class="part-number">{escape(roman)}</span><div><h3><a href="{part.filename}">{escape(title)}</a></h3><a class="part-link" href="{part.filename}">Read the introduction →</a></div></header>{chapter_links(pages,part.part)}</section>'
    hero += f'''</section><aside class="edition-note"><h2>About this edition</h2><p>Part VI revised September 24, 2026; Part VII worked examples September 24 (frontier snapshot September 23). Other parts retain their September 13 research cutoff unless explicitly dated. Prepared with AI-assisted writing, technical review, and editing. No independent human technical review or professional copy editing is claimed. CPU references are tested; accelerator execution and performance remain unverified here.</p><a href="{REPO}/blob/main/docs/release-readiness.md">Read the validation record ↗</a></aside>'''
    (output / 'index.html').write_text(shell(TITLE, 'index.html', hero, pages))
    search = '<header class="chapter-header"><p class="eyebrow">FIND AN IDEA</p><h1>Search the book</h1><p id="search-status" role="status">Enter a phrase in the search box above.</p></header><div id="search-results"></div><noscript><p>Search requires JavaScript. You can browse every chapter from the <a href="index.html#contents">table of contents</a>.</p></noscript>'
    (output / 'search.html').write_text(shell('Search', 'search.html', search, pages))
    (output / 'search-index.json').write_text(json.dumps(index, ensure_ascii=False))
    (output / '.nojekyll').write_text('')
    (output / 'robots.txt').write_text(f'User-agent: *\nAllow: /\nSitemap: {SITE_URL}sitemap.xml\n')
    urls = ['index.html', 'introduction.html', *[p.filename for p in pages if p.filename != 'introduction.html']]
    sitemap = '<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    sitemap += ''.join(f'<url><loc>{SITE_URL}{name}</loc></url>' for name in urls) + '</urlset>'
    (output / 'sitemap.xml').write_text(sitemap)
    result = {'chapters': len(chapters), 'parts': len(parts), 'figures': len(renderer.figures), 'reading_pages': len(pages)}
    (output / 'build-info.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output' / 'site')
    args = parser.parse_args()
    build(args.output)
