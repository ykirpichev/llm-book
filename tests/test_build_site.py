"""The web edition must preserve the manuscript and resolve its own navigation."""
from contextlib import redirect_stdout
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from urllib.parse import unquote, urlsplit

from src.build_site import ROOT, Renderer, build, equation_html, split_book


class Document(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.ids = []; self.links = []; self.code = []; self._pre = False
        self.feed(text)
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs: self.ids.append(attrs['id'])
        if tag == 'a' and 'href' in attrs: self.links.append(attrs['href'])
        if tag in {'script', 'img'} and 'src' in attrs: self.links.append(attrs['src'])
        if tag == 'link' and attrs.get('rel') == 'stylesheet': self.links.append(attrs['href'])
        if tag == 'pre': self._pre = True; self.code.append('')
    def handle_endtag(self, tag):
        if tag == 'pre': self._pre = False
    def handle_data(self, data):
        if self._pre: self.code[-1] += data


class SiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name)
        with redirect_stdout(io.StringIO()): cls.stats = build(cls.output)
        cls.pages = split_book(ROOT / 'manuscript')
        cls.documents = {p.name: Document(p.read_text()) for p in cls.output.glob('*.html')}
    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_all_chapters_and_figures_are_published_and_searchable(self):
        self.assertEqual(self.stats, {'chapters': 70, 'parts': 9, 'figures': 48, 'reading_pages': 80})
        index = json.loads((self.output/'search-index.json').read_text())
        self.assertEqual({p.filename for p in self.pages}, {p['url'] for p in index})
        self.assertTrue(all(p['text'].strip() for p in index))
        self.assertEqual(len(self.documents), 82)

    def test_local_links_images_and_fragments_resolve(self):
        for filename, doc in self.documents.items():
            self.assertEqual(len(doc.ids), len(set(doc.ids)), filename)
            for href in doc.links:
                url = urlsplit(href)
                if url.scheme or url.netloc: continue
                target = unquote(url.path) or filename
                self.assertTrue((self.output/target).is_file(), (filename,href))
                if url.fragment and target.endswith('.html'):
                    self.assertIn(unquote(url.fragment), self.documents[target].ids, (filename,href))

    def test_every_code_example_preserves_its_text(self):
        count = 0
        for page in self.pages:
            expected = re.findall(r'^```[^\n]*\n(.*?)^```', page.body, flags=re.M|re.S)
            actual = self.documents[page.filename].code
            self.assertEqual(actual, expected, page.filename)
            count += len(expected)
        source_count = sum(sum(line.startswith('```') for line in path.read_text().splitlines()) // 2
                           for path in (ROOT/'manuscript').glob('*.md'))
        self.assertEqual(count, source_count)
        self.assertGreater(count, 0)

    def test_special_blocks_are_all_rendered(self):
        sources = '\n'.join(p.body for p in self.pages)
        output = '\n'.join((self.output/p.filename).read_text() for p in self.pages)
        self.assertEqual(output.count('<figure>'), sources.count(':::diagram '))
        self.assertEqual(output.count('<figure class="equation">'), sources.count(':::equation '))
        self.assertEqual(output.count('<aside class="callout '), sources.count(':::callout '))
        for directive in (':::diagram ', ':::equation ', ':::callout ', 'LEAD: '):
            self.assertNotIn(directive, output)

    def test_equations_keep_operators_and_escape_html(self):
        self.assertEqual(equation_html('x_{i} < y^{2} & z'), 'x<sub>i</sub> &lt; y<sup>2</sup> &amp; z')

    def test_unknown_directives_and_unclosed_blocks_fail(self):
        renderer = Renderer(self.output)
        for content in (':::unsupported\n', ':::callout insight|Title\nMissing end', '```python\nx=1'):
            with self.subTest(content=content), self.assertRaises(ValueError):
                renderer.render(content)

    def test_headings_inside_code_do_not_split_chapters(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            (path/'01_part.md').write_text('# Part I - Test\n\n## First\n\n```text\n## Not a chapter\n```\n\n## Second\nBody\n')
            pages=split_book(path)
            self.assertEqual([p.title for p in pages], ['Part I - Test','First','Second'])
