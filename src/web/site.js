/* Progressive enhancement: reading and chapter navigation work without JavaScript. */
(() => {
  const menu = document.querySelector('.menu-toggle');
  const closeMenu = () => {
    document.body.classList.remove('nav-open');
    menu.setAttribute('aria-expanded', 'false');
  };
  menu.addEventListener('click', () => {
    const open = document.body.classList.toggle('nav-open');
    menu.setAttribute('aria-expanded', String(open));
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape') closeMenu();
  });
  document.querySelector('main').addEventListener('click', closeMenu);
  const current = document.querySelector('.sidebar [aria-current="page"]');
  if (current && window.matchMedia('(min-width: 741px)').matches) {
    const sidebar = document.querySelector('.sidebar');
    sidebar.scrollTop = Math.max(0, current.offsetTop - sidebar.clientHeight / 2);
  }
  document.querySelectorAll('pre > code').forEach(code => {
    const toolbar = document.createElement('div');
    toolbar.className = 'code-tools';
    const label = document.createElement('span');
    label.textContent = (code.className.match(/language-(\w+)/)?.[1] || 'code').toUpperCase();
    toolbar.append(label);
    if (navigator.clipboard && window.isSecureContext) {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = 'Copy';
      button.setAttribute('aria-label', 'Copy code example');
      button.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(code.textContent);
          button.textContent = 'Copied';
        } catch { button.textContent = 'Select to copy'; }
        window.setTimeout(() => { button.textContent = 'Copy'; }, 1800);
      });
      toolbar.append(button);
    }
    code.parentElement.prepend(toolbar);
  });
  const results = document.getElementById('search-results');
  if (!results) return;
  const status = document.getElementById('search-status');
  const query = (new URLSearchParams(location.search).get('q') || '').trim().slice(0, 200);
  document.getElementById('book-search').value = query;
  if (!query) return;
  const terms = query.toLocaleLowerCase().split(/\s+/u).filter(Boolean);
  status.textContent = 'Searching…';
  fetch('search-index.json').then(response => {
    if (!response.ok) throw new Error('Search index unavailable');
    return response.json();
  }).then(pages => {
    const matches = pages.map(page => {
      const title = page.title.toLocaleLowerCase();
      const text = page.text.toLocaleLowerCase();
      const all = title + ' ' + text;
      if (!terms.every(term => all.includes(term))) return null;
      const score = terms.reduce((n, term) => n + (title.includes(term) ? 12 : 0), 0)
        + (title.includes(query.toLocaleLowerCase()) ? 25 : 0);
      return {...page, score};
    }).filter(Boolean).sort((a, b) => b.score - a.score);
    status.textContent = matches.length
      ? `${matches.length} matching pages for “${query}”${matches.length > 25 ? ' · showing the first 25' : ''}`
      : `No results for “${query}”. Try a shorter term, such as “KV cache” or “distillation”.`;
    matches.slice(0, 25).forEach(page => {
      const article = document.createElement('article'); article.className = 'search-result';
      const part = document.createElement('p'); part.className = 'result-part'; part.textContent = page.part || 'Reading guide';
      const heading = document.createElement('h2');
      const link = document.createElement('a'); link.href = page.url; link.textContent = page.title; heading.append(link);
      const snippet = document.createElement('p');
      const lower = page.text.toLocaleLowerCase();
      const positions = terms.map(term => lower.indexOf(term)).filter(n => n >= 0);
      const start = Math.max(0, (positions.length ? Math.min(...positions) : 0) - 80);
      snippet.textContent = (start ? '…' : '') + page.text.slice(start, start + 260) + (page.text.length > start + 260 ? '…' : '');
      article.append(part, heading, snippet); results.append(article);
    });
  }).catch(() => {
    status.textContent = 'Search could not load. Try again, or use the chapter contents to browse the book.';
  });
})();
