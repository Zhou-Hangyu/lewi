/* The CSFRBench page: graph glyphs, the catalog, the results table, copy buttons, the BibTeX dialog. */
(function () {
  'use strict';
  var NS = 'http://www.w3.org/2000/svg';
  var root = document.documentElement;
  function $(id) { return document.getElementById(id); }
  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function html(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function safe(fn) { try { fn(); } catch (err) { if (window.console) console.error(err); } }
  var GRAPHS = window.LEWI_GRAPHS || [], DATA = window.CSFRB || {workloads: [], arms: [], aulc: {}};

  // ------------------------------------------------------------------ a graph glyph: inputs, local functions, output
  function glyph(title, opt) {
    opt = opt || {};
    var g = GRAPHS.filter(function (x) { return x.title === title; })[0];
    var svg = el('svg', {'aria-hidden': 'true', 'class': 'cb-glyph'});
    if (!g) return svg;
    var xs = g.nodes.map(function (n) { return n.x; }), ys = g.nodes.map(function (n) { return n.y; });
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs), top = Math.max.apply(null, ys);
    var W = opt.width || 150, H = opt.height || 96, pad = 8;
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    function PX(n) { return pad + (n.x - x0) / ((x1 - x0) || 1) * (W - 2 * pad); }
    function PY(n) { return H - pad - n.y / (top || 1) * (H - 2 * pad); }
    var gE = el('g', {'class': 'bg-e'}, svg);
    g.edges.forEach(function (e) {
      var a = g.nodes[e[0]], b = g.nodes[e[1]];
      el('line', {x1: PX(a), y1: PY(a), x2: PX(b), y2: PY(b)}, gE);
    });
    var many = g.nodes.length > 20;
    g.nodes.forEach(function (n) {
      el('circle', {cx: PX(n), cy: PY(n), r: n.kind === 'input' ? (many ? 2.1 : 2.7) : (many ? 3.6 : 4.4), 'class': 'bg-' + n.kind}, svg);
    });
    return svg;
  }

  function strip() {
    var host = $('cb-strip');
    if (!host) return;
    ['Depth-4 tree', 'Depth-5 tree', 'Depth-3 reuse DAG', 'Sampled depth-4 DAG', 'Flat modules', 'High-arity node']
      .forEach(function (t) { host.appendChild(glyph(t)); });
  }

  // ------------------------------------------------------------------ the catalog
  function catalog() {
    var host = $('cb-cards');
    if (!host) return;
    DATA.workloads.forEach(function (w) {
      var card = html('figure', 'cb-card');
      card.appendChild(glyph(w.glyph));
      var cap = html('figcaption');
      cap.appendChild(html('b', null, w.name));
      cap.appendChild(html('code', null, w.id));
      cap.appendChild(html('span', 'cb-tests', w.tests));
      var inst = w.id === 'sampled_dags' ? 32 : 16;
      var sizes = w.id === 'depth5_tree' ? '2¹¹ … 2¹⁷' : '2⁹ … 2¹⁵';
      cap.appendChild(html('span', 'cb-meta', w.functions + ' · ' + w.nodes + (w.nodes === 1 ? ' function' : ' functions') +
                                             ' · ' + inst + ' instances · n = ' + sizes));
      card.appendChild(cap);
      host.appendChild(card);
    });
  }

  // ------------------------------------------------------------------ the results table
  function results() {
    var host = $('cb-results');
    if (!host) return;
    var buttons = Array.prototype.slice.call(document.querySelectorAll('#results .seg-b'));
    var order = ['flat_modules', 'depth4_tree', 'reuse_dag', 'sampled_dags', 'fourier_tree', 'fourier_reuse_dag',
                 'high_arity_node', 'depth5_tree'];
    var names = {};
    DATA.workloads.forEach(function (w) { names[w.id] = w.name; });
    function fmt(v) { return v.toFixed(3).replace(/^0/, '').replace(/^-0/, '-'); }
    function draw(setting) {
      var table = DATA.aulc[setting] || {};
      while (host.firstChild) host.removeChild(host.firstChild);
      var t = html('table', 'cb-table cb-aulc'), head = html('thead'), tr = html('tr');
      tr.appendChild(html('th', null, 'Method'));
      order.forEach(function (w) { tr.appendChild(html('th', null, names[w])); });
      head.appendChild(tr);
      t.appendChild(head);
      var best = {};
      order.forEach(function (w) {
        var row = table[w] || {}, b = null;
        DATA.arms.forEach(function (a) {
          if (a.kind !== 'learned' || !row[a.id]) return;
          if (b === null || row[a.id][0] < row[b][0]) b = a.id;
        });
        best[w] = b;
      });
      var body = html('tbody'), firstRef = true;
      DATA.arms.forEach(function (a) {
        var present = order.some(function (w) { return table[w] && table[w][a.id]; });
        if (!present) return;
        var cls = a.kind === 'reference' ? 'cb-ref' + (firstRef ? ' cb-ref-first' : '') : (a.id === 'lewi' ? 'cb-lewi' : null);
        if (a.kind === 'reference') firstRef = false;
        var r = html('tr', cls);
        r.appendChild(html('th', null, a.name));
        order.forEach(function (w) {
          var v = table[w] && table[w][a.id], td = html('td');
          if (v) {
            td.appendChild(html('span', 'cb-m', fmt(v[0])));
            td.appendChild(html('span', 'cb-h', '±' + fmt(v[1])));
            if (best[w] === a.id) td.className = 'cb-best';
          } else {
            td.textContent = '–';
            td.className = 'cb-na';
          }
          r.appendChild(td);
        });
        body.appendChild(r);
      });
      t.appendChild(body);
      host.appendChild(t);
      buttons.forEach(function (b) { b.setAttribute('aria-selected', String(b.getAttribute('data-setting') === setting)); });
    }
    buttons.forEach(function (b) { b.addEventListener('click', function () { draw(b.getAttribute('data-setting')); }); });
    draw('primary');
  }

  // ------------------------------------------------------------------ copy buttons and the BibTeX dialog
  function copyText(text, btn, label) {
    function done() {
      btn.textContent = 'Copied';
      btn.classList.add('done');
      setTimeout(function () { btn.textContent = label; btn.classList.remove('done'); }, 1600);
    }
    function fallback() {
      var area = document.createElement('textarea');
      area.value = text;
      area.setAttribute('readonly', '');
      area.style.position = 'fixed';
      area.style.opacity = '0';
      document.body.appendChild(area);
      area.select();
      try { if (document.execCommand('copy')) done(); else btn.textContent = 'Press Ctrl+C'; }
      catch (err) { btn.textContent = 'Press Ctrl+C'; }
      document.body.removeChild(area);
    }
    if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(text).then(done, fallback);
    else fallback();
  }

  function codeCopy() {
    Array.prototype.forEach.call(document.querySelectorAll('pre.cb-code'), function (pre) {
      var btn = html('button', 'chip cb-copy', 'Copy');
      btn.type = 'button';
      btn.setAttribute('aria-label', 'Copy the code');
      btn.addEventListener('click', function () {
        var text = pre.querySelector('code').textContent.split('\n').map(function (line) {
          return line.replace(/\s+#.*$/, '').replace(/\s+$/, '');
        }).join('\n').replace(/\n+$/, '');
        copyText(text, btn, 'Copy');
      });
      pre.appendChild(btn);
    });
  }

  function bib() {
    var dlg = $('bib-dlg'), close = $('bib-close'), copy = $('copy-bib'), pre = $('bibtex');
    var openers = document.querySelectorAll('#bib-open, .bib-open');
    if (!dlg || !openers.length) return;
    function shut() { if (dlg.close) dlg.close(); else dlg.removeAttribute('open'); }
    Array.prototype.forEach.call(openers, function (b) {
      b.addEventListener('click', function () { if (dlg.showModal) dlg.showModal(); else dlg.setAttribute('open', ''); });
    });
    if (close) close.addEventListener('click', shut);
    dlg.addEventListener('click', function (ev) { if (ev.target === dlg) shut(); });
    if (copy && pre) copy.addEventListener('click', function () { copyText(pre.textContent, copy, 'Copy'); });
  }

  // ------------------------------------------------------------------ the background sheet: its tints drift with the scroll
  function sheet() {
    var WARM = [[242, 107, 29], [204, 121, 167], [230, 159, 0], [242, 107, 29]];
    var COOL = [[0, 114, 178], [0, 158, 115], [0, 114, 178], [0, 140, 150]];
    var ticking = false;
    function mix(stops, p) {
      var x = p * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x)), t = x - i, a = stops[i], b = stops[i + 1];
      return [0, 1, 2].map(function (k) { return Math.round(a[k] + (b[k] - a[k]) * t); });
    }
    function token(name) { return getComputedStyle(root).getPropertyValue(name).trim(); }
    function paint() {
      ticking = false;
      var max = Math.max(1, root.scrollHeight - window.innerHeight), p = Math.min(1, Math.max(0, window.scrollY / max));
      var aw = token('--sheet-alpha-w') || '0.1', ac = token('--sheet-alpha-c') || '0.085';
      root.style.setProperty('--sheet-w', 'rgba(' + mix(WARM, p).join(',') + ',' + aw + ')');
      root.style.setProperty('--sheet-c', 'rgba(' + mix(COOL, p).join(',') + ',' + ac + ')');
    }
    function later() { if (!ticking) { ticking = true; requestAnimationFrame(paint); } }
    window.addEventListener('scroll', later, {passive: true});
    window.addEventListener('resize', later);
    if (window.matchMedia) {
      var mq = window.matchMedia('(prefers-color-scheme: dark)');
      if (mq.addEventListener) mq.addEventListener('change', paint);
    }
    paint();
  }

  root.classList.add('js');
  safe(sheet);
  safe(strip);
  safe(catalog);
  safe(results);
  safe(codeCopy);
  safe(bib);
})();
