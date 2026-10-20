/* LeWi project page: the hero training replay, the motivation figures (one function or seven pieces, the gap),
   LeWi in four steps, the results pattern, the learned-wiring gallery and the world-model grid with task clips.
   Plain JavaScript, no dependencies. The theme follows the reader's system setting. */
(function () {
  'use strict';

  var root = document.documentElement;
  root.classList.add('js');
  var REDUCE = !!(window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches);
  var SVGNS = 'http://www.w3.org/2000/svg';

  function $(id) { return document.getElementById(id); }
  function el(tag, attrs, parent) {
    var e = document.createElementNS(SVGNS, tag);
    if (attrs) for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function txt(parent, x, y, s, cls, extra) {
    var t = el('text', {x: x, y: y, 'class': cls || ''}, parent);
    if (extra) for (var k in extra) t.setAttribute(k, extra[k]);
    t.textContent = s;
    return t;
  }
  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }
  function fmtInt(n) { return Math.round(n).toLocaleString('en-US'); }
  function bytes(b64) {
    var s = atob(b64), out = new Uint8Array(s.length);
    for (var i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
    return out;
  }
  function hexRgb(h) {
    h = h.replace('#', '');
    return [parseInt(h.substr(0, 2), 16), parseInt(h.substr(2, 2), 16), parseInt(h.substr(4, 2), 16)];
  }
  function rgba(h, a) { var c = hexRgb(h); return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + a + ')'; }
  function token(name) { return getComputedStyle(root).getPropertyValue(name).trim(); }

  // ------------------------------------------------------------------ theme: the colours the canvases need, read from CSS
  var PAL = null, themeListeners = [];
  function readPal() {
    PAL = {cube: token('--v3-cube'), out: token('--v3-out'), outline: token('--v3-outline'), dim: token('--v3-dim'),
           edgeDim: token('--v3-edge-dim'), gain: parseFloat(token('--v3-gain')) || 1, track: token('--v3-track'),
           lewi: token('--lewi')};
  }
  readPal();
  function onTheme(fn) { themeListeners.push(fn); }
  function themeChanged() { readPal(); themeListeners.forEach(function (fn) { safe(fn); }); }
  if (window.matchMedia) {
    var mqDark = matchMedia('(prefers-color-scheme: dark)');
    if (mqDark.addEventListener) mqDark.addEventListener('change', themeChanged); else if (mqDark.addListener) mqDark.addListener(themeChanged);
  }

  function sup(n) {
    var d = '⁰¹²³⁴⁵⁶⁷⁸⁹';
    return String(n).split('').map(function (c) { return d[+c]; }).join('');
  }
  function onVisible(node, cb) {
    if (!('IntersectionObserver' in window)) { cb(true); return; }
    new IntersectionObserver(function (es) { es.forEach(function (e) { cb(e.isIntersecting); }); },
                             {rootMargin: '80px'}).observe(node);
  }

  // ------------------------------------------------------------------ tooltip
  var tip = $('tip');
  function showTip(html, x, y) { tip.innerHTML = html; tip.classList.add('on'); placeTip(x, y); }
  function placeTip(x, y) {
    var w = tip.offsetWidth, h = tip.offsetHeight, left = x + 16, top = y + 16;
    if (left + w > innerWidth - 8) left = x - w - 16;
    if (top + h > innerHeight - 8) top = y - h - 16;
    tip.style.left = Math.max(8, left) + 'px';
    tip.style.top = Math.max(8, top) + 'px';
  }
  function hideTip() { tip.classList.remove('on'); }

  // ------------------------------------------------------------------ a recorded network
  // positions (int16, scaled), the association of every neuron with a target node (-1: none) and the binary
  // mask of every layer transition (one bit per connection; rows = receiving neurons), frame by frame
  function Net(run) {
    var c = run.counts, L = c.length, U = 0;
    this.run = run; this.counts = c; this.L = L;
    this.start = []; this.level = [];
    for (var l = 0; l < L; l++) {
      this.start.push(U);
      for (var q = 0; q < c[l]; q++) this.level.push(l);
      U += c[l];
    }
    this.U = U;
    this.frames = run.frames || 1;
    this.xyz = new Int16Array(bytes(run.xyz).buffer);
    this.assoc = new Int8Array(bytes(run.assoc).buffer);
    this.bits = bytes(run.masks);
    this.boff = [];
    var P = 0;
    for (var b = 0; b < L - 1; b++) { this.boff.push(P); P += c[b + 1] * c[b]; }
    this.P = P;
    this.scale = run.xyz_scale;
    this.colors = run.teacher.colors;
    this.cache = {};
    // marker per level: inputs a circle, hidden layers square / triangle / diamond / inverted triangle (hollow
    // from the fifth hidden layer on), the output a star
    this.shape = []; this.hollow = [];
    for (var lv = 0; lv < L; lv++) {
      var k4 = (lv - 1) % 4;
      this.shape.push(lv === 0 ? 'o' : lv === L - 1 ? '*' : 's^Dv'[k4]);
      this.hollow.push(lv > 4 && lv < L - 1);
    }
  }
  Net.prototype.edges = function (f) {
    if (this.cache[f]) return this.cache[f];
    var out = [], c = this.counts, bits = this.bits, base = f * this.P;
    for (var b = 0; b < this.L - 1; b++) {
      var R = c[b + 1], S = c[b], off = base + this.boff[b], s0 = this.start[b], r0 = this.start[b + 1];
      for (var r = 0; r < R; r++) {
        for (var s = 0; s < S; s++) {
          var i = off + r * S + s;
          if ((bits[i >> 3] >> (7 - (i & 7))) & 1) out.push(s0 + s, r0 + r);
        }
      }
    }
    var arr = new Int32Array(out);
    this.cache[f] = arr;
    return arr;
  };
  Net.prototype.activeShare = function (f) { return this.edges(f).length / 2 / this.P; };
  // the farthest neuron from the centre and the largest coordinate, over every frame
  Net.prototype.bounds = function () {
    if (this.b) return this.b;
    var xyz = this.xyz, k = this.scale, reach = 0, ext = 0;
    for (var i = 0; i < xyz.length; i += 3) {
      var x = xyz[i] * k, y = xyz[i + 1] * k, z = xyz[i + 2] * k;
      reach = Math.max(reach, Math.sqrt(x * x + y * y + z * z));
      ext = Math.max(ext, Math.abs(x), Math.abs(y), Math.abs(z));
    }
    return (this.b = {reach: reach, ext: ext});
  };

  function marker(ctx, shape, x, y, r) {
    ctx.beginPath();
    if (shape === 'o') ctx.arc(x, y, r, 0, 6.2832);
    else if (shape === 's') ctx.rect(x - r * 0.86, y - r * 0.86, r * 1.72, r * 1.72);
    else if (shape === '^') { ctx.moveTo(x, y - r * 1.15); ctx.lineTo(x + r * 1.05, y + r * 0.8); ctx.lineTo(x - r * 1.05, y + r * 0.8); ctx.closePath(); }
    else if (shape === 'v') { ctx.moveTo(x, y + r * 1.15); ctx.lineTo(x + r * 1.05, y - r * 0.8); ctx.lineTo(x - r * 1.05, y - r * 0.8); ctx.closePath(); }
    else if (shape === 'D') { ctx.moveTo(x, y - r * 1.2); ctx.lineTo(x + r * 0.95, y); ctx.lineTo(x, y + r * 1.2); ctx.lineTo(x - r * 0.95, y); ctx.closePath(); }
    else {
      for (var i = 0; i < 10; i++) {
        var a = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? r * 0.8 : r * 1.9;
        if (i) ctx.lineTo(x + rr * Math.cos(a), y + rr * Math.sin(a)); else ctx.moveTo(x + rr * Math.cos(a), y + rr * Math.sin(a));
      }
      ctx.closePath();
    }
  }

  // ------------------------------------------------------------------ 3-D view on a canvas
  function Viewer(canvas, opts) {
    var self = this;
    this.cv = canvas; this.ctx = canvas.getContext('2d');
    this.yaw = opts.yaw; this.pitch = opts.pitch; this.zoom = opts.zoom || 1;
    this.spin = opts.spin || 0;
    this.sway = opts.sway || null; this.baseYaw = opts.yaw; this.phase = 0;
    this.fill = opts.fill || 0.36;
    this.drag = null; this.lastUser = -1e9; this.dirty = true;
    this.order = null; this.P = null;
    canvas.addEventListener('pointerdown', function (e) {
      if (e.button !== undefined && e.button !== 0) return;
      self.drag = {x: e.clientX, y: e.clientY, yaw: self.yaw, pitch: self.pitch};
      try { canvas.setPointerCapture(e.pointerId); } catch (err) { /* older browsers */ }
      canvas.classList.add('dragging');
      if (opts.onInteract) opts.onInteract();
    });
    canvas.addEventListener('pointermove', function (e) {
      if (!self.drag) return;
      self.yaw = self.drag.yaw + (e.clientX - self.drag.x) * 0.0085;
      self.pitch = clamp(self.drag.pitch + (e.clientY - self.drag.y) * 0.0065, -1.25, 1.25);
      self.lastUser = performance.now();
      self.turned = true;
      self.dirty = true;
    });
    function end() {
      if (!self.drag) return;
      self.drag = null;
      if (self.sway) self.baseYaw = self.yaw - self.sway.amp * Math.sin(self.phase);
      canvas.classList.remove('dragging');
      self.lastUser = performance.now();
    }
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', end);
    canvas.addEventListener('lostpointercapture', end);
  }
  Viewer.prototype.resize = function () {
    var r = this.cv.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
    var w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (w !== this.cv.width || h !== this.cv.height) { this.cv.width = w; this.cv.height = h; this.dirty = true; }
    this.dpr = dpr; this.W = r.width; this.H = r.height;
  };
  // a slow sway about the opening view keeps the structure readable; a free spin is the other option
  Viewer.prototype.tick = function (dt, now) {
    if (this.drag || now - this.lastUser < 2600) return;
    if (this.sway) { this.phase += dt * this.sway.rate; this.yaw = this.baseYaw + this.sway.amp * Math.sin(this.phase); this.dirty = true; }
    else if (this.spin) { this.yaw += this.spin * dt; this.dirty = true; }
  };
  Viewer.prototype.setView = function (yaw, pitch) { this.yaw = this.baseYaw = yaw; this.pitch = pitch; this.phase = 0; this.turned = false; this.dirty = true; };
  // The fitted scale (st.fit) keeps every neuron, and the cube when drawn, inside the canvas. While the view follows its
  // own sway it uses the largest extent over the sway range (every frame, each axis); once the reader has turned the
  // view it uses the bound for any angle: under perspective a point at distance R projects at most R / sqrt(1 - (R/depth)^2)
  // from the centre. The scale eases from one to the other.
  Viewer.prototype.fitTarget = function (net, st, W, H, C, depth) {
    var pad = st.pad || 12, rx = W / 2 - Math.abs(st.dx || 0) * W - pad, ry = H / 2 - Math.abs(st.dy || 0) * H - pad;
    var cube = st.cube !== false;
    if (this.turned || !this.sway) {
      var R = cube ? Math.sqrt(3) * C : net.bounds().reach, rho = Math.min(0.9, R / depth);
      return Math.max(1, Math.min(rx, ry)) * Math.sqrt(1 - rho * rho) / R;
    }
    var e = this.ext;
    if (!e || e.net !== net || e.base !== this.baseYaw || e.pitch !== this.pitch || e.cube !== cube || e.C !== C) {
      e = this.ext = {net: net, base: this.baseYaw, pitch: this.pitch, cube: cube, C: C, x: 1e-6, y: 1e-6};
      var cp = Math.cos(this.pitch), sp = Math.sin(this.pitch), xyz = net.xyz, k = net.scale, U = net.U, F = net.frames;
      var fs = F > 100 ? 2 : 1, corners = [];
      for (var i = 0; i < 8; i++) corners.push([i & 1 ? C : -C, i & 2 ? C : -C, i & 4 ? C : -C]);
      for (var yi = 0; yi <= 16; yi++) {
        var yaw = this.baseYaw + this.sway.amp * (yi / 8 - 1), cy = Math.cos(yaw), sy = Math.sin(yaw);
        var reach = function (x, y, z) {
          var x1 = cy * x + sy * z, z1 = -sy * x + cy * z, y1 = cp * y - sp * z1, g = 1 / (1 - (sp * y + cp * z1) / depth);
          e.x = Math.max(e.x, Math.abs(x1 * g)); e.y = Math.max(e.y, Math.abs(y1 * g));
        };
        if (cube) corners.forEach(function (q) { reach(q[0], q[1], q[2]); });
        for (var f = 0; f < F; f += fs) {
          for (var u = 0; u < U; u++) { var j = (f * U + u) * 3; reach(xyz[j] * k, xyz[j + 1] * k, xyz[j + 2] * k); }
        }
      }
    }
    return Math.max(1, Math.min(rx / e.x, ry / e.y));
  };
  // positions are interpolated between frames fa and fb (weight t); masks and colors come from frame fm
  Viewer.prototype.draw = function (net, fa, fb, t, fm, st) {
    var ctx = this.ctx, W = this.W, H = this.H, U = net.U, xyz = net.xyz, k = net.scale;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    var cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    var lim = net.run.lim3, s = Math.min(W, H) * this.fill / lim * this.zoom, depth = lim * 7, C = lim * 0.92;
    // fitted: the scale that keeps every neuron (and the cube, when drawn) inside the canvas at any rotation; under
    // perspective a point at distance R from the centre projects at most R / sqrt(1 - (R / depth)^2) from it
    if (st.fit) {
      if (st.cube !== false) C = net.bounds().ext * 1.04;
      var target = this.fitTarget(net, st, W, H, C, depth);
      this.fitS = this.fitNet === net && this.fitS ? this.fitS + (target - this.fitS) * 0.12 : target;
      this.fitNet = net;
      s = this.fitS * this.zoom;
    }
    var ox = W / 2 + (st.dx || 0) * W, oy = H / 2 + (st.dy || 0) * H;
    if (!this.P || this.P.length !== U * 3) { this.P = new Float32Array(U * 3); this.order = new Array(U); }
    var P = this.P, u, i;
    for (u = 0; u < U; u++) {
      var ia = (fa * U + u) * 3, ib = (fb * U + u) * 3;
      var x = (xyz[ia] + (xyz[ib] - xyz[ia]) * t) * k;
      var y = (xyz[ia + 1] + (xyz[ib + 1] - xyz[ia + 1]) * t) * k;
      var z = (xyz[ia + 2] + (xyz[ib + 2] - xyz[ia + 2]) * t) * k;
      var x1 = cy * x + sy * z, z1 = -sy * x + cy * z;
      var y1 = cp * y - sp * z1, z2 = sp * y + cp * z1;
      var g = 1 / (1 - z2 / depth);
      P[u * 3] = ox + x1 * s * g; P[u * 3 + 1] = oy - y1 * s * g; P[u * 3 + 2] = z2;
    }
    // the cube the embedding lives in, drawn faintly for depth (the hero leaves it out)
    var corners = [];
    for (i = 0; i < 8; i++) {
      var cx = i & 1 ? C : -C, cyy = i & 2 ? C : -C, cz = i & 4 ? C : -C;
      var a1 = cy * cx + sy * cz, b1 = -sy * cx + cy * cz, a2 = cp * cyy - sp * b1, b2 = sp * cyy + cp * b1, gg = 1 / (1 - b2 / depth);
      corners.push([ox + a1 * s * gg, oy - a2 * s * gg]);
    }
    if (st.cube !== false) {
      ctx.strokeStyle = PAL.cube;
      ctx.lineWidth = 1;
      ctx.beginPath();
      [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]].forEach(function (e) {
        ctx.moveTo(corners[e[0]][0], corners[e[0]][1]); ctx.lineTo(corners[e[1]][0], corners[e[1]][1]);
      });
      ctx.stroke();
    }
    // active connections, grouped by the colour of the receiving neuron
    var E = net.edges(fm), groups = {}, assoc = net.assoc, base = fm * U, key;
    var only = st.only === undefined ? null : st.only;
    for (i = 0; i < E.length; i += 2) {
      if (only !== null && net.level[E[i]] !== only) continue;
      var node = assoc[base + E[i + 1]];
      key = node >= 0 ? node : -1;
      (groups[key] || (groups[key] = [])).push(E[i], E[i + 1]);
    }
    ctx.lineWidth = st.edgeWidth;
    ctx.lineCap = 'round';
    for (key in groups) {
      var kk = +key, list = groups[key];
      ctx.strokeStyle = kk >= 0 ? rgba(net.colors[kk], Math.min(0.9, st.edgeAlpha * PAL.gain)) : 'rgba(' + PAL.edgeDim + ',' + st.edgeAlphaDim + ')';
      ctx.beginPath();
      for (var j = 0; j < list.length; j += 2) {
        var p = list[j] * 3, q = list[j + 1] * 3;
        ctx.moveTo(P[p], P[p + 1]); ctx.lineTo(P[q], P[q + 1]);
      }
      ctx.stroke();
    }
    // neurons, far ones first
    var order = this.order;
    for (u = 0; u < U; u++) order[u] = u;
    order.sort(function (a, b) { return P[a * 3 + 2] - P[b * 3 + 2]; });
    var L = net.L, rad = st.radius;
    ctx.lineWidth = 0.9;
    for (i = 0; i < U; i++) {
      u = order[i];
      var lv = net.level[u], z2v = P[u * 3 + 2], gz = 1 / (1 - z2v / depth);
      var nd = assoc[base + u], col, alpha;
      if (lv === L - 1) { col = PAL.out; alpha = 1; }
      else if (nd >= 0) { col = net.colors[nd]; alpha = 1; }
      else { col = PAL.dim; alpha = st.dimAlpha; }
      var r = rad * gz * (lv === 0 ? 1.05 : 1);
      if (only !== null && lv !== only && lv !== only + 1) alpha *= st.otherAlpha;
      ctx.globalAlpha = alpha * clamp(0.72 + z2v / lim * 0.2, 0.5, 1);
      marker(ctx, net.shape[lv], P[u * 3], P[u * 3 + 1], r);
      if (net.hollow[lv]) { ctx.strokeStyle = col; ctx.lineWidth = 1.2; ctx.stroke(); ctx.lineWidth = 0.9; }
      else { ctx.fillStyle = col; ctx.fill(); ctx.strokeStyle = PAL.outline; ctx.stroke(); }
    }
    ctx.globalAlpha = 1;
    this.dirty = false;
  };

  // ------------------------------------------------------------------ a target graph on a night plate
  function targetGraph(svg, t, opt) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var leaves = t.leaves, N = t.parents.length, maxLv = Math.max.apply(null, t.level);
    var xs = t.leaf_x.concat(t.node_x), minX = Math.min.apply(null, xs), maxX = Math.max.apply(null, xs);
    var outX = t.readout.reduce(function (a, r) { return a + t.node_x[r]; }, 0) / t.readout.length;
    var W = opt.width || 140, pad = opt.pad || 7, lh = opt.levelH || 22, top = maxLv + 1;
    var H = top * lh + 2 * pad, sx = (W - 2 * pad) / Math.max(1, maxX - minX);
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    function X(x) { return pad + (x - minX) * sx; }
    function Y(l) { return H - pad - l * lh; }
    function at(ref) { return ref < leaves ? [X(t.leaf_x[ref]), Y(0)] : [X(t.node_x[ref - leaves]), Y(t.level[ref - leaves])]; }
    var gE = el('g', {'class': 'tg-e', 'stroke-width': opt.edge || 1}, svg);
    for (var v = 0; v < N; v++) {
      t.parents[v].forEach(function (ref) {
        var a = at(ref);
        el('line', {x1: a[0], y1: a[1], x2: X(t.node_x[v]), y2: Y(t.level[v])}, gE);
      });
    }
    t.readout.forEach(function (r) { el('line', {x1: X(t.node_x[r]), y1: Y(t.level[r]), x2: X(outX), y2: Y(top)}, gE); });
    var lr = opt.leafR || 2.3, nr = opt.nodeR || 4.8;
    for (var i = 0; i < leaves; i++) {
      el('circle', {cx: X(t.leaf_x[i]), cy: Y(0), r: lr, fill: t.colors[t.leaf_owner[i]], opacity: 0.9}, svg);
    }
    for (v = 0; v < N; v++) {
      el('circle', {cx: X(t.node_x[v]), cy: Y(t.level[v]), r: nr, fill: t.colors[v], 'class': 'tg-n'}, svg);
    }
    el('circle', {cx: X(outX), cy: Y(top), r: nr, 'class': 'tg-out'}, svg);
  }

  // ------------------------------------------------------------------ hero: real training runs, replayed one after another
  // The depth-4 tree comes with the page (hero.js); the other gallery runs load one at a time while the hero plays
  // (teaser-<run>.js). Each run plays from the first step to the last, holds, and fades into the next.
  function hero() {
    var first = window.LEWI_HERO, cv = $('hero-net'), plate = document.querySelector('.hv-target');
    if (!first || !cv) return;
    var ORDER = ['tree', 'dag', 'rand', 'flat', 'dense', 'tree5'], runs = {tree: first}, nets = {}, asked = {};
    var tag = document.querySelector('script[src*="assets/data/hero.js"]'), ver = tag ? (tag.src.split('?')[1] || '') : '';
    var view = new Viewer(cv, {yaw: first.view.yaw, pitch: first.view.pitch, sway: REDUCE ? null : {amp: 0.6, rate: 0.2}, fill: 0.42});
    var style = {edgeWidth: 0.7, edgeAlpha: 0.24, edgeAlphaDim: 0.07, radius: 3.5, dimAlpha: 0.55, cube: false, fit: true, pad: 10, dx: -0.02, dy: 0.03};
    var PLAY = 17000, HOLD = 4500, FADE = 450, GAMMA = 1.8;
    var k = 0, run = null, net = null, st = {playing: !REDUCE, phase: 'play', t: 0, step: 0};
    function load(stem) {
      if (runs[stem] || asked[stem]) return;
      asked[stem] = true;
      var s = document.createElement('script');
      s.src = 'assets/data/teaser-' + stem + '.js' + (ver ? '?' + ver : '');
      s.onload = function () { if (window.LEWI_TEASER && window.LEWI_TEASER[stem]) runs[stem] = window.LEWI_TEASER[stem]; };
      document.head.appendChild(s);
    }
    function use(i) {
      k = i;
      run = runs[ORDER[k]];
      net = nets[ORDER[k]] || (nets[ORDER[k]] = new Net(run));
      var big = run.counts[1] > 32;
      style.radius = big ? 2.2 : 3.5;
      style.edgeAlpha = big ? 0.07 : 0.24;
      style.edgeAlphaDim = big ? 0.025 : 0.07;
      if ($('hero-target')) targetGraph($('hero-target'), run.teacher, big ? {width: 132, levelH: 15, nodeR: 3.3, leafR: 1.5, pad: 5} : {width: 132, levelH: 21});
      view.setView(run.view.yaw, run.view.pitch);
      load(ORDER[(k + 1) % ORDER.length]);
    }
    // the frames around a training step (the runs' snapshots need not be evenly spaced)
    function frameOf(step) {
      var s = run.steps, lo = 0, hi = s.length - 1;
      if (step <= s[0]) return [0, 0, 0];
      if (step >= s[hi]) return [hi, hi, 0];
      while (hi - lo > 1) { var mid = (lo + hi) >> 1; if (s[mid] <= step) lo = mid; else hi = mid; }
      return [lo, hi, (step - s[lo]) / (s[hi] - s[lo])];
    }
    function render() {
      view.resize();
      var fr = frameOf(st.step);
      view.draw(net, fr[0], fr[1], fr[2], Math.round(fr[0] + fr[2]), style);
    }
    function fade(a) { cv.style.opacity = a; if (plate) plate.style.opacity = a; }
    use(0);
    if (REDUCE) st.step = run.best_step;
    var visible = true, last = performance.now();
    onVisible(cv, function (v) { visible = v; });
    onTheme(function () { view.dirty = true; render(); });
    function loop(now) {
      var dt = Math.min(now - last, 80);
      last = now;
      if (visible && !document.hidden) {
        if (st.playing) {
          st.t += dt;
          if (st.phase === 'play') {
            st.step = run.total_steps * Math.pow(clamp(st.t / PLAY, 0, 1), GAMMA);
            fade(st.t < FADE ? String(st.t / FADE) : '');
            if (st.t >= PLAY) { st.phase = 'hold'; st.t = 0; }
          } else if (st.phase === 'hold') {
            if (st.t >= HOLD) { st.phase = 'fade'; st.t = 0; }
          } else {
            fade(String(Math.max(0, 1 - st.t / FADE)));
            if (st.t >= FADE) {
              var next = (k + 1) % ORDER.length;
              if (runs[ORDER[next]]) use(next);      // a run still loading is skipped for one turn
              st.phase = 'play'; st.t = 0; st.step = 0;
            }
          }
        }
        view.tick(dt / 1000, now);
        render();
      }
      requestAnimationFrame(loop);
    }
    render();
    requestAnimationFrame(loop);
  }
  // ------------------------------------------------------------------ helpers shared by the new figures
  function sub(parent, x, y, base, index, cls, extra) {
    var t = txt(parent, x, y, base, cls, extra);
    var s = el('tspan', {'baseline-shift': 'sub', 'font-size': '0.68em'}, t);
    s.textContent = index;
    return t;
  }
  // A two-state figure that follows the scroll position: its second state shows once the figure's centre has risen above
  // `enter` of the viewport height, and the first returns when the centre sinks below `leave` again (scrolling back up).
  // Only crossings act, so a click on the figure's own switch holds until the next crossing.
  function scrolly(node, enter, leave, onIn, onOut) {
    var state = null, ticking = false;
    function check() {
      ticking = false;
      var r = node.getBoundingClientRect(), vh = window.innerHeight || document.documentElement.clientHeight;
      var c = (r.top + r.height / 2) / vh;
      if (state !== 'in' && c < enter) { state = 'in'; onIn(); }
      else if (state !== 'out' && c > leave) { state = 'out'; onOut(); }
    }
    function later() { if (!ticking) { ticking = true; requestAnimationFrame(check); } }
    window.addEventListener('scroll', later, {passive: true});
    window.addEventListener('resize', later);
    check();
  }
  function aulc(points) {
    var s = 0;
    for (var i = 1; i < points.length; i++) {
      s += (Math.log2(points[i].n) - Math.log2(points[i - 1].n)) * (points[i].mean + points[i - 1].mean) / 2;
    }
    return s / (Math.log2(points[points.length - 1].n) - Math.log2(points[0].n));
  }
  function armOf(panel, test) { return panel.arms.filter(test)[0]; }

  // ------------------------------------------------------------------ the background sheet: its two tints drift with the scroll
  function sheet() {
    var WARM = [[242, 107, 29], [204, 121, 167], [230, 159, 0], [242, 107, 29]];
    var COOL = [[0, 114, 178], [0, 158, 115], [0, 114, 178], [0, 140, 150]];
    var ticking = false;
    function mix(stops, p) {
      var x = p * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x)), t = x - i, a = stops[i], b = stops[i + 1];
      return [0, 1, 2].map(function (k) { return Math.round(a[k] + (b[k] - a[k]) * t); });
    }
    function paint() {
      ticking = false;
      var d = document.documentElement, max = Math.max(1, d.scrollHeight - window.innerHeight);
      var p = clamp(window.scrollY / max, 0, 1), aw = token('--sheet-alpha-w') || '0.1', ac = token('--sheet-alpha-c') || '0.085';
      var w = mix(WARM, p), c = mix(COOL, p);
      root.style.setProperty('--sheet-w', 'rgba(' + w.join(',') + ',' + aw + ')');
      root.style.setProperty('--sheet-c', 'rgba(' + c.join(',') + ',' + ac + ')');
    }
    function later() { if (!ticking) { ticking = true; requestAnimationFrame(paint); } }
    window.addEventListener('scroll', later, {passive: true});
    window.addEventListener('resize', later);
    onTheme(paint);
    paint();
  }

  // ------------------------------------------------------------------ motivation: one function of 8 inputs, or 7 pieces of 2
  function piecesFig() {
    var D = window.LEWI_PIECES, svg = $('pieces-svg');
    if (!D || !svg) return;
    var W = 800, H = 384, X_IN = 64, X_OUT = 752, Y0 = 64, PITCH = 40, TILE = 62;
    var FX0 = 150, FX1 = 676, FY0 = 34, FY1 = 374, XL = {1: 278, 2: 428, 3: 578};
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    function iy(k) { return Y0 + k * PITCH; }
    var YC = iy(3.5);
    var gW = el('g', {'class': 'pc-g pc-whole'}, svg), gP = el('g', {'class': 'pc-g pc-pieces'}, svg);
    for (var k = 0; k < 8; k++) el('line', {x1: X_IN + 7, y1: iy(k), x2: FX0, y2: iy(k), 'class': 'pc-wire'}, gW);
    el('rect', {x: FX0, y: FY0, width: FX1 - FX0, height: FY1 - FY0, rx: 14, 'class': 'pc-box'}, gW);
    txt(gW, (FX0 + FX1) / 2, YC + 10, 'f(x₁, …, x₈)', 'pc-f', {'text-anchor': 'middle'});
    el('line', {x1: FX1, y1: YC, x2: X_OUT - 8, y2: YC, 'class': 'pc-wire'}, gW);
    el('rect', {x: FX0, y: FY0, width: FX1 - FX0, height: FY1 - FY0, rx: 14, 'class': 'pc-frame'}, gP);
    [1, 2, 3].forEach(function (lv) { txt(gP, XL[lv], FY0 - 12, 'Level ' + lv, 'pc-lvl', {'text-anchor': 'middle'}); });
    var centre = D.node_x.map(function (nx, v) { return [XL[D.level[v]], iy(nx)]; });
    function slot(r) { return r < D.A ? D.leaf_slot[r] : D.node_x[r - D.A]; }
    D.parents.forEach(function (refs, v) {
      var c = centre[v];
      refs.slice().sort(function (a, b) { return slot(a) - slot(b); }).forEach(function (r, j) {
        var ty = c[1] + (j === 0 ? -TILE / 4 : TILE / 4), sx, sy;
        if (r < D.A) { sx = X_IN + 7; sy = iy(D.leaf_slot[r]); }
        else { sx = centre[r - D.A][0] + TILE / 2; sy = centre[r - D.A][1]; }
        el('line', {x1: sx, y1: sy, x2: c[0] - TILE / 2, y2: ty, 'class': 'pc-wire'}, gP);
      });
    });
    D.level.forEach(function (lv, v) {
      var c = centre[v];
      el('image', {href: 'assets/img/piece-' + v + '.png', x: c[0] - TILE / 2, y: c[1] - TILE / 2, width: TILE, height: TILE, preserveAspectRatio: 'none'}, gP);
      el('rect', {x: c[0] - TILE / 2, y: c[1] - TILE / 2, width: TILE, height: TILE, fill: 'none', stroke: D.colors[v], 'stroke-width': 3}, gP);
    });
    var root = centre[D.readout[0]];
    el('line', {x1: root[0] + TILE / 2, y1: root[1], x2: X_OUT - 8, y2: YC, 'class': 'pc-wire'}, gP);
    for (k = 0; k < 8; k++) {
      el('circle', {cx: X_IN, cy: iy(k), r: 7, 'class': 'pc-in'}, svg);
      sub(svg, X_IN - 16, iy(k) + 5, 'x', String(k + 1), 'pc-lab', {'text-anchor': 'end'});
    }
    el('circle', {cx: X_OUT, cy: YC, r: 8, 'class': 'pc-out'}, svg);
    txt(svg, X_OUT + 16, YC + 7, 'y', 'pc-lab', {'font-size': '20px'});
    var buttons = document.querySelectorAll('.pieces .seg-b'), counts = document.querySelectorAll('.counts .count');
    function show(view) {
      svg.classList.toggle('show-pieces', view === 'pieces');
      buttons.forEach(function (b) { b.setAttribute('aria-selected', String(b.getAttribute('data-view') === view)); });
      counts.forEach(function (c) { c.classList.toggle('on', c.getAttribute('data-view') === view); });
    }
    buttons.forEach(function (b) { b.addEventListener('click', function () { show(b.getAttribute('data-view')); }); });
    show('whole');
    scrolly(svg, 0.5, 0.64, function () { show('pieces'); }, function () { show('whole'); });
  }

  // ------------------------------------------------------------------ the gap: Dense and baselines against GraphOracle, then LeWi
  function gapFig() {
    var P = (window.LEWI_CURVES || [])[0], svg = $('gap-svg');
    if (!P || !svg) return;
    var W = 860, H = 440, x0 = 80, x1 = 836, y0 = 14, y1 = 372;
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    var ns = P.window, lx0 = Math.log2(ns[0]) - 0.2, lx1 = Math.log2(ns[ns.length - 1]) + 0.2;
    function X(n) { return x0 + (Math.log2(n) - lx0) / (lx1 - lx0) * (x1 - x0); }
    function Y(v) { return y1 - v * (y1 - y0); }
    [0, 0.25, 0.5, 0.75, 1].forEach(function (v) {
      el('line', {x1: x0, x2: x1, y1: Y(v), y2: Y(v), 'class': v ? 'gp-g' : 'gp-a'}, svg);
      txt(svg, x0 - 10, Y(v) + 4, v ? v.toFixed(2) : '0', 'gp-t', {'text-anchor': 'end'});
    });
    ns.forEach(function (n) {
      txt(svg, X(n), y1 + 22, n >= 1024 ? (n / 1024) + 'k' : String(n), 'gp-t', {'text-anchor': 'middle'});
    });
    txt(svg, (x0 + x1) / 2, y1 + 52, 'Training samples', 'gp-axt', {'text-anchor': 'middle'});
    txt(svg, 18, (y0 + y1) / 2, 'Test error (NMSE)', 'gp-axt', {'text-anchor': 'middle', transform: 'rotate(-90 18 ' + (y0 + y1) / 2 + ')'});
    var dense = armOf(P, function (a) { return a.kind === 'dense'; });
    var lewi = armOf(P, function (a) { return a.kind === 'lewi'; });
    var oracle = armOf(P, function (a) { return a.key === 'graph_oracle(levels=all)'; });
    var bases = P.arms.filter(function (a) { return a.kind === 'baseline'; });
    function path(a) { return a.points.map(function (q, i) { return (i ? 'L' : 'M') + X(q.n).toFixed(1) + ' ' + Y(q.mean).toFixed(1); }).join(''); }
    var fill = dense.points.map(function (q) { return X(q.n).toFixed(1) + ' ' + Y(q.mean).toFixed(1); })
      .concat(oracle.points.slice().reverse().map(function (q) { return X(q.n).toFixed(1) + ' ' + Y(q.mean).toFixed(1); }));
    el('path', {d: 'M' + fill.join('L') + 'Z', 'class': 'gp-fill'}, svg);
    bases.forEach(function (a) { el('path', {d: path(a), 'class': 'gp-base'}, svg); });
    el('path', {d: path(oracle), 'class': 'gp-oracle'}, svg);
    oracle.points.forEach(function (q) { el('circle', {cx: X(q.n), cy: Y(q.mean), r: 4.5, 'class': 'gp-dot-o'}, svg); });
    el('path', {d: path(dense), 'class': 'gp-dense'}, svg);
    dense.points.forEach(function (q) { el('circle', {cx: X(q.n), cy: Y(q.mean), r: 5, 'class': 'gp-dot-d'}, svg); });
    var gL = el('g', {'class': 'gp-lewi-g'}, svg);
    var lp = el('path', {d: path(lewi), 'class': 'gp-lewi'}, gL);
    lewi.points.forEach(function (q) { el('circle', {cx: X(q.n), cy: Y(q.mean), r: 5, 'class': 'gp-dot-l'}, gL); });
    var gap = aulc(dense.points) - aulc(oracle.points);
    var shareL = (aulc(dense.points) - aulc(lewi.points)) / gap, best = null;
    bases.forEach(function (a) { var s = (aulc(dense.points) - aulc(a.points)) / gap; if (!best || s > best.s) best = {a: a, s: s}; });
    txt(gL, x0 + 30, Y(0.2), 'LeWi closes ' + Math.round(shareL * 100) + '% of the gap', 'gp-note');
    txt(gL, x0 + 30, Y(0.2) + 26, 'best baseline (' + best.a.name + '): ' + Math.round(best.s * 100) + '%', 'gp-note2');
    var lx = x1 - 220, ly = y0 + 22;
    function legend(g, y, cls, label, dash) {
      el('line', {x1: lx, x2: lx + 42, y1: y, y2: y, 'class': cls}, g);
      txt(g, lx + 54, y + 5, label, 'gp-leg');
    }
    legend(svg, ly, 'gp-dense', 'Dense');
    legend(svg, ly + 30, 'gp-base', '6 baselines');
    legend(gL, ly + 60, 'gp-lewi', 'LeWi');
    legend(svg, ly + 90, 'gp-oracle', 'GraphOracle');
    txt(svg, lx + 54, ly + 111, 'given the target graph', 'gp-leg-k');
    var len = lp.getTotalLength ? lp.getTotalLength() : 0, buttons = document.querySelectorAll('.gapbox .seg-b');
    function show(on, animate) {
      svg.classList.toggle('with-lewi', on);
      buttons.forEach(function (b) { b.setAttribute('aria-selected', String((b.getAttribute('data-lewi') === 'on') === on)); });
      if (on && animate && len && !REDUCE) {
        lp.style.transition = 'none'; lp.style.strokeDasharray = len; lp.style.strokeDashoffset = len;
        lp.getBoundingClientRect();
        lp.style.transition = 'stroke-dashoffset 1.6s ease'; lp.style.strokeDashoffset = '0';
      }
    }
    buttons.forEach(function (b) { b.addEventListener('click', function () { show(b.getAttribute('data-lewi') === 'on', true); }); });
    show(false);
    scrolly(svg, 0.5, 0.64, function () { show(true, true); }, function () { show(false); });
  }

  // ------------------------------------------------------------------ how it works: four steps, as in the talk
  function stepperFig() {
    var svg = $('stepper-svg'), box = $('stepper');
    if (!svg || !box) return;
    var W = 640, H = 451, RAD = 135;
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    svg.style.color = 'var(--ink-2)';
    var defs = el('defs', null, svg), gr = el('radialGradient', {id: 'svShade'}, defs);
    el('stop', {offset: '0%', 'stop-color': 'currentColor', 'stop-opacity': 0.2}, gr);
    el('stop', {offset: '75%', 'stop-color': 'currentColor', 'stop-opacity': 0.07}, gr);
    el('stop', {offset: '100%', 'stop-color': 'currentColor', 'stop-opacity': 0.03}, gr);
    el('rect', {x: 0, y: 0, width: W, height: H, 'class': 'sv-field'}, svg);
    var N = [[81, 60], [487, 98], [183, 182], [310, 161], [101, 254], [370, 285], [410, 397]];
    var O = [[550, 65], [525, 316], [102, 384]];
    var P0 = [282, 239], P1 = [209, 210], A = 2, C = 5, MID = 0.4;
    // the force of each linked neuron on e_i: +1 pulls, -1 pushes, with its strength; together they move e_i from P0
    // towards P1, step 3 over the first MID of the way, step 4 over the rest (where C drops out and neuron 4 joins)
    var FORCE = {2: [1, 1], 3: [-1, 0.35], 5: [-1, 1]};
    // Step 1 starts from the usual picture of two layers with every pair connected: the sending neurons (circles) in one
    // column, the receiving ones (squares, e_i among them) in the next. The neurons then move to their positions.
    function column(ps, x, y0, y1) {
      var out = [];
      ps.map(function (p, k) { return k; }).sort(function (a, b) { return ps[a][1] - ps[b][1]; })
        .forEach(function (k, r) { out[k] = [x, y0 + r * (y1 - y0) / (ps.length - 1)]; });
      return out;
    }
    var colN = column(N, 150, 50, 400), colO = column(O.concat([P0]), 490, 105, 345);
    var gR = el('g', null, svg), gDense = el('g', null, svg), gLinks = el('g', null, svg), gGone = el('g', null, svg);
    var gArrows = el('g', null, svg), gNodes = el('g', null, svg);
    var ringR = el('circle', {r: RAD, 'class': 'sv-r'}, gR);
    var gRl = el('g', null, gR), rLine = el('line', {'class': 'sv-rl'}, gRl), rLab = txt(gRl, 0, 0, 'R', 'sv-lab');
    var dense = [];
    N.forEach(function (p, i) { [0, 1, 2, 3].forEach(function (j) { dense.push({i: i, j: j, l: el('line', {'class': 'sv-dense'}, gDense)}); }); });
    var oEls = O.map(function () { return el('rect', {width: 18, height: 18, 'class': 'sv-o'}, gNodes); });
    var nodeEls = N.map(function () { return el('circle', {r: 10, 'class': 'sv-n'}, gNodes); });
    var focusRing = el('circle', {r: 26, 'class': 'sv-ring'}, gNodes);
    var eNode = el('rect', {width: 22, height: 22, 'class': 'sv-e'}, gNodes);
    var eLab = sub(gNodes, 0, 0, 'e', 'i', 'sv-lab', {'text-anchor': 'middle'});
    var links = N.map(function () { return el('line', {'class': 'sv-link'}, gLinks); });
    var gone = el('line', {'class': 'sv-gone'}, gGone);
    var forceEls = N.map(function (p, i) { return FORCE[i] ? el('path', {'class': FORCE[i][0] > 0 ? 'sv-pull' : 'sv-push'}, gArrows) : null; });
    // the labels stay where the arrows start
    var mA = [(P0[0] + N[A][0]) / 2, (P0[1] + N[A][1]) / 2];
    txt(gArrows, mA[0] - 18, mA[1] - 20, 'pull', 'sv-pull-t', {'text-anchor': 'middle'});
    txt(gArrows, N[C][0] + 6, N[C][1] + 44, 'push', 'sv-push-t', {'text-anchor': 'middle'});
    var st = {step: -1, u: 0, e: P0.slice()}, sig = function (s) { return 1 / (1 + Math.exp(-s)); }, TOP = sig(1.5);
    function ramp(u, a, b) { return clamp((u - a) / (b - a), 0, 1); }
    function ease(t) { return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; }
    function lerp(a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]; }
    function line(l, a, b) { l.setAttribute('x1', a[0]); l.setAttribute('y1', a[1]); l.setAttribute('x2', b[0]); l.setAttribute('y2', b[1]); }
    function chevrons(c, u, size, two) {
      var n = [-u[1], u[0]], d = '';
      (two ? [-5.5, 5.5] : [0]).forEach(function (off) {
        var t = [c[0] + u[0] * off, c[1] + u[1] * off];
        d += 'M' + (t[0] - u[0] * size + n[0] * size) + ' ' + (t[1] - u[1] * size + n[1] * size) + 'L' + t[0] + ' ' + t[1] +
             'L' + (t[0] - u[0] * size - n[0] * size) + ' ' + (t[1] - u[1] * size - n[1] * size);
      });
      return d;
    }
    // Each step is a function of its share u (0 to 1) of the scroll, so every scroll changes the picture and the steps
    // join without jumps: (1) the layers turn into points, (2) the radius appears and the links to the neurons inside it
    // grow, nearest first, (3) every link pulls or pushes and e_i starts to move, (4) e_i moves on and its links change.
    function draw() {
      var s = st.step, u = st.u, m = s === 0 ? ease(ramp(u, 0.03, 0.9)) : 1;
      var nP = N.map(function (p, i) { return lerp(colN[i], p, m); }), oP = O.map(function (p, j) { return lerp(colO[j], p, m); });
      var e = s === 0 ? lerp(colO[3], P0, m) : st.e, rec = oP.concat([e]);
      var fade = s === 0 ? 1 - ramp(u, 0, 0.4) : 0;
      gDense.style.display = fade > 0 ? '' : 'none';
      gDense.style.opacity = fade;
      if (fade > 0) dense.forEach(function (d) { line(d.l, nP[d.i], rec[d.j]); });
      nodeEls.forEach(function (c, i) { c.setAttribute('cx', nP[i][0]); c.setAttribute('cy', nP[i][1]); });
      oEls.forEach(function (r, j) { r.setAttribute('x', oP[j][0] - 9); r.setAttribute('y', oP[j][1] - 9); });
      eNode.setAttribute('x', e[0] - 11); eNode.setAttribute('y', e[1] - 11);
      eLab.setAttribute('x', e[0]); eLab.setAttribute('y', e[1] + 42);
      eLab.style.opacity = s === 0 ? ramp(u, 0.7, 1) : 1;
      focusRing.setAttribute('cx', e[0]); focusRing.setAttribute('cy', e[1]);
      focusRing.style.opacity = s === 0 ? ramp(u, 0.7, 1) : s === 1 ? 1 - ramp(u, 0, 0.2) : 0;
      gR.style.opacity = s === 0 ? 0 : s === 1 ? ramp(u, 0, 0.25) : 1;
      ringR.setAttribute('cx', e[0]); ringR.setAttribute('cy', e[1]);
      var ang = 2.95, rEnd = [e[0] + RAD * Math.cos(ang), e[1] + RAD * Math.sin(ang)];
      line(rLine, e, rEnd);
      rLab.setAttribute('x', (e[0] + rEnd[0]) / 2 - 4); rLab.setAttribute('y', (e[1] + rEnd[1]) / 2 - 12);
      gRl.style.opacity = s === 1 ? ramp(u, 0.1, 0.35) : s === 2 ? 1 - ramp(u, 0, 0.2) : 0;
      var near = N.map(function (p, i) { return {i: i, d: Math.hypot(p[0] - e[0], p[1] - e[1]) / RAD}; })
        .filter(function (q) { return q.d < 1; }).sort(function (a, b) { return a.d - b.d; });
      links.forEach(function (l) { l.style.display = 'none'; });
      if (s >= 1) near.forEach(function (q, k) {
        var g = s === 1 ? ramp(u, 0.15 + k * 0.25, 0.5 + k * 0.25) : 1, l = links[q.i];
        if (g <= 0) return;
        var w = (sig(1.5 * (1 - q.d * q.d)) - 0.5) / (TOP - 0.5);
        l.style.display = '';
        line(l, e, lerp(e, N[q.i], g));
        l.setAttribute('stroke-width', (2 + 4 * w).toFixed(2));
        l.setAttribute('stroke-opacity', (0.45 + 0.55 * w).toFixed(2));
      });
      gone.style.display = s === 3 && Math.hypot(N[C][0] - e[0], N[C][1] - e[1]) >= RAD ? '' : 'none';
      line(gone, e, N[C]);
      gArrows.style.opacity = s === 2 ? ramp(u, 0, 0.2) : s === 3 ? 1 - ramp(u, 0, 0.2) : 0;
      forceEls.forEach(function (p, i) {
        if (!p) return;
        var q = N[i], d = Math.hypot(q[0] - e[0], q[1] - e[1]), f = FORCE[i];
        if (d >= RAD) { p.style.display = 'none'; return; }
        p.style.display = '';
        p.setAttribute('d', chevrons(lerp(e, q, 0.5), [(q[0] - e[0]) / d * f[0], (q[1] - e[1]) / d * f[0]], f[1] > 0.5 ? 7 : 5.5, f[1] > 0.5));
      });
    }
    // The figure is pinned while the reader scrolls through a track below it: each step owns an equal stretch of the
    // track (SPAN of a viewport height), so scrolling down plays the steps and scrolling up plays them back.
    var tabs = box.querySelectorAll('.st-tab'), rows = box.querySelectorAll('.st-math li'), track = $('st-track');
    var STEPS = 4, SPAN = 0.55, pinTop = 16, dragged = false, ticking = false;
    function setStep(i) {
      if (st.step === i) return;
      st.step = i;
      eNode.style.cursor = i >= 1 ? 'grab' : '';
      tabs.forEach(function (t, k) { t.setAttribute('aria-selected', String(k === i)); });
      rows.forEach(function (r) {
        var from = +r.getAttribute('data-from');
        r.classList.toggle('later', from > i);
        r.classList.toggle('past', from < i);
      });
    }
    function layout() {
      var h = box.offsetHeight, vh = window.innerHeight || document.documentElement.clientHeight;
      pinTop = Math.max(16, (vh - h) / 2);
      box.style.top = pinTop + 'px';
      track.style.height = Math.round(h + STEPS * SPAN * vh) + 'px';
    }
    function progress() {
      var total = track.offsetHeight - box.offsetHeight;
      return total > 0 ? clamp((pinTop - track.getBoundingClientRect().top) / total, 0, 1) : 0;
    }
    function update() {
      ticking = false;
      var p = progress() * STEPS, i = Math.min(STEPS - 1, Math.floor(p));
      setStep(i);
      st.u = clamp(p - i, 0, 1);
      if (!dragged) st.e = i === 2 ? lerp(P0, P1, MID * st.u) : i === 3 ? lerp(P0, P1, MID + (1 - MID) * st.u) : P0.slice();
      draw();
    }
    function later() { dragged = false; if (!ticking) { ticking = true; requestAnimationFrame(update); } }
    window.addEventListener('scroll', later, {passive: true});
    window.addEventListener('resize', function () { layout(); later(); });
    window.addEventListener('load', function () { layout(); later(); });
    // a tab scrolls the page to the end of its step
    tabs.forEach(function (t, k) {
      t.addEventListener('click', function () {
        var total = track.offsetHeight - box.offsetHeight;
        var y = window.scrollY + track.getBoundingClientRect().top - pinTop + (k + 0.97) / STEPS * total;
        window.scrollTo({top: Math.round(y), behavior: REDUCE ? 'auto' : 'smooth'});
      });
    });
    // from step 2 on, the reader can drag the neuron e_i: the connections follow until the next scroll
    function toSvg(ev) { var pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY; return pt.matrixTransform(svg.getScreenCTM().inverse()); }
    var drag = false;
    eNode.addEventListener('pointerdown', function (ev) {
      if (st.step < 1) return;
      ev.preventDefault();
      drag = true; dragged = true;
      window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
    });
    function move(ev) { if (!drag) return; var q = toSvg(ev); st.e = [clamp(q.x, 14, W - 14), clamp(q.y, 14, H - 14)]; draw(); }
    function up() { drag = false; window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); }
    layout();
    update();
  }

  // ------------------------------------------------------------------ results: the pattern across target structures
  function patternFig() {
    var C = window.LEWI_CURVES, G = window.LEWI_GRAPHS, svg = $('pattern-svg');
    if (!C || !svg) return;
    var ROWS = [{wl: 'tree_functional_anova', g: 'Depth-4 tree', label: 'Depth-4 tree'},
                {wl: 'dag_functional_anova', g: 'Depth-3 reuse DAG', label: 'Reuse DAG'},
                {wl: 'rand_functional_anova', g: 'Sampled depth-4 DAG', label: 'Sampled DAGs'},
                {wl: 'flat_functional_anova', g: 'Flat modules', label: 'Flat modules'},
                {wl: 'dense_fourier', g: 'High-arity node', label: 'High-arity node', note: 'Dense better', oracleIsDense: true}];
    var W = 1000, top = 70, rowH = 70, H = top + ROWS.length * rowH + 64, px0 = 400, px1 = 800, XMAX = 0.7;
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    function X(v) { return px0 + v / XMAX * (px1 - px0); }
    [0, 0.2, 0.4, 0.6].forEach(function (v) {
      el('line', {x1: X(v), x2: X(v), y1: top - 20, y2: top + ROWS.length * rowH - 26, 'class': 'pt-g'}, svg);
      txt(svg, X(v), top + ROWS.length * rowH - 4, v ? v.toFixed(1) : '0', 'pt-t', {'text-anchor': 'middle'});
    });
    txt(svg, px0 - 14, top + ROWS.length * rowH - 4, '← better', 'pt-ax', {'text-anchor': 'end'});
    txt(svg, (px0 + px1) / 2, top + ROWS.length * rowH + 30, 'average test error over training-set sizes', 'pt-ax', {'text-anchor': 'middle'});
    var LEG = [['Dense', 'var(--dense)'], ['LeWi', 'var(--lewi)'], ['GraphOracle', 'var(--oracle)']], lx = px0 + 20;
    LEG.forEach(function (l, i) {
      el('circle', {cx: lx + i * 140, cy: 22, r: 8, fill: l[1]}, svg);
      txt(svg, lx + i * 140 + 16, 27, l[0], 'pt-leg');
    });
    ROWS.forEach(function (row, i) {
      var P = C.filter(function (p) { return p.key === row.wl; })[0];
      if (!P) return;
      var y = top + i * rowH + 10;
      var dense = aulc(armOf(P, function (a) { return a.kind === 'dense'; }).points);
      var lewi = aulc(armOf(P, function (a) { return a.kind === 'lewi'; }).points);
      // the high-arity target is one node that reads every input: GraphOracle's masks keep every connection there, so it
      // is the Dense network (the paper runs no separate arm)
      var oa = armOf(P, function (a) { return a.key === 'graph_oracle(levels=all)'; });
      var oracle = oa ? aulc(oa.points) : row.oracleIsDense ? dense : null;
      var g = (G || []).filter(function (x) { return x.title === row.g; })[0];
      if (g) icon(g, 10, y - 24, 110, 48);
      txt(svg, 140, y + 6, row.label, 'pt-row');
      if (oracle !== null && X(dense) - X(oracle) >= 1) el('rect', {x: X(oracle), y: y - 6, width: X(dense) - X(oracle), height: 12, rx: 6, 'class': 'pt-band'}, svg);
      var dots = [[dense, 'var(--dense)', 9, 'Dense'], [lewi, 'var(--lewi)', 10, 'LeWi']];
      if (oracle !== null) dots.unshift([oracle, 'var(--oracle)', 8, 'GraphOracle', row.oracleIsDense ? 'Keeps every connection here, so it is Dense.' : '']);
      // dots that would overlap are spread over the row's height: GraphOracle above, Dense below
      var tight = dots.some(function (a, k) { return dots.some(function (b, l) { return k < l && Math.abs(X(a[0]) - X(b[0])) < 20; }); });
      var DY = {GraphOracle: -18, LeWi: 0, Dense: 19};
      dots.forEach(function (d) {
        var c = el('circle', {cx: X(d[0]), cy: y + (tight ? DY[d[3]] : 0), r: d[2], fill: d[1], stroke: 'var(--paper-2)', 'stroke-width': 2}, svg);
        c.addEventListener('pointermove', function (ev) { showTip('<div class="tt-h">' + row.label + '</div><div>' + d[3] + ': ' + d[0].toFixed(3) + '</div>' + (d[4] ? '<div class="tt-dim">' + d[4] + '</div>' : ''), ev.clientX, ev.clientY); });
        c.addEventListener('pointerleave', hideTip);
      });
      if (row.note) txt(svg, 830, y + 6, row.note, row.note === 'Dense better' ? 'pt-note' : 'pt-vs');
      else {
        var t = txt(svg, 830, y + 7, '−' + Math.round((1 - lewi / dense) * 100) + '%', 'pt-pct');
        txt(svg, 830 + 62, y + 6, 'vs Dense', 'pt-vs');
      }
    });
    function icon(g, x, y, w, h) {
      var xs = g.nodes.map(function (n) { return n.y; }), ys = g.nodes.map(function (n) { return n.x; });
      var mx = Math.max.apply(null, xs), y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
      function PX(n) { return x + n.y / (mx || 1) * w; }
      function PY(n) { return y + (n.x - y0) / ((y1 - y0) || 1) * h; }
      var gI = el('g', null, svg);
      g.edges.forEach(function (e) { var a = g.nodes[e[0]], b = g.nodes[e[1]]; el('line', {x1: PX(a), y1: PY(a), x2: PX(b), y2: PY(b), 'class': 'pt-ic-e'}, gI); });
      g.nodes.forEach(function (n) {
        el('circle', {cx: PX(n), cy: PY(n), r: n.kind === 'input' ? 2.2 : 3.6, 'class': n.kind === 'input' ? 'pt-ic-in' : n.kind === 'readout' ? 'pt-ic-out' : 'pt-ic-fn'}, gI);
      });
    }
  }
  // ------------------------------------------------------------------ CSFRBench: the hidden target, its examples, the learner
  function benchFig() {
    var run = window.LEWI_HERO, G = window.LEWI_GRAPHS;
    var tg = $('bench-target');
    if (tg && run) targetGraph(tg, run.teacher, {width: 220, levelH: 30, nodeR: 6.4, leafR: 3.2, pad: 10});
    // examples: eight inputs and the output per row (a schematic: random inputs on [-1, 1])
    var ex = $('bench-examples');
    if (ex) {
      var Wd = 236, rows = 7, cell = 17, gap = 3, seed = 7;
      function rnd() { seed = (seed * 16807) % 2147483647; return seed / 2147483647 * 2 - 1; }
      ex.setAttribute('viewBox', '0 0 ' + Wd + ' ' + (22 + rows * (cell + gap)));
      txt(ex, 4 * (cell + gap) - gap / 2, 13, 'x', 'bx-h', {'text-anchor': 'middle'});
      txt(ex, 8 * (cell + gap) + 22 + cell / 2, 13, 'y', 'bx-h', {'text-anchor': 'middle'});
      for (var r = 0; r < rows; r++) {
        var y = 22 + r * (cell + gap), sum = 0;
        for (var c = 0; c < 8; c++) {
          var v = rnd(); sum += v;
          el('rect', {x: c * (cell + gap), y: y, width: cell, height: cell, rx: 3, 'class': 'bx-in', 'fill-opacity': (0.12 + 0.5 * (v + 1) / 2).toFixed(2)}, ex);
        }
        el('path', {d: 'M' + (8 * (cell + gap) + 3) + ' ' + (y + cell / 2) + 'h12m-4 -4l4 4l-4 4', 'class': 'bx-arrow'}, ex);
        el('rect', {x: 8 * (cell + gap) + 22, y: y, width: cell, height: cell, rx: 3, 'class': 'bx-out', 'fill-opacity': (0.2 + 0.6 * Math.abs(Math.tanh(sum / 3))).toFixed(2)}, ex);
      }
    }
    // the learner: a dense network that sees only the examples
    var ln = $('bench-learner');
    if (ln) {
      var cols = [8, 6, 6, 1], Wl = 220, Hl = 170, xs = [18, 82, 146, 204];
      ln.setAttribute('viewBox', '0 0 ' + Wl + ' ' + Hl);
      function py(n, i) { return n === 1 ? Hl / 2 : 14 + i * (Hl - 28) / (n - 1); }
      var gL = el('g', {'class': 'bl-e'}, ln);
      for (var l = 0; l < cols.length - 1; l++) {
        for (var a = 0; a < cols[l]; a++) for (var b = 0; b < cols[l + 1]; b++) {
          el('line', {x1: xs[l], y1: py(cols[l], a), x2: xs[l + 1], y2: py(cols[l + 1], b)}, gL);
        }
      }
      cols.forEach(function (n, l) {
        for (var i = 0; i < n; i++) el('circle', {cx: xs[l], cy: py(n, i), r: l === 3 ? 6 : 4.6, 'class': l === 3 ? 'bl-out' : 'bl-n'}, ln);
      });
    }
    // the six graph families
    var host = $('bench-graphs');
    if (!host || !G) return;
    var FAM = [['Depth-4 tree', 'Depth-4 tree', 'hierarchy'], ['Depth-5 tree', 'Depth-5 tree', 'deeper, 16 inputs'],
               ['Depth-3 reuse DAG', 'Reuse DAG', 'shared pieces'], ['Sampled depth-4 DAG', 'Sampled DAGs', 'random graphs'],
               ['Flat modules', 'Flat modules', 'parallel pieces'], ['High-arity node', 'High-arity node', 'no small pieces']];
    FAM.forEach(function (f) {
      var g = G.filter(function (x) { return x.title === f[0]; })[0];
      if (!g) return;
      var card = document.createElement('figure');
      card.className = 'bg-card';
      var svg = el('svg', {'aria-hidden': 'true'}, card);
      var xs2 = g.nodes.map(function (n) { return n.x; }), ys2 = g.nodes.map(function (n) { return n.y; });
      var x0 = Math.min.apply(null, xs2), x1 = Math.max.apply(null, xs2), top = Math.max.apply(null, ys2);
      var Wg = 150, Hg = 92, pad = 8;
      svg.setAttribute('viewBox', '0 0 ' + Wg + ' ' + Hg);
      function PX(n) { return pad + (n.x - x0) / ((x1 - x0) || 1) * (Wg - 2 * pad); }
      function PY(n) { return Hg - pad - n.y / (top || 1) * (Hg - 2 * pad); }
      var gE = el('g', {'class': 'bg-e'}, svg);
      g.edges.forEach(function (e) { var a = g.nodes[e[0]], b = g.nodes[e[1]]; el('line', {x1: PX(a), y1: PY(a), x2: PX(b), y2: PY(b)}, gE); });
      g.nodes.forEach(function (n) {
        el('circle', {cx: PX(n), cy: PY(n), r: n.kind === 'input' ? (g.nodes.length > 20 ? 2.2 : 2.8) : 4.4, 'class': 'bg-' + n.kind}, svg);
      });
      var cap = document.createElement('figcaption');
      cap.innerHTML = '<b>' + f[1] + '</b><span>' + f[2] + '</span>';
      card.appendChild(cap);
      host.appendChild(card);
    });
  }

  // ------------------------------------------------------------------ world-model grid (Figure 7)
  var RDBU = ['#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7', '#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f'];
  function isDark() { return token('--wm-mid').toLowerCase() !== '#f7f7f7'; }
  function lerpHex(a, b, t) {
    var x = hexRgb(a), y = hexRgb(b);
    return 'rgb(' + Math.round(x[0] + (y[0] - x[0]) * t) + ',' + Math.round(x[1] + (y[1] - x[1]) * t) + ',' + Math.round(x[2] + (y[2] - x[2]) * t) + ')';
  }
  function divColor(v, clip) {
    var t = clamp(v / clip, -1, 1);
    if (isDark()) {
      var mid = token('--wm-mid'), end = t < 0 ? token('--wm-neg') : token('--wm-pos');
      return lerpHex(mid, end, Math.pow(Math.abs(t), 0.8));
    }
    var x = (t + 1) / 2 * (RDBU.length - 1), i = Math.min(RDBU.length - 2, Math.floor(x));
    return lerpHex(RDBU[i], RDBU[i + 1], x - i);
  }
  function signed(v) { return (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(3); }
  function worldModel() {
    var D = window.LEWI_WM, host = $('wm');
    if (!D || !host) return;
    while (host.firstChild) host.removeChild(host.firstChild);
    var CLIP = 0.75, cw = 40, ch = 30, ggap = 12, L = 70, top = 30, Rr = 104, bottom = 104;
    var cols = [], x = L;
    D.groups.forEach(function (g, gi) {
      g.tasks.forEach(function () { cols.push({x: x, g: gi}); x += cw; });
      x += ggap;
    });
    var gridW = x - ggap, W = gridW + Rr, H = top + D.baselines.length * ch + bottom;
    var svg = el('svg', {viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Paired AULC differences between LeWi and seven baselines on 17 world-model tasks.'}, host);
    var ci = 0;
    D.groups.forEach(function (g) {
      var xa = cols[ci].x, xb = cols[ci + g.tasks.length - 1].x + cw;
      var label = g.name === 'Manipulation' ? 'Manip.' : g.name === 'Synthetic' ? 'Synth.' : g.name;
      txt(svg, (xa + xb) / 2, top - 11, label, 'wm-group', {'text-anchor': 'middle'});
      ci += g.tasks.length;
    });
    D.baselines.forEach(function (b, i) {
      txt(svg, L - 10, top + i * ch + ch / 2 + 4.5, b, 'wm-row', {'text-anchor': 'end'});
      D.tasks.forEach(function (task, j) {
        var v = D.value[i][j], c = cols[j];
        var rect = el('rect', {x: c.x, y: top + i * ch, width: cw, height: ch, fill: divColor(v, CLIP), 'class': 'wm-cell'}, svg);
        if (D.verdict[i][j] === 'tie') el('circle', {cx: c.x + cw / 2, cy: top + i * ch + ch / 2, r: 4.6, 'class': 'wm-tie'}, svg);
        rect.addEventListener('pointermove', function (e) {
          var verdict = D.verdict[i][j] === 'win' ? 'LeWi better' : D.verdict[i][j] === 'loss' ? b + ' better' : 'no clear difference';
          showTip('<div class="tt-h">' + task + '</div><img class="tip-clip" src="assets/img/tasks/' + D.keys[j] + '.webp" alt="">' +
            '<div>LeWi vs ' + b + ': <b>' + verdict + '</b></div>' +
            '<div class="tt-r"><span>AULC difference</span><span>' + signed(v) + '</span></div>' +
            '<div class="tt-dim">95% interval ' + signed(D.lo[i][j]) + ' to ' + signed(D.hi[i][j]) + '</div>', e.clientX, e.clientY);
        });
        rect.addEventListener('pointerleave', hideTip);
      });
    });
    D.tasks.forEach(function (task, j) {
      var c = cols[j], tx = c.x + cw / 2 + 4, ty = top + D.baselines.length * ch + 10;
      var lab = txt(svg, tx, ty, task, 'wm-task', {'text-anchor': 'end', transform: 'rotate(-42 ' + tx + ' ' + ty + ')'});
      lab.addEventListener('pointermove', function (e) {
        var rows = D.baselines.map(function (b, i) {
          var v = D.value[i][j], mark = D.verdict[i][j] === 'win' ? '' : D.verdict[i][j] === 'loss' ? ' tt-loss' : ' tt-dim';
          return '<div class="tt-r' + mark + '"><span>vs ' + b + '</span><span>' + signed(v) + '</span></div>';
        }).join('');
        showTip('<div class="tt-h">' + task + '</div><img class="tip-clip" src="assets/img/tasks/' + D.keys[j] + '.webp" alt="">' +
          '<div class="tt-dim">AULC difference, negative = LeWi better</div>' + rows, e.clientX, e.clientY);
      });
      lab.addEventListener('pointerleave', hideTip);
    });
    // colour scale: a vertical bar beside the grid, LeWi better (blue) at the bottom
    var sx = gridW + 18, sw = 11, sy = top, sh = D.baselines.length * ch;
    var defs = el('defs', null, svg), lg = el('linearGradient', {id: 'wmScale', x1: 0, y1: 1, x2: 0, y2: 0}, defs);
    for (var k = 0; k <= 10; k++) el('stop', {offset: k * 10 + '%', 'stop-color': divColor(-CLIP + k * 2 * CLIP / 10, CLIP)}, lg);
    el('rect', {x: sx, y: sy, width: sw, height: sh, fill: 'url(#wmScale)', rx: 3}, svg);
    [[CLIP, '+0.75'], [0, '0'], [-CLIP, '−0.75']].forEach(function (t) {
      var y = sy + (CLIP - t[0]) / (2 * CLIP) * sh;
      el('line', {x1: sx + sw, x2: sx + sw + 4, y1: y, y2: y, 'class': 'wm-scale-tick'}, svg);
      txt(svg, sx + sw + 8, y + 4, t[1], 'wm-scale-t');
    });
    txt(svg, sx, sy + sh + 22, 'blue:', 'wm-scale-k');
    txt(svg, sx, sy + sh + 38, 'LeWi better', 'wm-scale-k');
  }


  // ------------------------------------------------------------------ gallery: the final learned wiring per target
  function gallery() {
    var runs = window.LEWI_GALLERY, cv = $('gallery-net'), tabs = $('gallery-tabs');
    if (!runs || !cv) return;
    var nets = {}, cur = 0, net = null;
    var view = new Viewer(cv, {yaw: runs[0].view.yaw, pitch: runs[0].view.pitch, sway: REDUCE ? null : {amp: 0.7, rate: 0.22}, fill: 0.37});
    var style = {edgeWidth: 0.7, edgeAlpha: 0.34, edgeAlphaDim: 0.08, radius: 3.0, dimAlpha: 0.55, only: null, otherAlpha: 0.1, fit: true, pad: 14, dy: -0.03};
    var layersBar = $('gallery-layers'), baseAlpha = 0.34;
    function layerButtons(r) {
      while (layersBar.firstChild) layersBar.removeChild(layersBar.firstChild);
      var L = r.counts.length, opts = [null];
      for (var b = 0; b < L - 1; b++) opts.push(b);
      opts.forEach(function (b) {
        var btn = document.createElement('button');
        btn.type = 'button'; btn.className = 'lay';
        btn.textContent = b === null ? 'All layers' : (b === 0 ? 'Input' : 'L' + b) + ' → ' + (b === L - 2 ? 'Output' : 'L' + (b + 1));
        btn.setAttribute('aria-pressed', String(b === style.only));
        btn.addEventListener('click', function () {
          style.only = b;
          style.edgeAlpha = b === null ? baseAlpha : Math.min(0.75, baseAlpha * 2);
          layersBar.querySelectorAll('.lay').forEach(function (x) { x.setAttribute('aria-pressed', String(x === btn)); });
          view.dirty = true;
        });
        layersBar.appendChild(btn);
      });
    }
    var buttons = runs.map(function (r, i) {
      var b = document.createElement('button');
      b.type = 'button'; b.className = 'tab'; b.setAttribute('role', 'tab');
      b.textContent = r.title;
      b.addEventListener('click', function () { select(i); });
      tabs.appendChild(b);
      return b;
    });
    tabs.addEventListener('keydown', function (e) {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      var i = (cur + (e.key === 'ArrowRight' ? 1 : runs.length - 1)) % runs.length;
      select(i); buttons[i].focus(); e.preventDefault();
    });
    function select(i) {
      cur = i;
      var r = runs[i];
      net = nets[i] || (nets[i] = new Net(r));
      buttons.forEach(function (b, j) { b.setAttribute('aria-selected', String(j === i)); b.tabIndex = j === i ? 0 : -1; });
      targetGraph($('gallery-target'), r.teacher, {width: 260, levelH: r.teacher.leaves > 8 ? 26 : 30, nodeR: r.teacher.leaves > 8 ? 5 : 6.2, leafR: 2.8, pad: 9});
      $('gallery-demo').href = 'demo/index.html#run=' + r.stem;
      view.setView(r.view.yaw, r.view.pitch);
      style.radius = r.counts[1] > 32 ? 2.3 : 3.0;
      baseAlpha = r.counts[1] > 32 ? 0.16 : 0.34;
      style.only = null;
      style.edgeAlpha = baseAlpha;
      layerButtons(r);
      view.dirty = true;
    }
    select(0);
    onTheme(function () { view.dirty = true; });
    var visible = false, last = performance.now();
    onVisible(cv, function (v) { visible = v; });
    function loop(now) {
      var dt = Math.min(now - last, 80);
      last = now;
      if (visible && !document.hidden) {
        view.tick(dt / 1000, now);
        var w = cv.width;
        view.resize();
        if (view.dirty || w !== cv.width) view.draw(net, 0, 0, 0, 0, style);
      }
      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
  }

  // ------------------------------------------------------------------ alignment results under the gallery: grouped bars
  function alignFig() {
    var A = window.LEWI_ALIGN;
    if (!A) return;
    var VW = 400, x0 = 122, x1 = 386, bh = 8, gapB = 3, padG = 14;
    function X(v) { return x0 + clamp(v, 0, 1) * (x1 - x0); }
    function fmt(v) { return v.toFixed(2); }
    function panel(id, rows, series, tip) {
      var svg = $(id);
      if (!svg) return;
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var gh = series.length * bh + (series.length - 1) * gapB, top = 40, H = top + rows.length * (gh + padG) + 26;
      svg.setAttribute('viewBox', '0 0 ' + VW + ' ' + H);
      var lx = x0;
      series.forEach(function (sr) {
        el('rect', {x: lx, y: 9, width: 14, height: 8, rx: 2, fill: sr.color}, svg);
        txt(svg, lx + 20, 17, sr.name, 'al-leg');
        lx += 20 + sr.w;
      });
      var bottom = top + rows.length * (gh + padG) - padG + 6;
      [0, 0.5, 1].forEach(function (v) {
        el('line', {x1: X(v), x2: X(v), y1: top - 6, y2: bottom, 'class': 'pt-g'}, svg);
        txt(svg, X(v), bottom + 16, v ? String(v) : '0', 'pt-t', {'text-anchor': 'middle'});
      });
      rows.forEach(function (r, i) {
        var y = top + i * (gh + padG);
        txt(svg, 0, y + gh / 2 + 5, r.label, 'al-row');
        series.forEach(function (sr, j) {
          var d = sr.get(r), yy = y + j * (bh + gapB);
          var bar = el('rect', {x: x0, y: yy, width: Math.max(1, X(d.value) - x0), height: bh, rx: bh / 2, fill: sr.color}, svg);
          bar.addEventListener('pointermove', function (ev) { showTip(tip(r, sr, d), ev.clientX, ev.clientY); });
          bar.addEventListener('pointerleave', hideTip);
        });
      });
    }
    var LEVEL = [{name: 'Dense', color: 'var(--dense)', w: 50, get: function (r) { return r.Dense; }},
                 {name: 'LeWi', color: 'var(--lewi)', w: 48, get: function (r) { return r.LeWi; }},
                 {name: 'GraphOracle', color: 'var(--oracle)', w: 80, get: function (r) { return r.GraphOracle; }}];
    function levelTip(r, sr, d) {
      return '<div class="tt-h">' + r.label + ' pieces</div><div>' + sr.name + ': <b>' + fmt(d.value) + '</b> (hidden layer ' + d.layer + ')</div>' +
        '<div class="tt-dim">95% interval ' + fmt(d.lo) + ' to ' + fmt(d.hi) + '</div>';
    }
    panel('al-probe', A.probe, LEVEL, levelTip);
    panel('al-das', A.das, LEVEL, levelTip);
    var CONTROLS = [['init', 'initialization'], ['rewired', 'degree-preserving rewiring'], ['shuffled', 'modules shuffled']];
    function bestControl(r) {
      var b = null;
      CONTROLS.forEach(function (c) { if (r[c[0]] && (!b || r[c[0]].value > b.value)) b = r[c[0]]; });
      return b;
    }
    panel('al-f1', A.f1, [{name: 'Best control', color: 'var(--base)', w: 92, get: bestControl},
                          {name: 'LeWi', color: 'var(--lewi)', w: 48, get: function (r) { return r.trained; }}],
      function (r, sr, d) {
        var rows = CONTROLS.map(function (c) {
          return '<div class="tt-r tt-dim"><span>' + c[1] + '</span><span>' + fmt(r[c[0]].value) + '</span></div>';
        }).join('');
        return '<div class="tt-h">' + r.label + '</div><div class="tt-r"><span>LeWi, trained</span><span>' + fmt(r.trained.value) + '</span></div>' + rows;
      });
  }
  // ------------------------------------------------------------------ the BibTeX dialog behind the hero button
  function bibDialog() {
    var dlg = $('bib-dlg'), open = $('bib-open'), close = $('bib-close');
    if (!dlg || !open) return;
    function shut() { if (dlg.close) dlg.close(); else dlg.removeAttribute('open'); }
    open.addEventListener('click', function () { if (dlg.showModal) dlg.showModal(); else dlg.setAttribute('open', ''); });
    if (close) close.addEventListener('click', shut);
    dlg.addEventListener('click', function (ev) { if (ev.target === dlg) shut(); });
  }
  function copyBib() {
    var btn = $('copy-bib'), pre = $('bibtex');
    if (!btn || !pre) return;
    btn.addEventListener('click', function () {
      var text = pre.textContent;
      function done() {
        btn.textContent = 'Copied'; btn.classList.add('done');
        setTimeout(function () { btn.textContent = 'Copy'; btn.classList.remove('done'); }, 1600);
      }
      function fallback() {
        var r = document.createRange(); r.selectNodeContents(pre);
        var s = getSelection(); s.removeAllRanges(); s.addRange(r);
        try { if (document.execCommand('copy')) { done(); return; } } catch (err) { /* fall through */ }
        btn.textContent = 'Press Ctrl+C';
      }
      if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(text).then(done, fallback);
      else fallback();
    });
  }

  function safe(fn) { try { fn(); } catch (err) { if (window.console) console.error(err); } }
  window.__lewi = {Net: Net, Viewer: Viewer, targetGraph: targetGraph};
  safe(sheet);
  safe(hero);
  safe(piecesFig);
  safe(benchFig);
  safe(gapFig);
  safe(stepperFig);
  safe(patternFig);
  safe(gallery);
  safe(alignFig);
  safe(worldModel);
  safe(bibDialog);
  safe(copyBib);
  onTheme(worldModel);
})();
