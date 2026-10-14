/* LeWi project page: the hero training replay, the method sketch, the benchmark graphs, the result charts,
   the learned-wiring gallery and small page behaviours. Plain JavaScript, no dependencies. */
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
  function themeToggle() {
    var mq = window.matchMedia ? matchMedia('(prefers-color-scheme: dark)') : null;
    function effective() { return root.getAttribute('data-theme') || (mq && mq.matches ? 'dark' : 'light'); }
    function label() {
      var next = effective() === 'dark' ? 'light' : 'dark';
      document.querySelectorAll('[data-theme-toggle]').forEach(function (b) { b.setAttribute('aria-label', 'Switch to ' + next + ' mode'); });
    }
    document.querySelectorAll('[data-theme-toggle]').forEach(function (b) {
      b.addEventListener('click', function () {
        var next = effective() === 'dark' ? 'light' : 'dark';
        root.setAttribute('data-theme', next);
        try { localStorage.setItem('lewi-theme', next); } catch (err) { /* private mode: the choice lasts for this page */ }
        label();
        themeChanged();
      });
    });
    if (mq) {
      var f = function () { if (!root.getAttribute('data-theme')) { label(); themeChanged(); } };
      if (mq.addEventListener) mq.addEventListener('change', f); else if (mq.addListener) mq.addListener(f);
    }
    label();
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
  Viewer.prototype.setView = function (yaw, pitch) { this.yaw = this.baseYaw = yaw; this.pitch = pitch; this.phase = 0; this.dirty = true; };
  // positions are interpolated between frames fa and fb (weight t); masks and colors come from frame fm
  Viewer.prototype.draw = function (net, fa, fb, t, fm, st) {
    var ctx = this.ctx, W = this.W, H = this.H, U = net.U, xyz = net.xyz, k = net.scale;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    var cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    var lim = net.run.lim3, s = Math.min(W, H) * this.fill / lim * this.zoom, depth = lim * 7;
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
    var C = lim * 0.92, corners = [];
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

  // ------------------------------------------------------------------ hero: one real training run, replayed
  function hero() {
    var run = window.LEWI_HERO, cv = $('hero-net');
    if (!run || !cv) return;
    var net = new Net(run);
    var view = new Viewer(cv, {yaw: run.view.yaw, pitch: run.view.pitch, sway: REDUCE ? null : {amp: 0.6, rate: 0.2}, fill: 0.44});
    var style = {edgeWidth: 0.7, edgeAlpha: 0.24, edgeAlphaDim: 0.07, radius: 3.5, dimAlpha: 0.55, cube: false};
    var F = net.frames, total = run.total_steps, per = total / (F - 1);
    var PLAY = 17000, HOLD = 4500, FADE = 450, GAMMA = 1.8;
    var st = {playing: !REDUCE, phase: 'play', t: 0, step: REDUCE ? run.best_step : 0};
    function frameOf(step) {
      var x = clamp(step / per, 0, F - 1), a = Math.floor(x);
      return [a, Math.min(F - 1, a + 1), x - a];
    }
    function render() {
      view.resize();
      var fr = frameOf(st.step);
      view.draw(net, fr[0], fr[1], fr[2], Math.round(fr[0] + fr[2]), style);
    }
    // a click that is not a drag pauses or resumes the replay
    var down = null;
    cv.addEventListener('pointerdown', function (e) { down = {x: e.clientX, y: e.clientY}; });
    cv.addEventListener('pointerup', function (e) {
      if (down && Math.abs(e.clientX - down.x) + Math.abs(e.clientY - down.y) < 5) st.playing = !st.playing;
      down = null;
    });
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
            st.step = total * Math.pow(clamp(st.t / PLAY, 0, 1), GAMMA);
            cv.style.opacity = st.t < FADE ? String(st.t / FADE) : '';
            if (st.t >= PLAY) { st.phase = 'hold'; st.t = 0; }
          } else if (st.phase === 'hold') {
            if (st.t >= HOLD) { st.phase = 'fade'; st.t = 0; }
          } else {
            cv.style.opacity = String(1 - st.t / FADE);
            if (st.t >= FADE) { st.phase = 'play'; st.t = 0; st.step = 0; }
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

  // ------------------------------------------------------------------ the idea in three pictures: place, connect, rewire
  function stepsFigs() {
    var W = 300, H = 190, RAD = 70;
    var S0 = [[52, 52], [120, 150], [190, 58], [250, 140]], R0 = [[96, 98], [170, 112], [238, 72]];
    var S1 = [[58, 76], [66, 132], [212, 60], [240, 128]], R1 = [[104, 104], [110, 152], [238, 84]];
    var GROUPS = [{cx: 84, cy: 116, rx: 58, ry: 58, c: '#0072b2'}, {cx: 228, cy: 94, rx: 44, ry: 56, c: '#009e73'}];
    function base(id) {
      var svg = $(id);
      if (svg) svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
      return svg;
    }
    function send(svg, p, cls) { el('rect', {x: p[0] - 6, y: p[1] - 6, width: 12, height: 12, rx: 1.5, 'class': cls || 'st-send'}, svg); }
    function recv(svg, p, cls) { el('path', {d: 'M' + p[0] + ' ' + (p[1] - 8) + 'l7.5 13h-15z', 'class': cls || 'st-recv'}, svg); }
    function nodes(svg, S, R, cls) { S.forEach(function (p) { send(svg, p, cls); }); R.forEach(function (p) { recv(svg, p, cls); }); }
    function links(svg, S, R) {
      var top = 1 / (1 + Math.exp(-1.5));
      R.forEach(function (r) {
        S.forEach(function (q) {
          var d = Math.hypot(r[0] - q[0], r[1] - q[1]);
          if (d >= RAD) return;
          var g = 1 / (1 + Math.exp(-1.5 * (1 - (d / RAD) * (d / RAD)))), w = (g - 0.5) / (top - 0.5);
          el('line', {x1: q[0], y1: q[1], x2: r[0], y2: r[1], 'class': 'st-link', 'stroke-width': (1.3 + 2.4 * w).toFixed(2), 'stroke-opacity': (0.35 + 0.65 * w).toFixed(2)}, svg);
        });
      });
    }
    var a = base('step-1');
    if (a) {
      el('line', {x1: 16, y1: 176, x2: 58, y2: 176, 'class': 'st-axis'}, a);
      el('line', {x1: 16, y1: 176, x2: 16, y2: 134, 'class': 'st-axis'}, a);
      txt(a, 62, 180, 'e₁', 'st-lab'); txt(a, 10, 128, 'e₂', 'st-lab');
      nodes(a, S0, R0);
    }
    var b = base('step-2');
    if (b) {
      el('circle', {cx: R0[1][0], cy: R0[1][1], r: RAD, 'class': 'st-ring'}, b);
      el('line', {x1: R0[1][0], y1: R0[1][1], x2: R0[1][0] + RAD * 0.707, y2: R0[1][1] + RAD * 0.707, 'class': 'st-trail'}, b);
      txt(b, R0[1][0] + RAD * 0.36 + 6, R0[1][1] + RAD * 0.36 + 2, 'R', 'st-lab');
      links(b, S0, R0);
      nodes(b, S0, R0);
    }
    var c = base('step-3');
    if (c) {
      GROUPS.forEach(function (g) { el('ellipse', {cx: g.cx, cy: g.cy, rx: g.rx, ry: g.ry, fill: g.c, 'fill-opacity': 0.08, stroke: g.c, 'stroke-opacity': 0.35, 'stroke-dasharray': '3 4'}, c); });
      S0.concat(R0).forEach(function (p, i) {
        var q = i < S0.length ? S1[i] : R1[i - S0.length];
        el('line', {x1: p[0], y1: p[1], x2: q[0], y2: q[1], 'class': 'st-trail'}, c);
        el('circle', {cx: p[0], cy: p[1], r: 2.2, 'class': 'st-ghost'}, c);
      });
      links(c, S1, R1);
      nodes(c, S1, R1);
    }
  }

  // ------------------------------------------------------------------ method sketch: drag the neurons
  var gateDot = null;
  function method() {
    var svg = $('method-svg');
    if (!svg) return;
    var W = 640, H = 410, UNIT = 140, R2 = 1.5, R = Math.sqrt(R2) * UNIT;
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    var defs = el('defs', null, svg);
    var grad = el('radialGradient', {id: 'gateShade'}, defs);
    el('stop', {offset: '0%', 'stop-color': 'currentColor', 'stop-opacity': 0.13}, grad);
    el('stop', {offset: '70%', 'stop-color': 'currentColor', 'stop-opacity': 0.06}, grad);
    el('stop', {offset: '100%', 'stop-color': 'currentColor', 'stop-opacity': 0.02}, grad);
    svg.style.color = 'var(--ink)';
    el('rect', {x: 0, y: 0, width: W, height: H, 'class': 'ms-bg'}, svg);
    var gGrid = el('g', null, svg);
    for (var gx = 40; gx < W; gx += 40) for (var gy = 40; gy < H; gy += 40) el('circle', {cx: gx, cy: gy, r: 1, fill: 'var(--hair)'}, gGrid);
    var gRing = el('g', null, svg), gLink = el('g', null, svg), gNode = el('g', null, svg);
    var ring = el('circle', {r: R, 'class': 'ms-ring'}, gRing);
    var rLine = el('line', {'class': 'gp-r', opacity: 0}, gRing);
    var rLab = txt(gRing, 0, 0, 'R', 'ms-r-label', {opacity: 0});

    var nodes = [
      {k: 's', x: 120, y: 110}, {k: 's', x: 165, y: 262}, {k: 's', x: 300, y: 196}, {k: 's', x: 420, y: 92},
      {k: 's', x: 468, y: 300}, {k: 's', x: 578, y: 180},
      {k: 'r', x: 236, y: 318}, {k: 'r', x: 252, y: 92}, {k: 'r', x: 392, y: 214}, {k: 'r', x: 560, y: 332}, {k: 'r', x: 76, y: 196}
    ];
    var send = nodes.filter(function (n) { return n.k === 's'; }), recv = nodes.filter(function (n) { return n.k === 'r'; });
    var links = [];
    recv.forEach(function (r) {
      send.forEach(function (s) {
        var line = el('line', {'class': 'ms-link'}, gLink);
        var item = {r: r, s: s, line: line};
        line.addEventListener('pointerenter', function () { focusPair(item); });
        line.addEventListener('pointerleave', function () { focusPair(null); });
        links.push(item);
      });
    });
    nodes.forEach(function (n) {
      var g = el('g', {'class': 'ms-node'}, gNode);
      el('circle', {r: 22, 'class': 'halo'}, g);
      if (n.k === 's') el('rect', {x: -8, y: -8, width: 16, height: 16, rx: 2, 'class': 'ms-send'}, g);
      else el('path', {d: 'M0 -11 L10.5 7.5 L-10.5 7.5 Z', 'class': 'ms-recv'}, g);
      n.g = g;
      g.addEventListener('pointerdown', function (e) { startDrag(n, e); });
      g.addEventListener('pointerenter', function () { if (!drag) setFocus(n); });
      g.addEventListener('pointerleave', function () { if (!drag) setFocus(null); });
    });
    var keyY = H - 16;
    var kg = el('g', null, svg);
    el('rect', {x: 18, y: keyY - 9, width: 11, height: 11, rx: 1.5, 'class': 'ms-send'}, kg);
    txt(kg, 35, keyY, 'layer ℓ', 'ms-key');
    el('path', {d: 'M' + 104 + ' ' + (keyY - 10) + ' l7 12 h-14 z', 'class': 'ms-recv'}, kg);
    txt(kg, 117, keyY, 'layer ℓ+1', 'ms-key');

    var focus = null, pairFocus = null, drag = null, touched = null, idle = !REDUCE;
    var read = $('method-read');
    function gate(d) { var sc = R2 - d * d; return {s: sc, g: 1 / (1 + Math.exp(-sc)), on: sc > 0}; }
    function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y) / UNIT; }
    function setFocus(n) { focus = n; update(); }
    function focusPair(p) { pairFocus = p; if (p) touched = p; update(); }
    function update() {
      var active = 0;
      nodes.forEach(function (n) { n.g.setAttribute('transform', 'translate(' + n.x.toFixed(1) + ' ' + n.y.toFixed(1) + ')'); });
      links.forEach(function (l) {
        var d = dist(l.r, l.s), q = gate(d);
        l.d = d; l.q = q;
        if (!q.on) { l.line.style.display = 'none'; return; }
        active++;
        l.line.style.display = '';
        l.line.setAttribute('x1', l.s.x); l.line.setAttribute('y1', l.s.y);
        l.line.setAttribute('x2', l.r.x); l.line.setAttribute('y2', l.r.y);
        var w = (q.g - 0.5) / (1 / (1 + Math.exp(-R2)) - 0.5);
        l.line.setAttribute('stroke-width', (1.2 + 3.2 * w).toFixed(2));
        l.line.setAttribute('stroke-opacity', (0.3 + 0.7 * w).toFixed(2));
        var rel = focus ? (l.r === focus || l.s === focus) : pairFocus ? l === pairFocus : true;
        l.line.classList.toggle('dim', !rel);
        l.line.classList.toggle('hot', pairFocus === l);
      });
      var c = focus || (pairFocus && pairFocus.r);
      if (c) {
        ring.setAttribute('cx', c.x); ring.setAttribute('cy', c.y); ring.classList.add('on');
        rLine.setAttribute('x1', c.x); rLine.setAttribute('y1', c.y);
        rLine.setAttribute('x2', c.x + R * 0.7071); rLine.setAttribute('y2', c.y - R * 0.7071);
        rLine.setAttribute('opacity', 1);
        rLab.setAttribute('x', c.x + R * 0.36 + 6); rLab.setAttribute('y', c.y - R * 0.36 - 4); rLab.setAttribute('opacity', 1);
      } else { ring.classList.remove('on'); rLine.setAttribute('opacity', 0); rLab.setAttribute('opacity', 0); }
      var p = pairFocus || touched;
      if (!p && focus) {
        var best = null;
        links.forEach(function (l) { if ((l.r === focus || l.s === focus) && (!best || l.d < best.d)) best = l; });
        p = best;
      }
      var total = links.length;
      if (p) {
        read.textContent = 'd = ' + p.d.toFixed(2) + ' · s = R² − d² = ' + p.q.s.toFixed(2) + ' · ' +
          (p.q.on ? 'gate σ(s) = ' + p.q.g.toFixed(2) : 'outside R: masked') + ' · ' + active + ' of ' + total + ' connections active';
        if (gateDot) gateDot(p.d);
      } else {
        read.textContent = active + ' of ' + total + ' possible connections active. Drag a neuron.';
      }
    }
    function toSvg(e) {
      var pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
      return pt.matrixTransform(svg.getScreenCTM().inverse());
    }
    function startDrag(n, e) {
      e.preventDefault();
      idle = false;
      var p = toSvg(e);
      drag = {n: n, dx: n.x - p.x, dy: n.y - p.y, id: e.pointerId};
      n.g.classList.add('drag');
      setFocus(n);
      window.addEventListener('pointermove', moveDrag);
      window.addEventListener('pointerup', endDrag);
      window.addEventListener('pointercancel', endDrag);
    }
    function moveDrag(e) {
      if (!drag) return;
      var p = toSvg(e);
      drag.n.x = clamp(p.x + drag.dx, 14, W - 14);
      drag.n.y = clamp(p.y + drag.dy, 14, H - 30);
      var best = null;
      links.forEach(function (l) { if ((l.r === drag.n || l.s === drag.n) && (!best || l.d < best.d)) best = l; });
      touched = best;
      update();
    }
    function endDrag() {
      if (!drag) return;
      drag.n.g.classList.remove('drag');
      drag = null;
      window.removeEventListener('pointermove', moveDrag);
      window.removeEventListener('pointerup', endDrag);
      window.removeEventListener('pointercancel', endDrag);
      setFocus(null);
    }
    // shuffle: new positions, eased in
    var seed = 7;
    function rnd() { seed = (seed * 16807) % 2147483647; return (seed - 1) / 2147483646; }
    $('method-shuffle').addEventListener('click', function () {
      idle = false;
      var from = nodes.map(function (n) { return [n.x, n.y]; }), to = [];
      nodes.forEach(function (n, i) {
        var x, y, ok, tries = 0;
        do {
          x = 40 + rnd() * (W - 80); y = 36 + rnd() * (H - 96); ok = true;
          for (var j = 0; j < to.length; j++) if (Math.hypot(to[j][0] - x, to[j][1] - y) < 62) ok = false;
        } while (!ok && ++tries < 60);
        to.push([x, y]);
      });
      var t0 = performance.now();
      (function step(now) {
        var t = clamp((now - t0) / 650, 0, 1), e = 1 - Math.pow(1 - t, 3);
        nodes.forEach(function (n, i) { n.x = from[i][0] + (to[i][0] - from[i][0]) * e; n.y = from[i][1] + (to[i][1] - from[i][1]) * e; });
        update();
        if (t < 1) requestAnimationFrame(step);
      })(t0);
    });
    // until the reader touches the sketch, one neuron drifts through the others, so connections switch on and off
    var mover = recv[2], home = {x: mover.x, y: mover.y}, t0 = null, running = false;
    function drift(now) {
      if (!idle || !running) { running = false; return; }
      if (t0 === null) t0 = now;
      var t = (now - t0) / 1000;
      mover.x = home.x + Math.sin(t * 0.42) * 165;
      mover.y = home.y + Math.sin(t * 0.84) * 70;
      touched = null;
      update();
      requestAnimationFrame(drift);
    }
    onVisible(svg, function (v) {
      if (v && idle && !running) { running = true; t0 = null; requestAnimationFrame(drift); }
      if (!v) running = false;
    });
    svg.addEventListener('pointerdown', function () { idle = false; });
    update();
  }

  // ------------------------------------------------------------------ gate profile: the multiplier against distance
  function gateProfile() {
    var svg = $('gate-profile');
    if (!svg) return;
    var W = 640, H = 196, x0 = 54, x1 = 620, y0 = 18, y1 = 150, R2 = 1.5, R = Math.sqrt(R2), DMAX = 2.0;
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    function X(d) { return x0 + d / DMAX * (x1 - x0); }
    function Y(v) { return y1 - v * (y1 - y0); }
    function sig(s) { return 1 / (1 + Math.exp(-s)); }
    [0, 0.5, sig(R2)].forEach(function (v, i) {
      el('line', {x1: x0, x2: x1, y1: Y(v), y2: Y(v), 'class': 'gp-axis', opacity: i ? 0.6 : 1}, svg);
      txt(svg, x0 - 8, Y(v) + 4, i === 2 ? v.toFixed(2) : String(v), 'gp-tick', {'text-anchor': 'end'});
    });
    [0, 0.5, 1, 1.5, 2].forEach(function (d) { txt(svg, X(d), y1 + 18, d.toFixed(1), 'gp-tick', {'text-anchor': 'middle'}); });
    txt(svg, x1, y1 + 38, 'distance d between the two neurons', 'gp-label', {'text-anchor': 'end'});
    txt(svg, x0 - 44, y0 - 4, 'multiplier on W', 'gp-label');
    var soft = '', on = '', area = 'M' + X(0) + ' ' + Y(0);
    for (var i = 0; i <= 200; i++) {
      var d = DMAX * i / 200, v = sig(R2 - d * d);
      soft += (i ? 'L' : 'M') + X(d).toFixed(1) + ' ' + Y(v).toFixed(1);
      if (d <= R) { on += (i ? 'L' : 'M') + X(d).toFixed(1) + ' ' + Y(v).toFixed(1); area += 'L' + X(d).toFixed(1) + ' ' + Y(v).toFixed(1); }
    }
    area += 'L' + X(R) + ' ' + Y(0.5) + 'L' + X(R) + ' ' + Y(0) + 'Z';
    on += 'L' + X(R) + ' ' + Y(0.5) + 'L' + X(R) + ' ' + Y(0) + 'L' + X(DMAX) + ' ' + Y(0);
    el('path', {d: area, 'class': 'gp-area'}, svg);
    el('path', {d: soft, 'class': 'gp-soft'}, svg);
    el('path', {d: on, 'class': 'gp-on'}, svg);
    el('line', {x1: X(R), x2: X(R), y1: y0 - 6, y2: y1, 'class': 'gp-r'}, svg);
    txt(svg, X(R) + 6, y0 + 4, 'd = R', 'gp-label');
    txt(svg, X(1.42), Y(0.47), 'σ(s) without the mask', 'gp-tick');
    var dot = el('circle', {r: 5.5, cx: X(0.6), cy: Y(sig(R2 - 0.36)), 'class': 'gp-dot'}, svg);
    gateDot = function (d) {
      var dd = clamp(d, 0, DMAX), inside = dd < R, v = inside ? sig(R2 - dd * dd) : 0;
      dot.setAttribute('cx', X(dd).toFixed(1));
      dot.setAttribute('cy', Y(v).toFixed(1));
      dot.classList.toggle('off', !inside);
    };
  }

  // ------------------------------------------------------------------ benchmark graphs
  function benchGraphs() {
    var G = window.LEWI_GRAPHS, host = $('graphs');
    if (!G || !host) return;
    var UNIT = 26, PADX = 0.7, cols = [];
    G.forEach(function (g) {
      var xs = g.nodes.map(function (n) { return n.x; }), ys = g.nodes.map(function (n) { return n.y; });
      var minX = Math.min.apply(null, xs) - PADX, maxX = Math.max.apply(null, xs) + PADX, maxY = Math.max.apply(null, ys);
      var w = (maxX - minX) * UNIT, padY = 14, h = 5 * 1.25 * UNIT * 0.62 + 2 * padY;
      var sy = (h - 2 * padY) / Math.max(1, maxY);
      cols.push('minmax(0, ' + (maxX - minX).toFixed(2) + 'fr)');
      var fig = document.createElement('figure');
      fig.className = 'graph';
      var svg = el('svg', {viewBox: '0 0 ' + w.toFixed(1) + ' ' + h.toFixed(1), role: 'img', 'aria-label': g.title}, fig);
      function X(x) { return (x - minX) * UNIT; }
      function Y(y) { return h - padY - y * sy; }
      var ge = el('g', null, svg);
      g.edges.forEach(function (e) {
        var a = g.nodes[e[0]], b = g.nodes[e[1]];
        el('line', {x1: X(a.x), y1: Y(a.y), x2: X(b.x), y2: Y(b.y), 'class': 'g-edge'}, ge);
      });
      g.nodes.forEach(function (n) {
        el('circle', {cx: X(n.x), cy: Y(n.y), r: n.kind === 'input' ? 4.2 : 7, 'class': n.kind === 'input' ? 'g-in' : n.kind === 'readout' ? 'g-out' : 'g-fn'}, svg);
      });
      var cap = document.createElement('figcaption');
      cap.textContent = g.title;
      fig.appendChild(cap);
      host.appendChild(fig);
    });
    host.style.setProperty('--cols', cols.join(' '));
  }

  // ------------------------------------------------------------------ learning curves (Figure 4)
  var ORACLE_VAR = {'graph_oracle(levels=1)': 'var(--oracle-1)', 'graph_oracle(levels=2)': 'var(--oracle-2)',
                    'graph_oracle(levels=all)': 'var(--oracle)', 'graph_oracle(levels=0)': 'var(--oracle-1)'};
  function armColor(a) {
    return a.kind === 'lewi' ? 'var(--lewi)' : a.kind === 'dense' ? 'var(--dense)' : a.kind === 'oracle' ? ORACLE_VAR[a.key] : 'var(--base)';
  }
  function fmtVal(v) { return v >= 100 ? v.toExponential(1).replace('e+', '×10^') : v >= 10 ? v.toFixed(1) : v.toFixed(3); }
  function curves() {
    var D = window.LEWI_CURVES, host = $('curves'), leg = $('curves-legend');
    if (!D || !host) return;
    var items = [['LeWi', 'var(--lewi)', 3, ''], ['Dense', 'var(--dense)', 2.2, ''], ['GraphOracle(1)', 'var(--oracle-1)', 1.9, '5 3.6'],
                 ['GraphOracle(2)', 'var(--oracle-2)', 1.9, '5 3.6'], ['GraphOracle(all)', 'var(--oracle)', 1.9, '5 3.6'],
                 ['Baselines: L0, RigL, BIMT, ℓ1, ℓ2, Fiedler', 'var(--base)', 1.3, '']];
    items.forEach(function (it) {
      var span = document.createElement('span');
      span.className = 'it';
      var s = el('svg', {viewBox: '0 0 26 10', 'aria-hidden': 'true'}, span);
      el('line', {x1: 1, x2: 25, y1: 5, y2: 5, stroke: it[1], 'stroke-width': it[2], 'stroke-dasharray': it[3], 'stroke-linecap': 'round'}, s);
      span.appendChild(document.createTextNode(it[0]));
      leg.appendChild(span);
    });
    var W = 330, H = 236, L = 44, Rm = 10, T = 12, B = 208;
    D.forEach(function (p) {
      var box = document.createElement('div');
      box.className = 'panel';
      var h3 = document.createElement('h3'); h3.textContent = p.title; box.appendChild(h3);
      var sub = document.createElement('p'); sub.className = 'sub';
      sub.textContent = p.instances + ' target instances · n = 2' + sup(Math.log2(p.window[0])) + '…2' + sup(Math.log2(p.window[p.window.length - 1]));
      box.appendChild(sub);
      var svg = el('svg', {viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Test NMSE against training samples, ' + p.title}, box);
      var lx0 = Math.log2(p.window[0] / 1.25), lx1 = Math.log2(p.window[p.window.length - 1] * 1.25);
      function X(n) { return L + (Math.log2(n) - lx0) / (lx1 - lx0) * (W - L - Rm); }
      function Y(v) { return B - v / p.top * (B - T); }
      var step = p.step || p.top / 4;
      for (var v = 0; v <= p.top + 1e-9; v += step) {
        el('line', {x1: L, x2: W - Rm, y1: Y(v), y2: Y(v), 'class': v === 0 ? 'ax-l' : 'ax-g'}, svg);
        txt(svg, L - 7, Y(v) + 3.5, (Math.round(v * 1000) / 1000).toString(), 'ax-t', {'text-anchor': 'end'});
      }
      var w0 = p.window, ticks = [w0[0], w0[Math.floor(w0.length / 2)], w0[w0.length - 1]];
      ticks.forEach(function (n) {
        el('line', {x1: X(n), x2: X(n), y1: B, y2: B + 4, 'class': 'ax-l'}, svg);
        var t = txt(svg, X(n), B + 18, '2', 'ax-t', {'text-anchor': 'middle'});
        var ts = el('tspan', {dy: -5, 'font-size': '8.5px'}, t); ts.textContent = String(Math.log2(n));
      });
      var order = ['baseline', 'oracle', 'dense', 'lewi'];
      var drawn = [];
      order.forEach(function (kind) {
        p.arms.filter(function (a) { return a.kind === kind; }).forEach(function (a) {
          var col = armColor(a), pts = a.points;
          if (a.band) {
            var up = [], dn = [];
            pts.forEach(function (q) { if (q.mean <= p.top) { up.push(X(q.n).toFixed(1) + ' ' + Y(Math.min(q.hi, p.top)).toFixed(1)); dn.unshift(X(q.n).toFixed(1) + ' ' + Y(Math.max(q.lo, 0)).toFixed(1)); } });
            if (up.length > 1) el('path', {d: 'M' + up.join('L') + 'L' + dn.join('L') + 'Z', 'class': 'bd ' + (kind === 'lewi' ? 'bd-lewi' : kind === 'dense' ? 'bd-dense' : 'bd-oracle'), fill: col}, svg);
          }
          var d = '', pen = false;
          pts.forEach(function (q) {
            if (q.mean > p.top) { pen = false; el('path', {d: 'M' + (X(q.n) - 4.5) + ' ' + (T - 3) + 'l4.5 -7 4.5 7z', 'class': 'caret'}, svg); return; }
            d += (pen ? 'L' : 'M') + X(q.n).toFixed(1) + ' ' + Y(q.mean).toFixed(1);
            pen = true;
          });
          var path = el('path', {d: d, 'class': 'ln ln-' + (kind === 'baseline' ? 'base' : kind), stroke: kind === 'oracle' ? col : null}, svg);
          if (kind !== 'oracle') path.removeAttribute('stroke');
          drawn.push({arm: a, path: path});
        });
      });
      var xh = el('line', {y1: T, y2: B, 'class': 'xh'}, svg);
      var dots = el('g', null, svg);
      var hit = el('rect', {x: L, y: T - 10, width: W - L - Rm, height: B - T + 10, 'class': 'hit'}, svg);
      function at(e) {
        var pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
        var q = pt.matrixTransform(svg.getScreenCTM().inverse()), best = 0;
        p.window.forEach(function (n, i) { if (Math.abs(X(n) - q.x) < Math.abs(X(p.window[best]) - q.x)) best = i; });
        return p.window[best];
      }
      hit.addEventListener('pointermove', function (e) {
        var n = at(e);
        box.classList.add('hover');
        xh.setAttribute('x1', X(n)); xh.setAttribute('x2', X(n));
        while (dots.firstChild) dots.removeChild(dots.firstChild);
        var rows = [];
        p.arms.forEach(function (a) {
          var q = a.points.filter(function (z) { return z.n === n; })[0];
          if (!q) return;
          rows.push({a: a, q: q});
          if (a.kind !== 'baseline' && q.mean <= p.top) el('circle', {cx: X(n), cy: Y(q.mean), r: a.kind === 'lewi' ? 4.2 : 3.3, fill: armColor(a), 'class': 'pt'}, dots);
        });
        rows.sort(function (u, w) { return u.q.mean - w.q.mean; });
        var html = '<div class="tt-h">' + p.title + ' · n = 2' + sup(Math.log2(n)) + ' = ' + fmtInt(n) + '</div>';
        rows.forEach(function (r) {
          var strong = r.a.kind === 'lewi' || r.a.kind === 'dense';
          html += '<div class="tt-r' + (strong ? '' : ' tt-dim') + '"><span><i class="sw" style="background:' + (r.a.kind === 'baseline' ? '#a3aab5' : r.a.color) + '"></i>' +
            (strong ? '<b>' + r.a.name + '</b>' : r.a.name) + '</span><span>' + fmtVal(r.q.mean) + '</span></div>';
        });
        showTip(html, e.clientX, e.clientY);
      });
      hit.addEventListener('pointerleave', function () { box.classList.remove('hover'); while (dots.firstChild) dots.removeChild(dots.firstChild); hideTip(); });
      host.appendChild(box);
    });
  }

  // ------------------------------------------------------------------ wiring alignment (Figure 6b)
  function alignment() {
    var A = window.LEWI_ALIGN, host = $('alignment');
    if (!A || !host) return;
    var COND = {trained: ['var(--lewi)', 'o'], rewired: ['var(--ink-2)', '^'], init: ['var(--faint)', 's'], shuffled: ['#a17a36', 'D']};
    var W = 560, L = 150, Rr = 16, lane = 15, gap = 22, top = 46;
    var nF = A.families.length, nC = A.conditions.length, H = top + nF * (nC * lane + gap) + 36;
    var svg = el('svg', {viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Edge F1 of the extracted module graph for trained LeWi and three controls in four settings.'}, host);
    function X(v) { return L + v * (W - L - Rr); }
    function shape(g, kind, x, y, r, col, filled) {
      var a = {fill: filled ? col : 'var(--paper)', stroke: col, 'stroke-width': 1.5};
      if (kind === 'o') a.d = 'M' + (x - r) + ' ' + y + 'a' + r + ' ' + r + ' 0 1 0 ' + 2 * r + ' 0a' + r + ' ' + r + ' 0 1 0 ' + (-2 * r) + ' 0';
      else if (kind === 's') a.d = 'M' + (x - r * 0.85) + ' ' + (y - r * 0.85) + 'h' + 1.7 * r + 'v' + 1.7 * r + 'h' + (-1.7 * r) + 'z';
      else if (kind === '^') a.d = 'M' + x + ' ' + (y - r * 1.1) + 'L' + (x + r) + ' ' + (y + r * 0.8) + 'L' + (x - r) + ' ' + (y + r * 0.8) + 'z';
      else a.d = 'M' + x + ' ' + (y - r * 1.15) + 'L' + (x + r * 0.95) + ' ' + y + 'L' + x + ' ' + (y + r * 1.15) + 'L' + (x - r * 0.95) + ' ' + y + 'z';
      return el('path', a, g);
    }
    // legend
    var lx = 0;
    A.conditions.forEach(function (c) {
      var k = COND[c.key];
      shape(svg, k[1], lx + 6, 12, 4.6, k[0], true);
      var t = txt(svg, lx + 16, 16, c.name, 'al-c');
      lx += 30 + c.name.length * 6.4;
    });
    shape(svg, 'o', 6, 32, 4.2, 'var(--ink-2)', true); txt(svg, 16, 36, 'all target edges', 'al-c');
    shape(svg, 'o', 146, 32, 4.2, 'var(--ink-2)', false); txt(svg, 156, 36, 'edges from target intermediates', 'al-c');
    [0, 0.25, 0.5, 0.75, 1].forEach(function (v) {
      el('line', {x1: X(v), x2: X(v), y1: top - 4, y2: H - 40, 'class': v === 0 ? 'al-l' : 'al-g'}, svg);
      txt(svg, X(v), H - 24, v === 0 ? '0' : v === 1 ? '1' : v.toFixed(2), 'al-t', {'text-anchor': 'middle'});
    });
    var byKey = {};
    A.rows.forEach(function (r) { byKey[r.family + '|' + r.condition + '|' + r.metric] = r; });
    A.families.forEach(function (f, i) {
      var y0 = top + i * (nC * lane + gap);
      var t = txt(svg, 0, y0 + nC * lane / 2 + 4, f.title, 'al-f');
      A.conditions.forEach(function (c, j) {
        var y = y0 + j * lane + lane / 2, k = COND[c.key];
        ['higher_f1', 'edge_f1'].forEach(function (m) {
          var r = byKey[f.key + '|' + c.key + '|' + m];
          if (!r) return;
          var yy = y + (m === 'edge_f1' ? -2.2 : 2.2);
          el('line', {x1: X(Math.max(0, r.lo)), x2: X(Math.min(1, r.hi)), y1: yy, y2: yy, stroke: k[0], 'stroke-width': 1.3, opacity: 0.7}, svg);
          var mk = shape(svg, k[1], X(r.mean), yy, c.key === 'trained' ? 4.6 : 3.9, k[0], m === 'edge_f1');
          mk.style.cursor = 'default';
          mk.addEventListener('pointermove', function (e) {
            showTip('<div class="tt-h">' + f.title + '</div><div>' + c.name + '</div><div class="tt-r"><span>' +
              (m === 'edge_f1' ? 'all target edges' : 'edges from intermediates') + '</span><span>' + r.mean.toFixed(3) + '</span></div>' +
              '<div class="tt-dim">95% interval ' + r.lo.toFixed(2) + ' to ' + r.hi.toFixed(2) + ' · ' + r.k + ' instances</div>', e.clientX, e.clientY);
          });
          mk.addEventListener('pointerleave', hideTip);
        });
      });
    });
    txt(svg, X(0.5), H - 4, 'Edge F1', 'al-t', {'text-anchor': 'middle'});
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
  function worldModel() {
    var D = window.LEWI_WM, host = $('wm');
    if (!D || !host) return;
    while (host.firstChild) host.removeChild(host.firstChild);
    var CLIP = 0.75, cw = 40, ch = 30, ggap = 12, L = 70, top = 30, Rr = 92, bottom = 104;
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
          showTip('<div class="tt-h">' + task + ' · LeWi vs ' + b + '</div><div class="tt-r"><span>ΔAULC</span><span>' + (v > 0 ? '+' : '') + v.toFixed(3) + '</span></div>' +
            '<div class="tt-dim">95% interval ' + D.lo[i][j].toFixed(3) + ' to ' + D.hi[i][j].toFixed(3) + '</div><div>' + verdict + '</div>', e.clientX, e.clientY);
        });
        rect.addEventListener('pointerleave', hideTip);
      });
      var cnt = D.counts[i];
      txt(svg, gridW + 16, top + i * ch + ch / 2 + 4.5, cnt.win + ' – ' + cnt.tie + ' – ' + cnt.loss, 'wm-tally');
    });
    txt(svg, gridW + 16, top - 11, 'W – T – L', 'wm-tally-h');
    D.tasks.forEach(function (task, j) {
      var c = cols[j], tx = c.x + cw / 2 + 4, ty = top + D.baselines.length * ch + 10;
      txt(svg, tx, ty, task, 'wm-task', {'text-anchor': 'end', transform: 'rotate(-42 ' + tx + ' ' + ty + ')'});
    });
    // colour scale
    var sx = gridW + 16, sw = 84, sy = top + D.baselines.length * ch + 30;
    var defs = el('defs', null, svg), lg = el('linearGradient', {id: 'wmScale'}, defs);
    for (var k = 0; k <= 10; k++) el('stop', {offset: k * 10 + '%', 'stop-color': divColor(-CLIP + k * 2 * CLIP / 10, CLIP)}, lg);
    el('rect', {x: sx, y: sy, width: sw, height: 9, fill: 'url(#wmScale)', rx: 2}, svg);
    txt(svg, sx, sy + 22, '−0.75', 'wm-scale-t');
    txt(svg, sx + sw, sy + 22, '+0.75', 'wm-scale-t', {'text-anchor': 'end'});
    txt(svg, sx, sy + 38, '← LeWi better', 'wm-scale-k');
  }

  // ------------------------------------------------------------------ gallery: the final learned wiring per target
  function gallery() {
    var runs = window.LEWI_GALLERY, cv = $('gallery-net'), tabs = $('gallery-tabs');
    if (!runs || !cv) return;
    var nets = {}, cur = 0, net = null;
    var view = new Viewer(cv, {yaw: runs[0].view.yaw, pitch: runs[0].view.pitch, sway: REDUCE ? null : {amp: 0.7, rate: 0.22}, fill: 0.37});
    var style = {edgeWidth: 0.7, edgeAlpha: 0.34, edgeAlphaDim: 0.08, radius: 3.0, dimAlpha: 0.55, only: null, otherAlpha: 0.1};
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
    function glyphs(r) {
      var c = r.counts, names = ['●', '■', '▲', '◆', '▼'], hollow = ['□', '△', '◇', '▽'], out = [];
      out.push('<span><i>●</i>input</span>');
      for (var l = 1; l < c.length - 1; l++) out.push('<span><i>' + (l <= 4 ? names[l] : hollow[(l - 1) % 4]) + '</i>layer ' + l + '</span>');
      out.push('<span><i>★</i>output</span>');
      return out.join('');
    }
    function select(i) {
      cur = i;
      var r = runs[i];
      net = nets[i] || (nets[i] = new Net(r));
      buttons.forEach(function (b, j) { b.setAttribute('aria-selected', String(j === i)); b.tabIndex = j === i ? 0 : -1; });
      targetGraph($('gallery-target'), r.teacher, {width: 260, levelH: r.teacher.leaves > 8 ? 26 : 30, nodeR: r.teacher.leaves > 8 ? 5 : 6.2, leafR: 2.8, pad: 9});
      $('gallery-note').textContent = r.note;
      var hidden = r.counts.length - 2;
      $('gallery-meta').innerHTML = '<dt>Network</dt><dd>' + hidden + ' × ' + r.counts[1] + '</dd>' +
        '<dt>Training samples</dt><dd>' + fmtInt(r.n_train) + '</dd>' +
        '<dt>Checkpoint</dt><dd>step ' + fmtInt(r.step) + '</dd>' +
        '<dt>Test NMSE</dt><dd>' + r.test_nmse.toFixed(3) + '</dd>' +
        '<dt>Active connections</dt><dd>' + Math.round(net.activeShare(0) * 100) + '%</dd>';
      $('gallery-glyphs').innerHTML = glyphs(r);
      $('gallery-demo').href = 'demo/index.html#run=' + r.stem;
      view.setView(r.view.yaw, r.view.pitch);
      style.radius = r.counts[1] > 32 ? 2.3 : 3.0;
      baseAlpha = r.counts[1] > 32 ? 0.16 : 0.34;
      style.only = 0;
      style.edgeAlpha = Math.min(0.75, baseAlpha * 2);
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

  // ------------------------------------------------------------------ navigation and small behaviours
  function nav() {
    var navEl = $('topnav'), heroEl = $('top');
    if (!navEl || !heroEl || !('IntersectionObserver' in window)) return;
    new IntersectionObserver(function (es) { es.forEach(function (e) { navEl.classList.toggle('show', !e.isIntersecting); }); },
                             {rootMargin: '-64px 0px 0px 0px'}).observe(heroEl);
    var links = {};
    navEl.querySelectorAll('.topnav-links a').forEach(function (a) { links[a.getAttribute('href').slice(1)] = a; });
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (!e.isIntersecting || !links[e.target.id]) return;
        for (var k in links) links[k].classList.toggle('active', k === e.target.id);
      });
    }, {rootMargin: '-35% 0px -60% 0px'});
    Object.keys(links).forEach(function (id) { var s = $(id); if (s) io.observe(s); });
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
  safe(themeToggle);
  safe(nav);
  safe(stepsFigs);
  safe(hero);
  safe(gateProfile);
  safe(method);
  safe(benchGraphs);
  safe(curves);
  safe(gallery);
  safe(alignment);
  safe(worldModel);
  safe(copyBib);
  onTheme(worldModel);
})();
