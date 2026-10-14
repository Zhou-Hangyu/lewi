/* The training-run player: the run replayed inside the learned 3-D embedding
 * space. Static data, no external requests.
 *
 * Reads a run bundle (assets/<run>.json + .bin) and drives the synchronized
 * views: the target-graph key, the rotatable 3-D embedding view with the
 * connections drawn per frame, every layer's soft-gate matrix (plus the
 * binary masks where they exist), and the validation-NMSE curve. The page-top
 * bar switches between two bundles; both runs share the target, datasets,
 * architecture, initialization, and training protocol. After a switch a
 * 'runchange' event carries the parsed run to the finished-network section.
 */
(function () {
  'use strict';

  var MAGMA = '00000401000501010601010802010902020b02020d03030f03031204041405041606051806051a07061c08071e0907200a08220b09240c09260d0a290e0b2b100b2d110c2f120d31130d34140e36150e38160f3b180f3d19103f1a10421c10441d11471e114920114b21114e22115024125325125527125829115a2a115c2c115f2d11612f116331116533106734106936106b38106c390f6e3b0f703d0f713f0f72400f74420f75440f764510774710784910784a10794c117a4e117b4f127b51127c52137c54137d56147d57157e59157e5a167e5c167f5d177f5f187f601880621980641a80651a80671b80681c816a1c816b1d816d1d816e1e81701f81721f817320817521817621817822817922827b23827c23827e24828025828125818326818426818627818827818928818b29818c29818e2a81902a81912b81932b80942c80962c80982d80992d809b2e7f9c2e7f9e2f7fa02f7fa1307ea3307ea5317ea6317da8327daa337dab337cad347cae347bb0357bb2357bb3367ab5367ab73779b83779ba3878bc3978bd3977bf3a77c03a76c23b75c43c75c53c74c73d73c83e73ca3e72cc3f71cd4071cf4070d0416fd2426fd3436ed5446dd6456cd8456cd9466bdb476adc4869de4968df4a68e04c67e24d66e34e65e44f64e55064e75263e85362e95462ea5661eb5760ec5860ed5a5fee5b5eef5d5ef05f5ef1605df2625df2645cf3655cf4675cf4695cf56b5cf66c5cf66e5cf7705cf7725cf8745cf8765cf9785df9795df97b5dfa7d5efa7f5efa815ffb835ffb8560fb8761fc8961fc8a62fc8c63fc8e64fc9065fd9266fd9467fd9668fd9869fd9a6afd9b6bfe9d6cfe9f6dfea16efea36ffea571fea772fea973feaa74feac76feae77feb078feb27afeb47bfeb67cfeb77efeb97ffebb81febd82febf84fec185fec287fec488fec68afec88cfeca8dfecc8ffecd90fecf92fed194fed395fed597fed799fed89afdda9cfddc9efddea0fde0a1fde2a3fde3a5fde5a7fde7a9fde9aafdebacfcecaefceeb0fcf0b2fcf2b4fcf4b6fcf6b8fcf7b9fcf9bbfcfbbdfcfdbf';
  var GLYPH = ['●', '■', '▲', '◆', '▼', '★'];
  var SHAPES = ['o', 's', '^', 'D', 'v', '*'];
  var BASE_FPS = 12;
  var INK = '#1a1a1a';
  var FUTURE = '#dcdcdc';
  var ACCENT = '#0f62fe';
  var CONN_OUT = '#d02670';
  var UNASSIGNED = 'unassigned: no correlation exceeds the threshold';

  // Every training view opens a little above the horizon, at the elevation of
  // the paper's embedding camera, so the cloud reads as 3-D from the start.
  var PITCH = 22 * Math.PI / 180;
  // Every view turns the same way: counterclockwise seen from above (yaw
  // grows, so the near side moves right), at one rate, in rad/s. The turn is
  // a choice (the "Auto-rotate" toggle) and is off when the page opens.
  var SPIN = 0.075;
  // `group` places a tab in the bar; `target` completes the section caption
  // and `intro` the opening paragraph; `hideDim` opens the run with its
  // unassigned neurons hidden; a `hidden` run keeps its bundle but gets no tab. The runs of the paper's
  // embedding figure carry the figure's own camera in their bundle
  // (meta.view), which replaces the yaw and pitch given here.
  var GROUPS = [
    {key: 'gallery', label: 'One LeWi training run per target structure'},
    {key: 'variants', label: 'Inner product variants'}
  ];
  var TREE7 = 'a seven-node tree over eight inputs';
  // the gallery follows the paper's graph templates (depth-4 tree, reuse DAG,
  // sampled DAG, flat modules, high-arity node); a second row holds the
  // depth-5 tree and the tree at d = 32
  var RUNS = [
    {stem: 'tree', group: 'gallery', label: 'Depth-4 tree', note: 'seven nodes in three levels',
     target: 'a depth-4 tree over eight inputs', intro: 'a depth-4 tree target',
     view: {spin: SPIN, zoom: 1}},
    {stem: 'dag', group: 'gallery', label: 'Reuse DAG', note: 'two nodes read twice',
     target: 'a reuse DAG over eight inputs, in which two nodes feed two others',
     intro: 'a reuse DAG target', view: {spin: SPIN, zoom: 1}},
    {stem: 'rand', group: 'gallery', label: 'Sampled DAGs', note: 'a random seven-node graph',
     target: 'a sampled seven-node DAG over eight inputs', intro: 'a sampled DAG target',
     view: {spin: SPIN, zoom: 1}},
    {stem: 'flat', group: 'gallery', label: 'Flat modules', note: 'four modules, summed',
     target: 'four flat modules over eight inputs, summed by the output',
     intro: 'a flat-modules target', view: {spin: SPIN, zoom: 1}},
    {stem: 'dense', group: 'gallery', label: 'High-arity node', note: 'one node reads all inputs',
     target: 'a high-arity node that reads all eight inputs', intro: 'a high-arity node target',
     view: {spin: SPIN, zoom: 1}},
    {stem: 'tree5', group: 'gallery', label: 'Depth-5 tree', note: '15 nodes over 16 inputs',
     target: 'a depth-5 tree of 15 nodes over 16 inputs', intro: 'a depth-5 tree target',
     view: {spin: SPIN, zoom: 1}, newRow: true, hideDim: true},
    {stem: 'fourier', group: 'gallery', label: 'Depth-4 tree', note: 'with 24 nuisance inputs',
     target: 'a depth-4 tree over 8 of 32 inputs, with the other 24 inputs as nuisance inputs',
     intro: 'a depth-4 tree target with 24 nuisance inputs',
     view: {yaw: Math.PI / 4, pitch: PITCH, spin: SPIN}},
    {stem: 'lewi', group: 'variants', hidden: true, label: 'inner product variant',
     note: 'first two masked \u00b7 recomputed every step',
     target: TREE7, view: {yaw: 1.22, pitch: PITCH, spin: SPIN}},
    {stem: 'trajectory', group: 'variants', hidden: true, label: 'inner product variant',
     note: 'first two masked \u00b7 re-selected every 2,000 steps',
     target: TREE7, view: {yaw: 2.35, pitch: PITCH, spin: SPIN}}
  ];
  var SHOWN = RUNS.filter(function (r) { return !r.hidden; });
  var VIEW0 = {yaw: 0.62, pitch: PITCH, spin: SPIN, zoom: 1.28};
  function runOf(stem) {
    for (var i = 0; i < RUNS.length; i++) if (RUNS[i].stem === stem) return RUNS[i];
    return null;
  }
  // the camera a run opens at: its entry above, then the figure's camera
  // where the bundle carries one
  function viewOf(stem, run) {
    var r = runOf(stem), v = {};
    var base = (r && r.view) || VIEW0;
    for (var k in VIEW0) v[k] = base[k] !== undefined ? base[k] : VIEW0[k];
    if (run && run.meta.view) {
      v.yaw = run.meta.view.yaw;
      v.pitch = run.meta.view.pitch;
    }
    return v;
  }

  var magmaLut = [];
  for (var mi = 0; mi < 256; mi++) {
    magmaLut.push([parseInt(MAGMA.substr(mi * 6, 2), 16),
                   parseInt(MAGMA.substr(mi * 6 + 2, 2), 16),
                   parseInt(MAGMA.substr(mi * 6 + 4, 2), 16)]);
  }

  // signed weights read on a diverging ramp: blue negative, white at zero,
  // red positive, so sign is visible at a glance and magnitude by saturation
  var divLut = [];
  for (var dv = -127; dv <= 127; dv++) {
    var t = Math.pow(Math.abs(dv) / 127, 0.62);
    divLut.push(dv >= 0
      ? [Math.round(255 - t * 77), Math.round(255 - t * 231), Math.round(255 - t * 212)]
      : [Math.round(255 - t * 222), Math.round(255 - t * 153), Math.round(255 - t * 83)]);
  }

  function hexRgb(h) {
    return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16),
            parseInt(h.slice(5, 7), 16)];
  }
  function rgba(h, a) {
    var c = hexRgb(h);
    return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + a + ')';
  }
  function fmt(n) { return n.toLocaleString('en-US'); }
  function sub(i) {
    return String(i).split('').map(function (c) { return '₀₁₂₃₄₅₆₇₈₉'[+c]; }).join('');
  }

  function shapePath(ctx, kind, x, y, r) {
    ctx.beginPath();
    if (kind === 's') { ctx.rect(x - r, y - r, 2 * r, 2 * r); }
    else if (kind === '^') {
      ctx.moveTo(x, y - r * 1.15); ctx.lineTo(x + r, y + r * 0.75);
      ctx.lineTo(x - r, y + r * 0.75); ctx.closePath();
    } else if (kind === 'v') {
      ctx.moveTo(x, y + r * 1.15); ctx.lineTo(x + r, y - r * 0.75);
      ctx.lineTo(x - r, y - r * 0.75); ctx.closePath();
    } else if (kind === 'D') {
      ctx.moveTo(x, y - r * 1.2); ctx.lineTo(x + r * 1.2, y);
      ctx.lineTo(x, y + r * 1.2); ctx.lineTo(x - r * 1.2, y); ctx.closePath();
    } else if (kind === '*') {
      for (var i = 0; i < 10; i++) {
        var rr = (i % 2 === 0) ? r * 1.5 : r * 0.62;
        var a = -Math.PI / 2 + i * Math.PI / 5;
        var px = x + rr * Math.cos(a), py = y + rr * Math.sin(a);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.closePath();
    } else { ctx.arc(x, y, r, 0, 2 * Math.PI); }
  }

  // every in/out connection of unit u at frame f: the kept ones on masked
  // boundaries, the soft ones at gate >= gateMin elsewhere; su feeds ru
  function connEdgesOf(R, f, u, gateMin) {
    var LS = R.levelStart, LC = R.meta.levels.counts;
    var out = [];
    R.gates.map(function (gate, b) {
      return {gate: gate, mask: R.masks[b],
              r0: LS[b + 1], s0: LS[b], nr: LC[b + 1], nc: LC[b]};
    })
      .forEach(function (B) {
        var base = f * B.nr * B.nc;
        var i, o, g;
        if (u >= B.r0 && u < B.r0 + B.nr) {
          var rr = u - B.r0;
          for (i = 0; i < B.nc; i++) {
            o = base + rr * B.nc + i;
            g = B.gate[o] / 255;
            if (B.mask ? !B.mask[o] : g < gateMin) continue;
            out.push({su: B.s0 + i, ru: u, g: g, kept: !!B.mask, dirIn: true});
          }
        }
        if (u >= B.s0 && u < B.s0 + B.nc) {
          var cc = u - B.s0;
          for (i = 0; i < B.nr; i++) {
            o = base + i * B.nc + cc;
            g = B.gate[o] / 255;
            if (B.mask ? !B.mask[o] : g < gateMin) continue;
            out.push({su: u, ru: B.r0 + i, g: g, kept: !!B.mask, dirIn: false});
          }
        }
      });
    return out;
  }

  // an arrowhead at the receiving end of the segment a -> b
  function arrowHead(ctx, a, b, col) {
    var dx = b[0] - a[0], dy = b[1] - a[1];
    var L = Math.sqrt(dx * dx + dy * dy);
    if (L < 14) return;
    var ux = dx / L, uy = dy / L;
    var tx = b[0] - ux * 8, ty = b[1] - uy * 8;
    ctx.beginPath();
    ctx.moveTo(tx, ty);
    ctx.lineTo(tx - ux * 8 - uy * 3.6, ty - uy * 8 + ux * 3.6);
    ctx.lineTo(tx - ux * 8 + uy * 3.6, ty - uy * 8 - ux * 3.6);
    ctx.closePath();
    ctx.fillStyle = col;
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 0.8;
    ctx.stroke();
  }

  // The network's output is not one of the target's variables, so it is drawn
  // in the output colour wherever it appears rather than in the colour of the
  // variable it happens to track -- the same choice the paper's figure makes.
  function unitCol(run, u, bucket, fallback) {
    if (run.levelOf[u] === run.meta.levels.counts.length - 1) {
      return run.meta.teacher.output_color || '#000000';
    }
    return bucket.lit ? run.meta.teacher.colors[bucket.node] : fallback;
  }

  // Lines are drawn at the other tabs' ink: up to INK_REF connections at full
  // strength, more of them each fainter by the square root of the excess.
  var INK_REF = 2400;
  function inkScale(n) { return Math.min(1, Math.sqrt(INK_REF / Math.max(1, n))); }

  // One neuron's marker: the level's shape, filled in the neuron's colour with
  // a white rim, or for the hollow shapes of the deeper layers of a deep
  // network white inside a rim of the neuron's colour.
  function drawMark(ctx, run, lv, x, y, r, col, alpha, rim) {
    shapePath(ctx, run.shapes[lv], x, y, r);
    ctx.globalAlpha = alpha;
    if (run.hollow[lv]) {
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      ctx.lineWidth = Math.max(1.1, r * 0.34);
      ctx.strokeStyle = col;
      ctx.stroke();
    } else {
      ctx.fillStyle = col;
      ctx.fill();
      ctx.lineWidth = rim;
      ctx.strokeStyle = 'rgba(255,255,255,0.9)';
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  window.__demoShared = {GLYPH: GLYPH, SHAPES: SHAPES, shapePath: shapePath,
                         drawMark: drawMark, inkScale: inkScale,
                         unitCol: unitCol,
                         rgba: rgba, sub: sub,
                         connEdgesOf: connEdgesOf, arrowHead: arrowHead};

  // ------------------------------------------------------- target-graph key

  function teacherSVG(t, opts) {
    var ns = 'http://www.w3.org/2000/svg';
    // a larger graph (the depth-5 tree: 15 nodes over 16 inputs) is drawn on a
    // wider canvas with smaller marks, so its labels keep clear of each other
    var dense = t.leaves > 8;
    var W = dense ? 276 : 250, H = dense ? 244 : 228;
    var padL = 30, padR = dense ? 30 : 48, padT = 16, padB = 26;
    var NODE_R = dense ? 5.6 : 7.6, READ_R = dense ? 7 : 9, LEAF = dense ? 6.4 : 9;
    var LABEL = dense ? 9.5 : 10.5, LDX = dense ? 6.5 : 10;
    var maxLv = 0, i, v;
    for (i = 0; i < t.level.length; i++) maxLv = Math.max(maxLv, t.level[i]);
    // the network's output is its own node above the last intermediate, as in
    // the paper's figure: the root is a target variable, the output is not
    var outLv = maxLv + 1;
    var xs = Math.max.apply(null, t.leaf_x.concat(t.node_x)) || 1;
    function X(x) { return padL + (x / xs) * (W - padL - padR); }
    function Y(lv) { return H - padB - (lv / outLv) * (H - padT - padB); }
    var svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'the target computation graph and color key');
    function el(name, at, parent) {
      var e = document.createElementNS(ns, name);
      for (var k in at) e.setAttribute(k, at[k]);
      (parent || svg).appendChild(e);
      return e;
    }
    for (v = 0; v < t.parents.length; v++) {
      var x1 = X(t.node_x[v]), y1 = Y(t.level[v]);
      for (i = 0; i < t.parents[v].length; i++) {
        var r = t.parents[v][i], x0, y0;
        if (r < t.leaves) { x0 = X(t.leaf_x[r]); y0 = Y(0); }
        else { x0 = X(t.node_x[r - t.leaves]); y0 = Y(t.level[r - t.leaves]); }
        el('line', {x1: x0, y1: y0, x2: x1, y2: y1,
                    stroke: '#d9d9d9', 'stroke-width': 1.1});
      }
    }
    // the output reads every readout node; it stands above their mean
    var outX = 0;
    for (i = 0; i < t.readout.length; i++) outX += t.node_x[t.readout[i]];
    outX = X(t.readout.length ? outX / t.readout.length : xs / 2);
    for (i = 0; i < t.readout.length; i++) {
      var rv0 = t.readout[i];
      el('line', {x1: X(t.node_x[rv0]), y1: Y(t.level[rv0]), x2: outX, y2: Y(outLv),
                  stroke: '#d9d9d9', 'stroke-width': 1.1});
    }
    for (i = 0; i < t.leaves; i++) {
      el('rect', {x: X(t.leaf_x[i]) - LEAF / 2, y: Y(0) - LEAF / 2, width: LEAF, height: LEAF,
                  fill: t.colors[t.leaf_owner[i]], stroke: '#fff',
                  'stroke-width': 0.9});
    }
    var groups = [];
    for (v = 0; v < t.parents.length; v++) {
      var g = el('g', {'class': opts.clickable ? 'tnode' : ''});
      var isRead = t.readout.indexOf(v) >= 0;
      el('circle', {cx: X(t.node_x[v]), cy: Y(t.level[v]), r: isRead ? READ_R : NODE_R,
                    fill: t.colors[v], stroke: '#fff', 'stroke-width': 1.1}, g);
      var tx = el('text', {x: X(t.node_x[v]) + LDX, y: Y(t.level[v]) - 5,
                           'font-size': LABEL, fill: '#4a4a4a',
                           'font-family': 'system-ui, sans-serif'}, g);
      tx.textContent = 'z' + sub(v);
      if (opts.clickable) {
        var halo = el('circle', {cx: X(t.node_x[v]), cy: Y(t.level[v]), r: NODE_R + 4.9,
                                 fill: 'none', stroke: t.colors[v],
                                 'stroke-width': 1.4, 'stroke-dasharray': '2.5 2.5',
                                 opacity: 0}, g);
        g.style.cursor = 'pointer';
        groups.push({g: g, halo: halo, v: v});
      }
    }
    for (var lv = 1; lv <= maxLv; lv++) {
      var d = el('text', {x: 2, y: Y(lv) + 3, 'font-size': 8,
                          fill: '#9a9a9a', 'font-family': 'system-ui, sans-serif'});
      d.textContent = 'level ' + lv;
    }
    if (t.readout.length) {
      var ox = outX, oy = Y(outLv);
      el('circle', {cx: ox, cy: oy, r: dense ? 7.2 : 8.4, fill: t.output_color || '#000000',
                    stroke: '#fff', 'stroke-width': 1.1});
      var ot = el('text', {x: ox + 10, y: oy - 5, 'font-size': 10.5,
                           fill: '#4a4a4a',
                           'font-family': 'system-ui, sans-serif'});
      ot.textContent = 'output';
    }
    var cap = el('text', {x: 2, y: H - 3, 'font-size': 8.2, fill: '#9a9a9a',
                          'font-family': 'system-ui, sans-serif'});
    cap.textContent = t.caption ||
      'squares: the 8 active inputs (of 32), paired per node';
    return {svg: svg, groups: groups};
  }

  // ------------------------------------------------------------- run store

  function parseRun(meta, buf, wbuf) {
    var S = meta.sections;
    function view(T, name) {
      var s = S[name], n = 1;
      for (var i = 0; i < s.shape.length; i++) n *= s.shape[i];
      return new T(buf, s.offset, n);
    }
    var run = {meta: meta};
    run.xyz = view(Int16Array, 'xyz');
    run.corr = view(Int8Array, 'corr');
    run.gates = [];
    for (var b = 0; b < meta.boundaries; b++) {
      run.gates.push(view(Uint8Array, 'gate' + b));
    }
    // how many transitions carry a binary mask is a property of the arm:
    // the paired runs mask the first two, the published method masks all five
    // a mask is one byte per connection, or one bit (numpy's packbits order)
    // in a bundle that packs it, unpacked here once
    function unpackBits(name) {
      var s = S[name], n = 1;
      for (var i = 0; i < s.shape.length; i++) n *= s.shape[i];
      var packed = new Uint8Array(buf, s.offset, (n + 7) >> 3), out = new Uint8Array(n);
      for (var j = 0; j < n; j++) out[j] = (packed[j >> 3] >> (7 - (j & 7))) & 1;
      return out;
    }
    run.masks = [];
    for (var mb = 0; mb < meta.boundaries; mb++) {
      var ms = S['mask' + mb];
      run.masks.push(!ms ? null : ms.dtype === 'bits' ? unpackBits('mask' + mb)
                                                      : view(Uint8Array, 'mask' + mb));
    }
    // the affine weights travel in their own buffer
    var WS = meta.wsections || {};
    run.weights = [];
    for (var wb = 0; wb < meta.boundaries; wb++) {
      var ws = WS['weight' + wb];
      if (!wbuf || !ws) { run.weights.push(null); continue; }
      var wn = 1;
      for (var wi = 0; wi < ws.shape.length; wi++) wn *= ws.shape[wi];
      run.weights.push(new Int8Array(wbuf, ws.offset, wn));
    }
    run.wScale = meta.w_scale || {};   // one scale per transition
    var counts = meta.levels.counts;
    // the share of active connections per frame (mask 1; where a transition
    // has no mask, a soft gate above one half), for the status line
    var pairsAll = 0, activeN = new Float64Array(meta.frames);
    for (var ab = 0; ab < meta.boundaries; ab++) {
      var perA = counts[ab + 1] * counts[ab], mA = run.masks[ab], gA = run.gates[ab];
      pairsAll += perA;
      for (var fa = 0; fa < meta.frames; fa++) {
        var cntA = 0, baseA = fa * perA, ia;
        if (mA) for (ia = 0; ia < perA; ia++) cntA += mA[baseA + ia];
        else for (ia = 0; ia < perA; ia++) cntA += gA[baseA + ia] >= 128 ? 1 : 0;
        activeN[fa] += cntA;
      }
    }
    run.activeFrac = function (f) { return pairsAll ? activeN[f] / pairsAll : 0; };
    // the display scale of each transition's forward-pass weight W g m: the
    // 99.5th percentile of its magnitude over the active connections of two
    // dozen frames of the run, so one scale serves the whole replay
    run.effScale = [];
    for (var eb = 0; eb < meta.boundaries; eb++) {
      var wq = run.weights[eb], gq = run.gates[eb], mq2 = run.masks[eb];
      if (!wq) { run.effScale.push(1); continue; }
      var perE = counts[eb + 1] * counts[eb], vals = [];
      var stepF = Math.max(1, Math.floor(meta.frames / 24));
      for (var fq = meta.frames - 1; fq >= 0; fq -= stepF) {
        for (var iq = 0, oq = fq * perE; iq < perE; iq++, oq++) {
          if (mq2 ? !mq2[oq] : gq[oq] < 128) continue;
          var vq = Math.abs(wq[oq]) * gq[oq];
          if (vq) vals.push(vq);
        }
      }
      vals.sort(function (a, b) { return a - b; });
      var ref = vals.length ? vals[Math.min(vals.length - 1, Math.floor(vals.length * 0.995))] : 0;
      run.effScale.push((ref || 127 * 255) / (127 * 255) * (run.wScale[eb] || 1));
    }
    // each level's marker: the inputs a circle; the hidden layers a square, a
    // triangle, a diamond and an inverted triangle, filled for the first four
    // and hollow for four more; the output a star
    run.glyphs = []; run.shapes = []; run.hollow = []; run.sizes = [];
    var nLv = counts.length;
    for (var lvm = 0; lvm < nLv; lvm++) {
      var first = lvm === 0, last = lvm === nLv - 1, k4 = (lvm - 1) % 4;
      run.glyphs.push(first ? '\u25cf' : last ? '\u2605'
        : (lvm > 4 ? '\u25a1\u25b3\u25c7\u25bd' : '\u25a0\u25b2\u25c6\u25bc')[k4]);
      run.shapes.push(first ? 'o' : last ? '*' : 's^Dv'[k4]);
      run.hollow.push(!first && !last && lvm > 4);
      run.sizes.push(first ? 4.6 : last ? 8.6 : counts[lvm] > 32 ? 3.0 : 3.6);
    }
    // the matrices' grid: one cell per neuron of the widest layer
    run.grid = Math.max.apply(null, counts.slice(1, -1));
    run.U = 0; run.levelStart = []; run.levelOf = []; run.idxIn = [];
    for (var li = 0; li < counts.length; li++) {
      run.levelStart.push(run.U);
      for (var q = 0; q < counts[li]; q++) {
        run.levelOf.push(li); run.idxIn.push(q);
      }
      run.U += counts[li];
    }
    run.NN = meta.teacher.parents.length;
    // the runs built from a capture carry the association per frame (the
    // node, or -1): the intermediate with the largest |r| on the diagnostic
    // examples, when that |r| exceeds the run's permutation threshold; a run
    // without the section gets the same rule from its correlations here
    run.assoc = S.assoc ? view(Int8Array, 'assoc') : null;
    run.evFrame = meta.events.map(function (e) {
      return meta.steps.indexOf(e); });
    function splitsOf(labels, cmp) {
      var out = [];
      for (var i = 1; i < labels.length; i++) {
        if (cmp ? cmp(labels[i], labels[i - 1])
                : labels[i] !== labels[i - 1]) out.push(i);
      }
      return out;
    }
    run.splitsOf = splitsOf;
    run.maxN = Math.max.apply(null, meta.nmse) * 1.05;

    // per-frame accessors
    run.P = function (f, u, k) {
      return run.xyz[(f * run.U + u) * 3 + k] * meta.xyz_scale; };
    run.rOf = function (f, u, k) { return run.corr[(f * run.U + u) * run.NN + k] / 127; };
    var outLevel = meta.levels.counts.length - 1;
    run.bestOf = function (f, u) {
      var bi = 0, bv = 0;
      for (var k = 0; k < run.NN; k++) {
        var a = Math.abs(run.rOf(f, u, k));
        if (a > bv) { bv = a; bi = k; }
      }
      if (run.assoc && run.levelOf[u] < outLevel) {
        var v = run.assoc[f * run.U + u];
        return v >= 0 ? {node: v, s: Math.abs(run.rOf(f, u, v)), lit: true}
                      : {node: bi, s: bv, lit: false};
      }
      return {node: bi, s: bv, lit: bv >= meta.floor};
    };

    // ---- the matrix registry: geometry, orders, bars, one entry per panel
    var ord = meta.orders;
    var LC = meta.levels.counts, LN = meta.levels.names;
    var seq = [];
    for (var n2 = 0; n2 < LC[0]; n2++) seq.push(n2);
    var tcol = meta.teacher.colors;
    function modBar(mods) {
      return mods.map(function (m) {
        var v = m < 0 ? -1 : ord.mod_node[m];
        return v < 0 ? '#dedede' : tcol[v];
      });
    }
    function nodeBar(nodes) {
      return nodes.map(function (v) { return v < 0 ? '#dedede' : tcol[v]; });
    }
    function withGeo(base, extra) {
      var out = {};
      for (var k in base) out[k] = base[k];
      for (k in extra) out[k] = extra[k];
      return out;
    }
    if (ord.byLevel) {
      // one entry per transition from the per-level display orders: rows the
      // receiving layer, columns the sending one, strips by tracked node
      run.matDefs = {};
      var BL = ord.byLevel;
      for (var tb = 0; tb < meta.boundaries; tb++) {
        var rl = tb + 1, outRow = rl === LC.length - 1;
        run.matDefs['g' + tb] = {
          data: run.gates[tb], gate: true, nr: LC[rl], nc: LC[tb], rlv: rl, slv: tb,
          rows: outRow ? [0] : BL[rl].order, cols: BL[tb].order,
          rowBar: outRow ? ['#555555'] : nodeBar(BL[rl].nodes), colBar: nodeBar(BL[tb].nodes),
          rowSplits: outRow ? [] : splitsOf(BL[rl].nodes), colSplits: splitsOf(BL[tb].nodes),
          rname: LN[rl], cname: tb === 0 ? 'input' : LN[tb]};
      }
    } else {
    var inBar = seq.map(function (i) {
      if (i >= ord.in_nodes.length) return null;
      return ord.in_nodes[i] < 0 ? '#dedede' : tcol[ord.in_nodes[i]]; });
    var sendSplits = splitsOf(ord.send_mod);
    var recvSplits = splitsOf(ord.recv_mod);
    var lv3Splits = splitsOf(ord.lv3_node);
    var lv4Splits = splitsOf(ord.lv4_node);
    var g0geo = {rows: ord.send, cols: ord['in'],
                 rowBar: modBar(ord.send_mod), colBar: inBar,
                 rowSplits: sendSplits, colSplits: ord.in_splits || [2, 4, 6, 8],
                 rname: LN[1], cname: 'input'};
    var g1geo = {rows: ord.recv, cols: ord.send,
                 rowBar: modBar(ord.recv_mod), colBar: modBar(ord.send_mod),
                 rowSplits: recvSplits, colSplits: sendSplits,
                 rname: LN[2], cname: LN[1]};
    run.matDefs = {
      g0: withGeo(g0geo, {data: run.gates[0], gate: true, nr: LC[1], nc: LC[0],
                          rlv: 1, slv: 0}),
      g1: withGeo(g1geo, {data: run.gates[1], gate: true, nr: LC[2], nc: LC[1],
                          rlv: 2, slv: 1}),
      g2: {data: run.gates[2], gate: true, nr: LC[3], nc: LC[2], rlv: 3, slv: 2,
           rows: ord.lv3, cols: ord.recv,
           rowBar: nodeBar(ord.lv3_node), colBar: modBar(ord.recv_mod),
           rowSplits: lv3Splits, colSplits: recvSplits,
           rname: LN[3], cname: LN[2]},
      g3: {data: run.gates[3], gate: true, nr: LC[4], nc: LC[3], rlv: 4, slv: 3,
           rows: ord.lv4, cols: ord.lv3,
           rowBar: nodeBar(ord.lv4_node), colBar: nodeBar(ord.lv3_node),
           rowSplits: lv4Splits, colSplits: lv3Splits,
           rname: LN[4], cname: LN[3]},
      g4: {data: run.gates[4], gate: true, nr: LC[5], nc: LC[4], rlv: 5, slv: 4,
           rows: [0], cols: ord.lv4,
           rowBar: ['#555555'], colBar: nodeBar(ord.lv4_node),
           rowSplits: [], colSplits: lv4Splits,
           rname: LN[5], cname: LN[4]}
    };
    }
    // a masked transition gets a second panel beside its gate, on the same
    // rows and columns, and its gate is drawn knowing which cells survive
    // each transition shows its gate, its mask where the arm applies one, and
    // the affine weight the two of them multiply -- all on the same rows and
    // columns, so a cell means the same connection across the three
    for (var qb = 0; qb < meta.boundaries; qb++) {
      var gdef = run.matDefs['g' + qb];
      if (run.masks[qb]) {
        gdef.masked = run.masks[qb];
        run.matDefs['m' + qb] = withGeo(gdef, {data: run.masks[qb], gate: false,
                                               masked: null});
      }
      if (run.weights[qb]) {
        run.matDefs['w' + qb] = withGeo(gdef, {data: run.weights[qb], gate: false,
                                               weight: true, masked: null});
      }
    }
    run.mats = [];
    for (var rb = 0; rb < meta.boundaries; rb++) {
      run.mats.push('g' + rb);
      if (run.masks[rb]) run.mats.push('m' + rb);
      if (run.weights[rb]) run.mats.push('w' + rb);
    }

    Object.keys(run.matDefs).forEach(function (k) {
      var def = run.matDefs[k];
      def.rowPos = {}; def.colPos = {};
      def.rows.forEach(function (u, i) { def.rowPos[u] = i; });
      def.cols.forEach(function (u, i) { def.colPos[u] = i; });
    });
    return run;
  }

  // Each run travels as a script (assets/<stem>.js, written by
  // build/pack_assets.py) that defines window.__demoRuns[stem] = {meta, hex}:
  // the anonymous host serves every file under a CSP sandbox without
  // allow-same-origin, which turns fetch() of the page's own files into
  // blocked cross-origin requests, while <script src> loads are allowed.
  var HEXV = new Uint8Array(128);
  for (var hv = 0; hv < 16; hv++) {
    HEXV['0123456789abcdef'.charCodeAt(hv)] = hv;
    HEXV['0123456789ABCDEF'.charCodeAt(hv)] = hv;
  }
  function decodeHex(text) {
    var n = text.length >> 1, bytes = new Uint8Array(n);
    for (var i = 0, j = 0; i < n; i++, j += 2) {
      bytes[i] = (HEXV[text.charCodeAt(j)] << 4) | HEXV[text.charCodeAt(j + 1)];
    }
    return bytes.buffer;
  }

  // the host's CDN caches script files for hours without a cache-control
  // header; the version query gives every publish a fresh cache key (bump
  // it here and on every ?v= of index.html together)
  var ASSET_VERSION = '20261003d';

  // Every run is loaded once per visit: its promise is kept, its text is
  // let go once decoded, and `ready` says which runs can be shown at once.
  var store = {}, ready = {};
  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      var el = document.createElement('script');
      el.src = src;
      el.onload = function () { resolve(); };
      el.onerror = function () { reject(new Error('did not load ' + src)); };
      document.head.appendChild(el);
    });
  }

  function loadRun(stem) {
    if (store[stem]) return store[stem];
    var runs = window.__demoRuns = window.__demoRuns || {};
    var wts = window.__demoWeights = window.__demoWeights || {};
    var q = '?v=' + ASSET_VERSION;
    store[stem] = Promise.all([
      runs[stem] ? Promise.resolve() : loadScript('assets/' + stem + '.js' + q),
      wts[stem] ? Promise.resolve() : loadScript('assets/' + stem + '-w.js' + q)
    ]).then(function () {
      if (!runs[stem]) throw new Error('the run script defined no data');
      var run = parseRun(runs[stem].meta, decodeHex(runs[stem].hex),
                         wts[stem] ? decodeHex(wts[stem].hex) : null);
      runs[stem].hex = null;
      if (wts[stem]) wts[stem].hex = null;
      ready[stem] = true;
      return run;
    });
    store[stem].catch(function () { delete store[stem]; });   // a failed load may be retried
    return store[stem];
  }

  // After the first run is on screen, the other tabs' runs load one after
  // another in the background, so a later tab opens at once.
  function preload() {
    var queue = SHOWN.map(function (r) { return r.stem; })
      .filter(function (s) { return !store[s]; });
    (function next() {
      if (!queue.length) return;
      var s = queue.shift();
      (store[s] || loadRun(s)).then(next, next);
    })();
  }

  function announce(stem, run) {
    var r = runOf(stem);
    document.dispatchEvent(new CustomEvent('runchange',
        {detail: {stem: stem, run: run, hideDim: !!(r && r.hideDim)}}));
  }

  // ------------------------------------------------------------------ boot

  var veil = document.getElementById('veil3');
  function fail(msg) {
    veil.innerHTML = '<div>' + msg + '</div>';
    veil.style.display = 'grid';
  }

  var wantRun = /[#&]run=([a-z0-9_-]+)/i.exec(location.hash);
  var firstStem = SHOWN[0].stem;
  if (wantRun) {
    for (var ri = 0; ri < SHOWN.length; ri++) {
      if (SHOWN[ri].stem === wantRun[1]) firstStem = SHOWN[ri].stem;
    }
  }
  loadRun(firstStem).then(function (run) { init(run); })
    .catch(function (err) {
      fail('Could not load the run data (' + err.message + '). ' +
           'Reloading the page usually resolves this.');
    });

  function init(run0) {
    var R = run0;
    var curStem = firstStem;
    var v0 = viewOf(curStem, R);
    var st = {
      f: 0, playing: false, speed: 1, focus: null,
      // each run opens with its own choice of hiding the unassigned neurons
      hideDim: !!(runOf(curStem) && runOf(curStem).hideDim),
      // the turn is off until the reader asks for it ("Auto-rotate"); once
      // on, it waits for the replay: it starts on the frame the first step is
      // shown
      spinning: false, spinArmed: false,
      pinned: {}, nPinned: 0, pinnedEdges: {}, conn: {}, hover: null,
      yaw: v0.yaw, pitch: v0.pitch, zoom: v0.zoom, spinRate: v0.spin,
      // which transitions' connections are drawn ('input', 'hidden', 'all'),
      // and what sets a line's opacity: the binary mask (one opacity for every
      // active connection) or the soft gate (its value, above a minimum)
      edgeSet: 'all', opacity: 'mask', gateMin: 0.5
    };

    // ------------------------------------------------------------ canvases
    function setupCanvas(cv, cssW, cssH) {
      var dpr = window.devicePixelRatio || 1;
      cv.width = Math.round(cssW * dpr);
      cv.height = Math.round(cssH * dpr);
      cv.style.width = cssW + 'px';
      cv.style.height = cssH + 'px';
      var ctx = cv.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return ctx;
    }

    var geoCv = document.getElementById('geo3');
    // gestures land on a transparent zone tracking the projected cube
    // (+10%); the rest of the panel stays free for page scrolling
    var geoZone = document.createElement('div');
    geoZone.className = 'c3d-zone';
    geoCv.parentNode.appendChild(geoZone);
    var curveCv = document.getElementById('curve3');
    // The connection matrices show ONE transition at a time (curB), picked by
    // a tab: its soft gate, its binary mask where the arm has one, and the
    // forward-pass weight the two of them multiply, on the same rows and
    // columns, so a cell means the same connection across the three.
    var KINDS = ['g', 'm', 'w'];
    var matCv = {g: document.getElementById('b3-g'), m: document.getElementById('b3-m'),
                 w: document.getElementById('b3-w')};
    var curB = 0;
    var mtabs = document.getElementById('mtabs3');
    function capName(nm) { return nm.charAt(0).toUpperCase() + nm.slice(1); }
    function transitionName(b) {
      var N = R.meta.levels.names;
      return capName(N[b]) + ' \u2192 ' + capName(N[b + 1]);
    }
    function show(id, on) {
      var el = document.getElementById(id);
      if (el) el.style.display = on ? '' : 'none';
    }
    function visibleKinds() {
      return KINDS.filter(function (k) { return !!R.matDefs[k + curB]; });
    }
    function syncMatTabs() {
      var bs = mtabs.children;
      for (var i = 0; i < bs.length; i++) {
        bs[i].className = 'mtab' + (i === curB ? ' on' : '');
        bs[i].setAttribute('aria-selected', i === curB ? 'true' : 'false');
      }
    }
    function setMatB(b) {
      curB = b;
      if (st.hover && st.hover.type === 'cell') { st.hover = null; tipHide(); }
      syncMatTabs();
      applyShape();
      layout();
      redrawMats();
      drawGeo(st.f);
    }
    // the tabs and the panels follow the loaded run: its number of transitions,
    // and which of them carry a mask and a weight
    function applyShape() {
      if (curB >= R.meta.boundaries) curB = 0;
      mtabs.innerHTML = '';
      for (var b = 0; b < R.meta.boundaries; b++) {
        var t = document.createElement('button');
        t.type = 'button';
        t.setAttribute('role', 'tab');
        t.textContent = transitionName(b);
        t.addEventListener('click', (function (bb) {
          return function () { if (bb !== curB) setMatB(bb); };
        })(b));
        mtabs.appendChild(t);
      }
      syncMatTabs();
      show('mwrap', !!R.matDefs['m' + curB]);
      show('wwrap', !!R.matDefs['w' + curB]);
    }
    applyShape();
    var ctxGeo, ctxCurve, ctxMat = {}, GW, GH, CW, CH, MW;
    var MPAD = 10;

    // The cell size is chosen by layout(). Sized by the width, a cell covers a
    // whole number of DEVICE pixels: the matrices are upscaled nearest-
    // neighbour from one pixel per cell, and a fractional cell would give
    // neighbouring cells different widths. Sized by the height of the
    // dashboard row (FRAC), the cell is exact so the three matrices fill the
    // room, and the cells are drawn smoothed so they stay the same size.
    var CS = 1, FRAC = false;
    function cellSize() { return CS; }
    // the module strips and the cluster splits are drawn on the same grid, as
    // filled rectangles rather than strokes: a stroke of fractional width
    // straddles a device pixel and softens, a snapped fill cannot
    function snap(v) {
      var dpr = window.devicePixelRatio || 1;
      return Math.round(v * dpr) / dpr;
    }
    function hairW() {
      var dpr = window.devicePixelRatio || 1;
      return Math.max(1, Math.round(dpr)) / dpr;
    }

    // a column is one cell wide on the 32-cell grid, or wider for a matrix
    // with fewer columns (the eight inputs), so every panel is one size
    function colW(def) { return cellSize() * R.grid / def.nc; }

    function matHeight(k) {
      var def = R.matDefs[k + curB];
      return MPAD + def.nr * cellSize() + 3;
    }

    function contentW(el) {
      var cs = getComputedStyle(el);
      return el.clientWidth - parseFloat(cs.paddingLeft) -
             parseFloat(cs.paddingRight);
    }

    function layout() {
      // measure every panel BEFORE resizing any canvas (a resize reflows the
      // grid), to the exact content width — a canvas set wider than its box
      // gets clamped by max-width while its inline height stays, which
      // squashes the aspect on narrow screens
      var geoBox = geoCv.parentNode;
      var gw = Math.max(160, Math.floor(contentW(geoBox)));
      var cw = Math.max(150, Math.floor(contentW(curveCv.parentNode)));
      var kinds = visibleKinds(), mw = Infinity;
      kinds.forEach(function (k) {
        mw = Math.min(mw, Math.floor(contentW(matCv[k].parentNode)));
      });
      // the embeddings are square and set the height of the dashboard row
      geoBox.style.minHeight = gw + 'px';
      GW = gw; GH = gw; ctxGeo = setupCanvas(geoCv, GW, GH);
      CW = cw; CH = Math.round(CW * 0.72);
      ctxCurve = setupCanvas(curveCv, CW, CH);
      // the matrices: a cell as wide as the column allows, or, beside the
      // embeddings (the dashboard), as the height the row leaves the stacked
      // three allows, whichever is smaller; the canvases are made tiny first
      // so the room is measured without them
      var dpr = window.devicePixelRatio || 1;
      var cs = Math.max(1, Math.floor((Math.max(90, mw) - MPAD - 2) / R.grid * dpr)) / dpr;
      FRAC = false;
      var matsPanel = document.getElementById('mats3');
      var trio = document.getElementById('trio3');
      if (matsPanel && trio && kinds.length && Math.abs(
            matsPanel.getBoundingClientRect().top -
            geoBox.parentNode.getBoundingClientRect().top) < 2) {
        kinds.forEach(function (k) { matCv[k].style.height = '1px'; });
        var room = trio.clientHeight, heads = 0;
        kinds.forEach(function (k) {
          var p = matCv[k].previousElementSibling, pm = getComputedStyle(p);
          heads += p.offsetHeight + parseFloat(pm.marginTop) + parseFloat(pm.marginBottom);
        });
        var gap = parseFloat(getComputedStyle(trio).rowGap) || 0;
        var nr = R.matDefs[kinds[0] + curB].nr;
        var each = (room - heads - gap * (kinds.length - 1)) / kinds.length - MPAD - 3;
        var csH = Math.floor(each / nr * 100) / 100;
        if (csH >= 1 / dpr && csH < cs) { cs = csH; FRAC = true; }
      }
      CS = cs;
      MW = Math.ceil(cs * R.grid) + MPAD + 2;
      kinds.forEach(function (k) {
        ctxMat[k] = setupCanvas(matCv[k], MW, matHeight(k));
      });
    }

    // -------------------------------------------------------- 3-D geometry
    function rotOf() {
      return {cy: Math.cos(st.yaw), sy: Math.sin(st.yaw),
              cp: Math.cos(st.pitch), sp: Math.sin(st.pitch),
              s: (GW * 0.32 / R.meta.lim3) * st.zoom};
    }
    function project(f, u, rot) {
      var x = R.P(f, u, 0), y = R.P(f, u, 1), z = R.P(f, u, 2);
      var x1 = rot.cy * x + rot.sy * z, z1 = -rot.sy * x + rot.cy * z;
      var y1 = rot.cp * y - rot.sp * z1, z2 = rot.sp * y + rot.cp * z1;
      return [GW / 2 + x1 * rot.s, GH / 2 - y1 * rot.s, z2];
    }

    var screenPts = [];

    function inFocus(b) {
      return st.focus === null || (b.lit && st.focus[b.node]);
    }

    function drawGeo(f) {
      var ctx = ctxGeo;
      ctx.clearRect(0, 0, GW, GH);
      var rot = rotOf();
      var u, i;

      var L = R.meta.lim3 * 0.92;
      var corners = [];
      for (i = 0; i < 8; i++) {
        var cx = (i & 1 ? L : -L), cyy = (i & 2 ? L : -L), cz = (i & 4 ? L : -L);
        var x1 = rot.cy * cx + rot.sy * cz, z1 = -rot.sy * cx + rot.cy * cz;
        var y1 = rot.cp * cyy - rot.sp * z1;
        corners.push([GW / 2 + x1 * rot.s, GH / 2 - y1 * rot.s]);
      }
      ctx.strokeStyle = 'rgba(0,0,0,0.07)';
      ctx.lineWidth = 1;
      [[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]]
        .forEach(function (e) {
          ctx.beginPath();
          ctx.moveTo(corners[e[0]][0], corners[e[0]][1]);
          ctx.lineTo(corners[e[1]][0], corners[e[1]][1]);
          ctx.stroke();
        });

      // the embedding axes through the true origin: coordinates are the raw
      // learned embeddings, so alignment and inner products read off these.
      // A run of the paper's figure is shown in the figure's frame instead:
      // its axes are the cube's, through the cube's centre, and unnamed.
      var raw = !R.meta.figure && !R.meta.display_frame;
      function axPt(k, v) {
        var q = [0, 0, 0];
        q[k] = v;
        var x1 = rot.cy * q[0] + rot.sy * q[2];
        var z1 = -rot.sy * q[0] + rot.cy * q[2];
        var y1 = rot.cp * q[1] - rot.sp * z1;
        return [GW / 2 + x1 * rot.s, GH / 2 - y1 * rot.s];
      }
      ctx.strokeStyle = 'rgba(0,0,0,0.16)';
      ctx.fillStyle = '#9a9a9a';
      ctx.font = '9px system-ui, sans-serif';
      for (i = 0; i < 3; i++) {
        var an = axPt(i, -L), ap = axPt(i, L);
        ctx.beginPath();
        ctx.moveTo(an[0], an[1]);
        ctx.lineTo(ap[0], ap[1]);
        ctx.stroke();
        if (raw) ctx.fillText('e' + '₁₂₃'[i], ap[0] + 3, ap[1] - 3);
      }
      if (raw) {
        ctx.beginPath();
        ctx.arc(GW / 2, GH / 2, 1.6, 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(0,0,0,0.35)';
        ctx.fill();
      }

      var bx = [Infinity, -Infinity], by = [Infinity, -Infinity];
      corners.forEach(function (q) {
        bx[0] = Math.min(bx[0], q[0]); bx[1] = Math.max(bx[1], q[0]);
        by[0] = Math.min(by[0], q[1]); by[1] = Math.max(by[1], q[1]);
      });
      var zpad = 0.1 * Math.max(bx[1] - bx[0], by[1] - by[0]);
      // cube + 10% each side, with a guaranteed 4% escape ring
      bx[0] = Math.max(0.04 * GW, bx[0] - zpad);
      bx[1] = Math.min(0.96 * GW, bx[1] + zpad);
      by[0] = Math.max(0.04 * GH, by[0] - zpad);
      by[1] = Math.min(0.96 * GH, by[1] + zpad);
      var zr = geoCv.getBoundingClientRect();
      var zsf = zr.width / GW;
      geoZone.style.left = (geoCv.offsetLeft + bx[0] * zsf) + 'px';
      geoZone.style.top = (geoCv.offsetTop + by[0] * zsf) + 'px';
      geoZone.style.width = ((bx[1] - bx[0]) * zsf) + 'px';
      geoZone.style.height = ((by[1] - by[0]) * zsf) + 'px';

      var best = [];
      for (u = 0; u < R.U; u++) best.push(R.bestOf(f, u));

      var items = [];
      var proj = [];
      for (u = 0; u < R.U; u++) {
        proj.push(project(f, u, rot));
        items.push({k: 0, d: proj[u][2], u: u});
      }
      var LS = R.levelStart, LC = R.meta.levels.counts;
      // the connections drawn: those of the transitions the "Edges" choice
      // names; one line per active connection (mask 1) in the binary-mask
      // view, one per connection whose soft gate clears the minimum in the
      // soft-gate view
      var soft = st.opacity === 'soft';
      var shown = 0;
      for (var bq = 0; bq < R.meta.boundaries; bq++) {
        if (!(st.edgeSet === 'all' || (st.edgeSet === 'input' ? bq === 0 : bq > 0))) continue;
        var gate = R.gates[bq], mask = R.masks[bq];
        var r0 = LS[bq + 1], s0 = LS[bq], nr = LC[bq + 1], nc = LC[bq];
        var stride = nr * nc, base = f * stride;
        for (var rr = 0; rr < nr; rr++) {
          for (var cc2 = 0; cc2 < nc; cc2++) {
            var o = base + rr * nc + cc2;
            var g = gate[o] / 255;
            if (soft ? g < st.gateMin : (mask ? !mask[o] : g < 0.5)) continue;
            shown++;
            var ra = proj[r0 + rr], sa = proj[s0 + cc2];
            items.push({k: 1, d: (ra[2] + sa[2]) / 2, a: ra, b: sa,
                        u: r0 + rr, su: s0 + cc2, g: g});
          }
        }
      }
      items.sort(function (a, b) { return a.d - b.d; });
      // a network with many more connections than the others (the depth-5
      // tree's ten thousand) keeps about the others' ink: each line fainter
      var dens = inkScale(shown);
      // while a path or a connection is highlighted, the rest steps back
      var quiet = st.nPinned || Object.keys(st.conn).length ||
                  Object.keys(st.pinnedEdges).length ||
                  (st.hover && st.hover.type === 'unit') ? 0.5 : 1;

      function trail(u2, strong) {
        // each segment wears the assignment color the neuron had at that
        // moment, so the path itself shows what it went through; the path so
        // far sits on a white halo, clear of the connections behind it
        function colAt(fr) {
          var lb = R.bestOf(fr, u2);
          return unitCol(R, u2, lb, '#8d8d8d');
        }
        var pts = [], seg;
        for (seg = 0; seg < R.meta.frames; seg++) pts.push(project(seg, u2, rot));
        ctx.lineCap = 'round'; ctx.lineJoin = 'round';
        ctx.beginPath();
        ctx.moveTo(pts[0][0], pts[0][1]);
        for (seg = 1; seg <= f; seg++) ctx.lineTo(pts[seg][0], pts[seg][1]);
        ctx.strokeStyle = 'rgba(255,255,255,0.9)';
        ctx.lineWidth = strong ? 4.8 : 4.2;
        ctx.stroke();
        for (seg = 1; seg < R.meta.frames; seg++) {
          ctx.beginPath();
          ctx.moveTo(pts[seg - 1][0], pts[seg - 1][1]);
          ctx.lineTo(pts[seg][0], pts[seg][1]);
          ctx.strokeStyle = rgba(colAt(seg), seg <= f ? (strong ? 0.95 : 0.88) : 0.15);
          ctx.lineWidth = seg <= f ? (strong ? 2.4 : 2) : 1;
          ctx.stroke();
        }
        ctx.lineCap = 'butt'; ctx.lineJoin = 'miter';
        ctx.beginPath(); ctx.arc(pts[0][0], pts[0][1], 2.6, 0, 2 * Math.PI);
        ctx.fillStyle = rgba(colAt(0), 0.95); ctx.fill();
      }
      for (var pk in st.pinned) trail(+pk, false);
      if (st.hover && st.hover.type === 'unit' && !(st.hover.u in st.pinned)) {
        trail(st.hover.u, true);
      }

      var ringUnits = [];
      if (st.hover) {
        if (st.hover.type === 'unit') ringUnits = [st.hover.u];
        else if (st.hover.units) ringUnits = st.hover.units;
      }

      var SIZES = R.sizes;
      items.forEach(function (it) {
        if (it.k === 1) {
          var eb = best[it.u];
          var ecol = unitCol(R, it.u, eb, '#b5b5b5');
          if (!inFocus(eb)) return;
          if (st.hideDim && (!eb.lit || !best[it.su].lit)) return;
          ctx.beginPath();
          ctx.moveTo(it.a[0], it.a[1]);
          ctx.lineTo(it.b[0], it.b[1]);
          // thin and light: the connections are the context, the highlighted
          // paths and traces are the figure
          ctx.globalAlpha = (soft ? 0.04 + 0.30 * it.g : 0.18) * dens * quiet;
          ctx.strokeStyle = ecol;
          ctx.lineWidth = 0.7;
          ctx.stroke();
          ctx.globalAlpha = 1;
          return;
        }
        var u3 = it.u;
        var b3 = best[u3];
        if (st.hideDim && !b3.lit) return;
        var p = proj[u3];
        var lv = R.levelOf[u3];
        var size = SIZES[lv] * (b3.lit ? 1 : 0.62) * Math.min(st.zoom, 1.6);
        var col = unitCol(R, u3, b3, '#c4c4c4');
        // an assigned neuron wears its node's exact colour: the colour of the
        // target graph, the matrix strips and the final-wiring panels
        var alpha = b3.lit ? 1 : 0.5;
        if (!inFocus(b3)) alpha *= 0.12;
        drawMark(ctx, R, lv, p[0], p[1], size, col, alpha, 0.7);
      });
      // matrix-selected connections: drawn last and unconditionally, so a
      // hovered or pinned cell is always visible whatever the view filters
      function hiEdge(ru, su, col) {
        var a = proj[ru], b = proj[su];
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
        ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 3.6; ctx.stroke();
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
        ctx.strokeStyle = col; ctx.lineWidth = 1.8; ctx.stroke();
        [ru, su].forEach(function (u4) {
          var q = proj[u4];
          ctx.beginPath();
          ctx.arc(q[0], q[1], SIZES[R.levelOf[u4]] + 3.5, 0, 2 * Math.PI);
          ctx.strokeStyle = col;
          ctx.lineWidth = 1.4;
          ctx.stroke();
        });
      }
      for (var pe in st.pinnedEdges) {
        var E = st.pinnedEdges[pe];
        hiEdge(E[0], E[1], ACCENT);
      }
      if (st.hover && st.hover.type === 'cell' && st.hover.edge) {
        hiEdge(st.hover.edge[0], st.hover.edge[1], INK);
      }
      // neuron tracing: every current in/out connection of each traced
      // neuron — blue into it, magenta out of it, arrows pointing forward
      for (var cu in st.conn) {
        var CU = +cu;
        connEdgesOf(R, f, CU, st.gateMin).forEach(function (Ez) {
          var pa = proj[Ez.su], pb = proj[Ez.ru];
          var ccol = Ez.dirIn ? ACCENT : CONN_OUT;
          ctx.globalAlpha = Ez.kept ? 0.95 : 0.3 + 0.65 * Ez.g;
          ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]);
          ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 4.2; ctx.stroke();
          ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]);
          ctx.strokeStyle = ccol; ctx.lineWidth = 2.1; ctx.stroke();
          arrowHead(ctx, pa, pb, ccol);
          ctx.globalAlpha = 1;
        });
        var selp = proj[CU];
        ctx.beginPath();
        ctx.arc(selp[0], selp[1], SIZES[R.levelOf[CU]] + 4.5, 0, 2 * Math.PI);
        ctx.strokeStyle = INK;
        ctx.lineWidth = 1.8;
        ctx.stroke();
      }
      for (var ri = 0; ri < ringUnits.length; ri++) {
        var pr = proj[ringUnits[ri]];
        ctx.beginPath();
        ctx.arc(pr[0], pr[1], SIZES[R.levelOf[ringUnits[ri]]] + 4.5, 0, 2 * Math.PI);
        ctx.strokeStyle = INK;
        ctx.lineWidth = 1.3;
        ctx.stroke();
      }
      screenPts = [];
      items.forEach(function (it) {
        if (it.k === 0) {
          screenPts.push({u: it.u, x: proj[it.u][0], y: proj[it.u][1]});
        }
      });
      var ec = document.getElementById('edge-count');
      if (ec) ec.textContent = shown ? fmt(shown) + ' connections drawn' : '';
    }

    // ---------------------------------------------------- matrix rendering
    var imgCache = {};

    function cellImage(k, f) {
      var b = curB, key = curStem + '|' + k + b + '|' + f;
      if (imgCache[key]) return imgCache[key];
      var def = R.matDefs[k + b];
      var img = new ImageData(def.nc, def.nr);
      var stride = def.nr * def.nc, base = f * stride;
      var G = R.gates[b], M = R.masks[b], W = R.weights[b];
      // the forward-pass weight W g m in units of the transition's display
      // scale, on the diverging ramp: blue negative, white zero, red positive
      var ws = W ? (R.wScale[b] || 1) / 127 / 255 / R.effScale[b] : 0;
      for (var r = 0; r < def.nr; r++) {
        for (var c = 0; c < def.nc; c++) {
          var at = base + def.rows[r] * def.nc + def.cols[c];
          var px, o = (r * def.nc + c) * 4;
          if (k === 'g') px = magmaLut[G[at]];
          else if (k === 'm') px = M[at] ? [26, 26, 26] : [255, 255, 255];
          else {
            var v = M && !M[at] ? 0 : W[at] * G[at] * ws;
            px = divLut[Math.max(-127, Math.min(127, Math.round(v * 127))) + 127];
          }
          img.data[o] = px[0]; img.data[o + 1] = px[1];
          img.data[o + 2] = px[2]; img.data[o + 3] = 255;
        }
      }
      imgCache[key] = img;
      return img;
    }

    var scratch = document.createElement('canvas');
    var scratchCtx = scratch.getContext('2d');
    var hires = document.createElement('canvas');
    var hiresCtx = hires.getContext('2d');

    function drawMat(k, f) {
      var ctx = ctxMat[k];
      var def = R.matDefs[k + curB];
      if (!ctx || !def) return;
      var cs = cellSize();
      var side = cs * R.grid;
      var cw = colW(def);              // an 8-input matrix fills the same width
      var h = def.nr * cs;
      var x0 = MPAD, y0 = MPAD;
      ctx.clearRect(0, 0, MW, matHeight(k));
      scratch.width = def.nc; scratch.height = def.nr;
      scratchCtx.putImageData(cellImage(k, f), 0, 0);
      if (FRAC) {
        // an exact cell: the cells are enlarged crisp to at least twice the
        // device resolution, then drawn smoothed, so every cell is one size
        var dprm = window.devicePixelRatio || 1;
        var kx = Math.ceil(cw * dprm) * 2, ky = Math.ceil(cs * dprm) * 2;
        hires.width = def.nc * kx; hires.height = def.nr * ky;
        hiresCtx.imageSmoothingEnabled = false;
        hiresCtx.drawImage(scratch, 0, 0, hires.width, hires.height);
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(hires, x0, y0, side, h);
      } else {
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(scratch, x0, y0, side, h);
      }

      var i, x, y;
      var hw = hairW(), gap = Math.max(hw, snap(cs) - snap(cs - 1));
      var barT = snap(3.2), barOff = snap(5);
      for (i = 0; i < def.nc; i++) {
        if (!def.colBar[i]) continue;
        ctx.fillStyle = def.colBar[i];
        x = snap(x0 + i * cw);
        ctx.fillRect(x, y0 - barOff, Math.max(hw, snap(cw) - gap), barT);
      }
      for (i = 0; i < def.nr; i++) {
        if (!def.rowBar[i]) continue;
        ctx.fillStyle = def.rowBar[i];
        y = snap(y0 + i * cs);
        ctx.fillRect(x0 - barOff, y, barT, Math.max(hw, snap(cs) - gap));
      }
      ctx.fillStyle = '#3a86c8';
      for (i = 0; i < def.colSplits.length; i++) {
        ctx.fillRect(snap(x0 + def.colSplits[i] * cw), y0, hw, h);
      }
      for (i = 0; i < def.rowSplits.length; i++) {
        ctx.fillRect(x0, snap(y0 + def.rowSplits[i] * cs), side, hw);
      }
      ctx.fillStyle = '#c9c9c9';
      ctx.fillRect(x0, y0, side, hw);
      ctx.fillRect(x0, snap(y0 + h) - hw, side, hw);
      ctx.fillRect(x0, y0, hw, h);
      ctx.fillRect(snap(x0 + side) - hw, y0, hw, h);

      function markCell(ru, su, col) {
        if (R.levelOf[ru] !== def.rlv || R.levelOf[su] !== def.slv) return;
        var rr = def.rowPos[ru - R.levelStart[def.rlv]];
        var cc = def.colPos[su - R.levelStart[def.slv]];
        if (rr === undefined || cc === undefined) return;
        ctx.strokeStyle = col;
        ctx.lineWidth = 1.6;
        ctx.strokeRect(snap(x0 + cc * cw) + 0.5, snap(y0 + rr * cs) + 0.5,
                       snap(cw) - 1, snap(cs) - 1);
      }
      for (var pe in st.pinnedEdges) {
        var E = st.pinnedEdges[pe];
        markCell(E[0], E[1], ACCENT);
      }
      if (st.hover && st.hover.type === 'cell' && st.hover.edge &&
          st.hover.m !== k) {
        markCell(st.hover.edge[0], st.hover.edge[1], INK);
      }
      if (st.hover && st.hover.type === 'cell' && st.hover.m === k) {
        x = x0 + st.hover.c * cw; y = y0 + st.hover.r * cs;
        ctx.strokeStyle = INK;
        ctx.lineWidth = 1.4;
        ctx.strokeRect(x, y, cw, cs);
        ctx.fillStyle = 'rgba(0,0,0,0.06)';
        ctx.fillRect(x0, y, side, cs);
        ctx.fillRect(x, y0, cw, h);
      }
    }

    // ----------------------------------------------------- curve rendering
    var CM = {l: 37, r: 8, t: 8, b: 21};
    function cx(s) { return CM.l + (s / R.meta.total_steps) * (CW - CM.l - CM.r); }
    function cy(v) { return CH - CM.b - (v / R.maxN) * (CH - CM.t - CM.b); }

    function drawCurve(f) {
      var ctx = ctxCurve;
      ctx.clearRect(0, 0, CW, CH);
      var i;
      var steps = R.meta.steps, nmse = R.meta.nmse, events = R.meta.events;
      ctx.strokeStyle = '#f0f0f0';
      ctx.lineWidth = 1;
      for (i = 0; i <= 4; i++) {
        var gy = cy(R.maxN * i / 4);
        ctx.beginPath(); ctx.moveTo(CM.l, gy); ctx.lineTo(CW - CM.r, gy); ctx.stroke();
      }
      for (i = 0; i < events.length; i++) {
        ctx.strokeStyle = events[i] <= steps[f] ? '#b9c6dd' : '#e8e8e8';
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        ctx.moveTo(cx(events[i]), CM.t); ctx.lineTo(cx(events[i]), CH - CM.b);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      function path(from, to) {
        ctx.beginPath();
        for (i = from; i <= to; i++) {
          var px = cx(steps[i]), py = cy(nmse[i]);
          if (i === from) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.stroke();
      }
      var F = R.meta.frames;
      ctx.strokeStyle = FUTURE; ctx.lineWidth = 1.6; path(f, F - 1);
      ctx.strokeStyle = INK; ctx.lineWidth = 2; path(0, f);
      ctx.beginPath();
      ctx.arc(cx(steps[f]), cy(nmse[f]), 3.4, 0, 2 * Math.PI);
      ctx.fillStyle = ACCENT; ctx.fill();
      ctx.lineWidth = 1.4; ctx.strokeStyle = '#fff'; ctx.stroke();

      ctx.fillStyle = '#9a9a9a';
      ctx.font = '9px system-ui, sans-serif';
      ctx.textAlign = 'right';
      for (i = 0; i <= 4; i += 2) {
        ctx.fillText((R.maxN * i / 4).toFixed(1), CM.l - 4, cy(R.maxN * i / 4) + 3);
      }
      ctx.textAlign = 'center';
      var tickStep = R.meta.total_steps > 96000 ? 32000
                   : R.meta.total_steps > 40000 ? 16000 : 8000;
      for (i = 0; i <= R.meta.total_steps; i += tickStep) {
        var lab = i >= 1000 ? (i / 1000) + 'k' : String(i);
        var lw = ctx.measureText(lab).width;
        ctx.fillText(lab, Math.min(cx(i), CW - lw / 2 - 1), CH - 8);
      }
      ctx.textAlign = 'left';
    }

    // ------------------------------------------------------ target-graph key
    // rebuilt on every run switch: the tabs have different targets
    var tk = null;
    var keyHolder = document.getElementById('teacher3-svg');
    function buildKey() {
      tk = teacherSVG(R.meta.teacher, {clickable: true});
      keyHolder.innerHTML = '';
      keyHolder.appendChild(tk.svg);
      tk.groups.forEach(function (g) {
        g.g.addEventListener('click', function (ev) {
          setFocus(g.v, ev.shiftKey);
        });
      });
    }
    buildKey();
    var focusChip = document.getElementById('focus3-chip');

    function focusList() {
      var vs = [];
      for (var v in st.focus) vs.push(+v);
      vs.sort();
      return vs;
    }
    function syncFocus() {
      tk.groups.forEach(function (g) {
        g.halo.setAttribute('opacity',
          st.focus !== null && st.focus[g.v] ? 1 : 0);
      });
      if (st.focus !== null) {
        var names = focusList().map(function (v) { return 'z' + sub(v); });
        focusChip.textContent = 'isolating ' + names.join(' + ') +
                                ' — click to show all';
        focusChip.classList.add('on');
      } else focusChip.classList.remove('on');
      drawGeo(st.f);
    }
    function setFocus(v, additive) {
      if (v === null) { st.focus = null; syncFocus(); return; }
      if (additive && st.focus !== null) {
        if (st.focus[v]) {
          delete st.focus[v];
          if (!focusList().length) st.focus = null;
        } else st.focus[v] = 1;
      } else {
        var only = st.focus !== null && st.focus[v] && focusList().length === 1;
        st.focus = only ? null : {};
        if (st.focus !== null) st.focus[v] = 1;
      }
      syncFocus();
    }
    focusChip.addEventListener('click', function () { setFocus(null); });

    // --------------------------------------------------------- transport
    var slider = document.getElementById('scrub3-range');
    var fill = document.getElementById('scrub3-fill');
    var ticks = document.getElementById('scrub3-ticks');
    var playBtn = document.getElementById('play3');
    var speedBtn = document.getElementById('speed3');
    var restartBtn = document.getElementById('restart3');
    var readout = document.getElementById('readout3');
    var evrow = document.getElementById('evrow3');
    var playSvg = '<svg width="15" height="16" viewBox="0 0 15 16"><path d="M2 1.5 L13.5 8 L2 14.5 Z" fill="currentColor"/></svg>';
    var pauseSvg = '<svg width="14" height="16" viewBox="0 0 14 16"><rect x="1.5" width="3.6" height="16" fill="currentColor"/><rect x="8.9" width="3.6" height="16" fill="currentColor"/></svg>';

    var tickEls = [], evBtns = [];
    function buildEvents() {
      ticks.innerHTML = '';
      while (evrow.children.length > 1) evrow.removeChild(evrow.lastChild);
      tickEls = []; evBtns = [];
      slider.max = R.meta.frames - 1;
      var F = R.meta.frames;
      R.meta.events.forEach(function (e, i) {
        var t = document.createElement('i');
        t.style.left = (R.evFrame[i] / (F - 1) * 100) + '%';
        ticks.appendChild(t);
        tickEls.push(t);
        var b = document.createElement('button');
        b.className = 'evbtn';
        b.textContent = (e / 1000) + 'k';
        b.title = 'jump to connectivity change ' + (i + 1) + ' (step ' + fmt(e) + ')';
        b.addEventListener('click', function () { seek(R.evFrame[i]); });
        evrow.appendChild(b);
        evBtns.push(b);
      });
      evrow.parentNode.style.display = R.meta.events.length ? '' : 'none';
    }

    function updateChrome(f) {
      slider.value = f;
      fill.style.width = (f / (R.meta.frames - 1) * 100) + '%';
      var steps = R.meta.steps, events = R.meta.events;
      for (var i = 0; i < events.length; i++) {
        tickEls[i].className = events[i] <= steps[f] ? 'fired' : '';
        evBtns[i].className = 'evbtn' + (events[i] <= steps[f] ? ' fired' : '');
      }
      readout.innerHTML = 'Step <b>' + fmt(steps[f]) + '</b> / ' +
        fmt(R.meta.total_steps) +
        ' &nbsp;·&nbsp; Validation NMSE <b>' + R.meta.nmse[f].toFixed(3) + '</b>' +
        ' &nbsp;·&nbsp; Active connections <b>' +
        (R.activeFrac(f) * 100).toFixed(0) + '%</b>';
    }

    function drawAll() {
      var f = st.f;
      drawGeo(f);
      visibleKinds().forEach(function (k) { drawMat(k, f); });
      drawCurve(f);
      updateChrome(f);
    }

    function seek(f) {
      st.f = Math.max(0, Math.min(R.meta.frames - 1, f));
      drawAll();
    }

    var rafLast = null, acc = 0;
    function setPlaying(p) {
      st.playing = p;
      holdUntil = 0;
      playBtn.innerHTML = p ? pauseSvg : playSvg;
      playBtn.setAttribute('aria-label', p ? 'pause' : 'play');
      rafLast = null;
    }
    playBtn.addEventListener('click', function () { setPlaying(!st.playing); });
    restartBtn.addEventListener('click', function () { seek(0); setPlaying(true); });
    var speeds = [1, 2, 4, 0.5];
    function setSpeed(v) {
      st.speed = v;
      speedBtn.textContent = (v + '×').replace('0.5', '½');
    }
    setSpeed(st.speed);
    speedBtn.addEventListener('click', function () {
      var i = (speeds.indexOf(st.speed) + 1) % speeds.length;
      setSpeed(speeds[i]);
    });
    slider.addEventListener('input', function () {
      setPlaying(false);
      seek(+slider.value);
    });

    var onScreen = true;
    if (window.IntersectionObserver) {
      var watched = document.getElementById('instrument3') || geoCv;
      new IntersectionObserver(function (entries) {
        entries.forEach(function (e) { onScreen = e.isIntersecting; });
      }, {rootMargin: '80px'}).observe(watched);
    }

    var END_HOLD = 10000;                  // ms to rest on the last frame
    var holdUntil = 0;
    var spinLast = null;                   // yaw only: a steady turn
    // After the reader turns or zooms the view, the rotation waits SPIN_REST
    // ms before it goes on (the pause button counts the wait down, so there
    // is time to stop it for good), then comes back to speed over SPIN_EASE.
    var SPIN_REST = 4000, SPIN_EASE = 900;
    // a tick counts at most this many ms, so a stall (a tab being built)
    // slows the replay for a moment instead of skipping steps
    var MAX_TICK = 100;
    var restUntil = 0, easeFrom = -Infinity;
    function now_() {
      return window.performance ? performance.now() : Date.now();
    }
    function holdSpin() {
      if (!st.spinning) return;
      restUntil = now_() + SPIN_REST;
      easeFrom = restUntil;
    }
    function syncRest(now) {
      if (!spinBtn) return;
      var left = st.spinning && !drag && now < restUntil
        ? (restUntil - now) / SPIN_REST : (st.spinning && drag ? 1 : 0);
      var on = left > 0;
      if (on !== spinBtn.classList.contains('resting')) {
        spinBtn.classList.toggle('resting', on);
        spinBtn.title = on ? 'the rotation resumes when this bar runs out' : '';
      }
      if (on) spinBtn.style.setProperty('--rest', left.toFixed(3));
    }
    function tick(now) {
      var live = onScreen && !document.hidden;
      var spin = st.spinning && st.spinArmed && live && !drag && now >= restUntil;
      if (spin) {
        if (spinLast === null) spinLast = now;
        var gain = Math.min(1, Math.max(0, (now - easeFrom) / SPIN_EASE));
        st.yaw += Math.min(now - spinLast, MAX_TICK) / 1000 * st.spinRate * gain;
        spinLast = now;
      } else spinLast = null;
      syncRest(now);
      var drew = false;
      if (st.playing && live && holdUntil) {
        rafLast = now;
        acc = 0;
        if (now >= holdUntil) {
          holdUntil = 0;
          seek(0);
          drew = true;
        }
      } else if (st.playing && live) {
        if (rafLast === null) rafLast = now;
        acc += Math.min(now - rafLast, MAX_TICK) / 1000 * BASE_FPS * st.speed;
        rafLast = now;
        var adv = Math.floor(acc);
        if (adv > 0) {
          acc -= adv;
          st.spinArmed = true;        // the turn starts with the replay's first step
          var nf = st.f + adv, last = R.meta.frames - 1;
          if (nf >= last) { nf = last; holdUntil = now + END_HOLD; }
          seek(nf);
          drew = true;
        }
      } else rafLast = null;
      if (spin && !drew) drawGeo(st.f);
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);

    document.addEventListener('keydown', function (e) {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (!onScreen) return;
      if (e.code === 'Space') { e.preventDefault(); setPlaying(!st.playing); }
      else if (e.key === 'ArrowRight') { setPlaying(false); seek(st.f + (e.shiftKey ? 10 : 1)); }
      else if (e.key === 'ArrowLeft') { setPlaying(false); seek(st.f - (e.shiftKey ? 10 : 1)); }
      else if (e.key === 'Escape') { setFocus(null); }
    });

    // ----------------------------------------------------------- tooltip
    var tip = document.getElementById('tip3');
    var inst = document.getElementById('instrument3');

    function tipShow(clientX, clientY, html) {
      var r = inst.getBoundingClientRect();
      tip.innerHTML = html;
      tip.style.display = 'block';
      var x = clientX - r.left + 14, y = clientY - r.top + 14;
      if (x > r.width - 260) x = clientX - r.left - tip.offsetWidth - 10;
      if (y > r.height - 150) y = clientY - r.top - tip.offsetHeight - 10;
      tip.style.left = x + 'px';
      tip.style.top = y + 'px';
    }
    function tipHide() { tip.style.display = 'none'; }

    function unitName(u) {
      var lv = R.levelOf[u];
      var names = R.meta.levels.names;
      return (lv === 0 ? 'input' : names[lv] + ' neuron') +
             (R.meta.levels.counts[lv] > 1 ? ' ' + R.idxIn[u] : '');
    }
    function inputNote(i) {
      var t = R.meta.teacher;
      for (var v = 0; v < t.active_raw.length; v++) {
        if (t.active_raw[v].indexOf(i) >= 0) {
          return 'active — feeds z' + sub(v);
        }
      }
      return 'inactive';
    }
    function profileHtml(f, u) {
      var rows = '';
      for (var k = 0; k < R.NN; k++) {
        var r = R.rOf(f, u, k);
        rows += '<span class="lab">z' + sub(k) + '</span>' +
          '<span class="bar"><i style="width:' + (Math.abs(r) * 100).toFixed(0) +
          '%;background:' + R.meta.teacher.colors[k] + '"></i></span>' +
          '<span class="val">' + (r >= 0 ? '+' : '−') +
          Math.abs(r).toFixed(2) + '</span>';
      }
      return '<div class="bars">' + rows + '</div>';
    }

    // ------------------------------------------------ 3-D panel interaction
    function setSpin(on) {
      st.spinning = on;
      if (on) st.spinArmed = true;
      restUntil = 0;
      if (spinBtn) {
        spinBtn.classList.toggle('on', on);
        spinBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
        if (!on) { spinBtn.classList.remove('resting'); spinBtn.title = ''; }
      }
    }
    function hitUnit(e) {
      var r = geoCv.getBoundingClientRect();
      var mx = e.clientX - r.left, my = e.clientY - r.top;
      var bu = null, bd = 13 * 13;
      for (var k = screenPts.length - 1; k >= 0; k--) {
        var s = screenPts[k];
        var d = (s.x - mx) * (s.x - mx) + (s.y - my) * (s.y - my);
        if (d < bd) { bd = d; bu = s.u; }
      }
      return bu;
    }
    var connNote = document.getElementById('conn3-note');
    function syncConnNote() {
      var n = Object.keys(st.conn).length;
      connNote.innerHTML = n
        ? 'tracing ' + n + ' — <span style="color:' + ACCENT +
          '">■</span> incoming · <span style="color:' + CONN_OUT +
          '">■</span> outgoing · arrows point forward'
        : '';
    }
    var lastConnTs = 0;
    function toggleConn(u) {
      if (u === null) return;
      if (st.conn[u]) delete st.conn[u];
      else if (Object.keys(st.conn).length < 8) st.conn[u] = 1;
      lastConnTs = Date.now();
      syncConnNote();
      drawGeo(st.f);
    }
    var drag = null;
    geoZone.addEventListener('pointerdown', function (e) {
      drag = {x: e.clientX, y: e.clientY, moved: false, t0: Date.now()};
      try { geoZone.setPointerCapture(e.pointerId); } catch (err) {}
    });
    geoZone.addEventListener('pointerup', function (e) {
      try { geoZone.releasePointerCapture(e.pointerId); } catch (err) {}
      if (drag && drag.moved) holdSpin();
      if (drag && !drag.moved && e.button !== 2) {
        var u = hitUnit(e);
        if (u !== null) {
          setSpin(false);          // the reader is studying one neuron
          if (e.shiftKey || Date.now() - drag.t0 > 500) {
            // shift-click, or a touch long-press: trace connections
            if (Date.now() - lastConnTs > 400) toggleConn(u);
          } else {
            if (u in st.pinned) { delete st.pinned[u]; st.nPinned--; }
            else if (st.nPinned < 12) { st.pinned[u] = 1; st.nPinned++; }
            drawGeo(st.f);
          }
        }
      }
      drag = null;
    });
    geoZone.addEventListener('contextmenu', function (e) {
      e.preventDefault();
      var u = hitUnit(e);
      if (u === null) return;
      setSpin(false);
      if (Date.now() - lastConnTs > 400) toggleConn(u);
    });
    geoZone.addEventListener('pointercancel', function () { drag = null; });
    geoZone.addEventListener('pointermove', function (e) {
      if (drag) {
        var dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        if (Math.abs(dx) + Math.abs(dy) > 2) drag.moved = true;
        if (drag.moved) {
          st.yaw += dx * 0.011;
          st.pitch = Math.max(-1.5, Math.min(1.5, st.pitch + dy * 0.011));
          drag.x = e.clientX; drag.y = e.clientY;
          holdSpin();
          tipHide();
          drawGeo(st.f);
          return;
        }
      }
      var bu = hitUnit(e);
      if (bu === null) {
        if (st.hover) { st.hover = null; tipHide(); drawGeo(st.f); redrawMats(); }
        return;
      }
      st.hover = {type: 'unit', u: bu};
      var b = R.bestOf(st.f, bu);
      tipShow(e.clientX, e.clientY,
        '<div class="tt-head">' + unitName(bu) + '</div>' +
        (R.levelOf[bu] === 0
          ? '<div class="tt-dim">' + inputNote(R.idxIn[bu]) + '</div>' : '') +
        (b.lit
          ? '<div>associated with <b>z' + sub(b.node) + '</b> · |r| = ' + b.s.toFixed(2) + '</div>'
          : '<div class="tt-dim">' + (R.assoc ? UNASSIGNED : 'below the permutation floor') +
            '</div>') +
        profileHtml(st.f, bu) +
        '<div class="tt-dim" style="margin-top:3px">click: highlight its path · ' +
        'shift-click / right-click: ' +
        (st.conn[bu] ? 'stop tracing' : 'trace its connections') + '</div>');
      drawGeo(st.f);
      redrawMats();
    });
    geoZone.addEventListener('pointerleave', function () {
      drag = null; st.hover = null; tipHide(); drawGeo(st.f); redrawMats();
    });
    geoZone.addEventListener('wheel', function (e) {
      e.preventDefault();
      var fz = Math.exp(-e.deltaY * 0.0015);
      st.zoom = Math.max(0.3, Math.min(12, st.zoom * fz));
      holdSpin();
      drawGeo(st.f);
    }, {passive: false});

    var resetBtn = document.getElementById('view3-reset');
    if (resetBtn) {
      resetBtn.addEventListener('click', function () {
        // the view AND the selections: pinned paths and connections,
        // node focus, hover
        applyView();                      // each run opens from its own angle
        st.pinned = {}; st.nPinned = 0; st.pinnedEdges = {}; st.conn = {};
        st.hover = null;
        restUntil = 0;
        tipHide();
        syncConnNote();
        setFocus(null);
        redrawMats();
      });
    }
    // a segmented choice: one of its buttons is on, and st[key] is its value
    function segControl(id, key, after) {
      var seg = document.getElementById(id);
      if (!seg) return function () {};
      var btns = seg.querySelectorAll('button');
      function sync() {
        for (var i = 0; i < btns.length; i++) {
          var on = btns[i].getAttribute('data-v') === st[key];
          btns[i].className = on ? 'on' : '';
          btns[i].setAttribute('aria-pressed', on ? 'true' : 'false');
        }
      }
      for (var i = 0; i < btns.length; i++) {
        btns[i].addEventListener('click', function (e) {
          st[key] = e.currentTarget.getAttribute('data-v');
          sync();
          if (after) after();
          drawGeo(st.f);
        });
      }
      sync();
      return sync;
    }
    // a toggle: filled while on
    function toggleBtn(id, key) {
      var b = document.getElementById(id);
      if (!b) return function () {};
      function sync() {
        b.classList.toggle('on', !!st[key]);
        b.setAttribute('aria-pressed', st[key] ? 'true' : 'false');
      }
      b.addEventListener('click', function () {
        st[key] = !st[key];
        sync();
        drawGeo(st.f);
      });
      sync();
      return sync;
    }
    segControl('edges3-seg', 'edgeSet');
    var gateCtl = document.getElementById('gate3-ctl');
    function syncGateCtl() { if (gateCtl) gateCtl.hidden = st.opacity !== 'soft'; }
    segControl('opacity3-seg', 'opacity', syncGateCtl);
    syncGateCtl();
    var syncHideBtn = toggleBtn('hide3-dim', 'hideDim');
    var spinBtn = document.getElementById('spin3');
    if (spinBtn) {
      spinBtn.addEventListener('click', function () { setSpin(!st.spinning); });
    }
    setSpin(st.spinning);
    var gateSld = document.getElementById('gate3-min');
    if (gateSld) {
      gateSld.addEventListener('input', function () {
        st.gateMin = parseFloat(gateSld.value);
        document.getElementById('gate3-minval').textContent =
          st.gateMin.toFixed(2);
        drawGeo(st.f);
      });
    }

    function redrawMats() {
      visibleKinds().forEach(function (k) { drawMat(k, st.f); });
    }

    // matrix hover: name the connection behind the cell. Gate, mask and weight
    // panels share this handler, so all three hover alike; a panel made later
    // for a deeper network is bound the same way.
    function bindMat(k) {
      var cv = matCv[k];
      if (!cv) return;
      cv.addEventListener('pointermove', function (e) {
        var def = R.matDefs[k + curB];
        if (!def) return;
        var rect = cv.getBoundingClientRect();
        var cs = cellSize();
        var c = Math.floor((e.clientX - rect.left - MPAD) / colW(def));
        var r = Math.floor((e.clientY - rect.top - MPAD) / cs);
        if (c < 0 || c >= def.nc || r < 0 || r >= def.nr) {
          if (st.hover) { st.hover = null; tipHide(); redrawMats(); drawGeo(st.f); }
          return;
        }
        var recvU = def.rows[r], sendU = def.cols[c];
        var stride = def.nr * def.nc;
        var at = st.f * stride + recvU * def.nc + sendU;
        var bq = curB;
        var g = R.gates[bq][at] / 255;
        var kept = R.masks[bq] ? R.masks[bq][at] : null;
        var wq = R.weights[bq];
        var wbase = wq ? wq[at] / 127 * (R.wScale[bq] || 1) : null;
        var weff = wbase === null ? null : wbase * g * (kept === null ? 1 : kept);
        var lhs = def.rname + (def.nr > 1 ? ' neuron ' + recvU : '');
        var rhs = def.cname +
          (def.cname === 'input' ? ' ' : ' neuron ') + sendU +
          (def.cname === 'input' ? ' (' + inputNote(sendU) + ')' : '');
        var ruG = R.levelStart[def.rlv] + recvU;
        var suG = R.levelStart[def.slv] + sendU;
        st.hover = {type: 'cell', m: k, r: r, c: c,
                    edge: [ruG, suG], units: [ruG, suG]};
        var pinKey = ruG + '-' + suG;
        tipShow(e.clientX, e.clientY,
          '<div class="tt-head">' + lhs + ' ← ' + rhs + '</div>' +
          '<div>soft gate ' + g.toFixed(2) +
          (kept === null
            ? ' · <span class="tt-dim">no binary mask at this transition</span>'
            : ' · ' + (kept ? '<b>mask 1: active</b>'
                            : '<span class="tt-dim">mask 0: inactive</span>')) +
          '</div>' +
          (wbase === null ? '' :
            '<div>base weight ' + wbase.toFixed(3) + ' · forward-pass weight <b>' +
            weff.toFixed(3) + '</b></div>') +
          '<div class="tt-dim" style="margin-top:3px">' +
          (st.pinnedEdges[pinKey] ? 'click to unpin this connection'
                                  : 'click to pin this connection') + '</div>');
        redrawMats();
        drawGeo(st.f);
      });
      cv.addEventListener('click', function () {
        if (!R.matDefs[k + curB]) return;
        if (!st.hover || st.hover.type !== 'cell' || !st.hover.edge) return;
        var key = st.hover.edge[0] + '-' + st.hover.edge[1];
        if (st.pinnedEdges[key]) delete st.pinnedEdges[key];
        else if (Object.keys(st.pinnedEdges).length < 12) {
          st.pinnedEdges[key] = st.hover.edge.slice();
        }
        drawGeo(st.f);
        redrawMats();
      });
      cv.addEventListener('pointerleave', function () {
        st.hover = null; tipHide(); redrawMats(); drawGeo(st.f);
      });
    }
    KINDS.forEach(bindMat);

    // curve: scrub by pointer
    function curveSeek(e) {
      var rect = curveCv.getBoundingClientRect();
      var s = (e.clientX - rect.left - CM.l) / (CW - CM.l - CM.r) * R.meta.total_steps;
      s = Math.max(0, Math.min(R.meta.total_steps, s));
      var bi = 0, bd = Infinity;
      for (var i = 0; i < R.meta.frames; i++) {
        var d = Math.abs(R.meta.steps[i] - s);
        if (d < bd) { bd = d; bi = i; }
      }
      setPlaying(false);
      seek(bi);
    }
    var curveDown = false;
    curveCv.addEventListener('pointerdown', function (e) {
      curveDown = true; curveCv.setPointerCapture(e.pointerId); curveSeek(e);
    });
    curveCv.addEventListener('pointermove', function (e) {
      if (curveDown) curveSeek(e);
    });
    curveCv.addEventListener('pointerup', function () { curveDown = false; });

    // shape key under the geometry panel
    var sk = document.getElementById('shape3-key');
    function buildShapeKey() {
      var skHtml = '';
      for (var g2 = 0; g2 < R.meta.levels.names.length; g2++) {
        skHtml += '<span><span class="g">' + R.glyphs[g2] + '</span>' +
                  R.meta.levels.names[g2] + '</span>';
      }
      skHtml += '<span><span class="g" style="color:#c4c4c4">●</span>' +
                (R.assoc ? 'unassigned'
                         : 'below the |r| ≥ ' + R.meta.floor.toFixed(2) + ' floor') +
                '</span>';
      sk.innerHTML = skHtml;
    }
    buildShapeKey();

    // -------------------------------------------------------- run switcher
    // Each tab carries a small drawing of its target graph (assets/graphs.js),
    // in the colours of the key: the tabs differ first of all by target.
    function thumbSVG(stem) {
      var g = (window.__demoGraphs || {})[stem];
      if (!g) return '';
      var DY = 1.25, top = g.out[1] * DY, xs = 0, i;
      for (i = 0; i < g.leaves.length; i++) xs = Math.max(xs, g.leaves[i][0]);
      for (i = 0; i < g.nodes.length; i++) xs = Math.max(xs, g.nodes[i][0]);
      function y(lv) { return (top - lv * DY).toFixed(3); }
      var out = '<svg class="rthumb" viewBox="-0.7 -0.7 ' + (xs + 1.4).toFixed(2) + ' ' +
                (top + 1.4).toFixed(2) + '" aria-hidden="true" focusable="false">';
      g.edges.forEach(function (e) {
        out += '<line x1="' + e[0] + '" y1="' + y(e[1]) + '" x2="' + e[2] + '" y2="' + y(e[3]) + '"/>';
      });
      g.leaves.forEach(function (l, k) {
        out += '<rect x="' + (l[0] - 0.3) + '" y="' + (top - 0.3).toFixed(3) +
               '" width="0.6" height="0.6" fill="' + l[1] + '"/>';
      });
      g.nodes.forEach(function (n) {
        out += '<circle cx="' + n[0] + '" cy="' + y(n[1]) + '" r="0.46" fill="' + n[2] + '"/>';
      });
      out += '<circle cx="' + g.out[0] + '" cy="' + y(g.out[1]) + '" r="0.46" fill="' + g.outColor + '"/>';
      return out + '</svg>';
    }

    var runBtns = {};
    var runbar = document.getElementById('runbar-top');
    GROUPS.forEach(function (g) {
      if (!SHOWN.some(function (r) { return r.group === g.key; })) return;
      var row = document.createElement('div');
      row.className = 'rungroup';
      row.setAttribute('role', 'group');
      row.setAttribute('aria-label', g.label);
      var cap = document.createElement('p');
      cap.className = 'rgcap';
      cap.textContent = g.label;
      var rows = document.createElement('div');
      rows.className = 'rgrows';
      row.appendChild(cap);
      row.appendChild(rows);
      var list = null;
      SHOWN.forEach(function (r) {
        if (r.group !== g.key) return;
        if (!list || r.newRow) {
          list = document.createElement('div');
          list.className = 'rglist';
          rows.appendChild(list);
        }
        var b = document.createElement('button');
        b.type = 'button';
        b.className = 'runbtn';
        b.innerHTML = thumbSVG(r.stem) + '<span class="rtext"><b>' + r.label +
                      '</b><span>' + r.note + '</span></span>';
        b.addEventListener('click', function () { switchRun(r.stem); });
        list.appendChild(b);
        runBtns[r.stem] = b;
      });
      runbar.appendChild(row);
    });
    function markTab() {
      for (var s2 in runBtns) {
        runBtns[s2].className = 'runbtn' + (s2 === curStem ? ' on' : '');
        runBtns[s2].setAttribute('aria-pressed', s2 === curStem ? 'true' : 'false');
      }
    }
    markTab();

    // the section caption names the loaded run's target and colour rule
    function syncCaption() {
      var r = runOf(curStem);
      var tgt = document.getElementById('run3-target');
      var rule = document.getElementById('run3-rule');
      var arch = document.getElementById('run3-arch');
      var LCc = R.meta.levels.counts;
      var WORDS = ['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight'];
      if (arch) arch.textContent = LCc[1] + ' neurons per hidden layer and ' +
        (WORDS[LCc.length - 2] || LCc.length - 2) + ' hidden layers';
      if (tgt && r) tgt.textContent = r.target;
      var intro = document.getElementById('intro-target');
      if (intro && r) intro.textContent = r.intro || r.target;
      if (rule) {
        // one rule for every run, with the run's own permutation threshold
        rule.innerHTML = 'Hidden neurons are colored by the target intermediate ' +
          'with which their activation has the largest absolute correlation on ' +
          'diagnostic examples, provided the correlation exceeds a ' +
          'permutation-derived threshold. The threshold of this run is ' +
          '|<var>r</var>| = ' + R.meta.floor.toFixed(3) + '.';
      }
    }
    syncCaption();

    function applyView() {
      var v = viewOf(curStem, R);
      st.yaw = v.yaw; st.pitch = v.pitch; st.zoom = v.zoom; st.spinRate = v.spin;
    }

    // a tab opens its run at the first step and plays it through, so every
    // run is seen whole; the turn starts with the replay's first step
    function switchRun(stem) {
      if (stem === curStem) return;
      setPlaying(false);
      st.spinArmed = false;           // the turn waits for the new run too
      if (!ready[stem]) veil.style.display = 'grid';
      loadRun(stem).then(function (run) {
        curStem = stem;
        R = run;
        markTab();
        st.hideDim = !!(runOf(stem) && runOf(stem).hideDim);
        syncHideBtn();
        st.pinned = {}; st.nPinned = 0; st.pinnedEdges = {}; st.conn = {};
        st.hover = null;
        restUntil = 0;
        syncConnNote();
        applyView();
        buildKey();
        buildShapeKey();
        syncCaption();
        setFocus(null);
        st.f = 0;
        acc = 0;
        buildEvents();
        applyShape();
        layout();
        drawAll();
        veil.style.display = 'none';
        setPlaying(true);
        announce(stem, run);
      }).catch(function (err) { fail('Could not load the run (' + err.message + ')'); });
    }

    // ---------------------------------------------------------------- go
    buildEvents();
    layout();
    var hash = /[#&]step=(\d+)/.exec(location.hash);
    if (hash) {
      var want = +hash[1], hb = 0, hd = Infinity;
      for (var hi = 0; hi < R.meta.frames; hi++) {
        if (Math.abs(R.meta.steps[hi] - want) < hd) {
          hd = Math.abs(R.meta.steps[hi] - want); hb = hi;
        }
      }
      st.f = hb;
    }
    drawAll();
    veil.style.display = 'none';
    announce(curStem, R);

    var resizeT = null;
    window.addEventListener('resize', function () {
      clearTimeout(resizeT);
      resizeT = setTimeout(function () {
        imgCache = {}; layout(); drawAll();
      }, 160);
    });

    if (!hash) setTimeout(function () { setPlaying(true); }, 500);
    // the other tabs' runs follow once this one plays
    setTimeout(preload, 1200);
  }
})();
