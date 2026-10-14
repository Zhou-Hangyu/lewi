/* Final learned wiring: the selected run's final (or selected) checkpoint,
 * drawn in the learned 3-D embedding space as one panel per layer transition
 * plus the full network. Every panel shares ONE camera: a drag or a zoom on
 * any panel turns them all, so the geometry reads the same across them.
 * Listens for the page-top run switcher's 'runchange' event; no fetches of
 * its own.
 *
 * A run of the paper's embedding figure shows its best checkpoint, in the
 * figure's frame and cube, opening at the figure's camera.
 */
(function () {
  'use strict';
  var TAU = Math.PI * 2;
  var PERIOD = 25000;                 // ms per turn while auto-rotate is on
  var RESUME_MS = 1800;
  var SH = window.__demoShared;

  var root = document.getElementById('static3d');
  if (!root) return;

  var state = {
    phase: 0, holdUntil: 0,
    spinning: true,                   // the final wiring turns until the reader stops it
    hideDim: false,                   // each run opens with its own default
    edgeSet: 'all',                   // 'input' | 'hidden' | 'all'
    opacity: 'mask',                  // 'mask': one opacity per active connection; 'soft': the gate's value
    gateMin: 0.5,
    cam: {yaw: 0.62, pitch: 0.42, zoom: 1},   // the one camera of every panel
    offLayers: {}, conn: {}
  };
  var R = null, F = 0, panels = [];
  var FIG = false;                  // the loaded run is one of the figure's
  var FRAMED = false;               // shown in a frame of its own (the figure's or a display frame)
  var BASE = {yaw: 0.62, pitch: 0.42};
  function camera() {
    return FIG && R.meta.view ? R.meta.view : BASE;
  }
  function resetCamera() {
    var c = camera();
    state.cam = {yaw: c.yaw, pitch: c.pitch, zoom: 1};
    state.phase = 0;
  }
  function capName(nm) { return nm.charAt(0).toUpperCase() + nm.slice(1); }
  function now_() {
    return window.performance ? performance.now() : Date.now();
  }
  function hold() { state.holdUntil = now_() + RESUME_MS; }

  // ------------------------------------------------------------- geometry

  function finalPoint(u) {
    return [R.P(F, u, 0), R.P(F, u, 1), R.P(F, u, 2)];
  }

  // every connection of every transition at the shown frame, with its soft
  // gate and its mask (a gate above one half where the transition has no
  // mask): the tools pick which of them are drawn, and how
  function collectEdges() {
    var LS = R.levelStart, LC = R.meta.levels.counts;
    var byB = [];
    for (var b = 0; b < R.meta.boundaries; b++) {
      var nr = LC[b + 1], nc = LC[b], stride = nr * nc, base = F * stride;
      var gate = R.gates[b], mask = R.masks[b], list = [];
      for (var r = 0; r < nr; r++) {
        for (var c = 0; c < nc; c++) {
          var o = base + r * nc + c, g = gate[o] / 255;
          list.push([LS[b + 1] + r, LS[b] + c, g, mask ? mask[o] : (g >= 0.5 ? 1 : 0)]);
        }
      }
      byB.push(list);
    }
    return byB;
  }
  function edgeOn(e) {
    return state.opacity === 'soft' ? e[2] >= state.gateMin : !!e[3];
  }
  function boundaryOn(b) {
    return state.edgeSet === 'all' || (state.edgeSet === 'input' ? b === 0 : b > 0);
  }

  // ------------------------------------------------------------ rendering

  function drawPanel(p) {
    var ctx = p.ctx, W = p.W, H = p.H;
    var yaw = state.cam.yaw + state.phase, pitch = state.cam.pitch;
    var cy = Math.cos(yaw), sy = Math.sin(yaw);
    var cp = Math.cos(pitch), sp = Math.sin(pitch);
    var s = p.pxPerUnit * state.cam.zoom;
    var c = p.center;
    function pr(q) {
      var x = q[0] - c[0], y = q[1] - c[1], z = q[2] - c[2];
      var x1 = cy * x + sy * z, z1 = -sy * x + cy * z;
      var y1 = cp * y - sp * z1, z2 = sp * y + cp * z1;
      return [W / 2 + x1 * s, H / 2 - y1 * s, z2];
    }
    ctx.clearRect(0, 0, W, H);
    var outLv = R.meta.levels.counts.length - 1;

    var us = p.units;
    var proj = {};
    var items = [];
    var i, u;
    for (i = 0; i < us.length; i++) {
      u = us[i];
      var b = R.finalBest[u];
      // the output is always drawn in a figure run, as the figure's star
      if (state.hideDim && !b.lit && !(FRAMED && R.levelOf[u] === outLv)) continue;
      if (p.kind === 'all' && state.offLayers[R.levelOf[u]]) continue;
      proj[u] = pr(R.finalPos[u]);
      items.push({k: 0, d: proj[u][2], u: u});
    }

    // the same frame in every panel — cube and axes — at one shared scale,
    // so the tightening of deeper layers reads directly; a raw-frame run
    // names its axes and marks the true origin, a figure run is in the
    // figure's frame, whose axes are unnamed
    var L = R.meta.lim3 * 0.92;
    var corners = [];
    for (i = 0; i < 8; i++) {
      corners.push(pr([(i & 1 ? L : -L), (i & 2 ? L : -L), (i & 4 ? L : -L)]));
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
    ctx.strokeStyle = 'rgba(0,0,0,0.16)';
    ctx.fillStyle = '#9a9a9a';
    ctx.font = '9px system-ui, sans-serif';
    for (i = 0; i < 3; i++) {
      var qn = [0, 0, 0], qp = [0, 0, 0];
      qn[i] = -L; qp[i] = L;
      var an = pr(qn), ap = pr(qp);
      ctx.beginPath(); ctx.moveTo(an[0], an[1]); ctx.lineTo(ap[0], ap[1]);
      ctx.stroke();
      if (!FRAMED) ctx.fillText('e' + '₁₂₃'[i], ap[0] + 3, ap[1] - 3);
    }
    if (!FRAMED) {
      var o0 = pr([0, 0, 0]);
      ctx.beginPath(); ctx.arc(o0[0], o0[1], 1.6, 0, TAU);
      ctx.fillStyle = 'rgba(0,0,0,0.35)';
      ctx.fill();
    }

    if (p.zone) {
      var bx = [Infinity, -Infinity], by = [Infinity, -Infinity];
      corners.forEach(function (q) {
        bx[0] = Math.min(bx[0], q[0]); bx[1] = Math.max(bx[1], q[0]);
        by[0] = Math.min(by[0], q[1]); by[1] = Math.max(by[1], q[1]);
      });
      var pad = 0.1 * Math.max(bx[1] - bx[0], by[1] - by[0]);
      // cube + 10% each side, but always leaving a 4% escape ring so the
      // page stays scrollable even when zoomed in
      bx[0] = Math.max(0.04 * W, bx[0] - pad);
      bx[1] = Math.min(0.96 * W, bx[1] + pad);
      by[0] = Math.max(0.04 * H, by[0] - pad);
      by[1] = Math.min(0.96 * H, by[1] + pad);
      var r0 = p.cv.getBoundingClientRect();
      var sf = r0.width / W;
      p.zone.style.left = (p.cv.offsetLeft + bx[0] * sf) + 'px';
      p.zone.style.top = (p.cv.offsetTop + by[0] * sf) + 'px';
      p.zone.style.width = ((bx[1] - bx[0]) * sf) + 'px';
      p.zone.style.height = ((by[1] - by[0]) * sf) + 'px';
    }

    // the connections: the transitions the "Edges" choice names in the
    // full-network panel, or the one transition of a transition panel; drawn
    // alike, by the binary mask or by the soft gate
    var soft = state.opacity === 'soft';
    var bs = [];
    if (p.kind === 'all') {
      for (i = 0; i < R.meta.boundaries; i++) if (boundaryOn(i)) bs.push(i);
    } else bs.push(p.lv);
    for (var bi = 0; bi < bs.length; bi++) {
      var set = R.edges[bs[bi]];
      for (i = 0; i < set.length; i++) {
        var e2 = set[i];
        if (!edgeOn(e2)) continue;
        var pa = proj[e2[0]], pb = proj[e2[1]];
        if (!pa || !pb) continue;
        items.push({k: 1, d: (pa[2] + pb[2]) / 2, a: pa, b: pb, u: e2[0], g: e2[2]});
      }
    }

    items.sort(function (a, b) { return a.d - b.d; });
    var nLines = 0;
    for (i = 0; i < items.length; i++) if (items[i].k === 1) nLines++;
    var dens = SH.inkScale(nLines);
    // while a neuron's connections are traced, the rest steps back
    var quiet = Object.keys(state.conn).length ? 0.5 : 1;
    p.screen = [];
    items.forEach(function (it) {
      if (it.k === 1) {
        var eb = R.finalBest[it.u];
        var col = SH.unitCol(R, it.u, eb, '#b5b5b5');
        ctx.beginPath();
        ctx.moveTo(it.a[0], it.a[1]);
        ctx.lineTo(it.b[0], it.b[1]);
        // thin and light: the connections are the context, the traces the figure
        ctx.globalAlpha = (soft ? 0.04 + 0.30 * it.g : 0.18) * dens * quiet;
        ctx.strokeStyle = col;
        ctx.lineWidth = 0.7;
        ctx.stroke();
        ctx.globalAlpha = 1;
        return;
      }
      var u2 = it.u;
      var b2 = R.finalBest[u2];
      var pt = proj[u2];
      var lv = R.levelOf[u2];
      var lit = b2.lit || (FRAMED && lv === outLv);
      var size = R.sizes[lv] * (lit ? 1 : 0.62) * Math.min(state.cam.zoom, 1.6);
      SH.drawMark(ctx, R, lv, pt[0], pt[1], size, SH.unitCol(R, u2, b2, '#c4c4c4'),
                  lit ? 1 : 0.55, 0.8);
      p.screen.push({u: u2, x: pt[0], y: pt[1]});
    });

    // neuron tracing: the in/out connections of each traced neuron in this
    // panel — blue into it, magenta out of it, arrows pointing forward;
    // endpoints living in other layers get a small marker
    for (var cu in state.conn) {
      var CU = +cu;
      if (p.kind === 'pair' && R.levelOf[CU] !== p.lv &&
          R.levelOf[CU] !== p.lv + 1) continue;
      var selp = proj[CU] || pr(R.finalPos[CU]);
      SH.connEdgesOf(R, F, CU, state.gateMin).forEach(function (Ez) {
        var pa = proj[Ez.su] || pr(R.finalPos[Ez.su]);
        var pb = proj[Ez.ru] || pr(R.finalPos[Ez.ru]);
        var ccol = Ez.dirIn ? '#0f62fe' : '#d02670';
        ctx.globalAlpha = Ez.kept ? 0.95 : 0.3 + 0.65 * Ez.g;
        ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]);
        ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 4.2; ctx.stroke();
        ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]);
        ctx.strokeStyle = ccol; ctx.lineWidth = 2.1; ctx.stroke();
        SH.arrowHead(ctx, pa, pb, ccol);
        var other = Ez.dirIn ? Ez.su : Ez.ru;
        var op = Ez.dirIn ? pa : pb;
        var ob = R.finalBest[other];
        ctx.beginPath(); ctx.arc(op[0], op[1], 2.4, 0, TAU);
        ctx.fillStyle = SH.unitCol(R, other, ob, '#c4c4c4');
        ctx.fill();
        ctx.lineWidth = 0.8; ctx.strokeStyle = '#ffffff'; ctx.stroke();
        ctx.globalAlpha = 1;
      });
      ctx.beginPath();
      ctx.arc(selp[0], selp[1], 8, 0, TAU);
      ctx.strokeStyle = '#1a1a1a';
      ctx.lineWidth = 1.8;
      ctx.stroke();
    }
  }

  function drawAllPanels() {
    panels.forEach(function (p) { drawPanel(p); });
  }

  var lastConnTs = 0;
  function toggleConn(u) {
    if (u === null) return;
    if (state.conn[u]) delete state.conn[u];
    else if (Object.keys(state.conn).length < 8) state.conn[u] = 1;
    lastConnTs = Date.now();
    syncConnNote();
    drawAllPanels();
  }

  // ------------------------------------------------------------- building

  function tooltipHtml(u) {
    // the same popup as the training-dynamics panel, at the final frame
    var b = R.finalBest[u];
    var lv = R.levelOf[u];
    var names = R.meta.levels.names;
    var name = (lv === 0 ? 'input' : names[lv] + ' neuron') +
               (R.meta.levels.counts[lv] > 1 ? ' ' + R.idxIn[u] : '');
    var html = '<div class="tt-head">' + name + '</div>';
    if (lv === 0) {
      var t = R.meta.teacher, note = 'inactive';
      for (var v = 0; v < t.active_raw.length; v++) {
        if (t.active_raw[v].indexOf(R.idxIn[u]) >= 0) {
          note = 'active — feeds z' + SH.sub(v);
        }
      }
      html += '<div class="tt-dim">' + note + '</div>';
    }
    html += b.lit
      ? '<div>associated with <b>z' + SH.sub(b.node) + '</b> · |r| = ' +
        b.s.toFixed(2) + '</div>'
      : '<div class="tt-dim">' + (R.assoc
          ? 'unassigned: no correlation exceeds the threshold'
          : 'below the permutation floor') + '</div>';
    var rows = '';
    for (var k = 0; k < R.NN; k++) {
      var r = R.rOf(F, u, k);
      rows += '<span class="lab">z' + SH.sub(k) + '</span>' +
        '<span class="bar"><i style="width:' + (Math.abs(r) * 100).toFixed(0) +
        '%;background:' + R.meta.teacher.colors[k] + '"></i></span>' +
        '<span class="val">' + (r >= 0 ? '+' : '−') +
        Math.abs(r).toFixed(2) + '</span>';
    }
    html += '<div class="bars">' + rows + '</div>';
    html += '<div class="tt-dim" style="margin-top:3px">' +
      (state.conn[u] ? 'click to stop tracing its connections'
                     : 'click to trace its connections') + '</div>';
    return html;
  }

  function makePanel(holder, kind, lv, W, H) {
    var box = document.createElement('div');
    box.className = 'c3d-panel' + (kind === 'all' ? ' s3-widebox' : '');
    var title = document.createElement('div');
    title.className = 'c3d-title';
    var N = R.meta.levels.names;
    if (kind === 'pair') {
      // each layer's name, then its marker
      title.innerHTML = capName(N[lv]) + ' <span class="gl">' + R.glyphs[lv] +
        '</span> → ' + capName(N[lv + 1]) + ' <span class="gl">' +
        R.glyphs[lv + 1] + '</span>';
    } else {
      title.textContent = 'Full network';
    }
    var cv = document.createElement('canvas');
    var dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    cv.style.width = W + 'px';
    cv.style.height = H + 'px';
    cv.style.maxWidth = '100%';
    cv.style.margin = '0 auto';
    var tip = document.createElement('div');
    tip.className = 'c3d-tip';
    // gestures land on a transparent zone tracking the projected cube
    // (+10%); the rest of the panel stays free for page scrolling
    var zone = document.createElement('div');
    zone.className = 'c3d-zone';
    box.appendChild(title); box.appendChild(cv); box.appendChild(zone);
    box.appendChild(tip);
    holder.appendChild(box);
    var ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    var units = [];
    for (var u = 0; u < R.U; u++) {
      var lu = R.levelOf[u];
      if (kind === 'all' || lu === lv || (kind === 'pair' && lu === lv + 1)) units.push(u);
    }
    var p = {cv: cv, zone: zone, ctx: ctx, W: W, H: H, kind: kind, lv: lv,
             units: units, screen: [], dragging: false,
             center: [0, 0, 0],
             // sized so the rotated cube plus the 10% zone padding stays
             // inside the panel: the ring outside it is page-scroll space
             pxPerUnit: (Math.min(W, H) * 0.32) / R.meta.lim3};

    function hitPt(e) {
      var r = cv.getBoundingClientRect();
      var sx = W / r.width;
      var mx = (e.clientX - r.left) * sx, my = (e.clientY - r.top) * sx;
      var best = null;
      for (var k = p.screen.length - 1; k >= 0; k--) {
        var s = p.screen[k];
        var dd = Math.pow(s.x - mx, 2) + Math.pow(s.y - my, 2);
        if (dd < 110 && (best === null || dd < best.dd)) best = {s: s, dd: dd};
      }
      return best;
    }
    function showTip(e) {
      var r = cv.getBoundingClientRect();
      var sx = W / r.width;
      var best = hitPt(e);
      if (!best) { tip.style.display = 'none'; return; }
      tip.innerHTML = tooltipHtml(best.s.u);
      tip.style.display = 'block';
      var lx = best.s.x / sx + 12;
      if (lx + tip.offsetWidth > r.width &&
          best.s.x / sx - 12 - tip.offsetWidth > 0) {
        lx = best.s.x / sx - 12 - tip.offsetWidth;
      }
      tip.style.left = Math.max(0, Math.min(lx, r.width - 40)) + 'px';
      tip.style.top = (best.s.y / sx + 14 + title.offsetHeight) + 'px';
    }
    zone.addEventListener('pointerdown', function (e) {
      p.dragging = true; p.lastX = e.clientX; p.lastY = e.clientY;
      p.moved = false;
      tip.style.display = 'none';
      hold();
      try { zone.setPointerCapture(e.pointerId); } catch (err) {}
    });
    function release(e) {
      if (!p.dragging) return;
      p.dragging = false;
      hold();
      try { zone.releasePointerCapture(e.pointerId); } catch (err) {}
    }
    zone.addEventListener('pointerup', function (e) {
      var tap = p.dragging && !p.moved && e.button !== 2;
      release(e);
      if (tap && Date.now() - lastConnTs > 400) {
        var hit = hitPt(e);
        toggleConn(hit ? hit.s.u : null);
      }
    });
    zone.addEventListener('contextmenu', function (e) {
      e.preventDefault();
      var hit = hitPt(e);
      if (hit && Date.now() - lastConnTs > 400) toggleConn(hit.s.u);
    });
    zone.addEventListener('pointercancel', release);
    zone.addEventListener('pointerleave', function (e) {
      release(e); tip.style.display = 'none'; hold();
    });
    zone.addEventListener('pointermove', function (e) {
      if (p.dragging) {
        if (Math.abs(e.clientX - p.lastX) + Math.abs(e.clientY - p.lastY) > 2) {
          p.moved = true;
        }
        // one camera: a drag here turns every panel
        state.cam.yaw += (e.clientX - p.lastX) * 0.011;
        state.cam.pitch = Math.max(-1.5, Math.min(1.5,
          state.cam.pitch + (e.clientY - p.lastY) * 0.011));
        p.lastX = e.clientX; p.lastY = e.clientY;
        hold();
        drawAllPanels();
        return;
      }
      hold();
      showTip(e);
    });
    zone.addEventListener('wheel', function (e) {
      e.preventDefault();
      var f = Math.exp(-e.deltaY * 0.0015);
      state.cam.zoom = Math.max(0.25, Math.min(14, state.cam.zoom * f));
      hold();
      drawAllPanels();
      showTip(e);
    }, {passive: false});
    panels.push(p);
    return p;
  }

  function layerPanelSize() {
    var w = root.clientWidth || 900;
    var per = w >= 1100 ? 5 : w >= 640 ? 3 : 2;
    return Math.floor((w - (per - 1) * 12) / per) - 10;
  }

  function render() {
    if (!R) return;
    panels = [];
    var grid = root.querySelector('.s3-grid');
    var wide = root.querySelector('.s3-wide');
    grid.innerHTML = '';
    wide.innerHTML = '';

    var side = layerPanelSize();
    grid.className = 's3-grid five';
    for (var b = 0; b < R.meta.boundaries; b++) makePanel(grid, 'pair', b, side, side);

    var w = (root.clientWidth || 900) - 10;
    makePanel(wide, 'all', -1, w, Math.min(Math.round(w * 0.66), 600));
    drawAllPanels();
    syncTools();
  }

  // ---------------------------------------------------------------- tools
  // the same controls as the training-dynamics view: which transitions'
  // connections, what sets a line's opacity, the unassigned neurons, the
  // turn, and the way back to the opening camera

  var tools = root.querySelector('.s3-tools');
  var last = null;
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }
  var syncs = [];
  function segment(label, key, options, after) {
    var wrap = el('span', 'ctl');
    wrap.appendChild(el('span', 'c3d-legkey', label));
    var seg = el('span', 'seg');
    seg.setAttribute('role', 'group');
    seg.setAttribute('aria-label', label);
    var btns = options.map(function (o) {
      var b = el('button', '', o[1]);
      b.type = 'button';
      b.setAttribute('data-v', o[0]);
      b.addEventListener('click', function () {
        state[key] = o[0];
        sync();
        if (after) after();
        drawAllPanels();
      });
      seg.appendChild(b);
      return b;
    });
    function sync() {
      btns.forEach(function (b) {
        var on = b.getAttribute('data-v') === state[key];
        b.className = on ? 'on' : '';
        b.setAttribute('aria-pressed', on ? 'true' : 'false');
      });
    }
    wrap.appendChild(seg);
    tools.appendChild(wrap);
    syncs.push(sync);
    sync();
    return sync;
  }
  function toggle(label, key, after) {
    var b = el('button', 'c3d-btn tog', label);
    b.type = 'button';
    function sync() {
      b.classList.toggle('on', !!state[key]);
      b.setAttribute('aria-pressed', state[key] ? 'true' : 'false');
    }
    b.addEventListener('click', function () {
      state[key] = !state[key];
      sync();
      if (after) after();
      drawAllPanels();
    });
    tools.appendChild(b);
    syncs.push(sync);
    sync();
    return sync;
  }
  segment('Edges', 'edgeSet', [['input', 'Input'], ['hidden', 'Hidden'], ['all', 'All']]);
  var gateCtl = el('span', 'ctl');
  gateCtl.appendChild(el('span', 'c3d-legkey', 'Minimum gate'));
  var sld = document.createElement('input');
  sld.type = 'range'; sld.min = '0'; sld.max = '1'; sld.step = '0.01';
  sld.value = String(state.gateMin);
  sld.className = 'c3d-slider';
  sld.setAttribute('aria-label', 'minimum gate');
  gateCtl.appendChild(sld);
  var sval = el('span', 'c3d-sval', state.gateMin.toFixed(2));
  gateCtl.appendChild(sval);
  sld.addEventListener('input', function () {
    state.gateMin = parseFloat(sld.value);
    sval.textContent = state.gateMin.toFixed(2);
    drawAllPanels();
  });
  function syncGate() { gateCtl.hidden = state.opacity !== 'soft'; }
  segment('Edge opacity', 'opacity', [['mask', 'Binary mask'], ['soft', 'Soft gate']], syncGate);
  tools.appendChild(gateCtl);
  syncGate();
  toggle('Hide unassigned neurons', 'hideDim');
  toggle('Auto-rotate', 'spinning', function () { last = null; });
  var resetBtn = el('button', 'c3d-btn', 'Reset view');
  resetBtn.type = 'button';
  resetBtn.addEventListener('click', function () {
    resetCamera();
    state.offLayers = {};
    state.conn = {};
    syncLayerChips();
    syncConnNote();
    drawAllPanels();
  });
  tools.appendChild(resetBtn);
  var cnote = el('span', 'c3d-legkey');
  tools.appendChild(cnote);
  function syncConnNote() {
    var n = Object.keys(state.conn).length;
    cnote.innerHTML = n
      ? 'tracing ' + n + ' — <span style="color:#0f62fe">■</span> incoming · ' +
        '<span style="color:#d02670">■</span> outgoing · arrows point forward'
      : '';
  }

  var lrow = root.querySelector('.s3-layers');
  var layerChips = [];
  function buildLayerChips() {
    lrow.innerHTML = '';
    layerChips = [];
    var lab = el('span', 'c3d-legkey', 'in the full network:');
    lrow.appendChild(lab);
    R.meta.levels.names.forEach(function (nm, gi) {
      var b = el('button', 'c3d-btn c3d-on', R.glyphs[gi] + ' ' + nm);
      b.type = 'button';
      b.addEventListener('click', function () {
        state.offLayers[gi] = !state.offLayers[gi];
        syncLayerChips();
        drawAllPanels();
      });
      lrow.appendChild(b);
      layerChips.push({b: b, gi: gi});
    });
  }
  function syncLayerChips() {
    layerChips.forEach(function (c) {
      c.b.className = 'c3d-btn' + (state.offLayers[c.gi] ? ' c3d-off' : ' c3d-on');
    });
  }
  function syncTools() {
    syncs.forEach(function (fn) { fn(); });
    syncGate();
  }

  // ------------------------------------------------------------ lifecycle

  var capEl = document.getElementById('s3-cap');
  var capDefault = capEl ? capEl.innerHTML : '';
  function fmtStep(n) { return n.toLocaleString('en-US'); }

  document.addEventListener('runchange', function (e) {
    R = e.detail.run;
    FIG = !!R.meta.figure;
    FRAMED = FIG || !!R.meta.display_frame;
    state.conn = {};
    state.hideDim = !!e.detail.hideDim;    // the run's own default
    resetCamera();
    syncConnNote();
    // the final wiring is the checkpoint the run selected; a figure run
    // names it, and it is the network the figure draws
    F = R.meta.best_frame !== undefined ? R.meta.best_frame : R.meta.frames - 1;
    if (capEl) {
      var how = ' All panels share the same camera. Drag to rotate, scroll to zoom, ' +
        'hover for neuron details, and click a neuron to trace its connections.';
      capEl.innerHTML = FIG
        ? 'The final state of the selected run, the checkpoint at step ' +
          fmtStep(R.meta.best_step) + ' of ' + fmtStep(R.meta.total_steps) +
          ' that the paper’s figure draws, shown for each layer transition and for ' +
          'the full network.' + how
        : R.meta.best_frame !== undefined
        ? 'The final state of the selected run, its selected checkpoint at step ' +
          fmtStep(R.meta.best_step) + ' of ' + fmtStep(R.meta.total_steps) +
          ', shown for each layer transition and for the full network.' + how
        : capDefault;
    }
    R.finalPos = [];
    R.finalBest = [];
    for (var u = 0; u < R.U; u++) {
      R.finalPos.push(finalPoint(u));
      R.finalBest.push(R.bestOf(F, u));
    }
    R.edges = collectEdges();
    state.offLayers = {};
    buildLayerChips();
    render();
  });

  var resizeT = null;
  window.addEventListener('resize', function () {
    clearTimeout(resizeT);
    resizeT = setTimeout(render, 180);
  });

  var anyVisible = true;
  if (window.IntersectionObserver) {
    new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { anyVisible = en.isIntersecting; });
    }, {rootMargin: '120px'}).observe(root);
  }
  function frame(now) {
    if (last === null) last = now;
    var dt = now - last;
    last = now;
    if (R && state.spinning && anyVisible && !document.hidden &&
        now >= state.holdUntil) {
      state.phase += (dt / PERIOD) * TAU;
      if (state.phase > TAU) state.phase -= TAU;
      drawAllPanels();
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
