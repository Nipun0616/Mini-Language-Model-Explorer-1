(function () {
  'use strict';
  var M = window.MiniLM;
  var $ = function (id) { return document.getElementById(id); };
  var hasPlotly = typeof window.Plotly !== 'undefined';

  /* ---------- helpers ---------- */
  function esc(s) { return String(s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); }
  function fmt(x, n) { n = n == null ? 2 : n; var s = Number(x).toFixed(n); if (/^-0(\.0*)?$/.test(s)) s = s.slice(1); return s; }
  function cv(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  var SERIES = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#D99100', '#56B4E9', '#8A6D00', '#6B7B8A'];

  var DEFAULTS = ['The cat sat on the mat.', 'The dog sat on the rug.', 'The cat likes the dog.', 'The sun is hot today.', 'I like my cat and dog.', 'The dog ran to the park.'];
  var CATS = [
    { key: 'Word', label: 'Words', css: '--c-word' },
    { key: 'Punctuation', label: 'Punctuation', css: '--c-punct' },
    { key: 'Character', label: 'Characters', css: '--c-char' },
    { key: 'Subword', label: 'Subwords', css: '--c-sub' },
    { key: 'Special', label: '[UNK]', css: '--c-special' },
    { key: 'Sentence', label: 'Training sentences', css: '--c-sent' },
    { key: 'Your sentence', label: 'Your sentences', css: '--c-user' },
    { key: 'Unseen', label: 'Unseen in training', css: '--c-unseen' }
  ];
  function catLabel(k) { for (var i = 0; i < CATS.length; i++) if (CATS[i].key === k) return CATS[i].label; return k; }

  var state = {
    sentences: DEFAULTS.slice(), vocab: null, corpus: null, model: null, vectors: null,
    dim: 8, epochs: 400, seed: 42, training: false,
    userSentences: ['The dog likes the cat.'], userShown: 0,
    view: '3d', drag: 'rotate', rev: 0,
    visible: { Word: true, Punctuation: true, Character: true, Subword: true, Special: true, Sentence: true, 'Your sentence': true, Unseen: false },
    sel: 'tok:cat', cmp: 'tok:dog', items: null, byKey: {}, pca: null, orderMain: [], order2d: []
  };

  /* ---------- plotting wrapper ---------- */
  function draw(id, traces, layout) {
    var el = $(id);
    if (!el) return Promise.resolve();
    if (!hasPlotly) { el.innerHTML = '<p class="note pad">Charts need the Plotly library, which could not be loaded here.</p>'; return Promise.resolve(); }
    return Plotly.react(el, traces, layout, { responsive: true, displaylogo: false, scrollZoom: true });
  }
  function base(extra) {
    return Object.assign({
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: 'Inter, system-ui, sans-serif', size: 12, color: cv('--fg') },
      margin: { l: 56, r: 16, t: 12, b: 48 }, showlegend: false
    }, extra || {});
  }
  function axis(title, extra) {
    return Object.assign({
      title: { text: title, font: { color: cv('--fg') } }, gridcolor: cv('--line'), zerolinecolor: cv('--muted'),
      linecolor: cv('--line'), tickfont: { color: cv('--muted') }
    }, extra || {});
  }

  /* ---------- vocabulary ---------- */
  function rebuildVocab() {
    state.vocab = M.buildVocab(state.sentences);
    state.corpus = M.buildCorpus(state.sentences, state.vocab, 2);
  }

  function renderVocabStats() {
    var v = state.vocab.validation, tg = state.vocab.target;
    var badge = v.ok ? '<span class="badge ok">Matches target ' + tg.total + '</span>' : '<span class="badge bad">Target is ' + tg.total + '</span>';
    $('vocabStats').innerHTML =
      '<div class="tile"><span class="small">Total vocabulary size</span><span class="big">' + v.total + '</span>' + badge + '</div>' +
      '<div class="tile"><span class="small">Word-level tokens</span><span class="big">' + v.wordLevel + '</span><span class="small">' + v.words + ' words + ' + v.punctuation + ' punctuation (target ' + tg.wordLevel + ')</span></div>' +
      '<div class="tile"><span class="small">Character / subword pieces</span><span class="big">' + v.pieces + '</span><span class="small">' + v.characters + ' characters + ' + v.subwords + ' subwords (target ' + tg.pieces + ')</span></div>' +
      '<div class="tile"><span class="small">Special tokens</span><span class="big">' + v.special + '</span><span class="small">[UNK] at ID 0</span></div>';
    var w = [];
    if (!v.unique) w.push('Some tokens are duplicated.');
    if (v.wordLevel < tg.wordLevel) w.push('This corpus has only ' + v.wordLevel + ' distinct words and punctuation marks, so the word-level part is ' + (tg.wordLevel - v.wordLevel) + ' short of ' + tg.wordLevel + '.');
    if (v.pieces < tg.pieces) w.push('Only ' + v.subwords + ' distinct two-letter pairs occur in the corpus words, so the pieces total ' + v.pieces + ' instead of ' + tg.pieces + '.');
    var el = $('vocabWarn');
    el.hidden = w.length === 0;
    el.textContent = w.length ? 'Vocabulary check: ' + w.join(' ') + ' The size is reported as it is, not padded.' : '';
  }

  function renderVocab() {
    var q = $('vSearch').value.trim().toLowerCase(), type = $('vType').value, sort = $('vSort').value;
    var rows = state.vocab.tokens.filter(function (t) {
      if (type !== 'all' && t.type !== type) return false;
      if (!q) return true;
      return t.token.toLowerCase().indexOf(q) >= 0 || String(t.id) === q;
    });
    if (sort === 'freq') {
      rows = rows.slice().sort(function (a, b) {
        var fa = a.freq == null ? -1 : a.freq, fb = b.freq == null ? -1 : b.freq;
        return fb - fa || a.id - b.id;
      });
    }
    var html = '<thead><tr><th class="num">ID</th><th>Token</th><th>Type</th><th class="num">Frequency</th></tr></thead><tbody>';
    rows.forEach(function (t) {
      html += '<tr><td class="num">' + t.id + '</td><td class="mono">' + esc(t.token) + '</td><td><span class="tag t-' + t.type + '">' + t.type + '</span></td><td class="num">' + (t.freq == null ? 'Unknown' : t.freq) + '</td></tr>';
    });
    $('vocabTable').innerHTML = html + '</tbody>';
    $('vocabCount').textContent = 'Showing ' + rows.length + ' of ' + state.vocab.tokens.length + ' tokens. Frequency counts how often the word, punctuation mark, letter or letter pair occurs in the corpus.';
  }

  /* ---------- tokenizer output ---------- */
  function chip(t) {
    var st = t.status === 'known' ? 'known' : t.status === 'piece' ? 'piece' : 'unknown';
    var stText = st === 'known' ? 'known' : st === 'piece' ? 'known piece' : 'unknown';
    return '<div class="chip t-' + t.type + (t.id === 0 ? ' is-unk' : '') + '"><span class="ctext">' + esc(t.text) + '</span><span class="cid">' + t.id + '</span><span class="ctype">' + t.type + '</span><span class="cstat ' + st + '">' + stText + '</span></div>';
  }
  function chipsHTML(toks) {
    var html = '', i = 0;
    while (i < toks.length) {
      var t = toks[i];
      if (t.fromUnknown) {
        var g = t.group, j = i, inner = '';
        while (j < toks.length && toks[j].group === g) { inner += chip(toks[j]); j++; }
        var parts = toks.slice(i, j), nUnk = parts.filter(function (x) { return x.id === 0; }).length;
        var label = esc(t.origin) + ' is not in the vocabulary' + (nUnk === parts.length ? ' and has no usable pieces' : ': split into ' + (parts.length - nUnk) + ' piece' + (parts.length - nUnk === 1 ? '' : 's'));
        html += '<div class="ugroup"><div class="ulabel">' + label + '</div><div class="chips">' + inner + '</div></div>';
        i = j;
      } else { html += chip(t); i++; }
    }
    return '<div class="chips">' + html + '</div>';
  }
  function tokenSummary(text, toks) {
    var chars = text.length, n = toks.length;
    var unkWords = {}; toks.forEach(function (t) { if (t.fromUnknown) unkWords[t.group] = true; });
    return '<div class="stats"><span>Total characters: <b>' + chars + '</b></span><span>Total tokens: <b>' + n + '</b></span><span>Average characters per token: <b>' + (n ? fmt(chars / n, 2) : '0.00') + '</b></span><span>Words split into pieces: <b>' + Object.keys(unkWords).length + '</b></span></div>';
  }
  function renderTokenizer() {
    var text = $('tokInput').value;
    if (!text.trim()) { $('tokOut').innerHTML = '<p class="note">Type a sentence above and press Tokenize.</p>'; return; }
    var toks = M.tokenize(text, state.vocab);
    $('tokOut').innerHTML =
      '<div><span class="lbl">Original sentence</span><p class="quote">“' + esc(text) + '”</p></div>' +
      '<div><span class="lbl">Tokens, with ID, type and status under each</span>' + chipsHTML(toks) + '</div>' +
      '<div><span class="lbl">Encoded token ID sequence</span><div class="ids">[' + toks.map(function (t) { return t.id; }).join(', ') + ']</div></div>' +
      tokenSummary(text, toks);
  }

  /* ---------- training ---------- */
  function newModel() {
    return M.createModel(state.vocab, state.corpus, { dim: state.dim, seed: state.seed, epochs: state.epochs, lr: 0.05, lrEnd: 0.005, negatives: 5 });
  }
  function setBusy(b) {
    state.training = b;
    ['btnTrain', 'btnRebuild', 'btnEmbed', 'dimRange', 'epRange', 'seedIn'].forEach(function (id) { $(id).disabled = b; });
  }
  function trainNow() {
    var m = newModel(); m.runEpochs(state.epochs); state.model = m; afterTrain();
  }
  function trainAnimated() {
    if (state.training) return;
    if (!state.corpus.pairs.length) { $('trainText').textContent = 'The corpus has no training pairs. Add a sentence with at least two tokens.'; return; }
    var m = newModel(); setBusy(true);
    var chunk = Math.max(2, Math.round(state.epochs / 60));
    function frame() {
      m.runEpochs(chunk);
      $('prog').value = Math.round(100 * m.epoch / state.epochs);
      var L = m.lossHistory;
      $('trainText').textContent = 'Training: epoch ' + m.epoch + ' of ' + state.epochs + ', loss ' + fmt(L[L.length - 1], 3);
      drawLoss(m);
      if (m.done()) { state.model = m; setBusy(false); afterTrain(); }
      else requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }
  function drawLoss(m) {
    var L = m.lossHistory, x = L.map(function (_, i) { return i + 1; });
    draw('lossPlot', [{ type: 'scatter', mode: 'lines', x: x, y: L, line: { color: cv('--ser-a'), width: 2 }, hovertemplate: 'Epoch %{x}<br>Loss %{y:.3f}<extra></extra>' }],
      base({ xaxis: axis('Epoch', { range: [0, Math.max(10, state.epochs)] }), yaxis: axis('Average loss per pair', { rangemode: 'tozero' }), margin: { l: 60, r: 16, t: 8, b: 44 } }));
  }
  function afterTrain() {
    state.vectors = state.model.vectors();
    var L = state.model.lossHistory;
    $('prog').value = 100;
    $('trainText').textContent = 'Done: ' + state.model.epoch + ' epochs, loss fell from ' + fmt(L[0], 3) + ' to ' + fmt(L[L.length - 1], 3) + ' across ' + state.corpus.pairs.length + ' training pairs.';
    $('statusBar').innerHTML = '<span class="dotok"></span><span>Embeddings trained</span><span><b>' + state.model.dim + '</b> dimensions</span><span><b>' + state.model.epoch + '</b> epochs</span><span>seed <b>' + state.seed + '</b></span><span>final loss <b>' + fmt(L[L.length - 1], 3) + '</b></span>';
    drawLoss(state.model);
    refreshItems(true);
    renderEmbTable(); renderSentTable();
  }

  /* ---------- items (tokens and sentences with vectors and PCA coordinates) ---------- */
  function sentenceVec(text) {
    var ids = M.tokenize(text, state.vocab).map(function (t) { return t.id; });
    return M.meanPool(ids, state.vectors);
  }
  function computeItems() {
    var V = state.vectors, trained = state.corpus.trained, items = [];
    state.vocab.tokens.forEach(function (t) {
      items.push({ key: 'tok:' + t.token, label: t.token, cat: trained[t.id] ? t.type : 'Unseen', kind: 'token', tokenId: t.id, type: t.type, vec: V[t.id] });
    });
    state.sentences.forEach(function (s, i) { items.push({ key: 'sent:' + i, label: 'S' + (i + 1), cat: 'Sentence', kind: 'sentence', text: s, vec: sentenceVec(s) }); });
    state.userSentences.forEach(function (s, i) { items.push({ key: 'user:' + i, label: 'U' + (i + 1), cat: 'Your sentence', kind: 'sentence', text: s, vec: sentenceVec(s) }); });
    var rows = items.filter(function (it) { return it.kind === 'token' && it.cat !== 'Unseen'; }).map(function (it) { return it.vec; });
    state.pca = rows.length >= 2 ? M.fitPCA(rows, 3) : null;
    items.forEach(function (it) { it.xyz = state.pca ? state.pca.project(it.vec) : [0, 0, 0]; });
    state.items = items; state.byKey = {};
    items.forEach(function (it) { state.byKey[it.key] = it; });
    if (!state.byKey[state.sel]) state.sel = firstWordKey();
    if (state.cmp && !state.byKey[state.cmp]) state.cmp = '';
  }
  function firstWordKey() {
    for (var i = 0; i < state.items.length; i++) if (state.items[i].cat === 'Word') return state.items[i].key;
    return state.items[0].key;
  }
  function refreshItems(full) {
    computeItems();
    renderSelectors(); renderCatBox();
    renderMain(); render2D(); renderInspector(); renderCosine(); renderSentCharts(); renderUserList(); renderUser();
  }

  /* ---------- embedding tables ---------- */
  function renderEmbTable() {
    var q = $('eSearch').value.trim().toLowerCase(), onlyTrained = $('eTrained').checked, d = state.model.dim, trained = state.corpus.trained;
    var head = '<thead><tr><th class="num">ID</th><th>Token</th><th>Type</th><th>Trained</th>';
    for (var i = 1; i <= d; i++) head += '<th class="num">d' + i + '</th>';
    var html = head + '</tr></thead><tbody>', shown = 0;
    state.vocab.tokens.forEach(function (t) {
      var isTrained = !!trained[t.id];
      if (onlyTrained && !isTrained) return;
      if (q && t.token.toLowerCase().indexOf(q) < 0 && String(t.id) !== q) return;
      shown++;
      html += '<tr class="' + (isTrained ? '' : 'unseen') + '"><td class="num">' + t.id + '</td><td class="mono">' + esc(t.token) + '</td><td><span class="tag t-' + t.type + '">' + t.type + '</span></td><td>' + (isTrained ? 'Yes' : 'No') + '</td>';
      state.vectors[t.id].forEach(function (x) { html += '<td class="num mono">' + fmt(x, 3) + '</td>'; });
      html += '</tr>';
    });
    $('embTable').innerHTML = html + '</tbody>';
    var nUn = state.vocab.tokens.filter(function (t) { return !trained[t.id]; }).length;
    $('embNote').textContent = 'Showing ' + shown + ' of ' + state.vocab.tokens.length + ' tokens, each with ' + d + ' values. ' + nUn + ' tokens never occur in the corpus, so they keep their small random starting vector and are marked “No”.';
  }
  function renderSentTable() {
    var html = '<thead><tr><th>#</th><th>Sentence</th><th class="num">Tokens</th><th>Sentence embedding (mean of token vectors)</th></tr></thead><tbody>';
    state.sentences.forEach(function (s, i) {
      var n = M.tokenize(s, state.vocab).length;
      html += '<tr><td>S' + (i + 1) + '</td><td>' + esc(s) + '</td><td class="num">' + n + '</td><td class="mono wrapcell">[' + sentenceVec(s).map(function (x) { return fmt(x, 3); }).join(', ') + ']</td></tr>';
    });
    $('sentTable').innerHTML = html + '</tbody>';
  }

  /* ---------- selectors and category box ---------- */
  function itemName(it) { return it.kind === 'token' ? it.label : it.label + ' “' + it.text + '”'; }
  function optionsHTML(selected, allowNone) {
    var html = allowNone ? '<option value="">None</option>' : '';
    CATS.forEach(function (c) {
      var its = state.items.filter(function (it) { return it.cat === c.key; });
      if (!its.length) return;
      html += '<optgroup label="' + esc(c.label) + '">';
      its.forEach(function (it) {
        var t = it.kind === 'token' ? it.label : it.label + ' – ' + (it.text.length > 34 ? it.text.slice(0, 33) + '…' : it.text);
        html += '<option value="' + esc(it.key) + '"' + (it.key === selected ? ' selected' : '') + '>' + esc(t) + '</option>';
      });
      html += '</optgroup>';
    });
    return html;
  }
  function renderSelectors() {
    $('selA').innerHTML = optionsHTML(state.sel, false);
    $('selB').innerHTML = optionsHTML(state.cmp, true);
    $('selA').value = state.sel; $('selB').value = state.cmp || '';
  }
  function renderCatBox() {
    var html = '';
    CATS.forEach(function (c) {
      var n = state.items.filter(function (it) { return it.cat === c.key; }).length;
      if (!n) return;
      html += '<label><input type="checkbox" data-cat="' + esc(c.key) + '"' + (state.visible[c.key] ? ' checked' : '') + '><span class="sw" style="background:var(' + c.css + ')"></span>' + esc(c.label) + ' (' + n + ')</label>';
    });
    $('catBox').innerHTML = html;
  }

  /* ---------- main 3D / 2D plots ---------- */
  function pcLabel(i) {
    var r = state.pca && state.pca.ratio[i];
    return 'PC' + (i + 1) + (r != null ? ' (' + (r * 100).toFixed(1) + '% of variance)' : ' (not available: only ' + state.model.dim + ' dimensions)');
  }
  function vecLines(v) {
    var out = [];
    for (var i = 0; i < v.length; i += 8) out.push(v.slice(i, i + 8).map(function (x) { return fmt(x, 2); }).join(', '));
    return out.join('<br>');
  }
  function hoverHTML(it) {
    var name = it.kind === 'token' ? it.label : it.label + ': ' + it.text;
    return '<b>' + esc(name) + '</b><br>' + esc(catLabel(it.cat)) + (it.kind === 'token' ? ' · ID ' + it.tokenId : '') + '<br>' + it.vec.length + '-dimensional vector:<br>' + vecLines(it.vec);
  }
  function buildTraces(mode) {
    var traces = [], order = [], is3 = mode === '3d', labelsOn = $('lblToggle').checked, fg = cv('--fg');
    CATS.forEach(function (c) {
      var its = state.items.filter(function (it) { return it.cat === c.key; });
      if (!its.length) return;
      var col = cv(c.css), isSent = c.key === 'Sentence' || c.key === 'Your sentence';
      var t = {
        type: is3 ? 'scatter3d' : 'scatter', mode: labelsOn ? 'markers+text' : 'markers', name: c.label,
        x: its.map(function (it) { return it.xyz[0]; }), y: its.map(function (it) { return it.xyz[1]; }),
        text: its.map(function (it) { return it.label; }), textposition: 'top center', textfont: { size: is3 ? 10 : 11, color: col },
        hovertext: its.map(hoverHTML), hoverinfo: 'text', hoverlabel: { align: 'left', font: { family: 'JetBrains Mono, monospace', size: 11 } },
        marker: { size: isSent ? (c.key === 'Your sentence' ? 10 : 8) : 5, color: col, symbol: isSent ? 'diamond' : 'circle' },
        visible: !!state.visible[c.key]
      };
      if (is3) t.z = its.map(function (it) { return it.xyz[2]; });
      traces.push(t); order.push(its.map(function (it) { return it.key; }));
    });
    var ring = [state.sel, state.cmp].filter(Boolean).map(function (k) { return state.byKey[k]; }).filter(function (it) { return it && state.visible[it.cat]; });
    if (ring.length) {
      var r = { type: is3 ? 'scatter3d' : 'scatter', mode: 'markers', name: 'Selected', hoverinfo: 'skip',
        x: ring.map(function (it) { return it.xyz[0]; }), y: ring.map(function (it) { return it.xyz[1]; }),
        marker: { symbol: 'circle-open', size: is3 ? 12 : 17, color: fg, line: { color: fg, width: 3 } } };
      if (is3) r.z = ring.map(function (it) { return it.xyz[2]; });
      traces.push(r); order.push(null);
    }
    return { traces: traces, order: order };
  }
  function dragMode(is3) {
    if (is3) return { rotate: 'turntable', pan: 'pan', zoom: 'zoom' }[state.drag];
    return { rotate: 'pan', pan: 'pan', zoom: 'zoom' }[state.drag];
  }
  function bindClick(id, key) {
    var el = $(id);
    if (!hasPlotly || el._bound || !el.on) return;
    el._bound = true;
    el.on('plotly_click', function (ev) {
      var p = ev.points && ev.points[0]; if (!p) return;
      var ord = state[key][p.curveNumber]; if (!ord) return;
      var k = ord[p.pointNumber]; if (!k) return;
      state.sel = k; $('selA').value = k; renderInspector(); renderMain(); render2D();
    });
  }
  function renderMain() {
    if (!state.items) return;
    var is3 = state.view === '3d', b = buildTraces(state.view); state.orderMain = b.order;
    var rev = state.view + ':' + state.rev, layout;
    if (is3) {
      var ax = function (t) { return axis(t, { showbackground: false, backgroundcolor: 'rgba(0,0,0,0)' }); };
      layout = base({ margin: { l: 0, r: 0, t: 0, b: 0 }, uirevision: rev, dragmode: undefined,
        scene: { xaxis: ax(pcLabel(0)), yaxis: ax(pcLabel(1)), zaxis: ax(pcLabel(2)), dragmode: dragMode(true), camera: { eye: { x: 1.5, y: 1.5, z: 1.1 } }, bgcolor: 'rgba(0,0,0,0)' } });
    } else {
      layout = base({ uirevision: rev, dragmode: dragMode(false), xaxis: axis(pcLabel(0)), yaxis: axis(pcLabel(1)) });
    }
    var p = draw('plotMain', b.traces, layout);
    if (p && p.then) p.then(function () { bindClick('plotMain', 'orderMain'); });
    var pc = state.pca;
    $('pcaNote').textContent = pc ? 'The ' + state.model.dim + '-dimensional vectors are projected onto the top ' + Math.min(3, state.model.dim) + ' principal components, which together keep ' + (pc.ratio.reduce(function (a, x) { return a + x; }, 0) * 100).toFixed(1) + '% of the variance among the trained tokens. Sentence points use the same projection.' : 'Not enough trained tokens for a projection.';
  }
  function render2D() {
    if (!state.items) return;
    var b = buildTraces('2d'); state.order2d = b.order;
    var p = draw('plot2d', b.traces, base({ uirevision: 'p2d', dragmode: 'pan', xaxis: axis(pcLabel(0)), yaxis: axis(pcLabel(1)) }));
    if (p && p.then) p.then(function () { bindClick('plot2d', 'order2d'); });
  }
  function setView(v) {
    state.view = v;
    $('vw3').setAttribute('aria-pressed', v === '3d'); $('vw2').setAttribute('aria-pressed', v === '2d');
    $('dgRot').disabled = v === '2d';
    if (v === '2d' && state.drag === 'rotate') setDrag('pan', true);
    if (hasPlotly) { try { Plotly.purge($('plotMain')); } catch (e) { /* nothing to purge */ } }
    $('plotMain')._bound = false;
    renderMain();
  }
  function setDrag(d, skipRender) {
    state.drag = d;
    $('dgRot').setAttribute('aria-pressed', d === 'rotate'); $('dgPan').setAttribute('aria-pressed', d === 'pan'); $('dgZoom').setAttribute('aria-pressed', d === 'zoom');
    if (skipRender) return;
    if (hasPlotly && $('plotMain').data) {
      var upd = {}; if (state.view === '3d') upd['scene.dragmode'] = dragMode(true); else upd.dragmode = dragMode(false);
      Plotly.relayout($('plotMain'), upd);
    }
  }

  /* ---------- inspector and bar chart ---------- */
  function nearestTo(a, k) {
    return state.items.filter(function (x) { return x !== a && x.cat !== 'Unseen'; })
      .map(function (x) { return { it: x, s: M.cosine(a.vec, x.vec) }; })
      .sort(function (p, q) { return q.s - p.s; }).slice(0, k);
  }
  function renderInspector() {
    var a = state.byKey[state.sel], b = state.cmp ? state.byKey[state.cmp] : null;
    if (!a) return;
    var col = 'var(' + (CATS.filter(function (c) { return c.key === a.cat; })[0] || CATS[0]).css + ')';
    var html = '<div class="insp-head"><span class="dot" style="background:' + col + '"></span><strong class="mono">' + esc(itemName(a)) + '</strong><span class="muted">' + esc(catLabel(a.cat)) + (a.kind === 'token' ? ' · ID ' + a.tokenId : '') + ' · ' + a.vec.length + ' dimensions</span></div>';
    html += '<div class="vec">[' + a.vec.map(function (x) { return fmt(x, 3); }).join(', ') + ']</div>';
    if (b) html += '<p>Cosine similarity with <strong class="mono">' + esc(itemName(b)) + '</strong>: <strong>' + fmt(M.cosine(a.vec, b.vec), 3) + '</strong></p>';
    html += '<div><span class="lbl">Closest vectors</span><ol class="near">' + nearestTo(a, 6).map(function (n) { return '<li><span class="mono">' + esc(itemName(n.it)) + '</span> · ' + fmt(n.s, 3) + '</li>'; }).join('') + '</ol></div>';
    $('inspectBody').innerHTML = html;
    renderBars();
  }
  function renderBars() {
    var a = state.byKey[state.sel], b = state.cmp ? state.byKey[state.cmp] : null; if (!a) return;
    var d = a.vec.length, x = []; for (var i = 1; i <= d; i++) x.push('d' + i);
    var traces = [{ type: 'bar', name: a.kind === 'token' ? a.label : a.label, x: x, y: a.vec, marker: { color: cv('--ser-a') }, hovertemplate: '%{x}: %{y:.3f}<extra>' + esc(a.label) + '</extra>' }];
    if (b) traces.push({ type: 'bar', name: b.label, x: x, y: b.vec, marker: { color: cv('--ser-b') }, hovertemplate: '%{x}: %{y:.3f}<extra>' + esc(b.label) + '</extra>' });
    draw('plotBars', traces, base({ barmode: 'group', showlegend: true, legend: { orientation: 'h', y: 1.12, font: { color: cv('--fg') } }, xaxis: axis('Dimension'), yaxis: axis('Value'), margin: { l: 56, r: 12, t: 28, b: 48 } }));
  }

  /* ---------- similarity charts ---------- */
  function renderCosine() {
    var toks = state.items.filter(function (it) { return it.kind === 'token' && (it.type === 'Word' || it.type === 'Punctuation') && it.cat !== 'Unseen'; });
    var labels = toks.map(function (t) { return t.label; });
    var z = toks.map(function (a) { return toks.map(function (b) { return M.cosine(a.vec, b.vec); }); });
    draw('plotCos', [{ type: 'heatmap', x: labels, y: labels, z: z, zmin: -1, zmax: 1, colorscale: 'RdBu', hovertemplate: '%{y} and %{x}: %{z:.2f}<extra></extra>', colorbar: { thickness: 12, tickfont: { color: cv('--muted') } } }],
      base({ xaxis: axis('', { tickangle: -45, tickfont: { size: 10, color: cv('--muted') }, side: 'bottom' }), yaxis: axis('', { autorange: 'reversed', tickfont: { size: 10, color: cv('--muted') } }), margin: { l: 56, r: 8, t: 8, b: 64 } }));
  }
  function renderSentCharts() {
    var sents = state.items.filter(function (it) { return it.kind === 'sentence'; });
    var d = state.model.dim, x = []; for (var i = 1; i <= d; i++) x.push(i);
    var traces = sents.map(function (s, i) {
      return { type: 'scatter', mode: 'lines+markers', name: s.label, x: x, y: s.vec, line: { color: s.cat === 'Your sentence' ? cv('--c-user') : SERIES[i % SERIES.length], width: s.cat === 'Your sentence' ? 3 : 1.6, dash: s.cat === 'Your sentence' ? 'dot' : 'solid' }, marker: { size: 5 }, hovertemplate: s.label + ' · d%{x}: %{y:.3f}<extra></extra>' };
    });
    draw('plotSentVec', traces, base({ showlegend: true, legend: { orientation: 'h', y: 1.2, font: { color: cv('--fg') } }, xaxis: axis('Dimension', { dtick: d > 16 ? 4 : 1 }), yaxis: axis('Value'), margin: { l: 56, r: 12, t: 30, b: 44 } }));
    var labels = sents.map(function (s) { return s.label; });
    var z = sents.map(function (a) { return sents.map(function (b) { return M.cosine(a.vec, b.vec); }); });
    draw('plotSentCos', [{ type: 'heatmap', x: labels, y: labels, z: z, zmin: -1, zmax: 1, colorscale: 'RdBu', text: z.map(function (r) { return r.map(function (v) { return fmt(v, 2); }); }), texttemplate: '%{text}', textfont: { size: 11 }, hovertemplate: '%{y} and %{x}: %{z:.3f}<extra></extra>', colorbar: { thickness: 12, tickfont: { color: cv('--muted') } } }],
      base({ xaxis: axis('', { side: 'bottom' }), yaxis: axis('', { autorange: 'reversed' }), margin: { l: 44, r: 8, t: 8, b: 36 } }));
  }

  /* ---------- Part 4: user explorer ---------- */
  function renderUserList() {
    $('uList').innerHTML = state.userSentences.map(function (s, i) {
      return '<button type="button" data-u="' + i + '" aria-pressed="' + (i === state.userShown) + '">U' + (i + 1) + ' · ' + esc(s.length > 30 ? s.slice(0, 29) + '…' : s) + '</button>';
    }).join('');
  }
  function renderUser() {
    var out = $('userOut');
    if (!state.userSentences.length) { out.innerHTML = '<p class="note">Embed a sentence to see its tokens, vectors and nearest neighbours here. It will also appear in the plot above.</p>'; return; }
    if (state.userShown >= state.userSentences.length) state.userShown = state.userSentences.length - 1;
    var text = state.userSentences[state.userShown], toks = M.tokenize(text, state.vocab), ids = toks.map(function (t) { return t.id; });
    var V = state.vectors, trained = state.corpus.trained, d = state.model.dim;
    var svec = M.meanPool(ids, V), me = state.byKey['user:' + state.userShown];
    var html = '<div><span class="lbl">Step 1 · Tokens</span>' + chipsHTML(toks) + '</div>' +
      '<div><span class="lbl">Step 2 · Token IDs</span><div class="ids">[' + ids.join(', ') + ']</div></div>' + tokenSummary(text, toks);
    var rows = '<thead><tr><th class="num">#</th><th>Token</th><th class="num">ID</th><th>Trained</th>';
    for (var i = 1; i <= d; i++) rows += '<th class="num">d' + i + '</th>';
    rows += '</tr></thead><tbody>';
    toks.forEach(function (t, k) {
      rows += '<tr class="' + (trained[t.id] ? '' : 'unseen') + '"><td class="num">' + (k + 1) + '</td><td class="mono">' + esc(t.text) + '</td><td class="num">' + t.id + '</td><td>' + (trained[t.id] ? 'Yes' : 'No') + '</td>';
      V[t.id].forEach(function (x) { rows += '<td class="num mono">' + fmt(x, 3) + '</td>'; });
      rows += '</tr>';
    });
    html += '<div><span class="lbl">Step 3 · One vector per token</span><div class="tablewrap" style="max-height:300px"><table>' + rows + '</tbody></table></div></div>';
    html += '<div><span class="lbl">Step 4 · Mean of the ' + toks.length + ' token vectors gives the sentence embedding</span><div class="vec">[' + svec.map(function (x) { return fmt(x, 3); }).join(', ') + ']</div></div>';
    html += '<div id="plotUserVec" class="plot short"></div>';
    var sims = state.items.filter(function (it) { return it.cat === 'Sentence'; }).map(function (it) { return { it: it, s: M.cosine(svec, it.vec) }; }).sort(function (a, b) { return b.s - a.s; });
    html += '<div class="grid2"><div class="sub"><span class="lbl">Most similar training sentences (cosine similarity)</span>' + sims.map(function (r) {
      return '<div class="simrow"><span class="mono">' + r.it.label + '</span><span class="txt" title="' + esc(r.it.text) + '">' + esc(r.it.text) + '</span><span class="bar"><i style="width:' + Math.max(0, Math.round(r.s * 100)) + '%"></i></span><span class="val">' + fmt(r.s, 3) + '</span></div>';
    }).join('') + '</div>';
    var near = state.items.filter(function (it) { return it.kind === 'token' && (it.type === 'Word' || it.type === 'Punctuation') && it.cat !== 'Unseen'; })
      .map(function (it) { return { it: it, s: M.cosine(svec, it.vec) }; }).sort(function (a, b) { return b.s - a.s; }).slice(0, 6);
    html += '<div class="sub"><span class="lbl">Closest word-level tokens to the sentence vector</span><ol class="near">' + near.map(function (n) { return '<li><span class="mono">' + esc(n.it.label) + '</span> · ' + fmt(n.s, 3) + '</li>'; }).join('') + '</ol></div></div>';
    html += '<p class="note">This sentence is plotted above as <b>' + (me ? me.label : '') + '</b>. Tokens marked “No” in the Trained column never occurred in the corpus, so their vectors are still near random and pull the average slightly off.</p>';
    out.innerHTML = html;
    draw('plotUserVec', [{ type: 'bar', x: svec.map(function (_, i) { return 'd' + (i + 1); }), y: svec, marker: { color: cv('--c-user') }, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' }],
      base({ xaxis: axis('Dimension'), yaxis: axis('Sentence vector value'), margin: { l: 60, r: 12, t: 8, b: 44 } }));
  }
  function embedUser() {
    if (state.training) return;
    var text = $('userInput').value.trim();
    if (!text) return;
    var idx = state.userSentences.indexOf(text);
    if (idx < 0) {
      state.userSentences.push(text);
      if (state.userSentences.length > 10) state.userSentences.shift();
      idx = state.userSentences.indexOf(text);
    }
    state.userShown = idx; state.sel = 'user:' + idx;
    refreshItems();
  }

  /* ---------- events ---------- */
  function applyCorpus() {
    var lines = $('corpus').value.split(/\n+/).map(function (s) { return s.trim(); }).filter(Boolean);
    var msg = $('corpusMsg');
    if (!lines.length) { msg.hidden = false; msg.textContent = 'Add at least one sentence.'; return; }
    msg.hidden = true;
    state.sentences = lines;
    rebuildVocab();
    if (!state.corpus.pairs.length) { msg.hidden = false; msg.textContent = 'These sentences give no training pairs. Use sentences with at least two tokens.'; return; }
    renderVocabStats(); renderVocab(); renderTokenizer();
    trainAnimated();
  }
  function bind() {
    $('btnRebuild').addEventListener('click', applyCorpus);
    $('btnReset').addEventListener('click', function () { $('corpus').value = DEFAULTS.join('\n'); applyCorpus(); });
    ['vSearch', 'vSort', 'vType'].forEach(function (id) { $(id).addEventListener('input', renderVocab); });
    $('btnTok').addEventListener('click', renderTokenizer);
    $('tokInput').addEventListener('keydown', function (e) { if (e.key === 'Enter') renderTokenizer(); });
    $('dimRange').addEventListener('input', function () { $('dimOut').textContent = this.value; });
    $('dimRange').addEventListener('change', function () { state.dim = +this.value; trainAnimated(); });
    $('epRange').addEventListener('input', function () { $('epOut').textContent = this.value; state.epochs = +this.value; });
    $('seedIn').addEventListener('change', function () { var v = parseInt(this.value, 10); state.seed = isNaN(v) ? 42 : v; this.value = state.seed; });
    $('btnTrain').addEventListener('click', function () {
      state.dim = +$('dimRange').value; state.epochs = +$('epRange').value;
      var v = parseInt($('seedIn').value, 10); state.seed = isNaN(v) ? 42 : v;
      trainAnimated();
    });
    $('eSearch').addEventListener('input', function () { if (state.vectors) renderEmbTable(); });
    $('eTrained').addEventListener('change', function () { if (state.vectors) renderEmbTable(); });
    $('vw3').addEventListener('click', function () { setView('3d'); });
    $('vw2').addEventListener('click', function () { setView('2d'); });
    $('dgRot').addEventListener('click', function () { setDrag('rotate'); });
    $('dgPan').addEventListener('click', function () { setDrag('pan'); });
    $('dgZoom').addEventListener('click', function () { setDrag('zoom'); });
    $('btnResetView').addEventListener('click', function () { state.rev++; renderMain(); });
    $('lblToggle').addEventListener('change', function () { renderMain(); render2D(); });
    $('catBox').addEventListener('change', function (e) {
      var c = e.target.getAttribute('data-cat'); if (!c) return;
      state.visible[c] = e.target.checked; renderMain(); render2D();
    });
    $('selA').addEventListener('change', function () { state.sel = this.value; renderInspector(); renderMain(); render2D(); });
    $('selB').addEventListener('change', function () { state.cmp = this.value; renderInspector(); renderMain(); render2D(); });
    $('btnEmbed').addEventListener('click', embedUser);
    $('userInput').addEventListener('keydown', function (e) { if (e.key === 'Enter') embedUser(); });
    $('btnClearUser').addEventListener('click', function () {
      state.userSentences = []; state.userShown = 0;
      if (state.sel.indexOf('user:') === 0) state.sel = firstWordKey();
      refreshItems();
    });
    $('uList').addEventListener('click', function (e) {
      var b = e.target.closest('button'); if (!b) return;
      state.userShown = +b.getAttribute('data-u'); state.sel = 'user:' + state.userShown;
      renderUserList(); renderUser(); renderInspector(); renderSelectors(); renderMain(); render2D();
    });
    function redraw() { if (!state.items) return; drawLoss(state.model); renderMain(); render2D(); renderBars(); renderCosine(); renderSentCharts(); renderUser(); }
    if (window.matchMedia) {
      var mq = window.matchMedia('(prefers-color-scheme: dark)');
      if (mq.addEventListener) mq.addEventListener('change', redraw);
    }
    new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  }

  /* ---------- start ---------- */
  $('corpus').value = state.sentences.join('\n');
  $('tokInput').value = 'The game sat on the mat.';
  rebuildVocab();
  renderVocabStats(); renderVocab(); renderTokenizer();
  bind();
  trainNow();
})();