/* Mini Language Model core: vocabulary, tokenizer, skip-gram negative sampling, PCA.
   Pure JS, no DOM. Works in the browser (global MiniLM) and in Node (module.exports). */
(function (root) {
  'use strict';

  var LETTERS = 'abcdefghijklmnopqrstuvwxyz'.split('');
  var TARGET = { special: 1, wordLevel: 20, pieces: 51, total: 72 };

  /* ---------- seeded random numbers ---------- */
  function mulberry32(seed) {
    var a = seed | 0;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* ---------- text splitting ---------- */
  function splitText(text) {
    return String(text).toLowerCase().match(/[a-z]+|[^\sa-z]/g) || [];
  }
  function isWord(t) { return /^[a-z]+$/.test(t); }

  /* ---------- vocabulary ---------- */
  // 1 [UNK] + up to 20 word-level tokens (words and punctuation, most frequent first)
  // + 26 single-letter pieces + the most frequent two-letter pieces, up to 51 pieces in total.
  // Pieces carry a ## prefix so the piece "##i" is a different token from the word "i".
  function buildVocab(sentences, target) {
    target = target || TARGET;
    var wl = new Map(), big = new Map(), chars = new Map(), seen = 0;
    sentences.forEach(function (s) {
      splitText(s).forEach(function (t) {
        if (!wl.has(t)) wl.set(t, { count: 0, first: seen++ });
        wl.get(t).count++;
        if (isWord(t)) {
          for (var c = 0; c < t.length; c++) chars.set(t[c], (chars.get(t[c]) || 0) + 1);
          for (var i = 0; i < t.length - 1; i++) {
            var b = t.slice(i, i + 2);
            if (!big.has(b)) big.set(b, { count: 0, first: big.size });
            big.get(b).count++;
          }
        }
      });
    });
    var byCount = function (a, b) { return b[1].count - a[1].count || a[1].first - b[1].first; };
    var wordTokens = Array.from(wl.entries()).sort(byCount).slice(0, target.wordLevel);
    var nSub = Math.max(0, target.pieces - LETTERS.length);
    var subTokens = Array.from(big.entries()).sort(byCount).slice(0, nSub);

    var tokens = [{ id: 0, token: '[UNK]', type: 'Special', freq: null }];
    wordTokens.forEach(function (e) {
      tokens.push({ id: tokens.length, token: e[0], type: isWord(e[0]) ? 'Word' : 'Punctuation', freq: e[1].count });
    });
    LETTERS.forEach(function (c) {
      tokens.push({ id: tokens.length, token: '##' + c, type: 'Character', freq: chars.get(c) || 0 });
    });
    subTokens.forEach(function (e) {
      tokens.push({ id: tokens.length, token: '##' + e[0], type: 'Subword', freq: e[1].count });
    });

    var map = new Map();
    tokens.forEach(function (t) { map.set(t.token, t); });
    var maxPieceLen = 1;
    tokens.forEach(function (t) {
      if (t.type === 'Subword') maxPieceLen = Math.max(maxPieceLen, t.token.length - 2);
    });
    var count = function (type) { return tokens.filter(function (t) { return t.type === type; }).length; };
    var wordLevel = count('Word') + count('Punctuation');
    var pieces = count('Character') + count('Subword');
    var validation = {
      total: tokens.length,
      target: target.total,
      ok: tokens.length === target.total && map.size === tokens.length,
      unique: map.size === tokens.length,
      special: count('Special'),
      words: count('Word'),
      punctuation: count('Punctuation'),
      wordLevel: wordLevel,
      characters: count('Character'),
      subwords: count('Subword'),
      pieces: pieces
    };
    return { tokens: tokens, map: map, maxPieceLen: maxPieceLen, validation: validation, target: target };
  }

  /* ---------- tokenizer ---------- */
  function decompose(word, vocab) {
    var out = [], i = 0;
    while (i < word.length) {
      var hit = null;
      for (var L = Math.min(vocab.maxPieceLen, word.length - i); L >= 1; L--) {
        var e = vocab.map.get('##' + word.slice(i, i + L));
        if (e) { hit = e; break; }
      }
      if (hit) { out.push(hit); i += hit.token.length - 2; }
      else { out.push(vocab.map.get('[UNK]')); i += 1; }
    }
    return out;
  }

  function tokenize(text, vocab) {
    var out = [], g = 0;
    splitText(text).forEach(function (raw) {
      var e = vocab.map.get(raw);
      if (e && (e.type === 'Word' || e.type === 'Punctuation')) {
        out.push({ text: e.token, id: e.id, type: e.type, status: 'known', origin: raw, group: g++, fromUnknown: false });
        return;
      }
      if (isWord(raw)) {
        var gid = g++;
        decompose(raw, vocab).forEach(function (p) {
          out.push({ text: p.token, id: p.id, type: p.type, status: p.id === 0 ? 'unknown' : 'piece', origin: raw, group: gid, fromUnknown: true });
        });
        return;
      }
      out.push({ text: '[UNK]', id: 0, type: 'Special', status: 'unknown', origin: raw, group: g++, fromUnknown: true });
    });
    return out;
  }

  /* ---------- training pairs ---------- */
  // Context pairs come from a sliding window over each sentence's token IDs.
  // Piece pairs tie every piece of a known word to that word and to its neighbouring pieces,
  // so character and subword tokens that occur in the corpus receive trained vectors too.
  function buildCorpus(sentences, vocab, windowSize) {
    var win = windowSize || 2, V = vocab.tokens.length;
    var seqs = [], pairs = [], ctxCount = new Array(V).fill(0), seen = new Array(V).fill(false);
    function add(c, o) { pairs.push([c, o]); ctxCount[o]++; seen[c] = true; seen[o] = true; }
    sentences.forEach(function (s) {
      var toks = tokenize(s, vocab), ids = toks.map(function (t) { return t.id; });
      seqs.push(ids);
      for (var i = 0; i < ids.length; i++) {
        for (var j = i - win; j <= i + win; j++) {
          if (j !== i && j >= 0 && j < ids.length) add(ids[i], ids[j]);
        }
      }
      toks.forEach(function (t) {
        if (t.type !== 'Word') return;
        var w = t.text, parts = [], x;
        for (x = 0; x < w.length; x++) {
          var one = vocab.map.get('##' + w[x]);
          if (one) parts.push(one);
          if (x < w.length - 1) {
            var two = vocab.map.get('##' + w.slice(x, x + 2));
            if (two) parts.push(two);
          }
        }
        parts.forEach(function (p) { add(p.id, t.id); add(t.id, p.id); });
        for (x = 0; x < w.length - 1; x++) {
          var la = vocab.map.get('##' + w[x]), lb = vocab.map.get('##' + w[x + 1]);
          if (la && lb) { add(la.id, lb.id); add(lb.id, la.id); }
        }
      });
    });
    return { seqs: seqs, pairs: pairs, ctxCount: ctxCount, trained: seen, window: win };
  }

  /* ---------- skip-gram with negative sampling ---------- */
  function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

  function createModel(vocab, corpus, cfg) {
    var V = vocab.tokens.length, d = cfg.dim, rng = mulberry32(cfg.seed);
    var Win = [], Wout = [];
    for (var i = 0; i < V; i++) {
      var a = new Float64Array(d);
      for (var j = 0; j < d; j++) a[j] = (rng() - 0.5) / d;
      Win.push(a); Wout.push(new Float64Array(d));
    }
    var initial = Win.map(function (v) { return Array.from(v); });

    var cand = [], cum = [], total = 0;
    corpus.ctxCount.forEach(function (c, id) {
      if (c > 0) { total += Math.pow(c, 0.75); cand.push(id); cum.push(total); }
    });
    function sampleNeg() {
      var r = rng() * total, lo = 0, hi = cum.length - 1;
      while (lo < hi) { var mid = (lo + hi) >> 1; if (cum[mid] < r) lo = mid + 1; else hi = mid; }
      return cand[lo];
    }

    var order = corpus.pairs.map(function (_, k) { return k; });
    var gv = new Float64Array(d);
    var model = {
      Win: Win, Wout: Wout, initial: initial, cfg: cfg, dim: d,
      epoch: 0, lossHistory: [], trained: corpus.trained,
      runEpochs: function (n) {
        for (var e = 0; e < n && model.epoch < cfg.epochs; e++) {
          for (var k = order.length - 1; k > 0; k--) {
            var r = Math.floor(rng() * (k + 1)), tmp = order[k]; order[k] = order[r]; order[r] = tmp;
          }
          var frac = cfg.epochs > 1 ? model.epoch / (cfg.epochs - 1) : 0;
          var lr = cfg.lr + (cfg.lrEnd - cfg.lr) * frac;
          var loss = 0;
          for (var p = 0; p < order.length; p++) {
            var pair = corpus.pairs[order[p]], c = pair[0], o = pair[1], v = Win[c];
            gv.fill(0);
            for (var s = 0; s <= cfg.negatives; s++) {
              var t, label;
              if (s === 0) { t = o; label = 1; }
              else {
                t = sampleNeg();
                var tries = 0;
                while (t === o && tries++ < 5) t = sampleNeg();
                if (t === o) continue;
                label = 0;
              }
              var u = Wout[t], dot = 0;
              for (var q = 0; q < d; q++) dot += v[q] * u[q];
              var sg = sigmoid(dot);
              loss += label ? -Math.log(sg + 1e-12) : -Math.log(1 - sg + 1e-12);
              var g = sg - label;
              for (q = 0; q < d; q++) { var uq = u[q]; gv[q] += g * uq; u[q] = uq - lr * g * v[q]; }
            }
            for (var z = 0; z < d; z++) v[z] -= lr * gv[z];
          }
          model.lossHistory.push(loss / Math.max(1, order.length));
          model.epoch++;
        }
        return model.lossHistory[model.lossHistory.length - 1];
      },
      done: function () { return model.epoch >= cfg.epochs; },
      vectors: function () { return Win.map(function (v) { return Array.from(v); }); }
    };
    return model;
  }

  /* ---------- vector helpers ---------- */
  function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function norm(a) { return Math.sqrt(dot(a, a)); }
  function cosine(a, b) {
    var na = norm(a), nb = norm(b);
    return na === 0 || nb === 0 ? 0 : dot(a, b) / (na * nb);
  }
  function meanPool(ids, vectors) {
    var d = vectors[0].length, out = new Array(d).fill(0);
    if (!ids.length) return out;
    ids.forEach(function (id) { for (var i = 0; i < d; i++) out[i] += vectors[id][i]; });
    return out.map(function (x) { return x / ids.length; });
  }

  /* ---------- PCA (Jacobi eigen decomposition of the covariance matrix) ---------- */
  function jacobiEigen(A) {
    var n = A.length, a = A.map(function (r) { return r.slice(); });
    var v = []; for (var i = 0; i < n; i++) { v.push(new Array(n).fill(0)); v[i][i] = 1; }
    for (var sweep = 0; sweep < 100; sweep++) {
      var off = 0;
      for (i = 0; i < n; i++) for (var j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
      if (off < 1e-22) break;
      for (var p = 0; p < n - 1; p++) {
        for (var q = p + 1; q < n; q++) {
          if (Math.abs(a[p][q]) < 1e-16) continue;
          var theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
          var t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          var c = 1 / Math.sqrt(t * t + 1), s = t * c, k;
          for (k = 0; k < n; k++) { var akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
          for (k = 0; k < n; k++) { var apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
          for (k = 0; k < n; k++) { var vkp = v[k][p], vkq = v[k][q]; v[k][p] = c * vkp - s * vkq; v[k][q] = s * vkp + c * vkq; }
        }
      }
    }
    var values = a.map(function (r, i) { return r[i]; });
    var vectors = values.map(function (_, col) { return v.map(function (row) { return row[col]; }); });
    return { values: values, vectors: vectors };
  }

  function fitPCA(rows, k) {
    k = k || 3;
    var n = rows.length, d = rows[0].length, mean = new Array(d).fill(0), i, j, m;
    rows.forEach(function (r) { for (j = 0; j < d; j++) mean[j] += r[j] / n; });
    var cov = []; for (i = 0; i < d; i++) cov.push(new Array(d).fill(0));
    rows.forEach(function (r) {
      for (i = 0; i < d; i++) for (j = i; j < d; j++) cov[i][j] += (r[i] - mean[i]) * (r[j] - mean[j]) / Math.max(1, n - 1);
    });
    for (i = 0; i < d; i++) for (j = 0; j < i; j++) cov[i][j] = cov[j][i];
    var eig = jacobiEigen(cov);
    var idx = eig.values.map(function (_, x) { return x; }).sort(function (x, y) { return eig.values[y] - eig.values[x]; });
    var totalVar = eig.values.reduce(function (s, x) { return s + Math.max(0, x); }, 0) || 1;
    var comps = [], ratio = [];
    for (m = 0; m < Math.min(k, d); m++) {
      var vec = eig.vectors[idx[m]].slice(), big = 0;
      for (j = 0; j < d; j++) if (Math.abs(vec[j]) > Math.abs(vec[big])) big = j;
      if (vec[big] < 0) vec = vec.map(function (x) { return -x; });
      comps.push(vec); ratio.push(Math.max(0, eig.values[idx[m]]) / totalVar);
    }
    return {
      mean: mean, components: comps, ratio: ratio, k: k,
      project: function (v) {
        var out = [];
        for (var c = 0; c < k; c++) {
          if (c >= comps.length) { out.push(0); continue; }
          var s = 0; for (var q = 0; q < v.length; q++) s += (v[q] - mean[q]) * comps[c][q];
          out.push(s);
        }
        return out;
      }
    };
  }

  var api = {
    TARGET: TARGET, mulberry32: mulberry32, splitText: splitText, buildVocab: buildVocab,
    decompose: decompose, tokenize: tokenize, buildCorpus: buildCorpus, createModel: createModel,
    cosine: cosine, meanPool: meanPool, fitPCA: fitPCA, jacobiEigen: jacobiEigen, dot: dot, norm: norm
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.MiniLM = api;
})(typeof window !== 'undefined' ? window : this);