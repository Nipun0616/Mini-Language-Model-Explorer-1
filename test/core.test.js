// Run with:  node test/core.test.js
// Checks the vocabulary, tokenizer, training and PCA code in js/core.js.
const assert = require('node:assert/strict');
const M = require('../js/core.js');

const SENTENCES = [
  'The cat sat on the mat.',
  'The dog sat on the rug.',
  'The cat likes the dog.',
  'The sun is hot today.',
  'I like my cat and dog.',
  'The dog ran to the park.'
];

let failed = 0;
function test(name, fn) {
  try { fn(); console.log('PASS', name); }
  catch (e) { failed++; console.error('FAIL', name, '\n    ', e.message); }
}

const vocab = M.buildVocab(SENTENCES);
const cfg = { dim: 8, seed: 42, epochs: 200, lr: 0.05, lrEnd: 0.005, negatives: 5 };

test('vocabulary has exactly 72 unique tokens', () => {
  assert.equal(vocab.validation.total, 72);
  assert.ok(vocab.validation.unique);
  assert.ok(vocab.validation.ok);
});

test('1 special + 20 word-level + 51 pieces', () => {
  assert.equal(vocab.validation.special, 1);
  assert.equal(vocab.validation.wordLevel, 20);
  assert.equal(vocab.validation.pieces, 51);
});

test('[UNK] is ID 0 and "the" is ID 1', () => {
  assert.equal(vocab.map.get('[UNK]').id, 0);
  assert.equal(vocab.map.get('the').id, 1);
});

test('known words get one token each', () => {
  const toks = M.tokenize('The cat sat on the mat.', vocab);
  assert.equal(toks.length, 7);
  assert.ok(toks.every((t) => t.status === 'known'));
});

test('unknown word "game" is split into pieces', () => {
  const toks = M.tokenize('The game sat on the mat.', vocab);
  assert.deepEqual(toks.slice(1, 5).map((t) => t.text), ['##g', '##a', '##m', '##e']);
});

test('characters with no piece become [UNK]', () => {
  assert.deepEqual(M.tokenize('42', vocab).map((t) => t.id), [0, 0]);
});

test('a smaller corpus gives a smaller vocabulary and does not pad', () => {
  const small = M.buildVocab(['The cat sat.']);
  assert.ok(small.validation.total < 72);
  assert.equal(small.validation.ok, false);
});

test('training is reproducible for a fixed seed', () => {
  const corpus = M.buildCorpus(SENTENCES, vocab, 2);
  const a = M.createModel(vocab, corpus, cfg); a.runEpochs(cfg.epochs);
  const b = M.createModel(vocab, corpus, cfg); b.runEpochs(cfg.epochs);
  assert.deepEqual(a.vectors(), b.vectors());
});

test('training loss goes down', () => {
  const corpus = M.buildCorpus(SENTENCES, vocab, 2);
  const m = M.createModel(vocab, corpus, cfg); m.runEpochs(cfg.epochs);
  assert.ok(m.lossHistory[cfg.epochs - 1] < m.lossHistory[0] * 0.5);
  assert.ok(m.vectors().every((v) => v.every(Number.isFinite)));
});

test('every token that occurs in the corpus is trained', () => {
  const corpus = M.buildCorpus(SENTENCES, vocab, 2);
  assert.ok(corpus.trained[vocab.map.get('cat').id]);
  assert.ok(corpus.trained[vocab.map.get('##c').id]);
  assert.equal(corpus.trained[vocab.map.get('##z').id], false);
});

test('mean pooling averages token vectors', () => {
  assert.deepEqual(M.meanPool([0, 1], [[0, 2], [2, 4]]), [1, 3]);
});

test('cosine similarity of a vector with itself is 1', () => {
  assert.ok(Math.abs(M.cosine([1, 2, 3], [1, 2, 3]) - 1) < 1e-12);
});

test('Jacobi eigen decomposition satisfies A v = lambda v', () => {
  const A = [[4, 1, 0.5], [1, 3, 0.2], [0.5, 0.2, 2]];
  const e = M.jacobiEigen(A);
  e.values.forEach((lam, k) => {
    for (let i = 0; i < 3; i++) {
      let av = 0; for (let j = 0; j < 3; j++) av += A[i][j] * e.vectors[k][j];
      assert.ok(Math.abs(av - lam * e.vectors[k][i]) < 1e-9);
    }
  });
});

test('PCA always projects to 3 coordinates, padding with zeros when d < 3', () => {
  const pca = M.fitPCA([[1, 2], [2, 1], [3, 5], [0, 1]], 3);
  const p = pca.project([1, 1]);
  assert.equal(p.length, 3);
  assert.equal(p[2], 0);
});

if (failed) { console.error('\n' + failed + ' test(s) failed'); process.exit(1); }
console.log('\nAll tests passed');