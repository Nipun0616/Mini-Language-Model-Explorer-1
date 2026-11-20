# Mini Language Model Explorer

An interactive page that shows how text becomes numbers inside a language model:

1. **Tokenization.** A 72-token vocabulary is built from six training sentences. A greedy longest-match tokenizer turns any sentence into token IDs and splits unknown words into character pieces.
2. **Embeddings.** Every token gets a vector (2 to 32 numbers) trained with skip-gram and negative sampling, using a fixed random seed so results repeat.
3. **Visualization.** PCA projects the vectors to 3D (or 2D) so you can rotate, zoom, hover and click them. Extra charts show a bar chart of one vector, cosine-similarity heatmaps and sentence comparisons.
4. **Your own sentence.** Type a sentence and see its tokens, vectors, mean-pooled sentence embedding and nearest training sentences.

There is no build step and no server code. It is plain HTML, CSS and JavaScript.

## Project structure

```
mini-language-model-explorer/
├── index.html          Page markup
├── css/style.css       Styles (light and dark theme)
├── js/core.js          Vocabulary, tokenizer, training, PCA (no DOM code)
├── js/app.js           Tables, charts and button handlers
├── test/core.test.js   Tests for js/core.js
├── package.json        npm scripts (test, start)
├── .gitignore
└── README.md
```

## Run it on your computer

Open `index.html` in Chrome, Edge or Firefox. You need an internet connection, because the charts library (Plotly) and the fonts load from CDNs.

To serve it from a local web server instead, use either of these from the project folder:

```bash
python3 -m http.server 8000      # then open http://localhost:8000
# or
npx serve .
```

## Run the tests

You need [Node.js](https://nodejs.org) 18 or newer. There is nothing to install.

```bash
npm test
```

## Put it on GitHub and deploy it

1. Create an **empty** repository on GitHub, for example `mini-language-model-explorer` (no README, no .gitignore).
2. In the project folder, run:

```bash
git init
git add .
git commit -m "Add Mini Language Model Explorer"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/mini-language-model-explorer.git
git push -u origin main
```

3. On GitHub open **Settings → Pages**. Under **Build and deployment**, set **Source** to **Deploy from a branch**, choose branch **main** and folder **/ (root)**, then save.
4. After a minute the site is live at `https://YOUR-USERNAME.github.io/mini-language-model-explorer/`.

Netlify and Vercel also work. Import the GitHub repository, leave the build command empty, and set the publish directory to the project root.

## How the vocabulary reaches 72 tokens

| Part | Count | Rule |
|---|---|---|
| `[UNK]` | 1 | Special token, ID 0 |
| Word-level tokens | 20 | The 20 most frequent words and punctuation marks in the corpus (19 words plus the period) |
| Character pieces | 26 | The letters a to z, written `##a` to `##z` |
| Subword pieces | 25 | The most frequent two-letter pairs inside corpus words |
| **Total** | **72** | |

The `##` prefix keeps the piece `##i` different from the word `i`. Nothing is padded or duplicated to reach 72. If you edit the corpus and it yields fewer distinct words or letter pairs, the page shows the real size and says which part falls short.

The target sizes are in `TARGET` at the top of `js/core.js`.

## How the embeddings are trained

- Training pairs come from a window of two tokens either side of each token.
- Piece tokens (letters and letter pairs) are paired with the words they appear in, so they get trained vectors too.
- For each pair the model learns to score the real context token higher than five random tokens.
- Vectors start from a seeded random draw. Output vectors start at zero. The learning rate falls from 0.05 to 0.005.
- Tokens that never occur in the corpus keep their random starting vector and are marked "Unseen".
- PCA is fitted on the trained token vectors only, and sentence points are projected with the same axes.
- A sentence embedding is the mean of its token vectors, so word order does not change it.

## Ideas for extending it

- Add a pre-trained comparison (for example GloVe) to see how a tiny corpus differs from a large one.
- Add a "nearest words" calculator for analogies such as king − man + woman.
- Export the embedding table to CSV.
