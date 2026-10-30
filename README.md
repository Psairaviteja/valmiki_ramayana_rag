
# Valmiki Ramayana RAG

A local retrieval-augmented generation (RAG) starter application that extracts pages from
`valmikiramayan.net`, indexes passages in ChromaDB, retrieves relevant passages, and answers
questions through a Streamlit interface using a local Ollama model.

## What it does

- Starts from the Bala Kanda, Sarga 1 frame page.
- Follows same-host HTML links and frame/iframe sources, up to a configurable page limit.
- Extracts readable page text and saves chunk metadata to `data/raw/manifest.jsonl`.
- Embeds passages with a multilingual sentence-transformer model.
- Stores vectors and metadata in ChromaDB.
- Retrieves top-k passages for a question and asks Ollama to answer only from those passages.
- Displays retrieved passages and source URLs for verification.

## Important source/coverage notes

1. The crawler only visits links reachable from the configured start URL on the same host. It may
   not reach every Kanda or every Sarga, and the site's frame/navigation layout can affect coverage.
2. After ingestion, inspect `data/raw/manifest.jsonl` to check the actual source URLs and text.
3. This starter uses page-level Kanda/Sarga inference from URL patterns. Some source URLs may be
   classified as `Unknown`; improve `infer_metadata()` if the site's URL structure requires it.
4. This is a retrieval prototype, not a verified scholarly edition. Compare citations against the
   source page, especially when page HTML contains multiple frames, navigation, or commentary.
5. Use the source website in accordance with its terms and applicable copyright rules. The ancient
   Sanskrit epic is traditional/public-domain material in many jurisdictions, but modern translations,
   annotations, and site-specific content can have separate rights.

## Requirements

- Python 3.10–3.12 recommended. Newer Python versions may not have compatible wheels for every
  ML dependency yet.
- Ollama installed and running locally.
- Internet access for initial page extraction and model download.

## Windows setup (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Install a local Ollama model. For example:

```powershell
ollama pull llama3.2
ollama serve
```

Keep Ollama running in another terminal. If you use a different model, update `OLLAMA_MODEL` in `.env`.

## Build the index

```powershell
python ingest.py
```

To start with a smaller crawl while testing:

```powershell
python ingest.py --max-pages 30
```

You can specify a different starting page:

```powershell
python ingest.py --start-url "https://www.valmikiramayan.net/baala/sarga1/bala_1_frame.htm" --max-pages 100
```

The ingestion step downloads the multilingual embedding model on first run and may take a while.
It rebuilds the Chroma collection each time you run it.

## Run the chat UI

In a new terminal, from the project folder with the virtual environment activated:

```powershell
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Suggested smoke tests

After indexing, ask questions that should be answerable from the pages you verified, such as:

- What does Valmiki ask Narada at the beginning of Bala Kanda, Sarga 1?
- Which qualities are listed in the opening description of the ideal person?
- What does the indexed passage say about Rama's conduct?

Then ask a question unrelated to the retrieved text. The correct behavior is to state that the indexed passages do not provide enough evidence, rather than inventing a source.

## Architecture

```text
HTML pages -> extraction and cleaning -> verse-aware chunks
           -> multilingual embeddings -> ChromaDB
Question -> embedding -> top-k retrieval -> local Ollama LLM
         -> answer with source references + inspectable retrieved passages
```

## Production improvements to do next

- Build a reliable sitemap/URL manifest for all seven Kandas rather than relying only on link crawling.
- Parse verse numbers and separate Sanskrit, transliteration, translation, and commentary into explicit fields.
- Add hybrid retrieval (BM25 + vector search) and reranking.
- Add an answerability threshold and source-citation validation.
- Add evaluation questions with expected source passages and track recall@k, MRR, citation precision, and groundedness.
- Add tests, structured logs, retries, rate limiting, and versioned indexes.
