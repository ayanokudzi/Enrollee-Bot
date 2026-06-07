# Enrollee-Bot — RAG chatbot for the FCS HSE admission campaign

A retrieval-augmented (RAG) question-answering chatbot that answers prospective
applicants' questions about admission to the Faculty of Computer Science, HSE University,
grounding every answer in the official admission documents and returning a source link.

**Author:** Alexey D. Kovyazin, group 245
**Supervisor:** Daria A. Andreeva
**Institution:** Faculty of Computer Science, HSE University, 2026

## What the project does

Each year the FCS admission campaign draws thousands of applicants, and the committee
cannot answer every repeated question by hand. A plain LLM would hallucinate deadlines
and scores, and a fixed rule-based bot cannot cover the variety of questions. This
project uses RAG: applicant questions are matched against a knowledge base built from the
official public HSE pages, and the retrieved fragments are passed to a language model
that answers only from them, citing the source. This combines flexibility with factual
accuracy, and the knowledge base can be updated when the rules change without retraining.

This repository contains the **experiments** behind the system: how the corpus is
preprocessed and chunked, which embedder and reranker are chosen, and how generation is
evaluated under a no-budget (free-models-only) constraint.

## Repository contents

| File | Description |
|------|-------------|
| `bot.py` | Telegram bot entry point (python-telegram-bot). |
| `rag.py` | RAG pipeline: preprocessing, chunking, retrieval, reranking, generation, fallback. |
| `build_index.py` | One-time index builder: scrapes the corpus, chunks, embeds, saves the FAISS index. |
| `config.py` | Model choices and toggles (token read from the `TELEGRAM_BOT_TOKEN` env variable). |
| `experiments.ipynb` | Full experiment pipeline: preprocessing, chunking, embedder selection, reranking, generation + evaluation. Runs in Google Colab on a free T4 GPU. |
| `data/fcs_hse_qa_dataset.json` | Gold benchmark: 50 question/reference-answer pairs across 12 categories, verified against the official documents. |
| `data/text.txt` | Knowledge corpus. Rebuilt automatically by the notebook from the public HSE pages (the original was lost). |
| `results.json` | Reference results produced by the notebook; every table in the report is populated from this file. |
| `fill_report.py` | Prints `results.json` as LaTeX table rows. |
| `requirements.txt` | Dependencies. |

The full written report (with the title page, sent to SmartLMS) is the separate
`report.pdf`.

## How to reproduce

1. Open `experiments.ipynb` in Google Colab.
2. `Runtime -> Change runtime type -> GPU (T4)`.
3. Upload `data/fcs_hse_qa_dataset.json` (keep the `data/` folder).
4. `Runtime -> Run all`.

A full run takes about **45-60 minutes** on a free Colab T4. This is normal: the time is
dominated by downloading the models (five embedders, four rerankers, three generators,
roughly 12-15 GB from Hugging Face on a fresh runtime), not by computation. Free Colab
can disconnect on idle, so keep the tab active. The committed `results.json` is the
reference output and can be compared against directly without waiting for a full rerun.

## Run the bot

The bot implements the final configuration: `multilingual-e5-base` + FAISS, the
`bge-reranker-v2-m3` cross-encoder, and `Qwen2.5-1.5B-Instruct` with a context-only
system prompt and a fallback to the admissions email.

```bash
pip install -r requirements.txt
export TELEGRAM_BOT_TOKEN="your-token-from-BotFather"   # Windows: set TELEGRAM_BOT_TOKEN=...
python build_index.py     # once: builds the knowledge base into data/index/
python bot.py             # starts polling; talk to the bot in Telegram
```

The LLM and reranker are heavy: on a CPU-only machine answers take tens of seconds. On a
machine with a GPU they are fast. For weak hardware, set `USE_LLM = False` (extractive
mode: replies with the best retrieved fragment) or `USE_RERANKER = False` in `config.py`.

## Metric

A retrieved chunk counts as a *hit* for a question if it shares **at least two lemmas**
(via `pymorphy3`) with the reference answer - lemma overlap rather than exact or substring
match, because Russian is heavily inflected. **Hit Rate@5** is the share of questions with
at least one hit in the top-5; **MRR@10** is the mean reciprocal rank of the first hit.

## Results in brief

Retrieval quality was improved step by step. The main findings:

- **Preprocessing** - whitespace cleaning gives the clearest gain (Hit Rate@5 0.80 -> 0.86); header removal and Unicode normalisation keep quality stable while cleaning the corpus.
- **Chunking** - a quality/cost trade-off; recursive 800-character chunks balance Hit Rate (0.84), MRR@10 (0.813) and chunk length, and avoid cutting mid-sentence.
- **Embedder** - `multilingual-e5-base` is a strong, well-documented choice (0.90 / 0.802), with `paraphrase-MiniLM-L12-v2` an equally valid alternative (0.94 / 0.826).
- **Reranker** - only the multilingual `bge-reranker-v2-m3` helps, raising Hit Rate@5 to **0.94** and MRR@10 to **0.85** at 0.82 s per query, inside the 5-second budget.
- **Generation** - a few-shot prompt raises out-of-domain refusal accuracy from 0.2 to 0.6 with no loss of in-domain quality; the 1.5B model with few-shot prompting is chosen.

**Final configuration:** four-step preprocessing -> recursive 800/10% chunking ->
`multilingual-e5-base` -> `bge-reranker-v2-m3` (top-20 -> top-5) over a FAISS index ->
`Qwen2.5-1.5B-Instruct` with a few-shot prompt -> Telegram Bot API.

## Note on the generation experiment

Generation uses free open-weight instruct models (Qwen2.5) and free, reproducible proxy
metrics (refusal accuracy, embedding-cosine correctness, an embedding-cosine faithfulness
proxy) instead of the RAGAS LLM-judge metrics, which require a paid judge model. This
substitution is stated explicitly in the report, and the proxies are treated as weaker
than RAGAS.

## AI usage

AI assistants were used for translation, drafting and restructuring the report, and for
writing the experiment code; candidate references were located with AI help and then
verified by the author against the original sources. The project definition, the
construction and verification of the gold dataset, the execution of the experiments, the
interpretation of the numbers, and the conclusions were carried out by the author. The
report contains a full AI-usage disclosure section.
