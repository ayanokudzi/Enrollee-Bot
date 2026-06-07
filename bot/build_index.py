"""Build the knowledge-base index once, then the bot loads it from disk.

Scrapes the public HSE admission pages (keeping the source URL per chunk so the bot
can cite it). If scraping yields too little text, falls back to a corpus built from the
gold answers and prints a warning. Run:  python build_index.py
"""
import os
import re
import json
import pickle

import numpy as np
import faiss
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

import config
from rag import preprocess, chunk_text

URLS = [
    "https://cs.hse.ru/abitur/",
    "https://cs.hse.ru/abitur/ba",
    "https://cs.hse.ru/abitur/ma",
    "https://cs.hse.ru/spravochnik",
    "https://ba.hse.ru/",
    "https://www.hse.ru/ba/ami/",
    "https://www.hse.ru/ba/se/",
    "https://www.hse.ru/ba/data/",
    "https://www.hse.ru/ba/sec/",
]
HEADERS = {"User-Agent": "Mozilla/5.0 (admission FAQ bot; corpus builder)"}


def fetch(url: str) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        main = soup.find("div", class_=re.compile("content|main|post", re.I)) or soup.body or soup
        return main.get_text(separator="\n")
    except Exception as e:
        print("  skip", url, "->", type(e).__name__)
        return ""


def build_corpus():
    """Return (full_text, list_of_(text, source_url) pages)."""
    pages = []
    for u in URLS:
        t = fetch(u)
        if len(t.strip()) > 400:
            pages.append((u, t))
            print(f"  ok  {u}  ({len(t)} chars)")
    total = sum(len(t) for _, t in pages)
    if total < 8000:
        print("\n[!] Too little scraped text. Falling back to a corpus from the gold answers.")
        qa = json.loads(open(config.QA_PATH, encoding="utf-8").read())
        pages = [("https://cs.hse.ru/abitur/", "\n\n".join(f"{x['context']}. {x['answer']}" for x in qa))]
    return pages


def main():
    os.makedirs(config.INDEX_DIR, exist_ok=True)
    print("Building corpus...")
    pages = build_corpus()

    # keep the source URL for every chunk
    full_text_parts, chunks, sources = [], [], []
    for url, raw in pages:
        clean = preprocess(raw)
        full_text_parts.append(clean)
        for ch in chunk_text(clean, 800, 0.1):
            chunks.append(ch)
            sources.append(url)
    print(f"corpus: {len(pages)} pages, {len(chunks)} chunks")

    with open(config.CORPUS_PATH, "w", encoding="utf-8") as f:
        f.write("\n\n".join(full_text_parts))

    print(f"Embedding with {config.EMBEDDER} ...")
    emb = SentenceTransformer(config.EMBEDDER)
    vecs = emb.encode(["passage: " + c for c in chunks],
                      normalize_embeddings=True, show_progress_bar=True).astype("float32")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    faiss.write_index(index, os.path.join(config.INDEX_DIR, "faiss.index"))
    with open(os.path.join(config.INDEX_DIR, "chunks.pkl"), "wb") as f:
        pickle.dump({"chunks": chunks, "sources": sources}, f)
    print(f"Saved index to {config.INDEX_DIR}/ ({len(chunks)} chunks).")


if __name__ == "__main__":
    main()
