"""RAG pipeline for the Enrollee-Bot.

Mirrors the final configuration from the report: 4-step preprocessing, recursive
800/10% chunking, multilingual-e5-base bi-encoder over a FAISS index, optional
bge-reranker-v2-m3 cross-encoder, and a Qwen2.5-1.5B-Instruct generator with a
context-only system prompt and a few-shot example.

Build the index once with `python build_index.py`, then the bot loads it from disk.
"""
import os
import re
import json
import pickle
import unicodedata

import numpy as np
import faiss
import torch

import config

# ----------------------------------------------------------------------------
# preprocessing (identical to the experiments)
# ----------------------------------------------------------------------------
def clean_ws(t: str) -> str:
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    t = re.sub(r"(?<=[а-яёa-z,])\n(?=[а-яёa-z])", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def strip_headers(t: str) -> str:
    pats = [r"^\s*НИУ ВШЭ\s*$", r"^\s*стр\.?\s*\d+.*$", r"^\s*\d+\s*/\s*\d+\s*$",
            r"^\s*©.*$", r"^\s*Факультет компьютерных наук\s*$",
            r"^\s*Мы используем файлы cookies.*$"]
    for p in pats:
        t = re.sub(p, "", t, flags=re.MULTILINE | re.IGNORECASE)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def norm_unicode(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    for a, b in {"«": '"', "»": '"', "“": '"', "”": '"', "—": "-", "–": "-",
                 "\u2212": "-", "\xa0": " ", "\u2009": " "}.items():
        t = t.replace(a, b)
    return t


def preprocess(t: str) -> str:
    return norm_unicode(strip_headers(clean_ws(t)))


def chunk_text(text: str, size: int = 800, overlap: float = 0.1):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=int(size * overlap),
        separators=["\n\n", "\n", ". ", " ", ""])
    return [c for c in splitter.split_text(text) if c.strip()]


# ----------------------------------------------------------------------------
# prompts
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Ты — ассистент приёмной комиссии Факультета компьютерных наук НИУ ВШЭ. "
    "Отвечай только на основе приведённого контекста, не используй внешние знания и "
    "ничего не выдумывай. Если в контексте нет ответа — честно скажи об этом и "
    f"предложи обратиться на {config.CONTACT}. Не давай прогнозов по проходным баллам, "
    "не давай официальных обещаний и не веди разговоры не по теме поступления. "
    "Отвечай вежливо, кратко и на русском языке."
)

FEWSHOT = (
    "\n\nПример 1:\nКонтекст: Приём документов идёт с 20 июня по 25 июля 2026 года.\n"
    "Вопрос: Когда подавать документы?\n"
    "Ответ: Приём документов идёт с 20 июня по 25 июля 2026 года.\n\n"
    "Пример 2:\nКонтекст: [нет данных]\nВопрос: Какая завтра погода?\n"
    "Ответ: Я отвечаю только на вопросы о поступлении на ФКН НИУ ВШЭ."
)

DISCLAIMER = "\n\n_Ответ носит справочный характер и не является официальной офертой._"


# ----------------------------------------------------------------------------
# pipeline
# ----------------------------------------------------------------------------
class RAGPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_index()
        self._load_models()

    # --- index ---
    def _load_index(self):
        idx_file = os.path.join(config.INDEX_DIR, "faiss.index")
        meta_file = os.path.join(config.INDEX_DIR, "chunks.pkl")
        if not (os.path.exists(idx_file) and os.path.exists(meta_file)):
            raise FileNotFoundError(
                "Index not found. Build it first with: python build_index.py")
        self.index = faiss.read_index(idx_file)
        with open(meta_file, "rb") as f:
            meta = pickle.load(f)
        self.chunks = meta["chunks"]
        self.sources = meta["sources"]

    # --- models ---
    def _load_models(self):
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(config.EMBEDDER, device=self.device)

        self.reranker = None
        if config.USE_RERANKER:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(config.RERANKER, device=self.device)

        self.tok = self.llm = None
        if config.USE_LLM:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(config.GENERATOR)
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.llm = AutoModelForCausalLM.from_pretrained(
                config.GENERATOR, torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None)
            if self.device == "cpu":
                self.llm = self.llm.to("cpu")

    # --- retrieval ---
    def retrieve(self, query: str):
        qv = self.embedder.encode(["query: " + query], normalize_embeddings=True).astype("float32")
        scores, idxs = self.index.search(qv, config.TOP_N)
        idxs, scores = idxs[0].tolist(), scores[0].tolist()
        top_cos = scores[0] if scores else 0.0
        cand = [(i, self.chunks[i], self.sources[i]) for i in idxs if i >= 0]

        if self.reranker is not None and cand:
            pairs = [[query, c[1]] for c in cand]
            order = np.argsort(-self.reranker.predict(pairs))
            cand = [cand[j] for j in order]

        return cand[:config.TOP_K], top_cos

    # --- generation ---
    def generate(self, query: str, contexts):
        ctx = "\n".join(f"- {c[1]}" for c in contexts)
        system = SYSTEM_PROMPT + FEWSHOT
        user = f"Контекст:\n{ctx}\n\nВопрос: {query}"
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ins = self.tok(prompt, return_tensors="pt").to(self.llm.device)
        with torch.no_grad():
            out = self.llm.generate(**ins, max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False)
        return self.tok.decode(out[0][ins.input_ids.shape[1]:], skip_special_tokens=True).strip()

    # --- full flow ---
    def answer(self, query: str) -> str:
        query = (query or "").strip()
        if not query:
            return "Задайте вопрос о поступлении на ФКН НИУ ВШЭ."

        contexts, top_cos = self.retrieve(query)
        if not contexts or top_cos < config.REL_THRESHOLD:
            return ("К сожалению, я не нашёл ответа в базе знаний. "
                    f"Рекомендую обратиться в приёмную комиссию ФКН: {config.CONTACT}")

        source = contexts[0][2]
        if not config.USE_LLM:                       # extractive mode
            body = contexts[0][1]
        else:
            try:
                body = self.generate(query, contexts)
            except Exception:
                body = contexts[0][1]                # graceful fallback to the best chunk

        text = body + DISCLAIMER
        if source:
            text += f"\n\nИсточник: {source}"
        return text
