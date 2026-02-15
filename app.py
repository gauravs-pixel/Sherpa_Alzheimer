
# app.py — Hugging Face Space entrypoint

import os
import re
import torch
import gradio as gr
from langdetect import detect, LangDetectException
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

# ---------------- Configuration ----------------
DB_DIR = "alzheimers_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

TOP_K = 5
MAX_CONTEXT_CHARS = 3500
MAX_NEW_TOKENS = 220

LANGS = {
    "en": "English",
    "es": "Spanish",
    "ca": "Catalan",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
}

# ---------------- Helpers ----------------
def detect_language_safe(text):
    try:
        return detect(text)
    except LangDetectException:
        return "en"

# ---------------- Load Vector DB ----------------
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectorstore = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

# ---------------- Load LLM ----------------
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=0.2,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
    return_full_text=False,
)
llm = HuggingFacePipeline(pipeline=pipe)

# ---------------- Prompt ----------------
PROMPT = PromptTemplate.from_template("""
You are Sherpa Alzheimer, a caregiver assistant.

You MUST reply ONLY in {language_name}.

If the answer is not in the context, say:
"I don't know based on the provided sources."

Question:
{question}

Context:
{context}

Answer:
""")

# ---------------- RAG ----------------
def ask(question):
    lang = detect_language_safe(question)
    if lang not in LANGS:
        lang = "en"

    docs = retriever.invoke(question)
    ctx, srcs = [], []
    for i, d in enumerate(docs, 1):
        text = re.sub(r"\s+", " ", d.page_content)[:1000]
        ctx.append(f"[Source {i}] {text}")
        srcs.append(d.metadata.get("source", "unknown"))

    prompt = PROMPT.format(
        question=question,
        context="\n\n".join(ctx),
        language_name=LANGS[lang],
    )
    answer = llm.invoke(prompt)
    return answer + "\n\nSources:\n" + "\n".join(srcs)

# ---------------- UI ----------------
def chat_fn(message, history):
    return ask(message)

with gr.Blocks(title="Sherpa Alzheimer") as demo:
    gr.Markdown("# 🧭 Sherpa Alzheimer")
    gr.ChatInterface(chat_fn)

if __name__ == "__main__":
    demo.queue()
    demo.launch()
