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


# -----------------------------
# Configuration
# -----------------------------
DB_DIR = "alzheimers_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

TOP_K = 5
MAX_NEW_TOKENS = 220


# -----------------------------
# Language helpers
# -----------------------------
LANG_NAME = {
    "en": "English",
    "es": "Español",
    "ca": "Català",
    "fr": "Français",
    "de": "Deutsch",
    "it": "Italiano",
    "pt": "Português",
}

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "en"

def build_style(lang_code: str) -> str:
    lang = LANG_NAME.get(lang_code, lang_code)
    return f"Reply in {lang}. Be calm, empathetic, and concise."


# -----------------------------
# Load vector store
# -----------------------------
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.load_local(
        DB_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )


# -----------------------------
# Load LLM
# -----------------------------
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.2,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )

    return HuggingFacePipeline(pipeline=gen_pipe)


# -----------------------------
# RAG logic
# -----------------------------
def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def format_docs(docs):
    context = []
    sources = []

    for i, d in enumerate(docs[:TOP_K], start=1):
        text = clean_text(d.page_content)[:1000]
        src = d.metadata.get("source", "unknown")

        context.append(f"[{i}] {text}")
        sources.append(f"- {src}")

    return "\n\n".join(context), "\n".join(sources)


PROMPT = PromptTemplate.from_template(
    ''' 
You are Sherpa, an Alzheimer caregiving assistant.

{style}

Rules:
- Use ONLY the context.
- If the answer is not present, say:
  "I don't know based on the provided sources."
- Cite evidence like [1], [2].

Question:
{question}

Context:
{context}

Answer:
'''.strip()
)


# -----------------------------
# App runtime
# -----------------------------
VECTORSTORE = load_vectorstore()
LLM = load_llm()


def answer_question(question: str, language_mode: str):
    lang = detect_language(question) if language_mode == "Auto" else language_mode
    style = build_style(lang)

    # ✅ Use public similarity_search to retrieve docs
    docs = VECTORSTORE.similarity_search(question, k=TOP_K)
    context, sources = format_docs(docs)

    if not context:
        return "I don't know based on the provided sources.\n\nSources:\n- (none)"

    prompt = PROMPT.format(
        style=style,
        question=question,
        context=context,
    )

    answer = LLM.invoke(prompt)
    return f"{answer}\n\n---\n**Sources**\n{sources}"


# -----------------------------
# Gradio UI
# -----------------------------
def chat_fn(message, history, language):
    return answer_question(message, language)


with gr.Blocks(title="Sherpa Alzheimer") as demo:
    gr.Markdown("# Sherpa Alzheimer 🧭\nCaregiver-focused RAG assistant")

    language = gr.Dropdown(
        ["Auto", "en", "es", "ca", "fr", "de", "it", "pt"],
        value="Auto",
        label="Language",
    )

    gr.ChatInterface(
        fn=lambda m, h: chat_fn(m, h, language.value),
        examples=[
            "What are early biomarkers of Alzheimer's disease?",
            "How do I respond when my parent insists they need to go home?",
            "Què és l'Alzheimer?"
        ],
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(share=True)
