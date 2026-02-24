# WhatsApp Integration Step 1
# ===========================
# This script implements a RAG (Retrieval-Augmented Generation) pipeline for Alzheimer-related information.
# It:
#   1. Loads URLs from an Excel file
#   2. Builds or loads a FAISS vector database from web pages
#   3. Loads a HuggingFace language model for LLM inference
#   4. Provides translation support for multiple languages
#   5. Implements a WhatsApp-ready question-answering function

%%writefile rag.py

import pandas as pd

# -------------------------------
# Step 1: Load URLs from Excel file
# -------------------------------
file_path = "sample_data/Dataset-sherpa-alzheimer-links.xlsx"
df = pd.read_excel(file_path)  # Read the Excel file into a pandas DataFrame

# Extract unique URLs from the "Enllaç" column, dropping any NaN values
urls = df["Enllaç"].dropna().astype(str).unique().tolist()

print(f"Loaded {len(urls)} URLs")
urls[:5]  # Show the first 5 URLs for inspection


# -------------------------------
# Step 2: Build Vector Database (FAISS)
# -------------------------------
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import os
import hashlib

# Constants
DB_PATH = "alzheimers_db"  # Folder to store FAISS index
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)  # Initialize embeddings

# Check if FAISS index exists
if not os.path.exists(DB_PATH):
    print("🔧 Building FAISS index (first time only)")

    # ---- 1) Load web documents from URLs ----
    all_docs = []
    for url in urls:
        try:
            loader = WebBaseLoader(url, header_template={"User-Agent": "Mozilla/5.0"})
            docs = loader.load()  # Load the page as documents
            all_docs.extend(docs)
            print(f"✅ Loaded: {url} ({len(docs)} docs)")
        except Exception as e:
            print(f"❌ Failed: {url} | Error: {e}")

    print(f"\nTotal loaded docs: {len(all_docs)}")

    # ---- 2) Split documents into smaller chunks for embedding ----
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,   # Maximum characters per chunk
        chunk_overlap=100 # Overlap to preserve context
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Split into {len(chunks)} chunks before deduplication")

    # ---- 3) Deduplicate chunks ----
    # Avoid storing identical chunks multiple times
    seen = set()
    deduped_chunks = []
    for d in chunks:
        h = hashlib.md5(d.page_content.strip().encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            deduped_chunks.append(d)

    chunks = deduped_chunks
    print(f"Chunks after deduplication: {len(chunks)}")

    # ---- 4) Create FAISS vectorstore ----
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("✅ FAISS index created")

    # Save FAISS index locally
    vectorstore.save_local(DB_PATH)
    print(f"✅ Saved FAISS DB to ./{DB_PATH}")

else:
    print("📦 Using existing FAISS index")


# =========================
# Load FAISS Vector Database
# =========================
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.load_local(
    "alzheimers_db",
    embeddings,
    allow_dangerous_deserialization=True  # Required for some LangChain versions
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 chunks

print("✅ Vector DB loaded")


# =========================
# Step 3: Load HuggingFace LLM
# =========================
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model with FP16 for efficiency
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)

# Create HuggingFace text-generation pipeline
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=180,  # Max length for generated response
    temperature=0.2,     # Low temperature for deterministic output
    do_sample=False,     # Disable sampling for consistent answers
    truncation=True,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    return_full_text=False
)

# Wrap pipeline for LangChain usage
llm = HuggingFacePipeline(pipeline=hf_pipeline)


# =========================
# Step 4: RAG + Query Setup
# =========================
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Function to format retrieved documents (takes the first chunk, most relevant)
def format_docs(docs):
    # Limit chunk to first 800 characters
    return docs[0].page_content[:800]

# Strict prompt template for medical assistant
prompt = PromptTemplate.from_template("""
You are a calm, empathetic medically informed assistant.
You speak to people who may be worried about memory loss or Alzheimer’s disease.

Rules:
- Use ONLY the provided context from the sources.
- Answer in ENGLISH.
- Explain things clearly and calmly, but stay concise.
- Do NOT add new symptoms, explanations, or advice that are not explicitly stated in the context.
- Do NOT repeat the same idea using different wording.
- Summarize only what is directly mentioned in the sources.
- If the sources do not clearly list the information, say: "I don’t have enough information in the provided sources to answer this question."

Question:
{input}

Context:
{context}

### FINAL ANSWER (do not repeat the rules):
""")


# -------------------------------
# Step 5: Language detection & normalization
# -------------------------------
import unicodedata
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # Ensure reproducible language detection

def normalize_text(text):
    """Normalize unicode text to standard form (NFKC)."""
    return unicodedata.normalize("NFKC", text)

def detect_language_safe(text):
    """Detect the language of a text with short-text handling."""
    text = normalize_text(text)

    # Expand short messages to stabilize detection
    if len(text) < 30:
        text = (text + " ") * 4

    try:
        return detect(text)
    except:
        return "en"  # Fallback to English


# -------------------------------
# Step 6: Utilities for RAG
# -------------------------------
def extract_sources(docs):
    """Extract unique source URLs from retrieved documents."""
    sources = []
    for doc in docs:
        src = doc.metadata.get("source")
        if src and src not in sources:
            sources.append(src)
    return sources


# -------------------------------
# Step 7: RAG English QA function
# -------------------------------
def ask_english(question_en: str):
    """
    Perform grounded QA in English using the FAISS retriever and LLM.
    Steps:
        1. Retrieve documents
        2. Format context
        3. Build prompt
        4. Generate answer
        5. Retry if empty
        6. Extract sources
        7. Append sources to answer
    """
    docs = retriever.invoke(question_en)
    if not docs:
        return "I don’t have enough information in the provided sources to answer this question."

    context = format_docs(docs)
    prompt_text = prompt.format(input=question_en, context=context)
    answer = llm.invoke(prompt_text).strip()

    if not answer:
        relaxed_prompt_text = prompt_text.replace(
            "Answer in the SAME LANGUAGE as the question.",
            ""
        )
        answer = llm.invoke(relaxed_prompt_text).strip()

    if not answer.strip():
        return "I don’t have enough information in the provided sources to answer this question."

    sources = extract_sources(docs)
    if not sources:
        return "I don’t have enough information in the provided sources to answer this question."

    if "Rules:" in answer:
        answer = answer.split("Rules:")[0].strip()

    answer += "\n\nSources:\n" + "\n".join(sources)
    return answer


# -------------------------------
# Step 8: Translation utilities
# -------------------------------
from transformers import MarianMTModel, MarianTokenizer

_translation_cache = {}  # Cache for loaded Marian models

def translate_text(text, src_lang, tgt_lang):
    """Translate text using MarianMT models with caching."""
    if src_lang == tgt_lang:
        return text

    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"

    if model_name not in _translation_cache:
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            _translation_cache[model_name] = (tokenizer, model)
        except Exception as e:
            print(f"⚠️ Translation model not available: {model_name}")
            return text

    tokenizer, model = _translation_cache[model_name]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_length=512,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def is_lang_preloaded(lang: str) -> bool:
    """Check if translation model for a language is already loaded."""
    if lang == "en":
        return True
    return (
        f"Helsinki-NLP/opus-mt-{lang}-en" in _translation_cache
        or f"Helsinki-NLP/opus-mt-en-{lang}" in _translation_cache
    )


# -------------------------------
# Step 9: Preload EU translation models
# -------------------------------
EU_MAIN_LANGS = ["en", "fr", "de", "es", "it", "pt", "nl", "ca"]

def preload_translation_models():
    """
    Preload Marian translation models for main EU languages.
    This avoids Twilio WhatsApp webhook timeouts during first request.
    """
    pairs = set()
    for lang in EU_MAIN_LANGS:
        if lang != "en":
            pairs.add((lang, "en"))
            pairs.add(("en", lang))

    print("🚀 Preloading translation models...", flush=True)
    for src, tgt in sorted(pairs):
        try:
            print(f"🔄 Loading {src} → {tgt}", flush=True)
            translate_text("Warm up.", src, tgt)
            print(f"✅ Ready {src} → {tgt}", flush=True)
        except Exception as e:
            print(f"⚠️ Failed {src} → {tgt}: {e}", flush=True)
    print("🎉 Translation preload complete", flush=True)

preload_translation_models()


# -------------------------------
# Step 10: WhatsApp RAG interface
# -------------------------------
def ask(question: str):
    """
    Main interface for WhatsApp questions.
    Steps:
        1. Normalize text
        2. Detect language
        3. Ensure language is preloaded
        4. Translate to English
        5. Ask English RAG model
        6. Translate back to original language
        7. Append sources
    """
    question = normalize_text(question)
    src_lang = detect_language_safe(question)

    if not is_lang_preloaded(src_lang):
        return (
            "I’m setting up support for your language. "
            "Please send your question again in a few seconds."
        )

    question_en = translate_text(question, src_lang, "en")
    answer_en = ask_english(question_en)

    if src_lang == "en":
        return answer_en

    if "\n\nSources:\n" in answer_en:
        answer_text, sources = answer_en.split("\n\nSources:\n", 1)
        sources = "\n\nSources:\n" + sources
    else:
        answer_text = answer_en
        sources = ""

    answer_translated = translate_text(answer_text, "en", src_lang)
    reply = answer_translated + sources

    if not reply.strip():
        reply = "I don’t have enough information in the provided sources to answer this question."

    return reply
