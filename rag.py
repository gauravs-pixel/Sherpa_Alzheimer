"""
rag_pipeline.py

This file builds the complete Retrieval-Augmented Generation (RAG) pipeline
used by the Alzheimer’s assistant.

It covers:
1. Loading trusted Alzheimer-related URLs from Excel
2. Crawling and cleaning web content
3. Chunking and deduplication
4. Embedding and FAISS vector database creation
5. Loading a lightweight local LLM (TinyLlama)
6. Strict prompt design for medical-safe answers
7. Multilingual support (Catalan, Spanish, etc.)
8. Final ask() function used by the UI and WhatsApp webhook
"""

# =====================================================
# 1️⃣ Load URL dataset from Excel
# =====================================================
import pandas as pd

# Path to Excel file containing trusted Alzheimer-related URLs
file_path = "sample_data/Dataset-sherpa-alzheimer-links.xlsx"

# Load spreadsheet
df = pd.read_excel(file_path)

# Extract unique, non-empty URLs from the "Enllaç" column
urls = df["Enllaç"].dropna().astype(str).unique().tolist()

print(f"Loaded {len(urls)} URLs")
urls[:5]  # preview


# =====================================================
# 2️⃣ Imports for document loading and vector DB
# =====================================================
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# =====================================================
# 3️⃣ Crawl and load web documents
# =====================================================
all_docs = []

for url in urls:
    try:
        # WebBaseLoader fetches and parses webpage text
        loader = WebBaseLoader(
            url,
            header_template={"User-Agent": "Mozilla/5.0"}  # avoids blocking
        )
        docs = loader.load()
        all_docs.extend(docs)
        print(f"✅ Loaded: {url}")
    except Exception as e:
        print(f"❌ Failed: {url} | Error: {e}")


# =====================================================
# 4️⃣ Split documents into chunks
# =====================================================
# Smaller chunks improve retrieval accuracy and reduce LLM hallucination
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,     # tuned for small LLM context windows
    chunk_overlap=100   # preserves semantic continuity
)

chunks = splitter.split_documents(all_docs)
print(f"Split into {len(chunks)} chunks before deduplication")


# =====================================================
# 5️⃣ Deduplicate chunks
# =====================================================
# Prevents repeated content from biasing retrieval
seen_texts = set()
unique_chunks = []

for chunk in chunks:
    text = chunk.page_content.strip()
    if text and text not in seen_texts:
        seen_texts.add(text)
        unique_chunks.append(chunk)

print(f"{len(unique_chunks)} unique chunks after deduplication")


# =====================================================
# 6️⃣ Create sentence embeddings
# =====================================================
# Lightweight, fast, CPU-friendly embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =====================================================
# 7️⃣ Build FAISS vector store
# =====================================================
vectorstore = FAISS.from_documents(unique_chunks, embeddings)


# =====================================================
# 8️⃣ Persist vector DB locally
# =====================================================
vectorstore.save_local("alzheimers_db")
print("✅ Vector DB saved locally as 'alzheimers_db'")


# =====================================================
# 9️⃣ Load vector DB for querying
# =====================================================
vectorstore = FAISS.load_local(
    "alzheimers_db",
    embeddings,
    allow_dangerous_deserialization=True
)

# Retriever returns top-k most relevant chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("✅ Vector DB loaded")


# =====================================================
# 🔟 Load Local LLM (TinyLlama)
# =====================================================
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)

# Deterministic generation for medical safety
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=0.2,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    return_full_text=False
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)


# =====================================================
# 1️⃣1️⃣ Prompt & RAG wiring
# =====================================================
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

def format_docs(docs):
    """
    Uses only the MOST relevant chunk
    to minimize hallucination.
    """
    return docs[0].page_content[:800]

prompt = PromptTemplate.from_template("""
SYSTEM INSTRUCTIONS (not part of the answer):
You are a calm, empathetic medical information assistant.

Rules:
- Use ONLY the provided context
- Answer in the SAME LANGUAGE as the question
- Do NOT repeat the question
- Do NOT invent new medical facts
- Be concise and clear
- If information is missing, say so

Question:
{input}

Context:
{context}

FINAL ANSWER:
""")

rag_pipeline = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
)


# =====================================================
# 1️⃣2️⃣ Language detection & translation
# =====================================================
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"


# =====================================================
# 1️⃣3️⃣ Core RAG execution (English only)
# =====================================================
def ask_english(question_en: str):
    docs = retriever.invoke(question_en)

    if not docs:
        return "I don’t have enough information in the provided sources to answer this question."

    context = format_docs(docs)
    answer = llm.invoke(prompt.format(input=question_en, context=context)).strip()

    if not answer or len(answer) < 30:
        return "I don’t have enough information in the provided sources to answer this question."

    sources = list({d.metadata.get("source") for d in docs if d.metadata.get("source")})
    answer += "\n\nSources:\n" + "\n".join(sources)

    return answer


# =====================================================
# 1️⃣4️⃣ Public function used by UI & WhatsApp
# =====================================================
from transformers import MarianMTModel, MarianTokenizer

_translation_cache = {}

def translate_text(text, src_lang, tgt_lang):
    if src_lang == tgt_lang:
        return text

    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"

    if model_name not in _translation_cache:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        _translation_cache[model_name] = (tokenizer, model)

    tokenizer, model = _translation_cache[model_name]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=512)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def ask(question: str):
    """
    Main entry point for the application.
    Handles multilingual questions safely.
    """
    src_lang = detect_language(question)

    # Translate to English for retrieval
    question_en = translate_text(question, src_lang, "en") if src_lang != "en" else question

    answer_en = ask_english(question_en)

    if src_lang == "en":
        return answer_en

    # Preserve sources in original language
    if "\n\nSources:\n" in answer_en:
        answer_text, sources = answer_en.split("\n\nSources:\n", 1)
        translated = translate_text(answer_text, "en", src_lang)
        return translated + "\n\nSources:\n" + sources

    return translate_text(answer_en, "en", src_lang)
