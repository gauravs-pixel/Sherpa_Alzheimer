
import pandas as pd

file_path = "sample_data/Dataset-sherpa-alzheimer-links.xlsx"
df = pd.read_excel(file_path)

urls = df["Enllaç"].dropna().astype(str).unique().tolist()

print(f"Loaded {len(urls)} URLs")
urls[:5]

# 1️⃣ Imports
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# 3️⃣ Load all documents
all_docs = []
for url in urls:
    try:
        loader = WebBaseLoader(url, header_template={"User-Agent": "Mozilla/5.0"})
        docs = loader.load()
        all_docs.extend(docs)
        print(f"✅ Loaded: {url}")
    except Exception as e:
        print(f"❌ Failed: {url} | Error: {e}")

# 4️⃣ Split into chunks
# Performance-tuned chunking for faster RAG inference
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,     # ⬅️ smaller chunks = fewer tokens at generation time
    chunk_overlap=100   # ⬅️ enough overlap to preserve meaning
)

chunks = splitter.split_documents(all_docs)
print(f"Split into {len(chunks)} chunks before deduplication")

# 5️⃣ Deduplicate chunks
seen_texts = set()
unique_chunks = []
for chunk in chunks:
    text = chunk.page_content.strip()
    if text and text not in seen_texts:
        seen_texts.add(text)
        unique_chunks.append(chunk)
print(f"{len(unique_chunks)} unique chunks after deduplication")

# 6️⃣ Create local HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 7️⃣ Build FAISS vector store
vectorstore = FAISS.from_documents(unique_chunks, embeddings)

# 8️⃣ Save the vector store
vectorstore.save_local("alzheimers_db")
print("✅ Vector DB saved locally as 'alzheimers_db'")

# ===== Cell 1: Vector DB =====
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.load_local(
    "alzheimers_db",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("✅ Vector DB loaded")


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

hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=None,
    max_length=None,
    temperature=0.2,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    return_full_text=False
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)


# ===== Cell 3: RAG + Query =====
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# 🔑 ADD THIS FUNCTION HERE
def format_docs(docs):
    # First retrieved chunk is the most semantically relevant
    return docs[0].page_content[:800]

def extract_sources(docs):
    sources = []
    for doc in docs:
        src = doc.metadata.get("source")
        if src and src not in sources:
            sources.append(src)
    return sources


from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("""
SYSTEM INSTRUCTIONS (not part of the answer):
You are a calm, empathetic medical information assistant.
You speak to people who may be worried about memory loss or Alzheimer’s disease.
Your role description is NOT part of the answer.

Rules:
- Use ONLY the provided context from the sources.
- Answer in the SAME LANGUAGE as the question.
- Do NOT repeat or rephrase the question.
- Do NOT copy sentences from the input.
- Do NOT mention your role, rules, or instructions.
- You must provide a helpful answer based on the context.
- If the question expresses sadness or concern, begin with empathy.
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



rag_pipeline = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
)


from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

def detect_language(text):
    try:
        return detect(text)   # returns ISO-639-1 code
    except:
        return "en"
def ends_with_complete_sentence(text):
      return bool(re.search(r'[.!?]\s*$', text))



def extract_sources(docs):
    """Extract unique source URLs from retrieved documents."""
    sources = []
    for doc in docs:
        src = doc.metadata.get("source")
        if src and src not in sources:
            sources.append(src)
    return sources

def is_definition_question(question: str) -> bool:
    q = question.lower()
    return any(
        q.startswith(x) for x in [
            "what is",
            "què és",
            "que es",
            "què significa",
            "what does"
        ]
    )

def ask_english(question_en: str):
    # 1️⃣ Retrieve relevant documents (ONCE)
    docs = retriever.invoke(question_en)
    if not docs:
        return "I don’t have enough information in the provided sources to answer this question."

    # 2️⃣ Build context once
    context = format_docs(docs)

    # 3️⃣ Build strict prompt once
    prompt_text = prompt.format(
        input=question_en,
        context=context
    )

    # 4️⃣ First generation attempt
    answer = llm.invoke(prompt_text).strip()

    # 5️⃣ Retry ONLY if model is silent (instruction backoff)
    if not answer:
        relaxed_prompt_text = prompt_text.replace(
            "Answer in the SAME LANGUAGE as the question.",
            ""
        )
        answer = llm.invoke(relaxed_prompt_text).strip()

    # 6️⃣ Optional safety check (keep conservative)
    if len(answer) < 30:
        return "I don’t have enough information in the provided sources to answer this question."

    # 7️⃣ Extract sources (receipts)
    sources = extract_sources(docs)
    if not sources:
        return "I don’t have enough information in the provided sources to answer this question."

    # 8️⃣ Clean accidental rule leakage
    if "Rules:" in answer:
        answer = answer.split("Rules:")[0].strip()

    # 9️⃣ Append sources
    answer += "\n\nSources:\n" + "\n".join(sources)

    return answer

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

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

def ask(question: str):
    src_lang = detect_language(question)

    # Translate to English if needed
    if src_lang != "en":
        question_en = translate_text(question, src_lang, "en")
    else:
        question_en = question

    # 🔧 IMPORTANT: retrieval must use ENGLISH
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
    return answer_translated + sources

