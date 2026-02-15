# ============================================================
# Sherpa Alzheimer – Vector Store Builder
# ============================================================
# This script:
# 1. Installs required dependencies
# 2. Loads Alzheimer-related URLs from an Excel file
# 3. Scrapes and cleans web content
# 4. Splits content into manageable text chunks
# 5. Creates vector embeddings
# 6. Stores them in a FAISS vector database
#
# This database is later used by the RAG assistant
# to retrieve trusted, source-backed information.
# ============================================================


# -----------------------------
# Step 1: Install Dependencies
# -----------------------------
# NOTE:
# These commands are typically used in:
# - Google Colab
# - Jupyter notebooks
# - Hugging Face Spaces (build step)
#
# They ensure consistent versions across environments.

!pip -q install --upgrade pip
!pip -q install \
  huggingface-hub==0.35.0 \
  transformers==4.45.2 \
  sentence-transformers==3.1.1 \
  langchain \
  langchain-community \
  faiss-cpu \
  torch \
  gradio \
  pandas \
  langdetect


# -----------------------------
# Step 2: Load Input Dataset
# -----------------------------
# The Excel file contains curated links to trusted
# Alzheimer and caregiving-related web pages.
#
# Business value:
# - Ensures content comes from vetted sources
# - Prevents hallucinations by grounding answers in real data

import pandas as pd

file_path = "Dataset-sherpa-alzheimer-links.xlsx"

# Read the Excel file into a DataFrame
df = pd.read_excel(file_path)

# Extract unique, non-empty URLs from the "Enllaç" column
urls = (
    df["Enllaç"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

print(f"Loaded {len(urls)} URLs")


# -----------------------------
# Step 3: Load Web Documents
# -----------------------------
# We fetch the HTML content of each URL and convert it
# into LangChain Document objects.
#
# Each document keeps metadata about its source URL,
# which allows us to later cite sources in answers.

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs = []

for url in urls:
    try:
        # Use a browser-like User-Agent to avoid blocking
        loader = WebBaseLoader(
            url,
            header_template={"User-Agent": "Mozilla/5.0"}
        )

        loaded_docs = loader.load()

        # Attach the source URL to each document's metadata
        for d in loaded_docs:
            d.metadata["source"] = url

        docs.extend(loaded_docs)
        print(f"Loaded: {url}")

    except Exception as e:
        # Failure on one URL should not stop the pipeline
        print(f"Failed: {url} | {e}")


# -----------------------------
# Step 4: Split Documents into Chunks
# -----------------------------
# Why chunking is necessary:
# - LLMs have context length limits
# - Smaller chunks improve semantic search accuracy
#
# Strategy:
# - 2000 characters per chunk
# - 200-character overlap to preserve context continuity

splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

print("Total chunks created:", len(chunks))


# -----------------------------
# Step 5: Deduplicate Chunks
# -----------------------------
# Web pages often contain repeated content (headers, footers).
# This step removes exact duplicates to:
# - Reduce vector database size
# - Improve retrieval quality
# - Save memory and compute

seen = set()
unique_chunks = []

for chunk in chunks:
    # Use (source URL + content) as a uniqueness key
    key = (
        chunk.metadata.get("source"),
        chunk.page_content.strip()
    )

    if key not in seen:
        seen.add(key)
        unique_chunks.append(chunk)

print("Unique chunks after deduplication:", len(unique_chunks))


# -----------------------------
# Step 6: Create Embeddings
# -----------------------------
# We convert text chunks into numerical vectors using
# a sentence-transformer model.
#
# Model choice:
# - all-MiniLM-L6-v2
# - Fast, lightweight, and strong semantic performance
#
# Business value:
# - Enables meaning-based search (not keyword matching)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# Step 7: Build FAISS Vector Store
# -----------------------------
# FAISS is a high-performance vector search library.
#
# It allows the assistant to:
# - Retrieve the most relevant knowledge chunks
# - Scale efficiently to thousands of documents
# - Run locally or in cloud environments

vectorstore = FAISS.from_documents(
    unique_chunks,
    embeddings
)


# -----------------------------
# Step 8: Save Vector Store to Disk
# -----------------------------
# This creates the `alzheimers_db/` directory containing:
# - index.faiss  → vector index
# - index.pkl    → document metadata
#
# This folder is loaded by app.py at runtime.

vectorstore.save_local("alzheimers_db")

print("Vector store successfully saved to 'alzheimers_db/'")


# ============================================================
# END OF SCRIPT
# ============================================================
# After running this file:
# - The RAG assistant can retrieve trusted knowledge
# - Answers will include citations
# - The system becomes explainable and reliable
#
# Next step:
# - Run app.py or server.py to launch the assistant
# ============================================================
