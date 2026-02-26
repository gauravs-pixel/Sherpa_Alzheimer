🧠 **Sherpa Alzheimer**

**An AI-Powered Caregiver Assistant for Alzheimer’s Disease**

Sherpa Alzheimer is a multilingual, AI-powered assistant designed to support caregivers and family members of people living with Alzheimer’s disease.
It delivers **reliable, empathetic, and source-backed answers using Retrieval-Augmented Generation (RAG)** and a curated Alzheimer’s knowledge base.

The application is live and publicly accessible, with both web and WhatsApp interfaces—no installation required.

🌍 **Live Application**

👉 Web App (Hugging Face Spaces)
https://huggingface.co/spaces/idofgaurav/Alzheimer

**What You Can Do**

Ask caregiver-focused questions about Alzheimer’s disease

Get evidence-based answers with citations

Use multiple languages:

  English, Spanish, Catalan, French, German, Italian, Portuguese

Test real caregiver scenarios such as:

  “How do I respond when my parent wants to go home?”

  “Què és l’Alzheimer?”

  “What are early signs of Alzheimer’s disease?”

⚠️** Disclaimer:**
This application is for educational and caregiver support purposes only and does not provide medical diagnoses.

**Why Sherpa Alzheimer (Business View)**

**Key Value Proposition**

Reduces caregiver stress and cognitive load

Provides fast, trustworthy, and explainable answers

Scales caregiver support without human staffing

Works across languages and regions

Accessible to non-technical and older users

**Core Principles**

Source-grounded responses (no hallucinations)

Empathetic and caregiver-friendly tone

Ethical and safety-first design

**🧠 How It Works (High Level)**

A caregiver asks a question (web or WhatsApp)

The system automatically detects the language

Relevant documents are retrieved from the Alzheimer knowledge base

The AI generates a clear, empathetic response

Sources are cited for transparency and trust

**This architecture is powered by Retrieval-Augmented Generation (RAG).**

**📚 Building the Alzheimer Knowledge Base (Vector Store)**

**Purpose**

The knowledge base ensures that all answers are grounded in trusted caregiving sources, not generic language model output.

**What the Vector Store Script Does**

Collects Alzheimer and caregiving information from trusted web sources

Cleans and splits long documents into manageable chunks

Converts text into semantic embeddings

Stores embeddings in a FAISS vector database

Preserves source metadata for answer citation

**Input Data**

Dataset-sherpa-alzheimer-links.xlsx

  Curated URLs related to Alzheimer’s disease and caregiving

  Each link is treated as a trusted knowledge source

  The column “Enllaç” is used to extract URLs

**Output**

After running the script, the following directory is created:

alzheimers_db/
├── index.faiss   # Vector index for semantic search
└── index.pkl     # Document metadata and source mapping

⚠️ The application **will not function correctly without this folder.**

**💬 WhatsApp Access (Twilio Sandbox)**

Sherpa Alzheimer is also accessible via WhatsApp, using the Twilio WhatsApp Sandbox.

**How to Join**

  Open WhatsApp on your mobile device

  Scan the provided QR code OR

  Send a message to the sandbox number

  Send the join code exactly as shown (example: join want-troops)

  Start chatting with the assistant like a normal WhatsApp conversation

**What This Enables**

  No app installation required

  Ideal for mobile and older caregivers

  Quick, conversational access

  Same knowledge base and safety logic as the web app

**Note**:
This uses the Twilio Sandbox for development and testing.
Production deployment requires a verified WhatsApp Business number.

**📄 File Guide**
app.py – **Core Application Logic**

The heart of the system.

**Responsibilities**

  Loads the FAISS vector database

  Loads the TinyLlama language model

  Automatically detects user language

  Retrieves relevant documents (RAG)

  Generates answers with citations

  Powers the Gradio web interface

**Audience**

  AI / ML engineers

  Developers extending prompts, models, or retrieval logic

**alzheimers_db/ – Knowledge Base (FAISS Vector Store)**

Contains the indexed Alzheimer caregiving knowledge.

**Files**

  index.faiss → Vector similarity index

  index.pkl → Metadata (sources, documents)

**Important Notes**

  Enables source-backed answers

  Must be rebuilt to update knowledge

  Required for correct app behavior

**requirements.txt**

Lists all required Python dependencies, including:

  torch

  transformers

  gradio

  langchain

  faiss-cpu

  langdetect

  twilio (WhatsApp integration)

**Used by:**

  Hugging Face Spaces

  Local development

  Cloud deployments

**⚙️ Installation & Local Setup**
**System Requirements**

  Python 3.10+

  Optional GPU recommended (TinyLlama 1.1B)

**Install Dependencies**

pip install torch transformers gradio langdetect faiss-cpu langchain langchain-community

**🛡️ Ethical & Safety Considerations**

No medical diagnoses are provided

All responses are grounded in curated caregiving sources

Encourages consultation with healthcare professionals

Designed for education, guidance, and emotional support

**🚀 Future Enhancements**

Persistent multi-turn chat memory

Mobile-optimized UI

Access control for caregiver organizations

Analytics dashboard (common questions, usage trends)

Production WhatsApp deployment

Cloud deployment (AWS, Render, Hugging Face)
