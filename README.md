The Sherpa Alzheimer application is deployed live on Hugging Face Spaces, allowing anyone to try the assistant directly in a web browser—no installation required.

**🧠 Building the Alzheimer Knowledge Base (Vector Store)**

**📄 Purpose of the Vector Store Script**

The vector store script is responsible for:

- Collecting Alzheimer and caregiving information from trusted web sources
- Cleaning and splitting long documents into manageable text chunks
- Converting text into semantic embeddings
- Storing embeddings in a FAISS vector database
- Preserving source metadata for answer citation

📥 **Input Data**

**Dataset-sherpa-alzheimer-links.xlsx**

- curated URLs related to Alzheimer’s disease and caregiving
- Each link is treated as a trusted knowledge source
- The column “Enllaç” is used to extract web URLs

This enables the assistant to provide accurate, explainable, and source-backed answers instead of generating unsupported responses.

**📁 Output**

After running the script, the following directory is created:

alzheimers_db/
├── index.faiss   # Vector index for semantic search
└── index.pkl     # Document metadata (sources, text mapping)

This project uses Retrieval-Augmented Generation (RAG) to ensure that all responses are grounded in trusted, real-world caregiving sources.
Before the assistant can answer questions, a vector database must be created from curated Alzheimer-related content.

This process is handled by the vector store builder script.

**💬 WhatsApp Integration (User Access)**

The Sherpa Alzheimer Assistant is also accessible via WhatsApp using the Twilio WhatsApp Sandbox.



<img width="2160" height="974" alt="image" src="https://github.com/user-attachments/assets/c3da1d37-cfc6-4461-bf01-97125753d375" />

**How to Join the WhatsApp Assistant**

1. Open WhatsApp on your mobile device
2. Scan the QR code shown above
OR
3. Send a WhatsApp message to the provided sandbox number
4. Send the join code exactly as displayed (for example:join want-troops)

Once joined, you can start chatting with the assistant directly on WhatsApp by sending your questions as normal messages.

**What This Enables**

- Caregivers and family members can access the assistant without installing any app
- Supports quick, conversational questions on mobile
- Ideal for non-technical users and older caregivers
- Same knowledge base and safety logic as the web application

Note: This setup uses the Twilio Sandbox, intended for development and testing.For production use, a dedicated WhatsApp Business number and approval are required.


🌍 **Live Application URL**

👉 https://huggingface.co/spaces/idofgaurav/Alzheimer

**What you can do in the live app**

- Ask caregiver-related questions about Alzheimer’s disease
- Use multiple languages (English, Spanish, Catalan, French, German, Italian, Portuguese)
- See source-backed answers generated using Retrieval-Augmented Generation (RAG)
- Test real caregiver scenarios such as:
  - “How do I respond when my parent wants to go home?”
  - “Què és l’Alzheimer?”
  - “What are early signs of Alzheimer’s disease?”

⚠️ Note: The app is for educational and caregiver support purposes only and is not a medical diagnosis tool.

**📁 Project Structure & File Guide**

Below is an explanation of each important file and directory in this repository and what role it plays in the system.

.
├── app.py
├── server.py
├── whatsapp_server.py
├── alzheimers_db/
│   ├── index.faiss
│   └── index.pkl
├── requirements.txt
└── README.md

**📄 app.py – Core Application Logic**

This is the heart of the project.

**What it does:**

- Loads the FAISS vector database
- Loads the TinyLlama language model
- Detects the user’s language automatically
- Retrieves relevant documents using RAG
- Generates answers with citations
- Powers the Gradio web interface

**Who should look here:**

- AI/ML engineers
- Developers modifying prompts, models, or retrieval logic

**📁 alzheimers_db/ – Knowledge Base (FAISS Vector Store)**

This directory contains the indexed Alzheimer’s caregiving knowledge.

**Files inside:**
- index.faiss → Vector index for similarity search
- index.pkl → Metadata (sources, document info)

**Important notes:**
- This is what enables source-based answers
- To update knowledge, you must rebuild the FAISS index
- The app will not work correctly without this folder

**📄 requirements.txt**

Lists all required Python dependencies such as:

- torch
- transformers
- gradio
- langchain
- faiss-cpu
- langdetect
- twilio (for WhatsApp)

**Used by:**
- Hugging Face Spaces
- Local installation
- Cloud deployments

**🧭 How the Live Hugging Face App Works (Business View)**

1. A caregiver enters a question in the web interface
2. The system detects the language automatically
3. Relevant knowledge is retrieved from the Alzheimer’s database
4. The AI generates a clear, empathetic answer
5. Sources are cited for transparency and trust

**Business value:**

- Reduces caregiver stress
- Improves access to reliable information
- Scales caregiver support without human staffing
- Works across languages and regions

**🌐 Public Access & Sharing**

- The Hugging Face app is publicly accessible
- No login required for basic usage
- The same backend can also be:
- Embedded in websites
- Connected to WhatsApp
- Extended to mobile apps or APIs

**🛡️ Ethical & Safety Considerations**

- The assistant does not provide medical diagnoses
- Responses are grounded in curated caregiving sources
- Users are encouraged to consult healthcare professionals
- Designed for support, education, and guidance only

🧭 Sherpa Alzheimer – Caregiver Handbook

## 1. Introduction
Sherpa Alzheimer is an AI-powered caregiver assistant designed specifically to help caregivers of Alzheimer’s patients. Its primary purpose is to provide quick, reliable, and empathetic guidance for caregiving decisions by combining retrieval-augmented generation (RAG) with a curated Alzheimer’s knowledge base.

Business and Functional Benefits:
- Reduces cognitive load by providing fast, evidence-backed answers.
- Delivers responses in a calm, empathetic, and concise style, improving caregiver confidence.
- Supports multiple languages: English, Spanish, Catalan, French, German, Italian, Portuguese.
- Allows integration into internal caregiver portals or public web interfaces.
- Ensures all answers include source citations for verification.

## 2. System Requirements
- Python 3.10 or higher.
- Optional GPU recommended for faster LLM inference (TinyLlama 1.1B).
- Required Python packages:

pip install torch transformers gradio langdetect faiss-cpu langchain langchain-community

Optional for updated Hugging Face modules:

pip install -U langchain-huggingface

## 3. Project Structure
Sherpa-Alzheimer/

app.py                 # Main application
alzheimers_db/         # FAISS vector store (knowledge base)
README.md              # Documentation
requirements.txt       # Optional Python dependencies

## 4. Installation & Setup
1. Create a project folder:
mkdir sherpa-alzheimer
cd sherpa-alzheimer

2. Prepare the Alzheimer’s knowledge base:
- Place your documents in the alzheimers_db/ folder as a FAISS vector store.
- If not yet created, generate embeddings using Sentence Transformers (all-MiniLM-L6-v2).

3. Install dependencies (see system requirements above).

## 5. Launching the App Locally
1. Run the app:
python app.py

2. Terminal output will look like:
Running on local URL: http://127.0.0.1:7860

3. Open this URL in a browser.
4. Example questions to ask:
- "What are early biomarkers of Alzheimer's disease?"
- "How do I respond when my parent insists they need to go home?"
- "Què és l'Alzheimer?"

## 6. Public Access
1. To allow others to access the app, edit app.py to launch Gradio with a public link:
demo.launch(share=True)

2. After running, the terminal will display a temporary public URL:
Running on public URL: https://abcd1234.gradio.app

3. Anyone with this link can access the app while your Python process is running.

## 7. Using the Chat Interface
- Enter your question in the chat box.
- Select the language (or leave "Auto" to detect automatically).
- Press Enter to submit.
- The assistant will respond with:
  - Answer based strictly on the retrieved documents.
  - Sources cited in a list format for verification.

Example Output:
Answer: Early signs include memory loss, confusion, and disorientation.

-----------------------------------------------------------------------------------
Sources:
- source1.txt
- source2.pdf

## 8. Safety & Best Practices
- Always verify critical medical advice with a healthcare professional.
- Use clear, short questions for the best response quality.
- Regularly update the FAISS knowledge base to maintain accuracy.
- Choose the appropriate language for clarity; Auto-detect works well for multi-lingual users.

## 9. Optional Debugging
- For advanced monitoring, add debug prints inside the answer_question function:

print(f"[DEBUG] Question: {question}")
print(f"[DEBUG] Retrieved docs: {len(docs)}")
print(f"[DEBUG] Prompt:\n{prompt}")
print(f"[DEBUG] Answer: {answer}")

- Check the terminal for live logs to verify what the assistant is doing for each question.

## 10. Future Enhancements
- Persistent chat memory for multi-turn conversations.
- Mobile-friendly UI for tablets and smartphones.
- Access control for internal caregiver portals.
- Analytics dashboard for tracking common questions and usage.
- Deployment on cloud platforms such as Hugging Face Spaces, AWS, or Render for continuous availability.



