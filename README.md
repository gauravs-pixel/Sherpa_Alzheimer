

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



