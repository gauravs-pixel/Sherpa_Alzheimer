"""
whatsapp_app.py

This file exposes a Flask webhook endpoint that connects:
WhatsApp (via Twilio) → RAG pipeline → WhatsApp response

Twilio sends incoming WhatsApp messages to this server.
The server forwards the message to the RAG logic (ask()),
then sends the generated answer back to the user on WhatsApp.
"""

from flask import Flask, request, Response
from twilio.twiml.messaging_response import MessagingResponse

# Import the RAG query function
# ask(question: str) -> str
# This function is expected to:
#   - retrieve relevant documents from the vector store
#   - generate an answer using the LLM
from rag import ask


# ---------------------------------
# Flask App Initialization
# ---------------------------------
app = Flask(__name__)


# ---------------------------------
# WhatsApp Webhook Endpoint
# ---------------------------------
@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    """
    This endpoint is called by Twilio whenever
    a WhatsApp message is received.

    Twilio sends data as application/x-www-form-urlencoded
    (not JSON), so we read from request.form.
    """

    # Log the full incoming payload for debugging purposes
    # Useful during development to inspect Twilio parameters
    form = request.form.to_dict(flat=True)
    print("INCOMING FORM:", form, flush=True)

    # Extract the user's message text
    incoming_msg = (request.form.get("Body") or "").strip()
    print("INCOMING MSG:", incoming_msg, flush=True)

    # ---------------------------------
    # Message Handling Logic
    # ---------------------------------
    if not incoming_msg:
        # Empty message safeguard
        reply = "Please send a question."

    elif incoming_msg.lower() == "ping":
        # Simple health check to confirm the webhook is alive
        reply = "pong"

    else:
        try:
            # Pass user message to the RAG pipeline
            reply = ask(incoming_msg)

            # Safety fallback if RAG returns nothing
            if not reply:
                reply = "I don't know."

        except Exception as e:
            # Never allow the webhook to crash
            # Twilio expects a valid response every time
            print("ERROR in ask():", repr(e), flush=True)
            reply = "Sorry — something went wrong on the server."

    # ---------------------------------
    # Build Twilio Response (TwiML)
    # ---------------------------------
    resp = MessagingResponse()
    resp.message(reply)

    # IMPORTANT:
    # Twilio requires an XML response with content-type text/xml
    return Response(
        str(resp),
        status=200,
        mimetype="text/xml"
    )


# ---------------------------------
# Local / Container Startup
# ---------------------------------
if __name__ == "__main__":
    """
    Runs the Flask app.

    host="0.0.0.0" allows external services (Twilio, Docker, HF Spaces)
    to reach the server.

    port=8081 must match the webhook URL configuration in Twilio.

    use_reloader=False prevents Flask from starting twice
    (important in notebooks and containers).

    threaded=True allows multiple concurrent WhatsApp requests.
    """
    app.run(
        host="0.0.0.0",
        port=8081,
        debug=False,
        use_reloader=False,
        threaded=True
    )
