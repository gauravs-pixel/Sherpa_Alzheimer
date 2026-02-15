from flask import Flask, request, Response
from twilio.twiml.messaging_response import MessagingResponse
from rag import ask


app = Flask(__name__)

@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    # Log incoming payload (helpful for debugging)
    form = request.form.to_dict(flat=True)
    print("INCOMING FORM:", form, flush=True)

    incoming_msg = (request.form.get("Body") or "").strip()
    print("INCOMING MSG:", incoming_msg, flush=True)

    # Decide reply
    if not incoming_msg:
        reply = "Please send a question."
    elif incoming_msg.lower() == "ping":
        reply = "pong"
    else:
        try:
            reply = ask(incoming_msg)
            if not reply:
                reply = "I don't know."
        except Exception as e:
            # Don't crash the webhook; return a safe fallback
            print("ERROR in ask():", repr(e), flush=True)
            reply = "Sorry — something went wrong on the server."

    # Build TwiML response
    resp = MessagingResponse()
    resp.message(reply)

    # IMPORTANT: return XML with the correct mimetype for Twilio
    return Response(str(resp), status=200, mimetype="text/xml")


if __name__ == "__main__":
    # use_reloader=False prevents double-start in notebooks
    app.run(host="0.0.0.0", port=8081, debug=False, use_reloader=False, threaded=True)
