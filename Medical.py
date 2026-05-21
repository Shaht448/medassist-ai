import os
import base64
import pdfplumber
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv
from io import BytesIO

load_dotenv()
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

SYSTEM_PROMPTS = {
    "consult": """You are MedAssist AI, a knowledgeable medical information assistant.
Structure every answer: (1) Direct answer, (2) Key details, (3) When to seek care.
Use plain language. Never diagnose. Always recommend consulting a licensed doctor.""",

    "symptoms": """You are MedAssist AI in Symptom Analysis mode.
ALWAYS respond using EXACTLY this format:

## Symptom Summary
Restate the symptoms clearly.

## Differential Analysis
List 3-5 possible conditions from most to least likely:
- **[Condition]** — Why this fits (1-2 sentences)

## Red Flags
Any symptoms needing emergency care. If none: "No immediate red flags."

## Recommended Next Steps
Self-care / see GP / urgent care / emergency room — be specific.

## Disclaimer
One sentence: this is not a clinical diagnosis.""",

    "drugs": """You are MedAssist AI in Drug Information mode — a clinical pharmacology reference.
For every medication respond with this EXACT structure:

## Drug Overview
- **Generic / Brand names:**
- **Drug class:**
- **Mechanism:** (1-2 sentences)

## Primary Indications
Bullet list of approved uses.

## Common Side Effects
- Mild (common):
- Serious (seek care):

## Key Drug Interactions
Most important. Mark DANGEROUS if severe.

## Contraindications
Who should NOT take this and why.

## Clinical Notes
Dosing, monitoring, pregnancy warnings.

End with: "Consult your pharmacist or physician before starting or stopping any medication." """,

    "terms": """You are MedAssist AI in Medical Terminology mode.
Respond using this EXACT structure:

## Plain English Definition
Explain as if speaking to someone with no medical background.

## Clinical Context
Where/when this term is used in medicine.

## Etymology
Break down the word roots.

## Related Terms
2-3 connected terms.

## What This Means for a Patient
If a doctor says this, what should the patient understand and ask?""",

    "report": """You are MedAssist AI in Clinical Documentation mode.
1. Identify the document type (SOAP, H&P, Discharge Summary, etc.)
2. Generate the full document in standard format
3. Use [PLACEHOLDER] wherever patient data is missing
4. Spell out abbreviations on first use
5. Suggest ICD-10 codes at the end

Add at top: "AI-generated template — review all [PLACEHOLDER] fields before clinical use" """,

    "scan": """You are MedAssist AI in Medical File Analysis mode.
When given a medical image, scan, report, or document:

## File Summary
What type of file/image this appears to be.

## Key Findings
List the most important observations in plain English.

## Medical Significance
What these findings typically indicate clinically.

## What to Discuss With Your Doctor
Specific questions the patient should ask their healthcare provider.

## Disclaimer
This is an AI analysis for educational purposes only — not a clinical report."""
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    # WHY READ API KEY FROM REQUEST?
    # The client sends their own OpenAI key with every request.
    # It never gets stored on the server — used once and discarded.
    # This means you don't need to put any key in .env or Render.
    api_key = request.form.get("api_key", "").strip()

    if not api_key:
        return jsonify({"error": "No API key provided. Please enter your OpenAI key on the settings screen."}), 401

    if not api_key.startswith("sk-"):
        return jsonify({"error": "Invalid API key format. OpenAI keys start with sk-"}), 401

    mode          = request.form.get("mode", "consult")
    messages_json = request.form.get("messages", "[]")
    user_text     = request.form.get("user_text", "")
    file          = request.files.get("file")

    import json
    messages = json.loads(messages_json)

    if mode not in SYSTEM_PROMPTS:
        return jsonify({"error": f"Unknown mode: {mode}"}), 400

    try:
        # Use the CLIENT's API key — not a server key
        client = OpenAI(api_key=api_key)
        model  = "gpt-4o"

        # Build user message content
        if file and file.filename:
            ext = file.filename.rsplit('.', 1)[-1].lower()

            if ext in ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp']:
                file_bytes = file.read()
                b64_image  = base64.b64encode(file_bytes).decode('utf-8')
                mime_map   = {'jpg':'jpeg','jpeg':'jpeg','png':'png','gif':'gif','webp':'webp','bmp':'bmp'}
                mime_type  = f"image/{mime_map.get(ext, 'jpeg')}"
                prompt_text = user_text if user_text else "Please analyze this medical image and explain what you see."
                current_content = [
                    {"type": "image_url", "image_url": {
                        "url": f"data:{mime_type};base64,{b64_image}",
                        "detail": "high"
                    }},
                    {"type": "text", "text": prompt_text}
                ]

            elif ext == 'pdf':
                file_bytes = file.read()
                pdf_text   = ""
                with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                    for page in pdf.pages:
                        extracted = page.extract_text()
                        if extracted:
                            pdf_text += extracted + "\n"

                if not pdf_text.strip():
                    return jsonify({"error": "Could not extract text from this PDF. It may be scanned or image-based."}), 400

                prompt_text = user_text if user_text else "Please analyze this medical document and explain the key findings."
                combined    = f"{prompt_text}\n\n--- DOCUMENT CONTENT ---\n{pdf_text[:6000]}"
                current_content = combined
            else:
                return jsonify({"error": "Unsupported file type. Please upload JPG, PNG, or PDF."}), 400
        else:
            current_content = user_text

        openai_messages = [
            {"role": "system", "content": SYSTEM_PROMPTS[mode]},
            *messages,
            {"role": "user", "content": current_content}
        ]

        response = client.chat.completions.create(
            model=model,
            max_tokens=1800,
            temperature=0.3,
            messages=openai_messages,
        )

        return jsonify({"reply": response.choices[0].message.content})

    except Exception as e:
        err = str(e)
        if "401" in err or "invalid_api_key" in err:
            return jsonify({"error": "Invalid API key. Please check your key and try again."}), 401
        if "429" in err:
            return jsonify({"error": "Rate limit hit. Please wait a moment and try again."}), 429
        if "insufficient_quota" in err:
            return jsonify({"error": "Your OpenAI account has no credits. Add credits at platform.openai.com"}), 402
        return jsonify({"error": f"Error: {err}"}), 500


if __name__ == "__main__":
    port  = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    print(f"\n✅ MedAssist AI → http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)