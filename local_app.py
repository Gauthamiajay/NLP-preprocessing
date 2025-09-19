# local_app.py
from flask import Flask, request, jsonify, send_from_directory
from transformers import pipeline
import torch

app = Flask(__name__)

device = 0 if torch.cuda.is_available() else -1
generator = pipeline("text2text-generation", model="google/flan-t5-small", device=device)

@app.route("/")
def index():
    return send_from_directory(".", "index_local.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    instruction = data.get("instruction", "Correct grammar")
    text = data.get("text", "")
    prompt = f"Instruction: {instruction}\nInput: {text}"
    out = generator(prompt, max_length=128, do_sample=False)[0]['generated_text']
    return jsonify({"result": out.strip()})

if __name__ == "__main__":
    app.run(debug=True)
