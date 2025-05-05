from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re
import json
import time
import gc
import os
import subprocess

app = Flask(__name__)
API_KEY = os.getenv('API_KEY')
MODEL_TYPE = "model_save"
device = torch.device(os.getenv('DEVICE', 'cpu'))

tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_TYPE).to(device)
    
@app.before_request
def before_request():
    api_key = request.headers.get('Authorization')
    if api_key != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized", "message": "Invalid API Key"}), 403
    load_model()


@app.route("/status", methods=["GET"])
def health_check():
    return jsonify({"success": True}), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_text = data.get("text", "")

    if len(input_text) > 256:
        return jsonify({"prediction": "尚不支援超過256字元之長文本"})

    input = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)["input_ids"]
    del input_text, data

    with torch.no_grad():
        output = model.generate(input, max_length=50, num_beams=5, early_stopping=True)[0]
    
    del inputs
    return jsonify({"prediction": re.sub(r'\s{0,1}', '', tokenizer.decode(output, skip_special_tokens=True))})

@app.after_request
def after_request(response):
    if response and response.get_json():
        data = response.get_json()

        data["time_request"] = int(time.time())
        data["version"] = "v1-LARGE"

        response.set_data(json.dumps(data))
        del data
    gc.collect()
    return response

if __name__ == "__main__":
    from waitress import serve  # Gunicorn 也可用
    serve(app, host="0.0.0.0", port=8084, threads=4)