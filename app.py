from flask import Flask, render_template, request, redirect, url_for, jsonify
import torch, os, io
from PIL import Image
import numpy as np

from config import Config
from models.model_loader import load_model
from utils.preprocessing import PathMNISTPreprocessor
from utils.xai_methods import XAIExplainer

app = Flask(__name__)
app.config.from_object(Config)

# Ensure folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OVERLAYS_FOLDER'], exist_ok=True)

# Load model
print("Loading model...")
device = torch.device(app.config['DEVICE'])
model = load_model(app.config['MODEL_PATH'], device=device)
print(f"Model loaded on {device}")

pre = PathMNISTPreprocessor(mean=app.config['IMAGENET_MEAN'], std=app.config['IMAGENET_STD'])
explainer = XAIExplainer(model, device=device)

@app.route('/')
def index():
    return render_template('index.html',
                           class_names=app.config['CLASS_SHORT'],
                           metrics=app.config['TEST_METRICS'])


@app.route('/predict', methods=['POST'])
def predict():
    print("POST /predict hit")
    if 'image' not in request.files:
        return "<div class='alert alert-danger'>No file uploaded</div>", 400
    f = request.files['image']
    print("Received file:", f.filename)
    if not f.filename:
        return "<div class='alert alert-danger'>No file selected</div>", 400

    img_bytes = f.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp.png')
    img.save(temp_path)

    x = pre.preprocess(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    # Build predictions for ALL 9 classes, ordered by probability desc for display in scroller
    ordered = sorted(list(enumerate(probs)), key=lambda t: t[1], reverse=True)
    predictions_sorted = [{
        'class_idx': idx,
        'class_name': app.config['CLASS_SHORT'][idx],
        'class_full': app.config['CLASS_NAMES'][idx],
        'probability': float(p)
    } for idx, p in ordered]

    # Also build an index-aligned list (0..8) for the dropdown percent labels
    idx_aligned = [{'class_idx': i,
                    'class_name': app.config['CLASS_SHORT'][i],
                    'class_full': app.config['CLASS_NAMES'][i],
                    'probability': float(probs[i])} for i in range(9)]

    default_target = ordered[0][0]
    default_method = 'gradcam'

    return render_template('result.html',
                           predictions=idx_aligned,   # dropdown expects index order for labels
                           top_prediction=predictions_sorted[0],  # highest probability prediction
                           overlay_base64=None,
                           heatmap_base64=None,
                           selected_class=default_target,
                           selected_method=default_method,
                           class_names=app.config['CLASS_SHORT'])

@app.route('/explain_page', methods=['POST'])
def explain_page():
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp.png')
    if not os.path.exists(temp_path):
        return render_template('result.html', predictions=[], overlay_base64=None, heatmap_base64=None)

    target_class = max(0, min(8, int(request.form.get('target_class', 0))))
    method = request.form.get('method', 'gradcam')

    img = Image.open(temp_path).convert('RGB')
    x = pre.preprocess(img).unsqueeze(0).to(device)

    original_np = pre.tensor_to_numpy(x)
    x.requires_grad = True

    if method == 'gradcam':
        heatmap = explainer.gradcam(x, target_class)
    else:
        heatmap = explainer.integrated_gradients(x, target_class)

    overlay = explainer.overlay(original_np, heatmap, alpha=0.5)
    overlay_b64 = explainer.to_base64(overlay)
    heatmap_b64 = explainer.to_base64(heatmap)

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    # Keep index-aligned predictions 0..8 for dropdown and scroller (we'll use same structure)
    predictions = [{'class_idx': i,
                    'class_name': app.config['CLASS_SHORT'][i],
                    'class_full': app.config['CLASS_NAMES'][i],
                    'probability': float(probs[i])} for i in range(9)]
    
    # Find the top prediction (highest probability)
    top_pred_idx = max(range(9), key=lambda i: probs[i])
    top_prediction = predictions[top_pred_idx]

    return render_template('result.html',
                           predictions=predictions,
                           top_prediction=top_prediction,
                           overlay_base64=overlay_b64,
                           heatmap_base64=heatmap_b64,
                           selected_class=target_class,
                           selected_method=method,
                           class_names=app.config['CLASS_SHORT'])


@app.route('/metrics')
def metrics():
    return jsonify(app.config['TEST_METRICS'])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
