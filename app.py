import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['MPLBACKEND'] = 'Agg'

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array
from keras.applications.efficientnet import preprocess_input
import cv2
from joblib import load
import pandas as pd
import shap
import base64
from io import BytesIO
import matplotlib.pyplot as plt

# --- LIME ---
try:
    from lime import lime_image
    LIME_AVAILABLE = True
    print("LIME loaded")
except ImportError:
    LIME_AVAILABLE = False
    print("LIME not installed")

app = Flask(__name__)

# ------------------- Load Models -------------------
print("Loading models...")
image_model = tf.keras.models.load_model("model/autism_efficientnet_fused_images.keras")
text_model = load("model/autism_text_new_model (1).pkl")
print("Models loaded.")

explainer = shap.TreeExplainer(text_model)

# --------------------------------------------------------------
# 1. LIME: BLUE-ORANGE HEATMAP (MEDICAL GRADE)
# --------------------------------------------------------------
if LIME_AVAILABLE:
    lime_explainer = lime_image.LimeImageExplainer(verbose=False)
else:
    lime_explainer = None

def generate_lime_heatmap(img_array, model):
    if not LIME_AVAILABLE:
        return img_array

    try:
        print("Generating BLUE-ORANGE LIME...")
        explanation = lime_explainer.explain_instance(
            img_array.astype('double'),
            model.predict,
            top_labels=1,
            hide_color=0,
            num_samples=200
        )

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=10,
            hide_rest=False
        )

        mask_float = mask.astype(float)
        mask_smooth = cv2.GaussianBlur(mask_float, (21, 21), 0)
        mask_norm = mask_smooth / (mask_smooth.max() + 1e-8)
        mask_resized = cv2.resize(mask_norm, (img_array.shape[1], img_array.shape[0]))

        heatmap = cv2.applyColorMap((mask_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = img_array.astype(np.float32)
        alpha = 0.6
        blended = cv2.addWeighted(heatmap.astype(np.float32), alpha, overlay, 1 - alpha, 0)
        return np.clip(blended, 0, 255).astype(np.uint8)

    except Exception as e:
        print("LIME FAILED:", e)
        return img_array

def analyze_image(file_stream):
    try:
        print("Analyzing image with LIME...")
        img = load_img(file_stream, target_size=(224, 224))
        x = img_to_array(img)
        x_pre = preprocess_input(x.copy())
        x_batch = np.expand_dims(x_pre, axis=0)

        preds = image_model.predict(x_batch, verbose=0)[0]
        pred_class = int(np.argmax(preds))
        conf = float(preds[pred_class])
        label = "Autistic" if pred_class == 0 else "Non-Autistic"
        print(f"IMAGE: {conf:.1%} → {label}")

        lime_overlay = generate_lime_heatmap(x, image_model)

        _, orig_buf = cv2.imencode('.png', cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_RGB2BGR))
        _, lime_buf = cv2.imencode('.png', cv2.cvtColor(lime_overlay, cv2.COLOR_RGB2BGR))

        orig_b64 = f"data:image/png;base64,{base64.b64encode(orig_buf).decode()}"
        lime_b64 = f"data:image/png;base64,{base64.b64encode(lime_buf).decode()}"

        return {
            'label': label,
            'confidence': f"{conf:.1%}",
            'conf_raw': conf,
            'original_b64': orig_b64,
            'heatmap_b64': lime_b64
        }
    except Exception as e:
        print(f"IMAGE ERROR: {e}")
        import traceback; traceback.print_exc()
        return None

# --------------------------------------------------------------
# 2. Risk + SHAP + FULL TABLE
# --------------------------------------------------------------
def get_risk(prob):
    if prob >= 0.7:   return "High Risk",   "#d9534f"
    if prob >= 0.4:   return "Medium Risk", "#f0ad4e"
    return "Low Risk", "#28a745"

def generate_shap_and_importance(input_df: pd.DataFrame):
    try:
        model_features = list(text_model.feature_names_in_)
        input_df = input_df.reindex(columns=model_features, fill_value=0)

        prob = text_model.predict_proba(input_df)[0][1]
        shap_values = explainer.shap_values(input_df)
        shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values[:, :, 1]
        base = float(np.asarray(explainer.expected_value[1]).flatten()[0]) if hasattr(explainer.expected_value, '__len__') else float(explainer.expected_value)

        plt.figure(figsize=(14, 4))
        shap.force_plot(base_value=base, shap_values=shap_vals[0], features=input_df.iloc[0],
                        feature_names=model_features, show=False, matplotlib=True)
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        shap_b64 = base64.b64encode(buf.read()).decode()
        plt.close('all')

        feature_data = []
        for feat in model_features:
            val = input_df.iloc[0][feat]
            if feat == "Qchat_10_Score":
                val_str = f"{int(val)}/10"
            elif feat == "Age_Years":
                val_str = str(int(val))
            elif feat.startswith("A") or feat in [
                "Sex", "Speech Delay/Language Disorder", "Genetic_Disorders", "Depression",
                "Global developmental delay/intellectual disability", "Social/Behavioural Issues",
                "Anxiety_disorder", "Jaundice", "Family_mem_with_ASD"
            ]:
                val_str = "Yes" if val == 1 else "No"
            else:
                val_str = str(int(val))

            impact = shap_vals[0][model_features.index(feat)]
            impact_str = f"{impact:+.3f}"
            feature_data.append({
                'name': feat.replace('_', ' ').title(),
                'value': val_str,
                'impact': impact,
                'impact_str': impact_str,
                'color': '#d9534f' if impact > 0 else '#28a745'
            })

        feature_data = sorted(feature_data, key=lambda x: abs(x['impact']), reverse=True)

        return {
            'shap_plot': f"data:image/png;base64,{shap_b64}",
            'features': feature_data,
            'prob_raw': prob
        }

    except Exception as e:
        print("SHAP/TABLE FAILED:", e)
        return {
            'shap_plot': "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iODAwIiBoZWlnaHQ9IjMwMCI+PHJlY3Qgd2lkdGg9IjEwMCUiIGhlaWdodD0iMTAwJSIgZmlsbD0iI2ZmNWE1YSIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LXNpemU9IjI0IiBmaWxsPSIjODgwMDAwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+U0hBUCBGQUlMRUQgLSBDSGVDSyBDT05TT0xFPC90ZXh0Pjwvc3ZnPg==",
            'features': [],
            'prob_raw': 0.5
        }

# --------------------------------------------------------------
# 3. Route — NO FUSION
# --------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    image_result = None
    text_result = None

    if request.method == "POST":
        # === IMAGE ===
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            file_stream = BytesIO(file.read())
            image_result = analyze_image(file_stream)

        # === TEXT ===
        if all(f"a{i}" in request.form for i in range(1, 11)):
            features = {}
            print("\n=== Q-CHAT DEBUG ===")
            for i in range(1, 11):
                form_val = request.form.get(f"a{i}")
                val = 1 if form_val == "1" else 0
                if i != 10: val = 1 - val
                key = f"A{i}" if i < 10 else "A10_Autism_Spectrum_Quotient"
                features[key] = val
                print(f"  {key}: {val}  (form: {form_val})")

            features["Age_Years"] = int(request.form.get("age", 5))
            features["Qchat_10_Score"] = sum(features.get(f"A{i}", 0) for i in range(1, 10)) + features.get("A10_Autism_Spectrum_Quotient", 0)
            print(f"  Qchat_10_Score: {features['Qchat_10_Score']}")
            print("=== END DEBUG ===\n")

            feature_map = {
                "speech_delay": "Speech Delay/Language Disorder",
                "genetic_disorders": "Genetic_Disorders",
                "depression": "Depression",
                "global_delay": "Global developmental delay/intellectual disability",
                "social_issues": "Social/Behavioural Issues",
                "anxiety": "Anxiety_disorder",
                "jaundice": "Jaundice",
                "family_asd": "Family_mem_with_ASD"
            }
            yes_no = {"Yes": 1, "No": 0}
            for k, v in feature_map.items():
                features[v] = yes_no.get(request.form.get(k), 0)

            features["Sex"] = 1 if request.form.get("sex") == "M" else 0
            features["Social_Responsiveness_Scale"] = 0

            input_df = pd.DataFrame([features]).reindex(columns=text_model.feature_names_in_, fill_value=0)
            result = generate_shap_and_importance(input_df)
            risk_label, risk_color = get_risk(result['prob_raw'])

            text_result = {
                'risk': risk_label,
                'probability': f"{result['prob_raw']:.1%}",
                'prob_raw': result['prob_raw'],
                'risk_color': risk_color,
                'shap_plot': result['shap_plot'],
                'features': result['features'],
                'qchat_score': features["Qchat_10_Score"]
            }

    return render_template("index.html", image_result=image_result, text_result=text_result)

if __name__ == "__main__":
    print("App running — NO FUSION + CLEAN + PATIENT-SAFE")
    app.run(host='0.0.0.0', port=5000, debug=False)


