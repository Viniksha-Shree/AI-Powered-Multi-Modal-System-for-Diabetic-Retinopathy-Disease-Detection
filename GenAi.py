import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from transformers import BartModel, BartTokenizer
from torchvision import models
import pandas as pd
import numpy as np
import time
from datetime import datetime
import base64
from g4f.client import Client

# Initialize g4f client
client = Client()

# -----------------------------------------------------------------------------
# Page & UI Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Advanced MultiModal Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI styling including Animate.css and Font Awesome
custom_css = """
<style>
@import url('https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

/* Global Styling */
body {
    background: #f8f9fa;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Fixed Header Styling */
header {
    position: center;
    top: 0;
    width: 100%;
    z-index: 100;
    background: linear-gradient(135deg, #667eea, #764ba2);
    padding: 15px 20px;
    color: white;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
header i {
    margin-right: 15px;
}
header h1 {
    display: inline-block;
    font-size: 2rem;
    margin: 0;
    vertical-align: middle;
}

/* Main Container pushed below header */
.main {
    margin-top: 80px;
    padding: 20px;
}

/* Sidebar Styling */
.css-1d391kg .css-1d391kg {
    background: linear-gradient(180deg, #764ba2, #667eea) !important;
    color: white !important;
}

/* Card Styling */
.card {
    background: white;
    border-radius: 12px;
    padding: 25px;
    margin: 20px 0;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

/* Custom Button Styling */
div.stButton > button {
    background-color: #667eea;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 10px 24px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: background-color 0.3s ease;
}
div.stButton > button:hover {
    background-color: #556cd6;
}

/* Tabs Styling */
.stTabs > div {
    margin-top: 20px;
}

/* DataFrame Styling */
.dataframe {
    background: white;
    border-radius: 8px;
    padding: 10px;
}

/* Download Link Styling */
.download-link {
    background: #667eea;
    color: white;
    padding: 8px 16px;
    border-radius: 6px;
    text-decoration: none;
    font-weight: 600;
}
.download-link:hover {
    background: #556cd6;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Fixed header with animated icon and title
st.markdown("""
    <header>
      <i class="fas fa-rocket animate__animated animate__pulse animate__infinite" style="font-size:48px;"></i>
      <h1>Advanced MultiModal Classifier</h1>
    </header>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Model & Resource Definitions
# -----------------------------------------------------------------------------
class MultiModalClassifier(nn.Module):
    def __init__(self, text_model, image_model, text_feat_dim, image_feat_dim, hidden_dim, num_classes):
        super(MultiModalClassifier, self).__init__()
        self.text_model = text_model
        self.image_model = image_model
        self.text_fc = nn.Linear(text_feat_dim, hidden_dim)
        self.image_fc = nn.Linear(image_feat_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, text_input=None, image_input=None):
        features = None
        if text_input is not None:
            text_input_filtered = {k: v for k, v in text_input.items() if k != "labels"}
            text_outputs = self.text_model(**text_input_filtered)
            pooled_text = text_outputs.last_hidden_state.mean(dim=1)
            text_features = self.text_fc(pooled_text)
            features = text_features if features is None else features + text_features

        if image_input is not None:
            image_features = self.image_model(image_input)
            image_features = self.image_fc(image_features)
            features = image_features if features is None else features + image_features

        if (text_input is not None) and (image_input is not None):
            features = features / 2

        logits = self.classifier(features)
        return logits

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

@st.cache_resource(show_spinner=False)
def load_resources():
    with st.spinner("Loading model and resources..."):
        model_name = "facebook/bart-base"
        tokenizer = BartTokenizer.from_pretrained(model_name)
        image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        text_df = pd.read_csv("dataset.csv")
        label_list = text_df['label'].unique().tolist()
        label2id = {label: idx for idx, label in enumerate(label_list)}
        id2label = {idx: label for label, idx in label2id.items()}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = "multimodal_model.pth"
        image_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        image_model.fc = nn.Identity()
        model = MultiModalClassifier(
            text_model=BartModel.from_pretrained(model_name),
            image_model=image_model,
            text_feat_dim=768,
            image_feat_dim=512,
            hidden_dim=512,
            num_classes=len(label_list)
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, tokenizer, image_transforms, device, id2label

# -----------------------------------------------------------------------------
# Prediction Functions
# -----------------------------------------------------------------------------
def inference_text(model, tokenizer, text, device, id2label, max_length=128, return_logits=False):
    encoding = tokenizer(text, padding="max_length", truncation=True,
                         max_length=max_length, return_tensors="pt")
    for key in encoding:
        encoding[key] = encoding[key].to(device)
    with torch.no_grad():
        logits = model(text_input=encoding, image_input=None)
    logits_np = logits.cpu().numpy().flatten()
    pred_id = int(np.argmax(logits_np))
    pred_label = id2label[pred_id]
    return (pred_label, logits_np) if return_logits else pred_label

def inference_image(model, image_file, transform, device, id2label, return_logits=False):
    image = Image.open(image_file).convert("RGB")
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        logits = model(text_input=None, image_input=image)
    logits_np = logits.cpu().numpy().flatten()
    pred_id = int(np.argmax(logits_np))
    pred_label = id2label[pred_id]
    return (pred_label, logits_np) if return_logits else pred_label

def inference_both(model, tokenizer, text, image_file, transform, device, id2label, max_length=128, return_logits=False):
    encoding = tokenizer(text, padding="max_length", truncation=True,
                         max_length=max_length, return_tensors="pt")
    for key in encoding:
        encoding[key] = encoding[key].to(device)
    image = Image.open(image_file).convert("RGB")
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        logits = model(text_input=encoding, image_input=image)
    logits_np = logits.cpu().numpy().flatten()
    pred_id = int(np.argmax(logits_np))
    pred_label = id2label[pred_id]
    return (pred_label, logits_np) if return_logits else pred_label

# Enhanced AI Suggestion Function
def get_ai_suggestion(pred_label, input_context, mode):
    try:
        prompt = f"""
        You are an AI assistant providing detailed, actionable suggestions based on a multimodal classifier's prediction.
        The classifier predicted the label '{pred_label}' for the following input:
        - Mode: {mode}
        - Input: {input_context}

        Provide an enhanced suggestion that includes:
        1. **Why**: Explain why this prediction might have been made based on the input.
        2. **How**: Describe how this prediction impacts the situation or context.
        3. **Step-by-Step Solutions**: Offer 3-5 specific, numbered steps to address or overcome the predicted outcome.
        
        Format your response clearly with these sections.
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            web_search=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI suggestion unavailable due to error: {str(e)}"

# -----------------------------------------------------------------------------
# Session State for Prediction History
# -----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

def add_history(record):
    st.session_state.history.append(record)

def download_history():
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        csv = df.to_csv(index=False).encode('utf-8')
        b64 = base64.b64encode(csv).decode()
        href = f'<a class="download-link" href="data:file/csv;base64,{b64}" download="prediction_history.csv">Download History as CSV</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
    else:
        st.sidebar.info("No history to download.")

# -----------------------------------------------------------------------------
# Sidebar Navigation & Options
# -----------------------------------------------------------------------------
def render_sidebar():
    st.sidebar.title("Navigation")
    mode = st.sidebar.radio("Select Inference Mode", 
                            ["Text Only", "Image Only", "Both", "Batch Images"])
    st.sidebar.markdown("---")
    st.sidebar.subheader("Advanced Options")
    show_details = st.sidebar.checkbox("Show Raw Logits & Probabilities", value=False)
    if st.sidebar.button("Clear History"):
        st.session_state.history = []
        st.sidebar.success("History cleared!")
    st.sidebar.markdown("---")
    download_history()
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        - **Text Only:** Input or select sample text for predictions.
        - **Image Only:** Upload a single image.
        - **Both:** Provide text and an image.
        - **Batch Images:** Upload multiple images at once.
        """
    )
    return mode, show_details

# -----------------------------------------------------------------------------
# Render Inference Tab
# -----------------------------------------------------------------------------
def render_inference_tab(model, tokenizer, image_transforms, device, id2label, mode, show_details):
    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.title("Advanced MultiModal Classifier")
    st.markdown("Combine text and image data to obtain predictions with cutting-edge AI models.")
    st.image("https://media.licdn.com/dms/image/v2/D4D12AQE-mU-zw8ePtQ/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1728155106122?e=2147483647&v=beta&t=zXlflFzaZ3hq5UTAmur34VnYkbi_IauarjD6kowDylQ", use_column_width=True)
    
    sample_texts = [
        "Increased microaneurysms, Cotton wool spots, Mild vision loss",
        "Severe headache with blurred vision and nausea",
        "Patient exhibits elevated blood sugar and blurred vision"
    ]
    
    if mode == "Text Only":
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Text Inference")
            col1, col2 = st.columns(2)
            with col1:
                text_input = st.text_area("Enter Text Input", "Enter your text here...")
            with col2:
                sample_option = st.selectbox("Or select a sample", sample_texts)
            final_text = text_input if text_input.strip() != "" else sample_option
            if st.button("Predict (Text)"):
                with st.spinner("Processing text..."):
                    if show_details:
                        pred_label, logits = inference_text(model, tokenizer, final_text, device, id2label, return_logits=True)
                    else:
                        pred_label = inference_text(model, tokenizer, final_text, device, id2label)
                    time.sleep(1)
                    st.success(f"Predicted Label: **{pred_label}**")
                    ai_suggestion = get_ai_suggestion(pred_label, final_text, "Text Only")
                    st.info(f"**AI Suggestion:**\n\n{ai_suggestion}")
                    add_history({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "mode": "Text Only",
                        "input": final_text,
                        "predicted_label": pred_label,
                        "ai_suggestion": ai_suggestion
                    })
                    if show_details:
                        st.markdown("**Raw Logits:**")
                        st.write(logits)
                        st.markdown("**Softmax Probabilities:**")
                        probs = softmax(logits)
                        prob_df = pd.DataFrame({
                            "Label": list(id2label.values()),
                            "Probability": probs
                        }).sort_values("Probability", ascending=False)
                        st.bar_chart(prob_df.set_index("Label"))
            st.markdown("</div>", unsafe_allow_html=True)
    
    elif mode == "Image Only":
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Image Inference")
            uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"], key="single_image")
            if uploaded_file is not None:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            if st.button("Predict (Image)"):
                if uploaded_file is None:
                    st.error("Please upload an image.")
                else:
                    with st.spinner("Processing image..."):
                        if show_details:
                            pred_label, logits = inference_image(model, uploaded_file, image_transforms, device, id2label, return_logits=True)
                        else:
                            pred_label = inference_image(model, uploaded_file, image_transforms, device, id2label)
                        time.sleep(1)
                        st.success(f"Predicted Label: **{pred_label}**")
                        ai_suggestion = get_ai_suggestion(pred_label, uploaded_file.name, "Image Only")
                        st.info(f"**AI Suggestion:**\n\n{ai_suggestion}")
                        add_history({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "mode": "Image Only",
                            "input": uploaded_file.name,
                            "predicted_label": pred_label,
                            "ai_suggestion": ai_suggestion
                        })
                        if show_details:
                            st.markdown("**Raw Logits:**")
                            st.write(logits)
                            st.markdown("**Softmax Probabilities:**")
                            probs = softmax(logits)
                            prob_df = pd.DataFrame({
                                "Label": list(id2label.values()),
                                "Probability": probs
                            }).sort_values("Probability", ascending=False)
                            st.bar_chart(prob_df.set_index("Label"))
            st.markdown("</div>", unsafe_allow_html=True)
    
    elif mode == "Both":
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Combined Text & Image Inference")
            text_input = st.text_area("Enter Text Input", "Enter your text here...", key="both_text")
            uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"], key="both_image")
            if uploaded_file is not None:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            if st.button("Predict (Both)"):
                if uploaded_file is None:
                    st.error("Please upload an image for combined inference.")
                else:
                    with st.spinner("Processing combined input..."):
                        if show_details:
                            pred_label, logits = inference_both(model, tokenizer, text_input, uploaded_file, image_transforms, device, id2label, return_logits=True)
                        else:
                            pred_label = inference_both(model, tokenizer, text_input, uploaded_file, image_transforms, device, id2label)
                        time.sleep(1)
                        st.success(f"Predicted Label: **{pred_label}**")
                        input_context = f"Text: {text_input[:30]}..., Image: {uploaded_file.name}"
                        ai_suggestion = get_ai_suggestion(pred_label, input_context, "Both")
                        st.info(f"**AI Suggestion:**\n\n{ai_suggestion}")
                        add_history({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "mode": "Both",
                            "input": input_context,
                            "predicted_label": pred_label,
                            "ai_suggestion": ai_suggestion
                        })
                        if show_details:
                            st.markdown("**Raw Logits:**")
                            st.write(logits)
                            st.markdown("**Softmax Probabilities:**")
                            probs = softmax(logits)
                            prob_df = pd.DataFrame({
                                "Label": list(id2label.values()),
                                "Probability": probs
                            }).sort_values("Probability", ascending=False)
                            st.bar_chart(prob_df.set_index("Label"))
            st.markdown("</div>", unsafe_allow_html=True)
    
    elif mode == "Batch Images":
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Batch Image Inference")
            uploaded_files = st.file_uploader("Upload Multiple Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="batch_images")
            if uploaded_files:
                cols = st.columns(3)
                predictions = {}
                for idx, file in enumerate(uploaded_files):
                    with cols[idx % 3]:
                        st.image(file, caption=file.name, use_column_width=True)
                        with st.spinner(f"Processing {file.name}..."):
                            if show_details:
                                pred_label, logits = inference_image(model, file, image_transforms, device, id2label, return_logits=True)
                            else:
                                pred_label = inference_image(model, file, image_transforms, device, id2label)
                            st.write(f"**{pred_label}**")
                            ai_suggestion = get_ai_suggestion(pred_label, file.name, "Batch Images")
                            st.info(f"**AI Suggestion:**\n\n{ai_suggestion}")
                            predictions[file.name] = pred_label
                            add_history({
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "mode": "Batch Image",
                                "input": file.name,
                                "predicted_label": pred_label,
                                "ai_suggestion": ai_suggestion
                            })
                if show_details and predictions:
                    st.markdown("**Batch Predictions Details:**")
                    st.write(predictions)
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Render History Tab
# -----------------------------------------------------------------------------
def render_history_tab():
    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.markdown("### Prediction History")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df)
    else:
        st.info("No predictions yet.")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Main Application Entry Point
# -----------------------------------------------------------------------------
def main():
    mode, show_details = render_sidebar()
    model, tokenizer, image_transforms, device, id2label = load_resources()

    tabs = st.tabs(["Inference", "History"])
    with tabs[0]:
        render_inference_tab(model, tokenizer, image_transforms, device, id2label, mode, show_details)
    with tabs[1]:
        render_history_tab()

if __name__ == '__main__':
    main()