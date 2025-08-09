import os
import shutil
import zipfile
import numpy as np
import streamlit as st
import gdown
import pydicom
from tensorflow.keras.models import load_model
from PIL import Image  # Pillow

# Ajuste para o seu modelo
IMG_SIZE = 128
MODEL_ID = "SEU_ID_DO_ARQUIVO"  # coloque aqui o ID do Google Drive
MODEL_PATH = "modelo_treinado.h5"
MODEL_URL = f"https://drive.google.com/uc?id=11oFWS9_ckKVAhSlbjcRglDfU6IyBGcjM"

# Baixa o modelo se não existir
if not os.path.exists(MODEL_PATH):
    with st.spinner("Baixando o modelo..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Carrega o modelo
modelo = load_model(MODEL_PATH)

st.title("Classificador de Imagens DICOM")

uploaded_zip = st.file_uploader("Envie o arquivo ZIP com imagens DICOM", type="zip")

def extract_dicom_images_from_folder(folder_path):
    dicom_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".dcm"):
                dicom_files.append(os.path.join(root, file))
    return dicom_files

if uploaded_zip:
    temp_dir = "temp_upload"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    zip_path = os.path.join(temp_dir, "uploaded.zip")
    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.getbuffer())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    dicom_paths = extract_dicom_images_from_folder(temp_dir)
    st.write(f"Arquivos DICOM encontrados: {len(dicom_paths)}")

    resultados = []
    for i, dicom_path in enumerate(dicom_paths):
        ds = pydicom.dcmread(dicom_path)
        img = ds.pixel_array.astype(np.float32)

        # Redimensiona com Pillow
        img_pil = Image.fromarray(img)
        img_resized = img_pil.resize((IMG_SIZE, IMG_SIZE))
        img_resized = np.array(img_resized)

        # Normaliza e adiciona canal e batch dimension
        img_norm = img_resized / 255.0
        img_norm = np.expand_dims(img_norm, axis=-1)  # canal
        img_norm = np.expand_dims(img_norm, axis=0)   # batch

        # Predição
        pred_prob = modelo.predict(img_norm)[0][0]
        pred_class = 1 if pred_prob > 0.5 else 0

        resultados.append((os.path.basename(dicom_path), pred_class, float(pred_prob)))
        st.write(f"Arquivo: {os.path.basename(dicom_path)} - Classe: {pred_class} - Probabilidade: {pred_prob:.3f}")

    st.success("Predições finalizadas!")
