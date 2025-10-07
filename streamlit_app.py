# @title Importando bibliotecas

import streamlit as st
import numpy as np
import os
import pydicom
import gdown
import zipfile
import shutil
import tensorflow as tf
#from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import io

import cv2  # já vamos usar opencv-python-headless no requirements
from skimage import measure
from skimage.segmentation import clear_border
from scipy.ndimage import binary_fill_holes

# @title Parâmetros
dicom_files = []
resultados = []

IMG_SIZE = 128
MODEL_ID = "1zfggM4S9HxRphPcGN2dCrWWswYr2kEMV"  # ID do Google Drive
MODEL_PATH = "classifier.h5"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"
NUM_COLS = 3

# ==========================
# Funções auxiliares
# ==========================

def funcObterArquivoDicom(dicom_dir):
    dicom_files = []
    for root, dirs, files in os.walk(dicom_dir):
        for file in files:
            if file.endswith(".dcm"):
                dicom_files.append(os.path.join(root, file))
    return dicom_files

def funcOrdenarFatias(dicom_files):
    slices = [pydicom.dcmread(f) for f in dicom_files]
    slices = [s for s in slices if hasattr(s, 'InstanceNumber')]
    slices.sort(key=lambda s: s.InstanceNumber)
    volume = np.stack([s.pixel_array for s in slices])
    return slices, volume

# ==========================
# Segmentação Pulmonar
# ==========================
def segment_lung_mask(image_hu, fill_lung_structures=True):
    binary_image = np.array(image_hu < -320, dtype=np.int8)
    labels = measure.label(binary_image)
    background_label = labels[0, 0]
    binary_image[labels == background_label] = 0

    if fill_lung_structures:
        binary_image = binary_fill_holes(binary_image)

    binary_image = clear_border(binary_image)
    labels = measure.label(binary_image)
    regions = measure.regionprops(labels)
    regions.sort(key=lambda x: x.area, reverse=True)

    final_mask = np.zeros_like(binary_image, dtype=bool)
    for region in regions[:2]:
        final_mask[labels == region.label] = True

    final_mask[:5, :] = False
    final_mask[-5:, :] = False
    final_mask[:, :5] = False
    final_mask[:, -5:] = False

    return final_mask

def preprocess_image_from_ds(ds):
    image = ds.pixel_array.astype(np.int16)
    intercept = ds.RescaleIntercept if 'RescaleIntercept' in ds else -1024
    slope = ds.RescaleSlope if 'RescaleSlope' in ds else 1
    image = slope * image + intercept

    lung_mask = segment_lung_mask(image, fill_lung_structures=True)
    image_segmented = image.copy()
    image_segmented[~lung_mask] = np.min(image)

    image_segmented = (image_segmented - np.min(image_segmented)) / (np.max(image_segmented) - np.min(image_segmented))
    image_resized = cv2.resize(image_segmented, (IMG_SIZE, IMG_SIZE))

    return image_resized


def predict_single_dicom(dicom_path, model):
    """
    Lê o DICOM, pré-processa, realiza predição e gera uma imagem
    final com a segmentação verde sobreposta.
    """
    ds = pydicom.dcmread(dicom_path)
    img_preprocessed = preprocess_image_from_ds(ds)  # imagem normalizada [0,1]
    img_array = np.expand_dims(img_preprocessed, axis=(0, -1))
    
    # Predição
    prob = model.predict(img_array, verbose=0)[0][0]
    pred_class = "Câncer" if prob > 0.5 else "Saudável"
    
    # Máscara segmentada
    lung_mask = segment_lung_mask(ds.pixel_array.astype(np.int16))
    
    # Redimensiona a máscara para o tamanho da imagem processada
    lung_mask_resized = cv2.resize(lung_mask.astype(np.uint8), (IMG_SIZE, IMG_SIZE))
    
    # Criar figura com matplotlib
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(img_preprocessed, cmap='gray')
    ax.imshow(lung_mask_resized, cmap='Greens', alpha=0.4)  # verde semitransparente
    ax.axis('off')
    
    # Salvar figura em memória para Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    
    return prob, pred_class, buf

# ==========================
# Layout Streamlit
# ==========================
st.set_page_config(page_title='Trabalho de Graduação', page_icon='🥼', layout='wide')
st.title('Classificação de Anomalias em TC de Tórax')
st.info('Identificação e localização de anomalias causadas por câncer de pulmão, em tomografias de tórax, utilizando inteligência artificial | Trabalho de Graduação referente ao curso de Engenharia Biomédica da UFABC')

st.sidebar.header("Menu")
st.sidebar.caption("Leitura de arquivos DICOM.")

uploaded_zip = st.file_uploader(label='Upload seu arquivo .zip com DICOMs', type="zip")

# ==========================
# Carrega modelo treinado
# ==========================
if not os.path.exists(MODEL_PATH):
    with st.spinner("Baixando o modelo..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

modelo = tf.keras.models.load_model(MODEL_PATH)

# ==========================
# Processamento do ZIP
# ==========================
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

    dicom_files = funcObterArquivoDicom(temp_dir)
    slices, volume = funcOrdenarFatias(dicom_files)

    if 'filtro' not in st.session_state:
        st.session_state.filtro = "Todos"

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Filtro")
        if st.button("Todos"):
            st.session_state.filtro = "Todos"
        if st.button("Apenas Câncer"):
            st.session_state.filtro = "Apenas Câncer"
        if st.button("Apenas Saudável"):
            st.session_state.filtro = "Apenas Saudável"
        st.write(f"Filtro ativo: {st.session_state.filtro}")

        resultados = []
        for dicom_path in dicom_files:
            try:
                # agora retorna a imagem com máscara verde sobreposta
                prob, pred_class, img_buf = predict_single_dicom(dicom_path, modelo)
                resultados.append({
                    "img": img_buf,  # BytesIO da imagem final
                    "pred_class": pred_class,
                    "pred_prob": prob,
                    "filename": os.path.basename(dicom_path)
                })
            except Exception as e:
                st.error(f"Erro ao processar {os.path.basename(dicom_path)}: {e}")

        num_cancer = [r for r in resultados if r['pred_class'] == 'Câncer']
        num_saudavel = [r for r in resultados if r['pred_class'] == 'Saudável']

        with col2:
            st.subheader("Dados")
            st.write(f"Arquivos DICOM encontrados: {len(dicom_files)}")
            st.write(f"Slices detectados com câncer: {len(num_cancer)}")
            st.write(f"Slices detectados como saudável: {len(num_saudavel)}")

        # Filtro
        if st.session_state.filtro == "Todos":
            filtrado = resultados
        elif st.session_state.filtro == "Apenas Câncer":
            filtrado = num_cancer
        else:
            filtrado = num_saudavel

        # Exibição em colunas
        st.subheader("Resultados")
        for i in range(0, len(filtrado), NUM_COLS):
            cols = st.columns(min(NUM_COLS, len(filtrado) - i))
            batch = filtrado[i:i+NUM_COLS]

            for j, item in enumerate(batch):
                with cols[j]:
                    st.image(item["img"], caption=f"{item['filename']}")
                    st.write(f"Predição: {item['pred_class']} ({item['pred_prob']:.3f})")
