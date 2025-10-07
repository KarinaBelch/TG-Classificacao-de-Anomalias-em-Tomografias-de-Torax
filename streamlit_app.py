# ==========================
# Importando bibliotecas
# ==========================
import streamlit as st
import os
import gdown
import zipfile
import shutil
import tensorflow as tf
import funcoes.processamento as funcao

# ==========================
# Parâmetros
# ==========================
dicom_files = []
resultados = []

IMG_SIZE = 128
MODEL_ID = "1zfggM4S9HxRphPcGN2dCrWWswYr2kEMV"  # ID do Google Drive
MODEL_PATH = "classifier.h5"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"
NUM_COLS = 3

# ==========================
# Layout Streamlit
# ==========================
st.set_page_config(page_title='Trabalho de Graduação', page_icon='🥼', layout='wide')
st.title('Classificação de Anomalias em TC de Tórax')

st.sidebar.header("Menu")
st.sidebar.caption('Identificação e localização de anomalias causadas por câncer de pulmão, em tomografias de tórax, utilizando inteligência artificial')
st.sidebar.caption('Trabalho de Graduação referente ao curso de Engenharia Biomédica da UFABC')
uploaded_zip = st.file_uploader(label='Upload seu arquivo .zip com DICOMs', type="zip")

# ==========================
# Carregar modelo
# ==========================
if not os.path.exists(MODEL_PATH):
    with st.spinner("Baixando o modelo..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

modelo = tf.keras.models.load_model(MODEL_PATH)

# ==========================
# Processamento ZIP
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

    dicom_files = funcao.funcObterArquivoDicom(temp_dir)
    slices, volume = funcao.funcOrdenarFatias(dicom_files)

    # ==========================
    # Predição e exibição
    # ==========================
    resultados = []
    for dicom_path in dicom_files:
        try:
            prob, pred_class, img_buf = funcao.predict_single_dicom(dicom_path, modelo, IMG_SIZE)
            resultados.append({
                "img": img_buf,
                "pred_class": pred_class,
                "pred_prob": prob,
                "filename": os.path.basename(dicom_path)
            })
        except Exception as e:
            st.error(f"Erro ao processar {os.path.basename(dicom_path)}: {e}")

    num_cancer = [r for r in resultados if r['pred_class'] == 'Câncer']
    num_saudavel = [r for r in resultados if r['pred_class'] == 'Saudável']

    col1, col2 = st.columns(2, border=True)

    # Filtro lateral
    if 'filtro' not in st.session_state:
        st.session_state.filtro = "Todos"

    with col1:
        with st.container():
            st.subheader("Filtro")
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                if st.button("Todos"):
                    st.session_state.filtro = "Todos"
            with col_f2:
                if st.button("Apenas Câncer"):
                    st.session_state.filtro = "Apenas Câncer"
            with col_f3:
                if st.button("Apenas Saudável"):
                    st.session_state.filtro = "Apenas Saudável"
            st.write(f"**Filtro ativo:** {st.session_state.filtro}")


    with col2:
        st.subheader("Dados")
        st.write(f"Arquivos DICOM encontrados: {len(dicom_files)}")
        st.write(f"Slices detectados com câncer: {len(num_cancer)}")
        st.write(f"Slices detectados como saudável: {len(num_saudavel)}")

    # Aplicar filtro
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
