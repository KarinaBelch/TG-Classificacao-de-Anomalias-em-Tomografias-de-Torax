import streamlit as st

# @title Layout

# # Titulo da página
st.set_page_config(page_title='Trabalho de Graduação', page_icon='🥼', layout='wide')
st.title('Classificação de Anomalias em TC de Tórax')
st.info('Trabalho de Graduação referente ao curso de Engenharia Biomédica da Universidade Federal do ABC | Identificação e localização de anomalias causadas por câncer de pulmão, em tomografias de tórax, utilizando inteligência artifical')
# 
# Menu Lateral
st.sidebar.header("Menu")
st.sidebar.caption("Leitura de arquivos DICOM.")
# 
# # Upload do arquivo
uploaded_zip = st.file_uploader(label='Upload your DICOM file:', type="zip")
