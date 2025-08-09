{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOohywmo97S7nmMks3ez3BI",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/KarinaBelch/TG-Classificacao-de-Anomalias-em-Tomografias-de-Torax/blob/main/streamlit_app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "cellView": "form",
        "id": "WY0xcXLlN9Ef"
      },
      "outputs": [],
      "source": [
        "# @title Importando bibliotecas\n",
        "\n",
        "import streamlit as st\n",
        "from tensorflow.keras.models import load_model\n",
        "import numpy as np\n",
        "import os\n",
        "import zipfile\n",
        "import gdown"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# @title Parâmetros\n",
        "\n",
        "# Listar arquivos .dcm\n",
        "dicom_files = []\n",
        "\n",
        "# Resultados armazenados em lista\n",
        "resultados = []"
      ],
      "metadata": {
        "cellView": "form",
        "id": "PvtVx6hWRrpn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# @title Funções\n",
        "\n",
        "# Função para obter os arquivos DICOM que estão no arquivo zipado enviado pelo usuário\n",
        "def funcObterArquivoDicom(dicom_dir):\n",
        "  for root, dirs, files in os.walk(dicom_dir):\n",
        "      for file in files:\n",
        "          if file.endswith(\".dcm\"):\n",
        "              dicom_files.append(os.path.join(root, file))\n",
        "\n",
        "  print(\"Arquivos DICOM encontrados:\", len(dicom_files))\n",
        "\n",
        "  return dicom_files\n",
        "\n",
        "# Função para ordenar as fatias do arquivo DICOM\n",
        "def funcOrdenarFatias(dicom_files):\n",
        "  # Ordenar por InstanceNumber (ordem axial)\n",
        "  slices = [pydicom.dcmread(f) for f in dicom_files]\n",
        "  slices = [s for s in slices if hasattr(s, 'InstanceNumber')]\n",
        "  slices.sort(key=lambda s: s.InstanceNumber)\n",
        "\n",
        "  # Converter para volume 3D\n",
        "  volume = np.stack([s.pixel_array for s in slices])\n",
        "\n",
        "  print(\"Volume 3D:\", volume.shape)  # (profundidade, altura, largura)\n",
        "\n",
        "  return slices, volume"
      ],
      "metadata": {
        "cellView": "form",
        "id": "GKjd9NwqSKGM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# @title Layout\n",
        "\n",
        "# # Titulo da página\n",
        "st.set_page_config(page_title='Trabalho de Graduação', page_icon='🥼', layout='wide')\n",
        "st.title('Classificação de Anomalias em TC de Tórax')\n",
        "st.info('Trabalho de Graduação referente ao curso de Engenharia Biomédica da Universidade Federal do ABC | Identificação e localização de anomalias causadas por câncer de pulmão, em tomografias de tórax, utilizando inteligência artifical')\n",
        "#\n",
        "# Menu Lateral\n",
        "st.sidebar.header(\"Menu\")\n",
        "st.sidebar.caption(\"Leitura de arquivos DICOM.\")\n",
        "#\n",
        "# # Upload do arquivo\n",
        "uploaded_zip = st.file_uploader(label='Upload your DICOM file:', type=\"zip\")"
      ],
      "metadata": {
        "cellView": "form",
        "id": "dYYxZTtrONgb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# @title Processamento dos arquivos DICOM\n",
        "\n",
        "\n",
        "model_path = \"lung_cancer_classifier.h5\"\n",
        "\n",
        "if not os.path.exists(model_path):\n",
        "    url = \"https://drive.google.com/file/d/11oFWS9_ckKVAhSlbjcRglDfU6IyBGcjM/view?usp=drive_link\"\n",
        "    gdown.download(url, model_path, quiet=False)\n",
        "\n",
        "# Carregando o modelo\n",
        "modelo = load_model(model_path)  # Nome do arquivo dentro do zip\n",
        "\n",
        "if uploaded_zip:\n",
        "    temp_dir = \"temp_upload\"\n",
        "\n",
        "    # Limpar e criar diretório temporário\n",
        "    if os.path.exists(temp_dir):\n",
        "        shutil.rmtree(temp_dir)\n",
        "    os.makedirs(temp_dir)\n",
        "\n",
        "    # Salvar arquivo zip\n",
        "    zip_path = os.path.join(temp_dir, \"uploaded.zip\")\n",
        "    with open(zip_path, \"wb\") as f:\n",
        "        f.write(uploaded_zip.getbuffer())\n",
        "\n",
        "    # Extrair arquivos do zip\n",
        "    with zipfile.ZipFile(zip_path, \"r\") as zip_ref:\n",
        "        zip_ref.extractall(temp_dir)\n",
        "\n",
        "    # Chamando a função para obter os arquivos DICOM do arquivo zipado\n",
        "    dicom_files = funcObterArquivoDicom(temp_dir)\n",
        "\n",
        "    st.write(\"Arquivos DICOM encontrados:\", len(dicom_files))\n",
        "\n",
        "    # Ler e ordenar as fatias\n",
        "    slices, volume = funcOrdenarFatias(dicom_files)\n",
        "\n",
        "    for i in range(len(slices)):\n",
        "      img_array = np.expand_dims(img_array, axis=0)  # shape (1, altura, largura, canais)\n",
        "      pred = modelo.predict(img_array)\n",
        "      resultados.append((slices))\n",
        "      print(slices)"
      ],
      "metadata": {
        "id": "5i3QC1GlReL9"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}