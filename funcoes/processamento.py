import pydicom
from PIL import Image
import cv2
from skimage import measure
import numpy as np
import io
import os
from skimage.segmentation import clear_border
from scipy.ndimage import binary_fill_holes

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

def preprocess_image_from_ds(ds, IMG_SIZE):
    image = ds.pixel_array.astype(np.int16)
    intercept = getattr(ds, 'RescaleIntercept', -1024)
    slope = getattr(ds, 'RescaleSlope', 1)
    image = slope * image + intercept

    lung_mask = segment_lung_mask(image, fill_lung_structures=True)
    image_segmented = image.copy()
    image_segmented[~lung_mask] = np.min(image)

    image_segmented = (image_segmented - np.min(image_segmented)) / (np.max(image_segmented) - np.min(image_segmented))
    image_resized = cv2.resize(image_segmented, (IMG_SIZE, IMG_SIZE))

    return image_resized

# ==========================
# Predição com visualização lado a lado
# ==========================
def predict_single_dicom(dicom_path, model, IMG_SIZE):
    ds = pydicom.dcmread(dicom_path)

    # --- Imagem original ---
    image_orig = ds.pixel_array.astype(np.int16)
    intercept = getattr(ds, 'RescaleIntercept', -1024)
    slope = getattr(ds, 'RescaleSlope', 1)
    image_orig = slope * image_orig + intercept

    image_orig_norm = (image_orig - np.min(image_orig)) / (np.max(image_orig) - np.min(image_orig))
    image_orig_resized = cv2.resize(image_orig_norm, (IMG_SIZE, IMG_SIZE))
    img_original = Image.fromarray((image_orig_resized*255).astype(np.uint8)).convert("RGB")

    # --- Pré-processamento para o modelo ---
    img_preprocessed = preprocess_image_from_ds(ds, IMG_SIZE)
    img_array = np.expand_dims(img_preprocessed, axis=(0, -1))
    prob = model.predict(img_array, verbose=0)[0][0]
    pred_class = "Câncer" if prob > 0.5 else "Saudável"

    # --- Máscara ---
    lung_mask = segment_lung_mask(image_orig)
    lung_mask_resized = cv2.resize(lung_mask.astype(np.uint8), (IMG_SIZE, IMG_SIZE))

    # --- Criar imagem com máscara verde ---
    img_mask = np.stack([image_orig_resized]*3, axis=-1)
    img_mask = (img_mask*255).astype(np.uint8)
    green = np.array([0,255,0], dtype=np.uint8)
    alpha = 0.4
    img_mask[lung_mask_resized>0] = ((1-alpha)*img_mask[lung_mask_resized>0] + alpha*green).astype(np.uint8)
    img_mask = Image.fromarray(img_mask)

    # --- Combinar lado a lado ---
    combined_width = img_original.width + img_mask.width
    combined_img = Image.new('RGB', (combined_width, IMG_SIZE))
    combined_img.paste(img_original, (0,0))
    combined_img.paste(img_mask, (img_original.width,0))

    buf = io.BytesIO()
    combined_img.save(buf, format='PNG')
    buf.seek(0)

    return prob, pred_class, buf