import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm

# ==========================

# CONFIGURATION

# ==========================

IMG_SIZE = (150, 150)

CLASS_NAMES = [
    "acne",
    "Eczema",
    "Psoriasis",
    "Vitiligo",
    "Unknown_Normal"
]

LAST_CONV_LAYER = "conv2d_1"

# ==========================

# LOAD MODEL

# ==========================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("skin_disease_model.keras")
    return model

model = load_model()

# ==========================

# GRAD-CAM MODEL

# ==========================

def find_last_conv_layer(m):
    for layer in reversed(m.layers):
        # prefer Conv2D layers by class, fallback to name containing 'conv'
        if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name.lower():
            return layer.name
    return None

# determine last conv layer name at runtime for robustness
LAST_CONV_LAYER_NAME = find_last_conv_layer(model) or LAST_CONV_LAYER

# ==========================

# IMAGE PREPROCESSING

# ==========================

def preprocess_image(image):
    image = image.resize(IMG_SIZE)

    img_array = np.array(image)

    img_array = img_array.astype("float32") / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array
# ==========================

# GRAD-CAM FUNCTION

# ==========================

def make_gradcam_heatmap(img_array):
    """Compute Grad-CAM heatmap for the top predicted class.

    img_array: preprocessed batch image, shape (1, H, W, C)
    returns: 2D numpy heatmap normalized to [0,1]
    """
    if LAST_CONV_LAYER_NAME is None:
        st.error("No convolutional layer found for Grad-CAM; skipping heatmap.")
        return None

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # Forward-pass helper that walks through layers (including nested models)
    def forward_through_layers(layers, x):
        conv_activation = None
        out = x
        for lyr in layers:
            # If the layer is itself a Model/Sequential, recurse into its layers
            if isinstance(lyr, tf.keras.Model):
                out, sub_conv = forward_through_layers(lyr.layers, out)
                if sub_conv is not None:
                    conv_activation = sub_conv
            else:
                # call layer with training=False when supported
                try:
                    out = lyr(out, training=False)
                except TypeError:
                    out = lyr(out)

                # record conv outputs when encountering Conv2D (or name includes 'conv')
                if isinstance(lyr, tf.keras.layers.Conv2D) or 'conv' in lyr.name.lower():
                    conv_activation = out

        return out, conv_activation

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)

        # run manual forward pass to capture last conv activations and predictions
        preds, last_conv = forward_through_layers(model.layers, img_tensor)

        # preds may be a tensor or nested; ensure predictions tensor
        predictions = preds
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    if last_conv is None:
        st.error("Could not locate conv activations during forward pass.")
        return None

    # Compute gradients of the class score w.r.t. the conv activations
    grads = tape.gradient(class_channel, last_conv)
    if grads is None:
        st.error("Unable to compute Grad-CAM gradients.")
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted sum of feature maps
    conv_outputs = last_conv[0]
    heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return tf.zeros_like(heatmap).numpy()
    heatmap /= (max_val + 1e-8)

    return heatmap.numpy()

# ==========================

# STREAMLIT PAGE

# ==========================

st.set_page_config(
page_title="Explainable Skincare Detection",
page_icon="🩺",
layout="wide"
)

st.title("🩺 Explainable Skincare Detection")
st.write(
"Upload a skin image to predict disease and visualize Grad-CAM heatmap."
)

# ==========================

# IMAGE UPLOAD

# ==========================

uploaded_file = st.file_uploader(
"Upload Skin Image",
type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    # ======================
    # PREDICTION
    # ======================

    img_array = preprocess_image(image)

    prediction = model.predict(
        img_array,
        verbose=0
    )

    predicted_index = np.argmax(
        prediction
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        np.max(prediction) * 100
    )

    with col2:

        st.success(
            f"Prediction: {predicted_class}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

    # ======================
    # GRAD-CAM
    # ======================
    try:
        heatmap = make_gradcam_heatmap(img_array)

        st.subheader("Grad-CAM Visualization")

        if heatmap is None:
            st.info("Grad-CAM unavailable for this image.")
        else:
            # overlay heatmap on the original uploaded image (preserve original size)
            orig_w, orig_h = image.size

            # resize heatmap to original image size and smooth
            heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))
            heatmap_resized = heatmap_resized.astype(np.float32)
            # optional smoothing to reduce speckle (adjust kernel for more/less smooth)
            heatmap_resized = cv2.GaussianBlur(heatmap_resized, (7, 7), 0)

            # threshold low activations to remove large blue-ish background
            pct = 60
            thresh = np.percentile(heatmap_resized, pct)
            mask = heatmap_resized > thresh
            if not np.any(mask):
                # fallback to lower threshold if nothing remains
                thresh = np.percentile(heatmap_resized, 40)
                mask = heatmap_resized > thresh

            # use a perceptually-uniform colormap with warm colors for hotspots
            cmap = cm.get_cmap("inferno")
            heatmap_rgb = cmap(heatmap_resized)[:, :, :3]

            # build RGBA heatmap where alpha is based on mask (so background is transparent)
            heatmap_img = Image.fromarray(np.uint8(heatmap_rgb * 255)).convert("RGBA")
            alpha_mask = (mask.astype(np.float32) * 0.6 * 255).astype(np.uint8)
            heatmap_img.putalpha(Image.fromarray(alpha_mask))

            original_rgba = image.convert("RGBA")

            # composite only the hotspot areas over the original
            overlay = Image.alpha_composite(original_rgba, heatmap_img)

            # show overlay slightly smaller than original for better fit
            display_width = int(orig_w * 0.7)
            st.image(overlay, width=display_width)

    except Exception as e:
        st.error(f"Grad-CAM failed: {e}")

    # ======================
    # PROBABILITIES
    # ======================

    st.subheader(
        "Class Probabilities"
    )

    for i, cls in enumerate(
        CLASS_NAMES
    ):

        st.write(
            f"{cls}: {prediction[0][i]*100:.2f}%"
        )
