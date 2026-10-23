import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.predict_tl import predict_array

st.set_page_config(page_title="NeuroVision AI", page_icon="🧠")
st.title("🧠 NeuroVision AI")
st.write("Classification d'IRM cérébrales : gliome, méningiome, tumeur de l'hypophyse, absence de tumeur.")

uploaded = st.file_uploader("Choisis une IRM", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is not None:
    data = np.frombuffer(uploaded.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)              # BGR, 3 canaux

    if img is None:
        st.error("Impossible de lire cette image.")
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img, caption="Image envoyée", width=350)
        if st.button("Analyser"):
            try:
                with st.spinner("Analyse en cours..."):
                    label, conf, probs = predict_array(img)
                st.success(f"Prédiction : **{label}** (confiance : {conf:.1%})")
                st.bar_chart(pd.Series(probs))
            except Exception as e:
                st.error(str(e))

st.caption("⚠️ Outil d'aide à la décision uniquement : il ne remplace jamais l'avis d'un médecin.")