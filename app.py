import numpy as np
import os
import streamlit as st
import sys
import pickle
import torch as t
from io import BytesIO
import zipfile
import base64
from datetime import datetime
import PIL.Image
sys.path.append('../stylegan2-ada-pytorch')
from util import *


force_cpu = False
model = "metfaces.pkl"

def init_w(G):
    if os.path.isfile("w1.npy"):
        w1 = np.load("w1.npy")
    else:
        w1 = get_random_w(G)
        np.save("w1", w1)
    if os.path.isfile("w2.npy"):
        w2 = np.load("w2.npy")
    else:
        w2 = get_random_w(G)
        np.save("w2", w2)
    return w1, w2

@st.cache(allow_output_mutation=True)
def load_model():
    print("Loading model... (this should only run once)")
    with open(model, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(get_device())
    return G

def main():
    # Load generator
    G = load_model()

    # Initalize w-space latent vector from last used via file or random
    w1, w2 = init_w(G)
    
    # Settings
    interpolation_steps = st.sidebar.slider("Interpolation steps", min_value=2, max_value=15, step=1)
    projection_steps = st.sidebar.slider("Projection iterations", min_value=100, max_value=10000, step=100)
    
    # Randomize start image
    if st.sidebar.button("Randomize Alpha", key="r1"):
        w1 = get_random_w(G)
        np.save("w1", w1)
        w1, w2 = init_w(G)

    # Use projector to start from custom image latent projection
    p1 = st.sidebar.file_uploader("Project alpha from file", type=['jpg'], key="p1")
    if p1 is not None:
        with open("p1.jpg", "wb") as f:
            f.write(BytesIO(p1.getvalue()).getbuffer())
        w1 = project_img(G, "p1.jpg", projection_steps)
        np.save("w1", w1)

    # Randomize end image
    if st.sidebar.button("Randomize Omega", key="r2"):
        w2 = get_random_w(G)
        np.save("w2", w2)
        w1, w2 = init_w(G)

    # Use projector to end on custom image latent projection
    p2 = st.sidebar.file_uploader("Project omega from file", type=['jpg'], key="p2")
    if p2 is not None:
        with open("p2.jpg", "wb") as f:
            f.write(BytesIO(p2.getvalue()).getbuffer())
        w2 = project_img(G, "p2.jpg", projection_steps)
        np.save("w2", w2)

    # Compute start image
    img1 = deprocess_img(G.synthesis(preprocess_w(w1), noise_mode='const', force_fp32=True))
    st.image(img1, key="a")

    # Compute interpolation
    ws = lerp_w(w1[0], w2[0], interpolation_steps)
    imgs = deprocess_img(G.synthesis(preprocess_w(ws), noise_mode='const', force_fp32=True))
    
    if st.sidebar.button("Download images", key="d"):
        date_time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        zip_fname = f"{date_time}.zip"
        f = zipfile.ZipFile(zip_fname, "w", zipfile.ZIP_DEFLATED)
        for n, img in enumerate(imgs):
          fname = f"{n}.jpg"
          PIL.Image.fromarray(img).save(fname)
          f.write(fname)
        with open(zip_fname, "rb") as f:
          bytes = f.read()
          b64 = base64.b64encode(bytes).decode()
          href = f'<a download={zip_fname} href="data:application/zip;base64,{b64}">{zip_fname}</a>'
          st.sidebar.markdown(href, unsafe_allow_html=True)
    
    for i in range(1, len(imgs)-1):
        st.image(imgs[i], key=i)
    
    # Compute end image
    img2 = deprocess_img(G.synthesis(preprocess_w(w2), noise_mode='const', force_fp32=True))
    st.image(img2, key="o")

if __name__ == "__main__":
    main()