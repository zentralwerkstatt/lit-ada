import numpy as np
import torch as t
import PIL.Image
import sys
sys.path.append('stylegan2-ada-pytorch')
from projector import project
from app import force_cpu


def get_device():
    device = "cuda" if t.cuda.is_available() else "cpu"
    if force_cpu: device = "cpu"
    return device

def lerp_w(w1, w2, n):
    ws = np.array([(1 - i) * w1 + i * w2 for i in np.linspace(0, 1, n)])
    return ws

def get_random_z(n=1, seed=None):
    if not seed: seed = np.random.randint(1000)
    z = np.random.RandomState(seed).randn(n, 512)
    return z

def get_random_w(G, n=1, seed=None):
    if not seed: seed = np.random.randint(1000)
    z = np.random.RandomState(seed).randn(n, 512)
    w = z_to_w(G, z)
    return w

def preprocess_w(w):
    tensor_w = t.from_numpy(w).to(get_device())
    return tensor_w

def deprocess_img(tensor_img):
    tensor_img = (tensor_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(t.uint8) # From generate.py
    img = tensor_img.detach().cpu().numpy()
    return img

def z_to_w(G, z):
    tensor_z = t.from_numpy(z).to(get_device())
    c = None  # No class labels
    tensor_w = G.mapping(tensor_z, c, truncation_psi=0.5, truncation_cutoff=8)
    w = tensor_w.detach().cpu().numpy()
    return w

def prepare_img_for_projection(G, fname):
    img = PIL.Image.open(fname).convert('RGB')
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    img = img.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    img = np.array(img, dtype=np.uint8)
    img = img.transpose([2, 0, 1])
    img = t.tensor(img).to(get_device())
    return img

def project_img(G, fname, num_steps):
    img = prepare_img_for_projection(G, fname)
    projected_w_steps = project(G, target=img, num_steps=num_steps, device=get_device(), verbose=True)
    w = projected_w_steps[-1].unsqueeze(0).detach().cpu().numpy()
    return w