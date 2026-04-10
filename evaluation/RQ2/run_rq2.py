import argparse
import json
import sys
from pathlib import Path
import pydantic

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from rich.console import Console


def compute_ssim(imageA, imageB, threshold=0.8):
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(grayA, grayB, full=True)
    return score


def compute_abs_diff(imageA, imageB, threshold=0.8):
    diff = cv2.absdiff(imageA, imageB)
    mean_diff = float(np.mean(diff))  # 0..255
    similarity = 1.0 - (mean_diff / 255.0)  # map to [0,1], higher is more similar
    similarity = max(0.0, min(1.0, similarity))
    return similarity


def compute_sift_matches(imageA, imageB, threshold=0.15):
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(grayA, None)
    kp2, des2 = sift.detectAndCompute(grayB, None)

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return 0.0, False

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = [m for m, n in matches if m.distance < 0.55 * n.distance]
    denom = max(len(kp1), len(kp2))
    similarity = len(good) / denom if denom > 0 else 0.0
    return similarity


def compute_clip_matches(imageA, imageB, threshold=0.8, device_pref: str):
    model_name = "openai/clip-vit-base-patch32"
    if device_pref == "cpu":
        device = "cpu"
    elif device_pref == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:  # auto
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    image_1 = Image.fromarray(cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB))
    image_2 = Image.fromarray(cv2.cvtColor(imageB, cv2.COLOR_BGR2RGB))

    with torch.no_grad():
        inputs = processor(images=image_1, return_tensors="pt").to(device)
        features_1 = model.get_image_features(**inputs)
        features_1 = features_1 / features_1.norm(p=2, dim=-1, keepdim=True)
        features_1 = features_1.squeeze(0).cpu()

        inputs = processor(images=image_2, return_tensors="pt").to(device)
        features_2 = model.get_image_features(**inputs)
        features_2 = features_2 / features_2.norm(p=2, dim=-1, keepdim=True)
        features_2 = features_2.squeeze(0).cpu()

    sim = torch.nn.functional.cosine_similarity(features_1, features_2, dim=0).item()
    return sim


