# -*- coding: utf-8 -*-
"""
Utility functions for Chatbot
"""
import re
import numpy as np
from pathlib import Path

def preprocess_text(text):
    """Chuẩn hóa: lowercase, xóa HTML/URL/email, ký tự đặc biệt"""
    text = text.lower()
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"(?<=\d)\.(?=\d)", "", text)
    text = re.sub(r"[^\w\sÀ-ỹ]", " ", text)
    return text.strip()


def split_sentences(text):
    """Tách thành câu bằng dấu câu"""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[\r\n]+", ". ", text)
    text = re.sub(r"\s+", " ", text)
    sentences = [s.strip() for s in re.split(r"[\.\!\?]+", text) if s.strip()]
    return sentences


def chunk_text(text, chunk_size=3):
    """Ghép câu thành chunks (mặc định 3 câu)"""
    sentences = split_sentences(text)
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = ". ".join(sentences[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def tokenize_for_match(text):
    """Tách text thành các token"""
    return re.findall(r"[\wÀ-ỹ]+", text.lower())


def normalize_display(text):
    """Chuẩn hóa text để hiển thị"""
    text = re.sub(r"\s*-\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def calculate_probability_score(similarity_scores):
    """
    Tính xác suất dựa trên similarity scores
    Sử dụng softmax để chuyển scores thành phân phối xác suất
    """
    exp_scores = np.exp(similarity_scores - np.max(similarity_scores))
    probabilities = exp_scores / exp_scores.sum()
    return probabilities


def ensure_directory_exists(path):
    """Đảm bảo thư mục tồn tại"""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_file_size_mb(filepath):
    """Lấy kích thước file (MB)"""
    return Path(filepath).stat().st_size / (1024 * 1024)
