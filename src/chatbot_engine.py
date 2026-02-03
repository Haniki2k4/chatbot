# -*- coding: utf-8 -*-
"""
Chatbot Engine với BERT Embedding, LLM Local và mô hình xác suất
- Tiền xử lý (preprocessing)
- Chunking (tách nhỏ documents)
- Embedding và semantic search
- LLM Local cho tạo câu trả lời
"""

import os
import logging
import time as time_module
from pathlib import Path
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .utils import (
    preprocess_text, split_sentences, chunk_text,
    tokenize_for_match, normalize_display, 
    calculate_probability_score, ensure_directory_exists
)
from config.settings import CHATBOT_CONFIG, LLM_CONFIG, PREPROCESS_VERSION


logger = logging.getLogger(__name__)

# Import LLM local (optional)
try:
    from llama_cpp import Llama
    HAS_LLM = True
except ImportError:
    HAS_LLM = False
    logger.warning("llama-cpp-python chưa được cài đặt. Chatbot sẽ chỉ dùng BERT embeddings.")


class DucGiangChatbot:
    """Chatbot Engine cho Bệnh viện Đức Giang"""
    
    def __init__(self, config=None):
        """
        Khởi tạo chatbot với cấu hình
        
        Args:
            config: Dictionary cấu hình (nếu None, dùng mặc định từ settings)
        """
        self.config = config or CHATBOT_CONFIG.copy()
        self.embedder = None
        self.llm = None
        self.chunks = []
        self.chunk_clean = []
        self.chunk_embeddings = None
        self.cache_file = self.config.get("cache_file")
        
        logger.info("Đang khởi tạo chatbot...")
        self._initialize()
        self._initialize_llm()
        logger.info("Chatbot đã sẵn sàng.")
    
    def _initialize_llm(self):
        """Khởi tạo LLM local"""
        use_llm = bool(self.config.get("use_llm", True))
        if not use_llm:
            logger.info("Chạy chế độ không dùng LLM (BERT embeddings only)")
            self.llm = None
            return

        if not HAS_LLM:
            logger.warning("llama-cpp-python chưa được cài đặt. Tắt LLM và dùng BERT embeddings.")
            self.llm = None
            self.config["use_llm"] = False
            return
        
        llm_path = self.config.get("llm_model_path")
        if not llm_path or not os.path.exists(llm_path):
            logger.warning(f"Không tìm thấy model LLM tại: {llm_path}. Tắt LLM và dùng BERT embeddings.")
            self.llm = None
            self.config["use_llm"] = False
            return
        
        try:
            logger.info(f"Đang load LLM từ: {llm_path}")
            self.llm = Llama(
                model_path=llm_path,
                **{k: v for k, v in LLM_CONFIG.items() if k not in ['temperature', 'top_p', 'max_tokens']}
            )
            logger.info("LLM đã sẵn sàng.")
        except Exception as e:
            raise RuntimeError(f"Lỗi khi load LLM: {e}")
    
    def _initialize(self):
        """Khởi tạo model và load dữ liệu"""
        # Nạp model embedding
        model_name = self.config.get("embedding_model")
        logger.info(f"Đang load model: {model_name}")
        self.embedder = SentenceTransformer(model_name)
        
        # Kiểm tra cache
        if os.path.exists(self.cache_file):
            logger.info("Tìm thấy cache, đang load...")
            self._load_cache()
        else:
            logger.info("Không tìm thấy cache, đang xây dựng index...")
            self._build_index()
            self._save_cache()
    
    def _load_texts(self):
        """Đọc tất cả file txt từ thư mục"""
        texts = []
        data_path = Path(self.config.get("data_folder"))
        
        if not data_path.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục: {data_path}")
        
        txt_files = list(data_path.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"Không có file txt trong thư mục: {data_path}")
        
        for filepath in sorted(txt_files):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    texts.append(content)
        
        logger.info(f"Đã load {len(texts)} file txt")
        return texts
    
    def _build_index(self):
        """Xây dựng index từ dữ liệu"""
        docs = self._load_texts()
        
        all_chunks = []
        all_chunk_clean = []
        total_sentences = 0
        chunk_size = self.config.get("chunk_size", 3)
        
        for doc in docs:
            sentences = split_sentences(doc)
            total_sentences += len(sentences)
            chunks = chunk_text(doc, chunk_size=chunk_size)
            for ch in chunks:
                all_chunks.append(normalize_display(ch))
                all_chunk_clean.append(preprocess_text(ch))
        
        self.chunks = all_chunks
        self.chunk_clean = all_chunk_clean
        logger.info(f"Tổng số câu: {total_sentences}")
        logger.info(f"Đã tạo {len(self.chunks)} chunks")
        
        # Tạo embeddings
        logger.info("Đang tạo embeddings...")
        self.chunk_embeddings = self.embedder.encode(
            self.chunk_clean,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        logger.info(f"Đã tạo embeddings: {self.chunk_embeddings.shape}")
    
    def _save_cache(self):
        """Lưu cache để tăng tốc lần sau"""
        ensure_directory_exists(os.path.dirname(self.cache_file))
        cache_data = {
            "chunks": self.chunks,
            "chunk_clean": self.chunk_clean,
            "embeddings": self.chunk_embeddings,
            "preprocess_version": PREPROCESS_VERSION
        }
        with open(self.cache_file, "wb") as f:
            pickle.dump(cache_data, f)
        logger.info(f"Đã lưu cache vào {self.cache_file}")
    
    def _load_cache(self):
        """Load cache đã lưu"""
        with open(self.cache_file, "rb") as f:
            cache_data = pickle.load(f)
        
        if cache_data.get("preprocess_version") != PREPROCESS_VERSION:
            logger.warning("Cache cũ không còn phù hợp, đang xây dựng lại index...")
            self._build_index()
            self._save_cache()
            return
        
        self.chunks = cache_data["chunks"]
        self.chunk_clean = cache_data.get("chunk_clean", [])
        self.chunk_embeddings = cache_data["embeddings"]
        logger.info(f"Đã load {len(self.chunks)} chunks từ cache")
    
    def get_response(self, user_query, top_k=None, return_scores=False):
        """
        Lấy câu trả lời cho câu hỏi của user
        
        Args:
            user_query: Câu hỏi từ người dùng
            top_k: Số lượng chunks tốt nhất cần lấy (mặc định từ config)
            return_scores: Có trả về scores không
        
        Returns:
            Câu trả lời hoặc (câu trả lời, scores, inference_time)
        """
        inference_start_time = time_module.time()
        
        if not user_query.strip():
            return "Vui lòng nhập câu hỏi."
        
        top_k = top_k or self.config.get("top_k", 5)
        threshold = self.config.get("similarity_threshold", 0.3)
        
        # Chuẩn hóa query
        query_clean = preprocess_text(user_query)
        
        # Tạo embedding cho query
        query_embedding = self.embedder.encode(
            query_clean,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Tính cosine similarity
        similarities = cosine_similarity([query_embedding], self.chunk_embeddings)[0]
        
        # Tính overlap score
        query_tokens = set(tokenize_for_match(query_clean))
        overlap_scores = np.zeros_like(similarities)
        if query_tokens:
            for i, ch in enumerate(self.chunk_clean):
                ch_tokens = set(tokenize_for_match(ch))
                if ch_tokens:
                    overlap_scores[i] = len(query_tokens & ch_tokens) / max(len(query_tokens), 1)
        
        # Combined score
        combined_scores = 0.85 * similarities + 0.15 * overlap_scores
        
        # Lọc theo threshold
        valid_indices = np.where(combined_scores >= threshold)[0]
        
        if len(valid_indices) == 0:
            inference_time = time_module.time() - inference_start_time
            response = "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong cơ sở dữ liệu. Bạn có thể hỏi về thông tin liên quan đến Bệnh viện Đức Giang."
            if return_scores:
                return response, [], inference_time
            return response
        
        # Lấy top_k best matches
        valid_scores = combined_scores[valid_indices]
        if len(valid_scores) > 0:
            top_valid_idx = valid_scores.argsort()[-top_k:][::-1]
            top_indices = valid_indices[top_valid_idx]
        else:
            top_indices = combined_scores.argsort()[-top_k:][::-1]
        
        # Tính xác suất
        top_scores = combined_scores[top_indices]
        probabilities = calculate_probability_score(top_scores)
        
        # Lấy responses và scores
        responses = []
        scores_info = []
        
        for idx, (chunk_idx, prob) in enumerate(zip(top_indices, probabilities)):
            if combined_scores[chunk_idx] >= threshold:
                responses.append(self.chunks[chunk_idx])
                scores_info.append({
                    "rank": idx + 1,
                    "similarity": float(combined_scores[chunk_idx]),
                    "probability": float(prob),
                    "text": self.chunks[chunk_idx]
                })
        
        # Xóa trùng lặp
        unique_responses = []
        seen = set()
        for resp in responses:
            resp_lower = resp.lower()
            if resp_lower not in seen:
                unique_responses.append(resp)
                seen.add(resp_lower)
        
        # Sinh câu trả lời
        if self.llm and unique_responses:
            max_contexts = int(self.config.get("max_contexts", 2))
            final_response = self._generate_llm_response(user_query, unique_responses[:max_contexts])
            if not final_response:
                final_response = self._combine_responses(unique_responses)
        else:
            final_response = self._combine_responses(unique_responses) if unique_responses else "Xin lỗi, tôi không tìm thấy thông tin phù hợp."
        
        inference_time = time_module.time() - inference_start_time
        
        if return_scores:
            return final_response, scores_info, inference_time
        return final_response
    
    def _combine_responses(self, responses):
        """Kết hợp các responses thành câu trả lời"""
        best_chunk = responses[0]
        extra_chunk = responses[1] if len(responses) > 1 else ""
        combined = best_chunk + (". " + extra_chunk if extra_chunk else "")
        combined = combined.replace(r"\s+", " ").strip()
        if not combined.endswith(('.', '!', '?')):
            combined += "."
        return f"Dựa trên thông tin tìm được, {combined}"
    
    def _generate_llm_response(self, query, contexts):
        """Sinh câu trả lời từ LLM"""
        if not self.llm:
            return None
        
        context_text = "\n- ".join(contexts)
        prompt = f"""<|im_start|>system
Bạn là trợ lý ảo của Bệnh viện Đức Giang.
    Ưu tiên trả lời ngắn gọn nhưng đầy đủ ý. Khi cần, có thể trả lời dài để đủ thông tin.
Chỉ dùng thông tin trong ngữ cảnh. Nếu không đủ thông tin, nói rõ là chưa tìm thấy.
Ưu tiên tiếng Việt, tránh suy đoán.
<|im_end|>
<|im_start|>user
Thông tin tham khảo:
{context_text}

Câu hỏi: {query}
<|im_end|>
<|im_start|>assistant
"""
        
        try:
            output = self.llm(
                prompt,
                max_tokens=LLM_CONFIG.get("max_tokens", 200),
                temperature=LLM_CONFIG.get("temperature", 0.3),
                top_p=LLM_CONFIG.get("top_p", 0.9),
                stop=["<|im_end|>", "\n\n"],
                echo=False
            )
            response = output["choices"][0]["text"].strip()
            return response if response else None
        except Exception as e:
            logger.warning(f"Lỗi khi sinh câu trả lời từ LLM: {e}")
            return None
    
    def get_stats(self):
        """Lấy thống kê về chatbot"""
        return {
            "total_chunks": len(self.chunks),
            "embedding_dim": self.chunk_embeddings.shape[1] if self.chunk_embeddings is not None else 0,
            "model": self.config.get("embedding_model"),
            "llm_enabled": self.llm is not None,
            "llm_model": self.config.get("llm_model_path") if self.llm else None
        }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    bot = DucGiangChatbot()
    print("\n" + "="*50)
    print("CHATBOT BỆNH VIỆN ĐỨC GIANG")
    print("="*50)
    print("Gõ 'exit' hoặc 'quit' để thoát\n")
    
    while True:
        user_input = input("Bạn: ").strip()
        if user_input.lower() in ["exit", "quit", "bye", "thoát"]:
            print("Bot: Cảm ơn bạn! Tạm biệt!")
            break
        if not user_input:
            continue
        
        response, scores, inference_time = bot.get_response(user_input, return_scores=True)
        print(f"Bot: {response}\nThời gian suy luận: {inference_time:.2f}s\n")
