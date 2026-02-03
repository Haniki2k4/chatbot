# -*- coding: utf-8 -*-
"""
Chatbot Engine với BERT Embedding, LLM Local và mô hình xác suất
- Tiền xử lý (preprocessing)
- Chunking (tách nhỏ documents)
- Embedding và semantic search
- LLM Local cho tạo câu trả lời (tùy chọn)
"""

import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path

# Import LLM local (optional)
try:
    from llama_cpp import Llama
    HAS_LLM = True
except ImportError:
    HAS_LLM = False
    print("⚠️  llama-cpp-python chưa được cài đặt. Chatbot sẽ chỉ sử dụng BERT embeddings.")


# ==================== PREPROCESSING & CHUNKING ====================

PREPROCESS_VERSION = "v4"

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


class DucGiangChatbot:
    def __init__(self, data_folder="duc_giang_txt", model_name="sentence-transformers/all-MiniLM-L6-v2", 
                 llm_model_path="E:/NLP/het-mon/chatbot/models/qwen2.5-1.5b-instruct-q4_k_m.gguf", use_llm=True):
        """
        Khởi tạo chatbot với BERT embeddings và LLM local 
        
        Args:
            data_folder: Thư mục chứa các file txt đã crawl
            model_name: Tên model BERT để embedding
            llm_model_path: Đường dẫn tới model GGUF (VD: "models/llama-3.2-1b-instruct-q4_k_m.gguf")
            use_llm: Có sử dụng LLM để sinh câu trả lời không (mặc định False)
        """
        self.data_folder = data_folder
        self.model_name = model_name
        self.llm_model_path = llm_model_path
        self.use_llm = use_llm
        self.embedder = None
        self.llm = None
        self.chunks = []
        self.chunk_clean = []
        self.chunk_embeddings = None
        self.cache_file = "chatbot_cache.pkl"
        
        print("🔄 Đang khởi tạo chatbot...")
        self._initialize()
        
        # Luôn khởi tạo LLM local
        self._initialize_llm()
        
        print("✅ Chatbot đã sẵn sàng!")
    
    def _initialize_llm(self):
        """Khởi tạo LLM local"""
        if not HAS_LLM:
            raise RuntimeError("llama-cpp-python chưa được cài đặt")
        
        if not self.llm_model_path or not os.path.exists(self.llm_model_path):
            raise RuntimeError(f"Không tìm thấy model LLM tại: {self.llm_model_path}")
        
        try:
            print(f"🤖 Đang load LLM từ: {self.llm_model_path}")
            self.llm = Llama(
                model_path=self.llm_model_path,
                n_ctx=512,  # Context window
                n_threads=max(1, os.cpu_count() // 2),
                n_batch=64,
                verbose=False
            )
            print("✅ LLM đã sẵn sàng!")
        except Exception as e:
            raise RuntimeError(f"Lỗi khi load LLM: {e}")
    
    def _initialize(self):
        """Khởi tạo model và load dữ liệu"""
        # Load BERT model
        print(f"📥 Đang load model: {self.model_name}")
        self.embedder = SentenceTransformer(self.model_name)
        
        # Kiểm tra cache
        if os.path.exists(self.cache_file):
            print("📂 Tìm thấy cache, đang load...")
            self._load_cache()
        else:
            print("🔨 Không tìm thấy cache, đang xây dựng index...")
            self._build_index()
            self._save_cache()
    
    def _load_texts(self):
        """Đọc tất cả file txt từ thư mục"""
        texts = []
        data_path = Path(self.data_folder)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục: {self.data_folder}")
        
        txt_files = list(data_path.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"Không có file txt trong thư mục: {self.data_folder}")
        
        for filepath in txt_files:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    texts.append(content)
        
        print(f"📖 Đã load {len(texts)} file txt")
        return texts
    
    def _build_index(self):
        """Xây dựng index từ dữ liệu"""
        # Load và preprocess texts
        docs = self._load_texts()
        
        # Tạo chunks với preprocessing
        all_chunks = []
        all_chunk_clean = []
        total_sentences = 0
        for doc in docs:
            # **Preprocessing + Chunking được thực hiện ở đây**
            sentences = split_sentences(doc)
            total_sentences += len(sentences)
            chunks = chunk_text(doc, chunk_size=3)
            for ch in chunks:
                all_chunks.append(self._normalize_display(ch))
                all_chunk_clean.append(preprocess_text(ch))
        
        self.chunks = all_chunks
        self.chunk_clean = all_chunk_clean
        print(f"📝 Tổng số câu: {total_sentences}")
        print(f"✂️ Đã tạo {len(self.chunks)} chunks")
        
        # Tạo embeddings
        print("🧮 Đang tạo embeddings...")
        self.chunk_embeddings = self.embedder.encode(
            self.chunk_clean,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        print(f"✅ Đã tạo embeddings: {self.chunk_embeddings.shape}")
    
    def _save_cache(self):
        """Lưu cache để tăng tốc lần sau"""
        cache_data = {
            "chunks": self.chunks,
            "chunk_clean": self.chunk_clean,
            "embeddings": self.chunk_embeddings,
            "preprocess_version": PREPROCESS_VERSION
        }
        with open(self.cache_file, "wb") as f:
            pickle.dump(cache_data, f)
        print(f"💾 Đã lưu cache vào {self.cache_file}")
    
    def _load_cache(self):
        """Load cache đã lưu"""
        with open(self.cache_file, "rb") as f:
            cache_data = pickle.load(f)
        if cache_data.get("preprocess_version") != PREPROCESS_VERSION:
            print("♻️ Cache cũ không còn phù hợp, đang xây dựng lại index...")
            self._build_index()
            self._save_cache()
            return
        self.chunks = cache_data["chunks"]
        self.chunk_clean = cache_data.get("chunk_clean", [])
        self.chunk_embeddings = cache_data["embeddings"]
        print(f"✅ Đã load {len(self.chunks)} chunks từ cache")
    
    def _calculate_threshold(self, query):
        """Ngưỡng similarity cố định"""
        return 0.3
    
    def _calculate_probability_score(self, similarity_scores):
        """
        Tính xác suất dựa trên similarity scores
        Sử dụng softmax để chuyển scores thành phân phối xác suất
        """
        # Softmax normalization
        exp_scores = np.exp(similarity_scores - np.max(similarity_scores))
        probabilities = exp_scores / exp_scores.sum()
        return probabilities

    def _tokenize_for_match(self, text):
        return re.findall(r"[\wÀ-ỹ]+", text.lower())

    def _normalize_display(self, text):
        text = re.sub(r"\s*-\s*", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    def _generate_llm_response(self, query, contexts):
        """
        Sinh câu trả lời từ LLM dựa trên contexts
        
        Args:
            query: Câu hỏi người dùng
            contexts: Danh sách các chunks có liên quan
            
        Returns:
            Câu trả lời từ LLM
        """
        if not self.use_llm or not self.llm:
            return None
        
        # Tạo context text
        context_text = "\n- ".join(contexts)
        
        # Tạo prompt theo định dạng Qwen/Llama
        prompt = f"""<|im_start|>system
Bạn là trợ lý ảo của Bệnh viện Đức Giang.
Trả lời ngắn gọn, tự nhiên, đúng trọng tâm.
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
            # Generate response
            output = self.llm(
                prompt,
                max_tokens=200,
                temperature=0.3,
                top_p=0.9,
                stop=["<|im_end|>", "\n\n"],
                echo=False
            )
            
            response = output["choices"][0]["text"].strip()
            return response if response else None
        except Exception as e:
            print(f"⚠️  Lỗi khi sinh câu trả lời từ LLM: {e}")
            return None
    
    def get_response(self, user_query, top_k=5, return_scores=False):
        """
        Lấy câu trả lời cho câu hỏi của user
        
        Args:
            user_query: Câu hỏi từ người dùng
            top_k: Số lượng chunks tốt nhất cần lấy
            return_scores: Có trả về scores không
        
        Returns:
            Câu trả lời hoặc (câu trả lời, scores, inference_time) 
        """
        import time as time_module
        inference_start_time = time_module.time()
        
        if not user_query.strip():
            return "Vui lòng nhập câu hỏi."
        
        # **Preprocessing: Chuẩn hóa câu hỏi**
        query_clean = preprocess_text(user_query)
        
        # Tạo embedding cho query
        query_embedding = self.embedder.encode(
            query_clean,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Tính cosine similarity
        similarities = cosine_similarity(
            [query_embedding],
            self.chunk_embeddings
        )[0]

        # Điểm overlap theo từ khóa
        query_tokens = set(self._tokenize_for_match(query_clean))
        overlap_scores = np.zeros_like(similarities)
        if query_tokens:
            for i, ch in enumerate(self.chunk_clean):
                ch_tokens = set(self._tokenize_for_match(ch))
                if ch_tokens:
                    overlap_scores[i] = len(query_tokens & ch_tokens) / max(len(query_tokens), 1)

        combined_scores = 0.85 * similarities + 0.15 * overlap_scores
        
        # Tính ngưỡng động
        threshold = self._calculate_threshold(query_clean)
        
        # Lọc theo ngưỡng
        valid_indices = np.where(combined_scores >= threshold)[0]
        
        if len(valid_indices) == 0:
            inference_time = time_module.time() - inference_start_time
            response = "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong cơ sở dữ liệu. Bạn có thể hỏi về thông tin liên quan đến Bệnh viện Đức Giang."
            if return_scores:
                return response, [], inference_time
            return response
        
        # Lấy top_k best matches trong các kết quả đạt ngưỡng (cố định top_k=5)
        top_k = 5
        valid_scores = combined_scores[valid_indices]
        if len(valid_scores) > 0:
            top_valid_idx = valid_scores.argsort()[-top_k:][::-1]
            top_indices = valid_indices[top_valid_idx]
        else:
            top_indices = combined_scores.argsort()[-top_k:][::-1]
        
        # Tính xác suất
        top_scores = combined_scores[top_indices]
        probabilities = self._calculate_probability_score(top_scores)
        
        # Lấy các chunks tương ứng
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
        
        # Loại bỏ trùng lặp
        unique_responses = []
        seen = set()
        for resp in responses:
            resp_lower = resp.lower()
            if resp_lower not in seen:
                unique_responses.append(resp)
                seen.add(resp_lower)
        
        # Nếu có LLM, dùng LLM để sinh câu trả lời
        if self.use_llm and unique_responses:
            llm_response = self._generate_llm_response(user_query, unique_responses[:2])
            
            if llm_response:
                final_response = llm_response
            else:
                # Fallback nếu LLM fail
                best_chunk = unique_responses[0]
                if len(unique_responses) > 1:
                    final_response = best_chunk + " " + unique_responses[1]
                else:
                    final_response = best_chunk
        else:
            # Không dùng LLM: kết hợp thành câu trả lời ngắn gọn
            if unique_responses:
                best_chunk = unique_responses[0]
                extra_chunk = unique_responses[1] if len(unique_responses) > 1 else ""
                combined = best_chunk + (". " + extra_chunk if extra_chunk else "")
                combined = re.sub(r"\s+", " ", combined).strip()
                if not combined.endswith(('.', '!', '?')):
                    combined += "."
                final_response = f"Dựa trên thông tin tìm được, {combined}"
            else:
                final_response = "Xin lỗi, tôi không tìm thấy thông tin phù hợp."
        
        if return_scores:
            inference_time = time_module.time() - inference_start_time
            return final_response, scores_info, inference_time
        
        return final_response
    
    def get_stats(self):
        """Lấy thống kê về chatbot"""
        return {
            "total_chunks": len(self.chunks),
            "embedding_dim": self.chunk_embeddings.shape[1] if self.chunk_embeddings is not None else 0,
            "model": self.model_name,
            "llm_enabled": self.use_llm,
            "llm_model": self.llm_model_path if self.use_llm else None
        }


# Test nếu chạy trực tiếp
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    bot = DucGiangChatbot()
    
    print("\n" + "="*50)
    print("CHATBOT BỆnh VIỆN ĐỨC GIANG")
    print("="*50)
    print("Gõ 'exit', 'quit' hoặc 'bye' để thoát\n")
    
    while True:
        user_input = input("Bạn: ").strip()
        
        if user_input.lower() in ["exit", "quit", "bye", "thoát"]:
            print("Bot: Cảm ơn bạn đã sử dụng dịch vụ. Tạm biệt!")
            break
        
        if not user_input:
            continue
        
        response, scores = bot.get_response(user_input, return_scores=True)
        print(f"Bot: {response}")
        
        if scores:
            print("\n📊 Thông tin chi tiết:")
            for score_info in scores[:2]:
                print(f"  - Độ tương đồng: {score_info['similarity']:.3f}")
                print(f"  - Xác suất: {score_info['probability']:.3f}")
        print()
