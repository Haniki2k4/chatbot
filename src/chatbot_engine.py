"""
Core chatbot engine with BERT Embedding and LLM support
"""

import os
import re
import time
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import logging

# Optional LLM support
try:
    from llama_cpp import Llama
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

logger = logging.getLogger(__name__)


# ==================== PREPROCESSING & CHUNKING ====================

def preprocess_text(text):
    """Normalize text: lowercase, remove HTML/URL/email, special chars"""
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
    """Split text into sentences using punctuation marks"""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[\r\n]+", ". ", text)
    text = re.sub(r"\s+", " ", text)
    sentences = [s.strip() for s in re.split(r"[\.\!\?]+", text) if s.strip()]
    return sentences


def chunk_text(text, chunk_size=3):
    """Group sentences into chunks"""
    sentences = split_sentences(text)
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = ". ".join(sentences[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


# ==================== CHATBOT ENGINE ====================

class DucGiangChatbot:
    """Vietnamese hospital chatbot with RAG and LLM support"""
    
    def __init__(self, config=None):
        """
        Initialize chatbot with configuration
        
        Args:
            config: Configuration object with all settings
        """
        if config is None:
            from config import get_config
            config = get_config()
        
        self.config = config
        self.embedder = None
        self.llm = None
        self.chunks = []
        self.chunk_clean = []
        self.chunk_embeddings = None
        
        logger.info("🔄 Initializing chatbot...")
        self._initialize()
        
        if self.config.USE_LLM:
            self._initialize_llm()
        
        logger.info("✅ Chatbot ready!")
    
    def _initialize_llm(self):
        """Initialize LLM model"""
        if not HAS_LLM:
            logger.warning("⚠️  llama-cpp-python not installed")
            return
        
        llm_path = self.config.LLM_MODEL_PATH
        if not Path(llm_path).exists():
            logger.error(f"❌ LLM model not found: {llm_path}")
            return
        
        try:
            logger.info(f"🤖 Loading LLM: {llm_path}")
            self.llm = Llama(
                model_path=str(llm_path),
                n_ctx=self.config.LLM_N_CTX,
                n_threads=self.config.LLM_N_THREADS,
                n_batch=self.config.LLM_N_BATCH,
                verbose=False
            )
            logger.info("✅ LLM loaded!")
        except Exception as e:
            logger.error(f"❌ Error loading LLM: {e}")
    
    def _initialize(self):
        """Initialize embedding model and load data"""
        logger.info(f"📥 Loading model: {self.config.EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(self.config.EMBEDDING_MODEL)
        
        # Check cache
        if self.config.CACHE_FILE.exists():
            logger.info("📂 Loading from cache...")
            self._load_cache()
        else:
            logger.info("🔨 Building index...")
            self._build_index()
            self._save_cache()
    
    def _load_texts(self):
        """Load all text files from documents directory"""
        texts = []
        docs_path = Path(self.config.DOCUMENTS_DIR)
        
        if not docs_path.exists():
            logger.warning(f"⚠️  Documents directory not found: {docs_path}")
            return texts
        
        txt_files = list(docs_path.glob("*.txt"))
        if not txt_files:
            logger.warning(f"⚠️  No text files in: {docs_path}")
            return texts
        
        for filepath in txt_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():
                        texts.append(content)
            except Exception as e:
                logger.error(f"Error reading {filepath}: {e}")
        
        logger.info(f"📖 Loaded {len(texts)} documents")
        return texts
    
    def _build_index(self):
        """Build embedding index from documents"""
        docs = self._load_texts()
        
        all_chunks = []
        all_chunk_clean = []
        total_sentences = 0
        
        for doc in docs:
            sentences = split_sentences(doc)
            total_sentences += len(sentences)
            chunks = chunk_text(doc, chunk_size=self.config.CHUNK_SIZE)
            
            for ch in chunks:
                all_chunks.append(self._normalize_display(ch))
                all_chunk_clean.append(preprocess_text(ch))
        
        self.chunks = all_chunks
        self.chunk_clean = all_chunk_clean
        
        logger.info(f"📝 Total sentences: {total_sentences}")
        logger.info(f"✂️ Created {len(self.chunks)} chunks")
        
        # Create embeddings
        logger.info("🧮 Creating embeddings...")
        self.chunk_embeddings = self.embedder.encode(
            self.chunk_clean,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        logger.info(f"✅ Embeddings shape: {self.chunk_embeddings.shape}")
    
    def _save_cache(self):
        """Save cache for faster loading"""
        self.config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "chunks": self.chunks,
            "chunk_clean": self.chunk_clean,
            "embeddings": self.chunk_embeddings,
            "preprocess_version": self.config.PREPROCESS_VERSION
        }
        with open(self.config.CACHE_FILE, "wb") as f:
            pickle.dump(cache_data, f)
        logger.info(f"💾 Cache saved to {self.config.CACHE_FILE}")
    
    def _load_cache(self):
        """Load cached embeddings"""
        try:
            with open(self.config.CACHE_FILE, "rb") as f:
                cache_data = pickle.load(f)
            
            if cache_data.get("preprocess_version") != self.config.PREPROCESS_VERSION:
                logger.info("♻️  Cache outdated, rebuilding...")
                self._build_index()
                self._save_cache()
                return
            
            self.chunks = cache_data["chunks"]
            self.chunk_clean = cache_data.get("chunk_clean", [])
            self.chunk_embeddings = cache_data["embeddings"]
            logger.info(f"✅ Loaded {len(self.chunks)} chunks from cache")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            self._build_index()
            self._save_cache()
    
    def _tokenize_for_match(self, text):
        """Tokenize text for keyword matching"""
        return re.findall(r"[\wÀ-ỹ]+", text.lower())
    
    def _normalize_display(self, text):
        """Normalize text for display"""
        text = re.sub(r"\s*-\s*", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    def _calculate_probability_score(self, similarity_scores):
        """Calculate probability using softmax"""
        exp_scores = np.exp(similarity_scores - np.max(similarity_scores))
        probabilities = exp_scores / exp_scores.sum()
        return probabilities
    
    def _generate_llm_response(self, query, contexts):
        """Generate response using LLM"""
        if not self.config.USE_LLM or not self.llm:
            return None
        
        context_text = "\n- ".join(contexts)
        
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
            logger.error(f"⚠️  LLM error: {e}")
            return None
    
    def get_response(self, user_query, top_k=None, return_scores=False):
        """
        Get chatbot response to user query
        
        Args:
            user_query: User question
            top_k: Number of top results (default from config)
            return_scores: Return confidence scores
        
        Returns:
            Response or (response, scores, inference_time)
        """
        inference_start = time.time()
        
        if top_k is None:
            top_k = self.config.TOP_K
        
        if not user_query.strip():
            return "Vui lòng nhập câu hỏi."
        
        # Preprocess query
        query_clean = preprocess_text(user_query)
        
        # Create embedding
        query_embedding = self.embedder.encode(
            query_clean,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Calculate similarity
        similarities = cosine_similarity([query_embedding], self.chunk_embeddings)[0]
        
        # Keyword overlap score
        query_tokens = set(self._tokenize_for_match(query_clean))
        overlap_scores = np.zeros_like(similarities)
        
        if query_tokens:
            for i, ch in enumerate(self.chunk_clean):
                ch_tokens = set(self._tokenize_for_match(ch))
                if ch_tokens:
                    overlap_scores[i] = len(query_tokens & ch_tokens) / max(len(query_tokens), 1)
        
        # Combined scoring
        combined_scores = 0.85 * similarities + 0.15 * overlap_scores
        threshold = self.config.SIMILARITY_THRESHOLD
        
        # Filter by threshold
        valid_indices = np.where(combined_scores >= threshold)[0]
        
        if len(valid_indices) == 0:
            inference_time = time.time() - inference_start
            response = "Xin lỗi, tôi không tìm thấy thông tin phù hợp."
            if return_scores:
                return response, [], inference_time
            return response
        
        # Get top-k results
        valid_scores = combined_scores[valid_indices]
        top_valid_idx = valid_scores.argsort()[-top_k:][::-1]
        top_indices = valid_indices[top_valid_idx]
        
        # Calculate probabilities
        top_scores = combined_scores[top_indices]
        probabilities = self._calculate_probability_score(top_scores)
        
        # Prepare responses
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
        
        # Remove duplicates
        unique_responses = []
        seen = set()
        for resp in responses:
            resp_lower = resp.lower()
            if resp_lower not in seen:
                unique_responses.append(resp)
                seen.add(resp_lower)
        
        # Generate response using LLM if available
        if self.config.USE_LLM and unique_responses and self.llm:
            llm_response = self._generate_llm_response(user_query, unique_responses[:2])
            
            if llm_response:
                final_response = llm_response
            else:
                best_chunk = unique_responses[0]
                extra = unique_responses[1] if len(unique_responses) > 1 else ""
                final_response = best_chunk + (". " + extra if extra else "")
        else:
            if unique_responses:
                best_chunk = unique_responses[0]
                extra = unique_responses[1] if len(unique_responses) > 1 else ""
                final_response = best_chunk + (". " + extra if extra else "")
                final_response = re.sub(r"\s+", " ", final_response).strip()
                if not final_response.endswith(('.', '!', '?')):
                    final_response += "."
                final_response = f"Dựa trên thông tin: {final_response}"
            else:
                final_response = "Xin lỗi, không tìm thấy thông tin phù hợp."
        
        inference_time = time.time() - inference_start
        
        if return_scores:
            return final_response, scores_info, inference_time
        
        return final_response
    
    def get_stats(self):
        """Get chatbot statistics"""
        return {
            "total_chunks": len(self.chunks),
            "embedding_dim": self.chunk_embeddings.shape[1] if self.chunk_embeddings is not None else 0,
            "model": self.config.EMBEDDING_MODEL,
            "llm_enabled": self.config.USE_LLM and self.llm is not None,
            "llm_model": str(self.config.LLM_MODEL_PATH) if self.config.USE_LLM else None
        }
