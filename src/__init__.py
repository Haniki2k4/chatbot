"""
Chatbot source code package
"""

from .chatbot_engine import DucGiangChatbot, preprocess_text, chunk_text

__version__ = "0.1.12"
__all__ = ["DucGiangChatbot", "preprocess_text", "chunk_text"]
