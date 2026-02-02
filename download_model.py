# -*- coding: utf-8 -*-
"""
Script tự động tải model LLM cho chatbot
Tải model GGUF nhỏ gọn từ HuggingFace
"""

import os
import sys
import urllib.request
from pathlib import Path


def download_file(url, output_path):
    """Tải file với progress bar"""
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        bar_length = 50
        filled_length = int(bar_length * percent / 100)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        sys.stdout.write(f'\r|{bar}| {percent:.1f}% ({downloaded/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB)')
        sys.stdout.flush()
    
    print(f"📥 Đang tải: {url}")
    urllib.request.urlretrieve(url, output_path, show_progress)
    print("\n✅ Tải xong!")


def main():
    """Tải model LLM"""
    print("="*60)
    print("🤖 DOWNLOAD MODEL LLM CHO CHATBOT")
    print("="*60)
    
    # Tạo thư mục models
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Chọn model (nhỏ và nhanh)
    print("\n📋 Các model có sẵn:")
    print("1. Qwen2.5-0.5B-Instruct-Q2_K (~200MB) - Nhẹ nhất, phù hợp CPU yếu")
    print("2. Llama-3.2-1B-Instruct-Q4_K_M (~550MB) - Cân bằng, khuyến nghị")
    print("3. Qwen2.5-1.5B-Instruct-Q4_K_M (~900MB) - Tốt hơn nhưng nặng hơn")
    
    choice = input("\nChọn model (1-3) [mặc định: 2]: ").strip() or "2"
    
    models = {
        "1": {
            "name": "qwen2.5-0.5b-instruct-q2_k.gguf",
            "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q2_k.gguf",
            "size": "~200MB"
        },
        "2": {
            "name": "llama-3.2-1b-instruct-q4_k_m.gguf",
            "url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            "size": "~550MB"
        },
        "3": {
            "name": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            "url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
            "size": "~900MB"
        }
    }
    
    if choice not in models:
        print("❌ Lựa chọn không hợp lệ")
        return
    
    model = models[choice]
    output_path = models_dir / model["name"]
    
    # Kiểm tra file đã tồn tại
    if output_path.exists():
        print(f"\n⚠️  File {model['name']} đã tồn tại")
        overwrite = input("Tải lại? (y/n) [n]: ").strip().lower()
        if overwrite != 'y':
            print("✅ Sử dụng file có sẵn")
            return
    
    # Tải model
    print(f"\n🔽 Model: {model['name']}")
    print(f"📦 Kích thước: {model['size']}")
    print(f"💾 Đường dẫn: {output_path.absolute()}")
    print("\n⏳ Bắt đầu tải... (có thể mất vài phút)")
    
    try:
        download_file(model["url"], str(output_path))
        print(f"\n✅ Hoàn tất! Model đã lưu tại: {output_path.absolute()}")
        print("\n💡 Tiếp theo:")
        print("   1. Chạy chatbot: python app.py")
        print("   2. Model sẽ tự động được load nếu ở đúng vị trí")
    except Exception as e:
        print(f"\n❌ Lỗi khi tải model: {e}")
        print("\n💡 Nếu lỗi, bạn có thể:")
        print(f"   1. Tải thủ công từ: {model['url']}")
        print(f"   2. Lưu vào: {output_path.absolute()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Đã hủy tải xuống")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
