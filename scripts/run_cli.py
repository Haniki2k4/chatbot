# -*- coding: utf-8 -*-
"""Setup and run the chatbot application"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chatbot_engine import DucGiangChatbot


def main():
    """Run chatbot in CLI mode"""
    import locale
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    
    print("\n" + "="*60)
    print("CHATBOT BỆNH VIỆN ĐỨC GIANG")
    print("="*60)
    print("Gõ 'exit' hoặc 'quit' để thoát\n")
    
    bot = DucGiangChatbot()
    
    while True:
        try:
            user_input = input("Bạn: ").strip()
            
            if user_input.lower() in ["exit", "quit", "bye", "thoát"]:
                print("\nBot: Cảm ơn bạn đã sử dụng dịch vụ. Tạm biệt!")
                break
            
            if not user_input:
                continue
            
            response, scores, inference_time = bot.get_response(user_input, return_scores=True)
            print(f"\nBot: {response}")
            print(f"Thời gian suy luận: {inference_time*1000:.0f}ms")
            
            if scores:
                print("Thông tin chi tiết:")
                for score_info in scores[:2]:
                    print(f"  - Độ tương đồng: {score_info['similarity']:.3f} ({score_info['similarity']*100:.1f}%)")
                    print(f"  - Xác suất: {score_info['probability']:.3f} ({score_info['probability']*100:.1f}%)")
            print()
        
        except KeyboardInterrupt:
            print("\n\nBot: Tạm biệt!")
            break
        except Exception as e:
            print(f"\nLỗi: {e}\n")


if __name__ == "__main__":
    main()
