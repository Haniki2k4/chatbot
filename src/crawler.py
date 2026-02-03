import requests
from bs4 import BeautifulSoup
import os
import time
import re
import hashlib
import nltk
from nltk.tokenize import sent_tokenize

# Tải bộ tách câu nếu chưa có
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")

# Cấu hình nguồn crawl
BASE_URL = "https://benhvienducgiang.com"
START_URL = "https://benhvienducgiang.com"
INTRO_URLS = [
    "https://benhvienducgiang.com/gioi-thieu/102-300.aspx",
    "https://benhvienducgiang.com/gioi-thieu-chung/102-648.aspx",
    "https://benhvienducgiang.com/chuc-nang-va-nhiem-vu/102-650.aspx",
]
OUTPUT_DIR = "duc_giang_txt"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Theo dõi URL đã crawl để tránh lặp
VISITED_URLS = set()
# Theo dõi nội dung trùng (hash) để tránh lưu lặp
CONTENT_HASHES = set()


def normalize_url(url):
    """Xóa query params và fragments"""
    url = url.split("?")[0].split("#")[0]
    return url.rstrip("/")


def get_content_hash(text):
    """MD5 hash 500 ký tự đầu để detect duplicate"""
    text_normalized = text.lower().strip()[:500]
    return hashlib.md5(text_normalized.encode()).hexdigest()


def get_page_text(url):
    """Lấy text từ <p> và div.content"""
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        # Lấy nội dung từ các thẻ phổ biến
        paragraphs = soup.find_all("p")
        divs = soup.find_all("div", class_=["content", "article-content", "post-content"])
        text_parts = []
        text_parts.extend([p.get_text(" ", strip=True) for p in paragraphs])
        text_parts.extend([d.get_text(" ", strip=True) for d in divs])
        return " ".join(text_parts)
    except Exception as e:
        print(f" X |  Lỗi khi crawl {url}: {str(e)}")
        return ""


def extract_article_links_from_page(url):
    """Trích link bài viết từ trang listing"""
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        article_links = set()
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            # Bỏ link không hợp lệ và quảng cáo
            if href.startswith("#") or href.startswith("javascript"):
                continue
            if "pk_advertisement" in href:
                continue
            # Chuẩn hóa link nội bộ
            if href.startswith("/"):
                href = BASE_URL + href
            elif not href.startswith("http"):
                continue
            href = normalize_url(href)
            # Lọc link ngoài và link đã crawl
            if BASE_URL not in href:
                continue
            if href in VISITED_URLS:
                continue
            # Bỏ các trang không cần thiết
            if any(skip in href.lower() for skip in ["?", "search", "trang-chu", "404", "login"]):
                continue
            article_links.add(href)
        return article_links
    except Exception as e:
        print(f" X |   Lỗi khi lấy link bài báo từ {url}: {str(e)}")
        return set()


def extract_links_from_page(url, keyword_filter=None):
    """Trích tất cả link từ trang"""
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        links = set()
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            # Bỏ link không hợp lệ và quảng cáo
            if href.startswith("#") or href.startswith("javascript"):
                continue
            if "pk_advertisement" in href:
                continue
            # Chuẩn hóa link nội bộ
            if href.startswith("/"):
                href = BASE_URL + href
            elif not href.startswith("http"):
                continue
            href = normalize_url(href)
            # Lọc theo keyword nếu có
            if keyword_filter and keyword_filter not in href:
                continue
            # Bỏ link đã crawl
            if href in VISITED_URLS:
                continue
            if BASE_URL in href:
                links.add(href)
        return links
    except Exception as e:
        print(f" X |   Lỗi khi lấy link từ {url}: {str(e)}")
        return set()


def get_intro_links(max_depth=2, save_text=False, file_counter=0):
    """Crawl trang giới thiệu (depth-first), save text nếu cần"""
    print(" Đang tìm kiếm các link giới thiệu...")
    # Duyệt theo hàng đợi và giới hạn độ sâu
    all_links = set()
    visited = set()
    queue = list(INTRO_URLS)
    depth = {url: 0 for url in INTRO_URLS}
    saved_count = file_counter
    while queue:
        url = queue.pop(0)
        url = normalize_url(url)
        # Bỏ link đã xử lý
        if url in visited or url in VISITED_URLS:
            continue
        # Dừng khi vượt độ sâu cho phép
        if depth.get(url, 0) >= max_depth:
            continue
        visited.add(url)
        VISITED_URLS.add(url)
        print(f"  Crawling: {url}")
        text = get_page_text(url)
        # Lưu nội dung đủ dài và chưa trùng
        if len(text) > 300:
            content_hash = get_content_hash(text)
            if content_hash not in CONTENT_HASHES:
                CONTENT_HASHES.add(content_hash)
                all_links.add(url)
                if save_text:
                    file_path = os.path.join(OUTPUT_DIR, f"duc_giang_{saved_count+1}.txt")
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(text)
                    print(f"  Đã lưu: {file_path}")
                    saved_count += 1
            else:
                print("   X |   Nội dung trùng lặp, bỏ qua")
        # Lấy link bài viết và link nội bộ để crawl tiếp
        article_links = extract_article_links_from_page(url)
        links = extract_links_from_page(url)
        for link in links:
            if link not in visited and link not in VISITED_URLS and depth.get(url, 0) < max_depth:
                queue.append(link)
                depth[link] = depth.get(url, 0) + 1
    print(f" Tìm thấy {len(all_links)} links giới thiệu")
    if save_text:
        print(f" Đã lưu {saved_count - file_counter} files")
    return list(all_links), article_links, saved_count


def get_internal_links(save_text=False, file_counter=0):
    """Crawl trang chủ, save homepage text nếu cần"""
    print(" Đang tìm kiếm các link từ trang chủ...")
    start = normalize_url(START_URL)
    saved_count = file_counter
    # Lưu nội dung trang chủ nếu cần
    if start not in VISITED_URLS:
        VISITED_URLS.add(start)
        if save_text:
            text = get_page_text(start)
            if len(text) > 300:
                content_hash = get_content_hash(text)
                if content_hash not in CONTENT_HASHES:
                    CONTENT_HASHES.add(content_hash)
                    file_path = os.path.join(OUTPUT_DIR, f"duc_giang_{saved_count+1}.txt")
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(text)
                    print(f"  Đã lưu trang chủ: {file_path}")
                    saved_count += 1
    links = extract_links_from_page(start)
    article_links = extract_article_links_from_page(start)
    print(f" Tìm thấy {len(links)} links từ trang chủ")
    return list(links), article_links, saved_count


def crawl_and_save(min_files):
    """Crawl và lưu dữ liệu raw text từ website"""
    # Tạo thư mục lưu dữ liệu và reset trạng thái
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    VISITED_URLS.clear()
    CONTENT_HASHES.clear()

    count = 0
    
    print("\n" + "="*60)
    print("Đang crawl các trang giới thiệu")
    print("="*60)
    intro_links, intro_articles, count = get_intro_links(max_depth=2, save_text=True, file_counter=count)
    
    print("\n" + "="*60)
    print("Đang crawl trang chủ")
    print("="*60)
    site_links, site_articles, count = get_internal_links(save_text=True, file_counter=count)

    # Gộp link và loại trùng
    all_links = []
    seen = set()
    for link in intro_links + site_links:
        if link not in seen and link not in VISITED_URLS:
            all_links.append(link)
            seen.add(link)

    # Crawl thêm nếu chưa đủ số file
    if count < min_files and all_links:
        print(f"\n" + "="*60)
        print("Đang crawl các link còn lại")
        print("="*60)
        print(f" Bắt đầu crawl {len(all_links)} links còn lại...")
        
        for i, link in enumerate(all_links, 1):
            if count >= min_files:
                break
            
            if link in VISITED_URLS:
                continue
                
            print(f"[{i}/{len(all_links)}] Đang crawl: {link}")
            text = get_page_text(link)

            # Bỏ nội dung quá ngắn
            if len(text) < 100:
                print(" X | Text quá ngắn (<100 ký tự), bỏ qua")
                VISITED_URLS.add(link)
                continue

            # Bỏ nội dung trùng
            content_hash = get_content_hash(text)
            if content_hash in CONTENT_HASHES:
                print(" X | Nội dung trùng lặp, bỏ qua")
                VISITED_URLS.add(link)
                continue
            
            CONTENT_HASHES.add(content_hash)

            file_path = os.path.join(OUTPUT_DIR, f"duc_giang_{count+1}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f" Đã lưu: {file_path}")
            VISITED_URLS.add(link)
            count += 1

            time.sleep(0.5)

    print(f"\n" + "="*70)
    print(f" Hoàn tất crawling!")
    print("="*70)
    print(f"- Tổng số file: {count}")
    print(f"- Từ giới thiệu: {len(intro_links)}")
    print(f"- Links bổ sung: {len(all_links)}")
    print(f"- URLs đã visit: {len(VISITED_URLS)}")
    print(f"- Content hashes: {len(CONTENT_HASHES)}")
    print("="*70)
    
    return count


def load_texts(folder):
    """Load raw text từ folder"""
    # Đọc toàn bộ file .txt theo thứ tự
    texts = []
    for file in sorted(os.listdir(folder)):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), encoding="utf-8") as f:
                texts.append(f.read())
    return texts


def verify_data():
    """Kiểm tra số lượng file crawled"""
    # Kiểm tra thư mục dữ liệu
    if not os.path.exists(OUTPUT_DIR):
        print(f" X |  Thư mục {OUTPUT_DIR} không tồn tại")
        return False
    # Lấy danh sách file txt
    txt_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.txt')]
    if not txt_files:
        print(f" X |  Không có file txt trong {OUTPUT_DIR}")
        return False
    print(f"\n Thống kê dữ liệu:")
    print(f"  - Số file: {len(txt_files)}")
    # Tính thống kê ký tự
    texts = load_texts(OUTPUT_DIR)
    total_chars = sum(len(t) for t in texts)
    print(f"  - Tổng số ký tự: {total_chars:,}")
    print(f"  - Trung bình: {total_chars // len(txt_files):,} ký tự/file")
    if len(txt_files) < 10:
        print(f"\n X |   CẢNH BÁO: Số file .txt ({len(txt_files)}) < 10!")
        return False
    return True


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("="*70)
    print("--- CRAWLER BỆNH VIỆN ĐỨC GIANG - V3 ---")
    
    num_files = crawl_and_save(min_files=25)
    
    if num_files > 0:
        print("\n" + "="*70)
        is_valid = verify_data()
        print("="*70)
        
        if is_valid:
            print("\nDữ liệu đã sẵn sàng!")
            print(f"- Thư mục: {os.path.abspath(OUTPUT_DIR)}")
            print(f"- URLs crawled: {len(VISITED_URLS)}")
            print(f"- Unique content: {len(CONTENT_HASHES)}")
        else:
            print("\n X |   Dữ liệu chưa đủ yêu cầu!")
            print(" Chạy lại crawler hoặc tăng min_files để crawl thêm")
    else:
        print("\n X |  Không crawl được dữ liệu. Vui lòng kiểm tra lại.")