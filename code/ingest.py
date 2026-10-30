import os
import json
import asyncio
import time
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Импорты для краулера (crawl4ai)
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# ==========================================
# 1. НАСТРОЙКИ
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = "./documents"
URLS_FILE = os.path.join(CURRENT_DIR, "urls.txt")
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "my_docs"
EMBEDDING_MODEL = "bge-m3"

# ==========================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================

def load_json_file(file_path):
    """Читает JSON файлы (поддержка графа и обычных списков)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        docs = []
        if isinstance(data, dict) and "nodes" in data:
            print(f"  [JSON] Формат графа. Узлов: {len(data['nodes'])}")
            for node in data["nodes"]:
                url = node.get("url", "")
                text_content = node.get("text", "") or node.get("text_excerpt", "")
                if not text_content.strip() or not node.get("crawled", False):
                    continue
                docs.append(Document(
                    page_content=text_content,
                    metadata={"source": url, "title": node.get("title", "")}
                ))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                content = json.dumps(item, ensure_ascii=False, indent=2)
                docs.append(Document(page_content=content, metadata={"source": file_path, "item_index": i}))
        else:
            content = json.dumps(data, ensure_ascii=False, indent=2)
            docs.append(Document(page_content=content, metadata={"source": file_path, "item_index": 0}))
        return docs
    except Exception as e:
        print(f"  [Ошибка JSON] {file_path}: {e}")
        return []

async def crawl_urls_flat(urls):
    """
    Скачивает ссылки по ОДНОЙ, чтобы не вешать процесс.
    С жестким таймаутом на каждую страницу.
    """
    if not urls:
        return []

    all_docs = []
    
    browser_config = BrowserConfig(headless=True, verbose=False)
    content_filter = PruningContentFilter(threshold=0.48, threshold_type="fixed")
    md_generator = DefaultMarkdownGenerator(content_filter=content_filter)
    
    run_config = CrawlerRunConfig(
        markdown_generator=md_generator,
        page_timeout=15000 # 15 секунд на загрузку страницы в браузере
    )

    print(f"  -> 🕸 Начинаем сканирование {len(urls)} ссылок (по одной)...")
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        for url in urls:
            print(f"     -> Читаем: {url}")
            try:
                # Оборачиваем в asyncio.wait_for: если 30 секунд нет ответа — убиваем
                result = await asyncio.wait_for(
                    crawler.arun(url=url, config=run_config),
                    timeout=10.0 
                )
                
                if result.success and result.markdown.fit_markdown:
                    all_docs.append(Document(
                        page_content=result.markdown.fit_markdown,
                        metadata={"source": result.url, "page": 1}
                    ))
                    print(f"        ✅ Успешно ({len(result.markdown.fit_markdown)} символов)")
                else:
                    print(f"        ❌ Ошибка: {getattr(result, 'error_message', 'Нет данных')}")
                    
            except asyncio.TimeoutError:
                print(f"        ⏰ Таймаут (30 сек). Пропускаем ссылку.")
            except Exception as e:
                print(f"        ❌ Критическая ошибка: {e}")
                
    return all_docs

# ==========================================
# ОСНОВНАЯ ФУНКЦИЯ
# ==========================================

async def main():
    all_docs = []

    # 1. ЧТЕНИЕ ИЗ ПАПКИ (PDF/TXT/JSON)
    if os.path.exists(DOCS_DIR):
        print(f"⏳ Сканируем папку: {DOCS_DIR}...")
        for filename in os.listdir(DOCS_DIR):
            file_path = os.path.join(DOCS_DIR, filename)
            if not os.path.isfile(file_path): continue
            
            print(f"  -> Читаем файл: {filename}")
            try:
                if filename.lower().endswith(".pdf"):
                    all_docs.extend(PyMuPDFLoader(file_path).load())
                elif filename.lower().endswith(".txt"):
                    all_docs.extend(TextLoader(file_path, encoding="utf-8").load())
                elif filename.lower().endswith(".json"):
                    all_docs.extend(load_json_file(file_path))
            except Exception as e:
                print(f"  [Ошибка] {filename}: {e}")

    # 2. ЧТЕНИЕ ИЗ URL (только плоский список)
    if os.path.exists(URLS_FILE):
        print(f"⏳ Читаем ссылки из: {URLS_FILE}...")
        with open(URLS_FILE, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]
            
        if urls:
            url_docs = await crawl_urls_flat(urls)
            all_docs.extend(url_docs)

    if not all_docs:
        print("❌ Не найдено источников.")
        return

    print(f"✅ Всего загружено документов: {len(all_docs)}")

    # 3. ЧАНКИНГ
    print("⏳ Режем на чанки...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = text_splitter.split_documents(all_docs)
    print(f"✅ Получено чанков: {len(chunks)}")

    # 4. СОХРАНЕНИЕ (УСКОРЕНО)
    print(f"⏳ Векторизуем и сохраняем (батчами по 100)...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    # Сохраняем по 100 штук с небольшой паузой, чтобы Ollama не падала
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"  -> Сохраняем {i+1} - {min(i+batch_size, len(chunks))} из {len(chunks)}...")
        vectorstore.add_documents(batch)
        time.sleep(0.5) # Уменьшили паузу с 1 сек до 0.5

    print(f"🎉 Готово! База обновлена.")

if __name__ == "__main__":
    asyncio.run(main())