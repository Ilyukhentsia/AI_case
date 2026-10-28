from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "my_docs"

# Подключаем ту же модель
embeddings = OllamaEmbeddings(model="bge-m3")

# Загружаем существующую базу
vectorstore = Chroma(
    persist_directory=CHROMA_DB_DIR, 
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME
)

# Задаем вопрос
# query = "О чем говорится в документе?" # <-- Напиши свой вопрос
query = input("Введите свой вопрос: ")

print(f"🔎 Ищем: {query}\n")
results = vectorstore.similarity_search(query, k=2)

for i, doc in enumerate(results):
    print(f"--- Результат {i+1} ---")
    # Вот тут мы выводим метаданные, которые сохранили!
    print(f"Источник: {doc.metadata.get('source')}")
    print(f"Страница: {doc.metadata.get('page', 'N/A')}")
    print(f"Текст: {doc.page_content[:300]}...\n")