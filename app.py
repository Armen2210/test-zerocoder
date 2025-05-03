from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла (если есть)
load_dotenv()

# Инициализация клиента OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("OPENAI_API_KEY is not set in environment variables.")

client = OpenAI(api_key=openai_api_key)

# Настройки Currents API
currents_api_key = os.getenv("CURRENTS_API_KEY")
currents_api_url = "https://api.currentsapi.services/v1/search"

app = FastAPI(title="AI Blog Generator", description="Генератор постов на основе темы и новостей")

class TopicRequest(BaseModel):
    topic: str

@app.get("/ping")
def ping():
    """Проверка работоспособности API"""
    return {"status": "ok"}

def fetch_news(topic: str) -> str:
    """Получить новости по теме с помощью Currents API"""
    if not currents_api_key:
        return ""

    params = {
        "apiKey": currents_api_key,
        "keywords": topic,
        "language": "ru"
    }
    try:
        response = requests.get(currents_api_url, params=params)
        response.raise_for_status()
        articles = response.json().get("news", [])
        if not articles:
            return ""
        news_summary = "\n".join(f"- {a['title']}: {a['description']}" for a in articles[:3])
        return news_summary
    except Exception as e:
        return ""  # Без новостей всё равно продолжаем

def generate_post_with_news(topic: str) -> dict:
    """Сгенерировать заголовок, мета и пост, используя OpenAI и новости как контекст"""
    news_context = fetch_news(topic)
    context_note = f"\n\nВот последние новости по теме:\n{news_context}" if news_context else ""

    prompt_title = f"Придумайте привлекательный заголовок для поста на тему: {topic}.{context_note}"
    response_title = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_title}],
        max_tokens=50,
        temperature=0.7,
    )
    title = response_title.choices[0].message.content.strip()

    prompt_meta = f"Напишите краткое, но информативное мета-описание для поста с заголовком: {title}.{context_note}"
    response_meta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_meta}],
        max_tokens=100,
        temperature=0.7,
    )
    meta_description = response_meta.choices[0].message.content.strip()

    prompt_post = (
        f"Напишите подробный и увлекательный пост для блога на тему: {topic}. "
        f"Используйте короткие абзацы, подзаголовки, примеры и ключевые слова для лучшего восприятия и SEO-оптимизации."
        f"{context_note}"
    )
    response_post = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt_post}],
        max_tokens=2048,
        temperature=0.7,
    )
    post_content = response_post.choices[0].message.content.strip()

    return {
        "title": title,
        "meta_description": meta_description,
        "post_content": post_content
    }

@app.post("/generate")
def generate(topic_req: TopicRequest):
    try:
        result = generate_post_with_news(topic_req.topic)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Запуск сервера:
# uvicorn main:app --reload
