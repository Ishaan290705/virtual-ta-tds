import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from discourse_content.process_data import find_similar_questions_later
from course_content.process_data import find_similar_questions_later_tds
from course_content.content_filtered import course_content, course_shrinked, other_covered
import requests
import json
import re
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

GPT_MODEL = "gpt-4"  # Changed from "o4-mini" to valid model name

# Create FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

@app.get("/")
async def health_check():
    return {"status": "TDS Virtual TA API is running 🚀", "version": "1.0"}

def get_ocr(image_data: str) -> Optional[str]:
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract text from this image."},
                        {"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{image_data}"}}
                    ]
                }
            ],
            "max_tokens": 300
        }
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"OCR Error: {str(e)}")
        return None

def compute_embedding(text: str) -> Optional[np.ndarray]:
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "text-embedding-3-small",
            "input": text.strip()[:2000]
        }
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers=headers,
            json=payload,
            timeout=5
        )
        response.raise_for_status()
        return np.array(response.json()["data"][0]["embedding"])
    except Exception as e:
        print(f"Embedding Error: {str(e)}")
        return None

def call_openai_chat(system_prompt: str, user_query: str) -> dict:
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": GPT_MODEL,
            "temperature": 0.3,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        }
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        
        # Extract JSON from markdown if present
        json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        
        return json.loads(content)
    except Exception as e:
        print(f"OpenAI API Error: {str(e)}")
        return {"answer": "error", "relevant": "error"}

def discourse_related(user_query: str, context: list) -> dict:
    system_prompt = f"""
    As a professor for 'Tools in Data Science', analyze the student's question and these similar Q&As:
    {context}
    Respond in JSON format: {{"answer": "...", "relevant": n}} where n is 1-3 or "error"
    """
    return call_openai_chat(system_prompt, user_query)

def tds_content_related(user_query: str, context: list) -> dict:
    system_prompt = f"""
    As a course professor, use these materials to answer:
    {context}
    Respond in JSON format: {{"answer": "...", "relevant": n}} where n is 1-3 or "error"
    """
    return call_openai_chat(system_prompt, user_query)

def course_related(user_query: str) -> dict:
    system_prompt = f"""
    You're an assistant for 'Tools in Data Science' (2025). Course topics: {course_shrinked}
    Related tech: {other_covered}
    Rules:
    1. Match queries to exact course_shrinked titles
    2. For "What is X?", direct to materials
    3. For logistics questions, use metadata
    4. For tool questions, mention all relevant technologies
    Respond with: {{"answer": "...", "topic": "exact_title_or_Course_Page"}}
    """
    return call_openai_chat(system_prompt, user_query)

@app.post("/api/")
async def answer_query(query: QueryRequest):
    try:
        # Process image if provided
        image_text = ""
        if query.image:
            image_text = get_ocr(query.image) or ""
            if image_text:
                image_text = f"\nImage context: {image_text}"

        full_query = f"{query.question}{image_text}"
        
        # Try discourse content first
        if embedding := compute_embedding(full_query):
            if matches := find_similar_questions_later(embedding):
                response = discourse_related(full_query, matches)
                if response.get("relevant") != "error":
                    match = matches[int(response["relevant"])-1][0]
                    return {
                        "answer": response["answer"],
                        "links": [{"url": match['url'], "text": match['answer'][:100]}]
                    }

            # Try TDS content next
            if matches := find_similar_questions_later_tds(embedding):
                response = tds_content_related(full_query, matches)
                if response.get("relevant") != "error":
                    match = matches[int(response["relevant"])-1][0]
                    return {
                        "answer": response["answer"],
                        "links": [{"url": match['url'], "text": match['question'][:100]}]
                    }

        # Fallback to course content
        response = course_related(full_query)
        url = course_content.get(response.get("topic", ""), "")
        return {
            "answer": response.get("answer", "No answer found"),
            "links": [{"url": url, "text": response.get("topic", "")}]
        }

    except Exception as e:
        print(f"Endpoint Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
