import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from discourse_content.process_data import find_similar_questions_later
from course_content.process_data import find_similar_questions_later_tds
from course_content.content_filtered import course_content, course_shrinked, other_covered
import requests
import json
import re
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_MODEL = "gpt-4o"  # or o4-mini if using AI proxy

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
    image: str = None

@app.get("/")
async def root():
    return {"status": "TDS Virtual TA API is running 🚀"}

def get_ocr(image_data):
    data_url = f"data:image/webp;base64,{image_data}"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "Please extract all text from this image."},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]}
        ],
        "max_tokens": 200
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    print(f"OCR Error: {response.status_code}: {response.text}")
    return ""

def compute_embedding(user_question):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "text-embedding-3-small",
        "input": [user_question.strip()[:2000]]
    }
    response = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=payload)
    response.raise_for_status()
    return np.array(response.json()["data"][0]["embedding"])

def fetch_response(model, system_prompt, user_query):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt.lower()},
            {"role": "user", "content": user_query.lower()}
        ]
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
        content = response.json()["choices"][0]["message"]["content"]
        try:
            return json.loads(re.search(r"```json\\s*(.*?)\\s*```", content, re.DOTALL).group(1))
        except:
            try:
                return json.loads(content)
            except:
                return {"answer": content.strip(), "relevant": "error"}
    return {"answer": f"Error {response.status_code}: {response.text}", "relevant": "error"}

@app.post("/api/")
async def answer_query(query: QueryRequest):
    try:
        image_data = get_ocr(query.image) if query.image else ""
        data = query.question + image_data
        embedding = compute_embedding(data)

        matches = find_similar_questions_later(embedding)
        if matches:
            for i in range(len(matches)):
                matches[i][0]['question'] = matches[i][0]['question'][:1500] + "....continued"
            response = discourse_related(data, matches)
            if response["relevant"] != "error":
                match, _ = matches[int(response["relevant"]) - 1]
                return {"answer": response["answer"], "links": [{"url": match['url'], "text": match['answer']}]}

        matches = find_similar_questions_later_tds(embedding)
        if matches:
            response = tds_content_related(data, matches)
            if response["relevant"] != "error":
                match, _ = matches[int(response["relevant"]) - 1]
                return {"answer": response["answer"], "links": [{"url": match['url'], "text": match['question']}]}

        response = course_related(data)
        topic = response.get("topic", "Course Page")
        return {
            "answer": response["answer"],
            "links": [{"url": course_content.get(topic, "None"), "text": topic}]
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"answer": "An error occurred...", "links": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
