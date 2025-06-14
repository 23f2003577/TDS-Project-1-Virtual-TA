# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "argparse",
#     "fastapi",
#     "httpx",
#     "markdownify",
#     "numpy",
#     "semantic-text-splitter",
#     "tqdm",
#     "uvicorn",
#     "google-genai",
#     "pillow",
#     "html2text",
#     "beautifulsoup4",
#     "python-dotenv",
#     "openai"
# ]
# ///

import argparse
import os
import base64
import httpx
import json
import numpy as np
import re
from pathlib import Path
from fastapi import FastAPI, Request
from pydantic import BaseModel
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions
import time
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

class RateLimiter:
    def __init__(self, requests_per_minute:60, requests_per_second:2):
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_second  
        self.request_times = []
        self.last_request_time = 0

    def wait_if_needed(self):
        current_time = time.time()

        #Per second rate limiting
        time_since_last = current_time - self.last_request_time
        if time_since_last < (1.0 / self.requests_per_second):
            time.sleep((1.0 / self.requests_per_second) - time_since_last)
        
        #Per minute rate limiting
        current_time = time.time()
        self.request_times = [t for t in self.request_times if current_time -t < 60]
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (current_time - self.request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
                current_time
                self.request_times = [t for t in self.request_times if current_time - t < 60]
        self.request_times.append(current_time)
        self.last_request_time = current_time

rate_limiter = RateLimiter(requests_per_minute=5, requests_per_second=2)

def get_image_description(image_path):
    client = genai.Client(api_key =  os.getenv("GEMINI_API_KEY"))

    image_data = client.files.upload(file = os.path.join('tools-in-data-science-public', image_path))
    #with open(os.path.join('tools-in-data-science-public', image_path), 'rb') as image_file:
    #    image_data = image_file.read()

    response = client.models.generate_content(
        model = "gemini-2.0-flash",
        content = [image_data, "Describe the contents of this image in detail, focusing on any text, objects, or relevant features that could help answer questions about it."]
    )
    
    return response.text

def get_embedding(text: str, max_retries: int = 3) -> list:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    for attempt in range(max_retries):
        try:
            rate_limiter.wait_if_needed()

            response = client.models.embed_content(
            model = "gemini-embedding-exp-03-07",
            contents = text
            )

            return response.embeddings[0].values
        except Exception as e:
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                wait_time = 2 ** attempt
                print(f"Rate limit exceeded, waiting for {wait_time} seconds...")
                time.sleep(wait_time)
            elif attempt == max_retries - 1:
                print(f"Failed to get embedding after {max_retries} attempts: {e}")
            else:
                print(f"Attempt {attempt + 1} failed: {e}, retrying...")
                time.sleep(1)

    raise Exception("Max retries exceeded")

def load_embeddings():
    data = np.load("merged_embeddings.npz", allow_pickle=True)
    return data["chunks"], data["embeddings"]

def generate_llm_response(question: str, context: str):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    system_prompt = "You are a helpful teaching assistant that answers questions based on provided context. Use the provided context to answer the question. Your response should be in markdown format, with necessary formatting for code blocks, lists and headings. If the question is not answerable from the context, respond with 'I don't know'."
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            system_prompt,
            f"Context: {context}", 
            f"Question: {question}"
        ],
        config=GenerateContentConfig(
            max_output_tokens=512,
            temperature=0.5,
            top_p=0.95,
            top_k=40,
        ),
    )
    return response.text

def answer(question: str, image: str = None):
    loaded_chunks, loaded_embeddings = load_embeddings()
    if image:
        image_description = get_image_description(f"data:image/jpeg;base64,{image}")
        question += f" {image_description}"

    question_embedding = get_embedding(question)
    similarities = np.dot(loaded_embeddings, question_embedding) / (np.linalg.norm(loaded_embeddings, axis=1) * np.linalg.norm(question_embedding))

    top_indices = np.argsort(similarities)[-10:][::-1]
    top_chunks = [loaded_chunks[i] for i in top_indices]

    response = generate_llm_response(question, "\n".join(top_chunks))
    return {
        "question": question,
        "response": response,
        "top_chunks": top_chunks
    }


@app.post("/api/")
async def api_answer(request: Request):
    try:
        data = await request.json()
        print(data)
        return answer(data.get('question'), data.get('image'))
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "Welcome to the Q&A API. Use POST /api/ with JSON payload {'question': 'your question', 'image': 'base64 image data'} to get answers."}

if __name__ == "__main__":
    import uvicorn
    print(os.getenv("GEMINI_API_KEY"))
    uvicorn.run(app, host="0.0.0.0", port=8000)
