from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

# Load API key from file
load_dotenv("config.env")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# FastAPI init
app = FastAPI(title="GenAI App", version="1.0")

class QueryRequest(BaseModel):
    system_prompt: str
    user_query: str

@app.post("/generate")
async def generate_text(req: QueryRequest):
    try:
        completion = client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            temperature=0.0,
            messages=[
                {"role": "system", "content": req.system_prompt},
                {"role": "user", "content": req.user_query},
            ],
        )

        content = completion.choices[0].message.content

        # If JSON, try parsing
        clean_text = content.strip().removeprefix("```json").removesuffix("```").strip()
        try:
            data = json.loads(clean_text)
            return {"success": True, "response": data}
        except Exception:
            return {"success": True, "response": content}

    except Exception as e:
        return {"success": False, "error": str(e)}
