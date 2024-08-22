
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from generation import generate
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/completion")
def complete(body: dict):
    data: list[list[str]] = body['data']
    prompt = data[-1][0]
    history = data[:-1]
    return generate(prompt, history=history)


if __name__ == '__main__':
  uvicorn.run(app, host="0.0.0.0", port=1300, workers=1)
