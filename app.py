from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

MODEL_DIR = "./saved_summary_model"

model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)

# device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)
model.eval()

templates = Jinja2Templates(directory="templates")


class DialogueInput(BaseModel):
    dialogue: str


def clean_data(text: str) -> str:
    text = re.sub(r"\r\n", " ", text)      # newlines
    text = re.sub(r"\s+", " ", text)       # extra whitespace
    text = re.sub(r"<.*?>", " ", text)     # html tags, e.g. <p> <h1>
    text = text.strip().lower()
    return text


@torch.inference_mode()
def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)

    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True,
    )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary


# API endpoints
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    if not dialogue_input.dialogue.strip():
        return {"summary": ""}
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")
