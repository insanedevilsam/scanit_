from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ocr_utils import extract_text_from_bytes  # NEW: import your OCR utility

MODEL_DIR = "final_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR, local_files_only=True)

app = FastAPI()

class SummarizeRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Model API is ready!"}

@app.post("/summarize/")
async def summarize(request: SummarizeRequest):
    text = request.text
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return {"summary": summary}

# NEW: OCR endpoint
@app.post("/ocr/")
async def ocr_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        text = extract_text_from_bytes(contents)
    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}
    return {"text": text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)