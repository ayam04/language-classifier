import uvicorn
from fastapi import FastAPI, File, UploadFile
from functions import *
import os
import tempfile

app = FastAPI()

@app.post("/classify-english")
async def main(video_input: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(await video_input.read())
            temp_file_path = temp_file.name
        
        transcript = transcribe_video(temp_file_path)
        response = classify_video(transcript)
        os.remove(temp_file_path)
        return response
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, port=8080, reload=True)
