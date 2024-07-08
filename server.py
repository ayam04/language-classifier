import uvicorn
from functions import *
from fastapi import FastAPI

app = FastAPI()

@app.post("/communication-score")
async def main(cand_id: str):
    try:
        video_output_path = process_candidate_videos(cand_id)
        transcript = transcribe_video(video_output_path)
        response = classify_video(transcript)
        return response
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, port=8080)
