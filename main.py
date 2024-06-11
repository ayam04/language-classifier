import os
import whisper
import subprocess
from dotenv import load_dotenv
from langchain_community.llms.huggingface_hub import HuggingFaceHub

load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_API_KEY")

model = whisper.load_model("base")
llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.3", model_kwargs={"temperature": 0.5, "max_new_tokens": 30000})

def transcribe_video(video_input):
    filename = os.path.basename(video_input)
    audio_output = "audio.mp3"
    ffmpeg_command = f"ffmpeg -i {video_input} -vn -c:a libmp3lame -b:a 192k {audio_output}"
    subprocess.run(ffmpeg_command, shell=True, check=True)
    response = model.transcribe(audio_output)  
    os.remove(audio_output)
    with open(f"Transcripts/{filename}.txt", "x") as f:
        f.write(response["text"])

def classify_video(transcript):
    prompt = f"""
    You are a professional English teacher. You are presented with a transcript of a candidate's interview for any role in general. You are supposed to catgeorize the candidate's performance into one of the following categories: Excellent, Good, Average, Poor, Very Poor. 
    
    You are also expected to rate the candidate on these metrics on the scale of 1 to 10:
    
    1. Fluency: How well the candidate speaks English.
    2. Grammar and Syntax: How well the candidate uses correct grammar and sentence structure.
    3. Vocabulary and Word Choice: How well the candidate uses a wide range of vocabulary and chooses the right words.
    4. Pronunciation and Accent: How well the candidate pronounces words and the accent they use.
    5. Comprehension and Responsiveness: How well the candidate understands questions and responds to them.

    Here is the transcript of the candidate's interview: {transcript}.

    Make sure to not explain anything in your response. Just provide the ratings and the category.
    """
    response = llm(prompt)
    print(response.replace(prompt,"").strip())

video_input = "Videos/video.mp4"

# transcribe_video(video_input)
with open("Transcripts/video.mp4.txt", "r") as f:
    transcript = f.read()
    classify_video(transcript)