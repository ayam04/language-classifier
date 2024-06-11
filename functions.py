import os
import json
import whisper
import warnings
import subprocess
from dotenv import load_dotenv
from langchain_community.llms.huggingface_hub import HuggingFaceHub

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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
    You are a professional English teacher. You are presented with a transcript of a candidate's interview for any role in general. You are supposed to catgeorize the candidate's overall performance on a scale of 1 to 100 Percent.
    
    You are also expected to rate the candidate on these metrics on the scale of 1 to 100:
    
    1. Fluency: How well the candidate speaks English.
    2. Grammar and Syntax: How well the candidate uses correct grammar and sentence structure.
    3. Vocabulary and Word Choice: How well the candidate uses a wide range of vocabulary and chooses the right words.
    4. Pronunciation and Accent: How well the candidate pronounces words and the accent they use.
    5. Comprehension and Responsiveness: How well the candidate understands questions and responds to them.

    Here is the transcript of the candidate's interview: {transcript}.

    Make sure to not explain anything in your response. Just provide the ratings and the category.

    Make sure that your output is a json response of the following format without any additional text or characters and no multiple lines. The response should be a single line of json. The response should be in the following format:
    {{
        "Overall Performance": 90,
        "Fluency": 88,
        "Grammar and Syntax": 97,
        "Vocabulary and Word Choice": 59,
        "Pronunciation and Accent": 96,
        "Comprehension and Responsiveness": 40
    }}
    """
    response = llm(prompt).replace(prompt, "").replace("\n", "").strip()

    try:
        json_response = json.loads(response)
    except json.JSONDecodeError as e:
        print(f"Initial JSON decode error: {e}")
        response = response.rstrip(",}")
        response = response + "}"
        try:
            json_response = json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Secondary JSON decode error: {e}")
            raise

    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(json_response, f, indent=4)
    

video_input = "Videos/video.mp4"

transcribe_video(video_input)
with open("Transcripts/video.mp4.txt", "r") as f:
    transcript = f.read()
    classify_video(transcript)