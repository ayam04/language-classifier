import os
import json
import whisper
import warnings
import subprocess
import numpy as np
from dotenv import load_dotenv
from langchain_community.llms.huggingface_hub import HuggingFaceHub

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_API_KEY")

model = whisper.load_model("base")
llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.3", model_kwargs={"temperature": 0.5, "max_new_tokens": 30000})

def transcribe_video(video_input):
    filename = os.path.basename(video_input).split(".")[0]
    audio_output = "audio.wav"
    ffmpeg_command = f"ffmpeg -i {video_input} -vn -c:a libmp3lame -b:a 192k {audio_output}"
    subprocess.run(ffmpeg_command, shell=True, check=True)
    
    model = whisper.load_model("base")
    response = model.transcribe(audio_output)  
    os.remove(audio_output)
    
    with open(f"{filename}.txt", "x") as f:
        f.write(response["text"].strip())

    total_confidence = 0
    total_length = 0
    
    for segment in response["segments"]:
        avg_logprob = segment['avg_logprob']
        confidence = np.exp(avg_logprob)
        segment_length = segment['end'] - segment['start']
        total_confidence += confidence * segment_length
        total_length += segment_length
    
    if total_length == 0:
        return 0
    
    average_confidence = total_confidence / total_length

    return (f"{average_confidence*10:.0f}", response["text"].strip())

def classify_video(conf_transcript):
    conf, transcript = conf_transcript
    conf = int(conf)
    prompt = f"""
    You are a professional English teacher. You are presented with a transcript of a candidate's interview for any role in general. You are supposed to categorize the candidate's overall performance on a scale of 1 to 10.
    
    You are also expected to rate the candidate on these metrics on the scale of 1 to 10:
    
    1. Fluency: How well the candidate speaks English.
    2. Grammar and Syntax: How well the candidate uses correct grammar and sentence structure.
    3. Vocabulary and Word Choice: How well the candidate uses a wide range of vocabulary and chooses the right words.
    4. Pronunciation and Accent: How well the candidate pronounces words and the accent they use.
    5. Comprehension and Responsiveness: How well the candidate understands questions and responds to them.

    Here is the transcript of the candidate's interview: {transcript}.

    Make sure to not explain anything in your response. Just provide the ratings and the category.

    Make sure that your output is a json response of the following format without any additional text or characters and no multiple lines. The response should be a single line of json. The response should be in the following format:
    {{
        "Overall Performance": 5,
        "Fluency": 6,
        "Grammar and Syntax": 5,
        "Vocabulary and Word Choice": 1,
        "Pronunciation and Accent": 0,
        "Comprehension and Responsiveness": 8
    }}
    This is just the format of the json. Do not send the above json as the response, unless you want to provide the same ratings for the candidate.
    Update the scores in the json response above with the ratings you want to provide for the candidate.
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

    json_response["Pronunciation and Accent"] = conf

    return json_response
    
# video_input = "Videos/video.mp4"

# transcript = transcribe_video(video_input)
# print(transcript)
# print(classify_video(transcript))