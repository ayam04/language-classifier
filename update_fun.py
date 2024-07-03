import os
import re
import json
import time
import whisper
import warnings
import subprocess
import numpy as np
from dotenv import load_dotenv
from langchain_community.llms.huggingface_hub import HuggingFaceHub

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

start = time.time()

load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_API_KEY")

model = whisper.load_model("base")
llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2", model_kwargs={"temperature": 0.5, "max_new_tokens": 25000})

def transcribe_video(audio_input):
    filename = os.path.basename(audio_input).split(".")[0]
    model = whisper.load_model("base")
    response = model.transcribe(audio_input)
    
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
    transcript = response["text"].strip()
    
    return (f"{average_confidence*10:.0f}", transcript)

def classify_video(conf_transcript):
    conf, transcript = conf_transcript
    conf = int(conf)
    prompt = f"""
    You are a professional English teacher. You are presented with a transcript of a candidate's interview for any role in general. You are supposed to categorize the candidate's overall performance on a scale of 1 to 10, while also providing a 1 line reason of why the score has been given. The overall performance's reason should be detailed and should cover all aspects of the candidate's performance in 3 lines.
    
    You are also expected to rate the candidate on these metrics on the scale of 1 to 10:
    
    1. Fluency: How well the candidate speaks English.
    2. Grammar and Syntax: How well the candidate uses correct grammar and sentence structure.
    3. Vocabulary and Word Choice: How well the candidate uses a wide range of vocabulary and chooses the right words.
    4. Pronunciation and Accent: How well the candidate pronounces words and the accent they use.
    5. Comprehension and Responsiveness: How well the candidate understands questions and responds to them.

    Here is the transcript of the candidate's interview: {transcript}.

    Make sure to not explain anything in your response, except for the 1 line description of why the score has been given. Just provide the ratings, reason and the category.

    Make sure that your output is a json response of the following format without any additional text or characters and no multiple lines. The response should be a single line of json. The response should be in the following format:
    {{"Overall Performance": [5,"The candidate spoke clearly and effectively about their professional experiences."], "Fluency": [6,"The candidate had clear fluency throught the interview"], "Grammar and Syntax": [5,"Minor errors in sentence structure."] ,"Vocabulary and Word Choice": [5,"Used some advanced vocabulary but could benefit from more varied word choice."], "Pronunciation and Accent": [9,"The candidate had very clear pronunciation"], "Comprehension and Responsiveness": [8,"The candidate demonstrated a good understanding of the questions and provided clear and concise responses."]}}
    This is just the format of the json. Do not send the above json as the response, unless you want to provide the same ratings or reasons for the candidate.
    Update the string values in the json above with the ratings you want to provide for the candidate.
    Update the scores in the json response above with the ratings you want to provide for the candidate.
    You response should be a single line json.
    """
    
    response = llm(prompt).replace(prompt, "").replace("\n", "").strip()
    response = re.sub(r'^.*?({.*?}).*$', r'\1', response)

    print(response)
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

    json_response["Pronunciation and Accent"][0] = conf
    end = time.time()
    print(f"Time taken: {end - start}")
    return json_response

# transcript = transcribe_video(video_input)
# print(transcript)
# print(classify_video(transcript))