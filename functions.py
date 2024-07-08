import os
import re
import json
import time
import whisper
import warnings
import subprocess
import numpy as np
import boto3
from dotenv import load_dotenv
from langchain_community.llms.huggingface_hub import HuggingFaceHub

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

start = time.time()

load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_API_KEY")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
region_name = os.getenv("AWS_REGION")
bucket_name = os.getenv("BUCKET_NAME")

s3 = boto3.client('s3', region_name=region_name)
# model = whisper.load_model("base", device="cuda", compute_type="int8")
model = whisper.load_model("base")
llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2", model_kwargs={"temperature": 0.5, "max_new_tokens": 25000})

def list_videos_for_candidate(bucket_name, candidate_assessment_id):
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=f"recording-new/{candidate_assessment_id}")
    # print(response)
    return [content['Key'] for content in response.get('Contents', [])]

def download_video_from_s3(bucket_name, video_key, download_path):
    try:
        s3.download_file(bucket_name, video_key, download_path)
    except Exception as e:
        print(f"Error downloading video from S3: {e}")
        pass

def concatenate_videos(video_paths, output_path):
    with open("filelist.txt", "w") as f:
        for video in video_paths:
            f.write(f"file './{video}'\n")

    ffmpeg_command = f"ffmpeg -f concat -safe 0 -i filelist.txt -c copy {output_path}"
    # print(f"Executing FFmpeg command: {ffmpeg_command}")
    subprocess.run(ffmpeg_command, shell=True, check=True)

    os.remove("filelist.txt")
    for video_path in video_paths:
        os.remove(video_path)

def process_candidate_videos(candidate_assessment_id):
    video_output_path = f"{candidate_assessment_id}_video.mp4"

    video_files = list_videos_for_candidate(bucket_name, candidate_assessment_id)
    print(f"Found videos: {video_files}")

    downloaded_videos = []
    for video_file in video_files:
        download_path = f"Videos/{os.path.basename(video_file)}"
        download_video_from_s3(bucket_name, video_file, download_path)
        downloaded_videos.append(download_path)

    concatenate_videos(downloaded_videos, video_output_path)
    print(f"Videos concatenated and saved to {video_output_path}")
    return video_output_path

def transcribe_video(video_input):
    audio_output = "audio.wav"
    ffmpeg_command = f"ffmpeg -i {video_input} -vn -c:a libmp3lame -b:a 192k {audio_output}"
    subprocess.run(ffmpeg_command, shell=True, check=True)
    os.remove(video_input)
    
    # audio = whisperx.load_audio(audio_output)
    response = model.transcribe(audio_output)  
    os.remove(audio_output)

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
    You are a professional English teacher. You are presented with a transcript of a candidate's interview for any role in general. You are supposed to categorize the candidate's overall performance on a scale of 1 to 10, while also providing a 1 line reason of why the score has been given. The overall performance's reason should be detailed and should cover all aspects of the candidate's performance in 30 words.
    
    The pronunciation, accent and fluency based confidence score is {conf}.
    
    You are also expected to rate the candidate on these metrics on the scale of 1 to 10:
    
    1. Fluency: How well the candidate speaks English.
    2. Grammar and Syntax: How well the candidate uses correct grammar and sentence structure.
    3. Vocabulary and Word Choice: How well the candidate uses a wide range of vocabulary and chooses the right words.
    4. Pronunciation and Accent: How well the candidate pronounces words and the accent they use.
    5. Comprehension and Responsiveness: How well the candidate understands questions and responds to them.

    Here is the transcript of the candidate's interview: {transcript}.

    Make sure to not explain anything in your response, except for the 1 line description of why the score has been given. Just provide the ratings, reason and the category.

    Make sure that your output is a json response of the following format without any additional text or characters and no multiple lines. The response should be a single line of json. The response should be in the following format:

    {{"Overall Performance": [5,"The candidate spoke clearly and effectively about their professional experiences."], "Fluency": [6,"The candidate had clear fluency throught the interview"], "Grammar and Syntax": [5,"Minor errors in sentence structure."],"Vocabulary and Word Choice": [5,"Used some advanced vocabulary but could benefit from more varied word choice."], "Pronunciation and Accent": [9,"The candidate had very clear pronunciation"], "Comprehension and Responsiveness": [8,"The candidate demonstrated a good understanding of the questions and provided clear and concise responses."]}}

    This is just the format of the json. Do not send the above json as the response, unless you want to provide the same ratings or reasons for the candidate.
    Update the string values in the json above with the ratings you want to provide for the candidate.
    The responses for individual categories should be at least 10 to 15 words.
    Update the scores in the json response above with the ratings you want to provide for the candidate.
    Your response should be a single line json.
    """
    
    response = llm(prompt).replace(prompt, "").replace("\n", "").strip()
    response = re.sub(r'^.*?({.*?}).*$', r'\1', response)

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

# Usage
candidate_id = "662f486dafd2bf001c8d56f1"
video_output_path = process_candidate_videos(candidate_id)
transcript = transcribe_video(video_output_path)
print(transcript)
print(classify_video(transcript))