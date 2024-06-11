import os
import subprocess
import whisper

model = whisper.load_model("base")

def transcribe_video(video_input):
    filename = os.path.basename(video_input)
    audio_output = "audio.mp3"
    ffmpeg_command = f"ffmpeg -i {video_input} -vn -c:a libmp3lame -b:a 192k {audio_output}"
    subprocess.run(ffmpeg_command, shell=True, check=True)
    response = model.transcribe(audio_output)  
    os.remove(audio_output)
    with open(f"Transcripts/{filename}.txt", "x") as f:
        f.write(response["text"])

def classify_video(video_input):
    pass

video_input = "Videos/video.mp4"