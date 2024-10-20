# English Language Classifier

## Project Overview

The English Language Classifier is a FastAPI-based application that allows users to upload a video file, transcribe the audio content, and evaluate the speaker's English proficiency. This project uses OpenAI's Whisper model for transcription and a HuggingFace model for classification.

## Features

- **Video Upload**: Upload a video file directly to the server.
- **Audio Extraction**: Extracts audio from the uploaded video.
- **Transcription**: Uses Whisper to transcribe the audio content.
- **Classification**: Evaluates the transcription for English proficiency based on various metrics.
- **JSON Response**: Provides a JSON response with the classification results.

## Installation

### Prerequisites

- Python 3.8+
- ffmpeg (choco install ffmpeg)

### Steps

1. Clone the repository:
    ```bash
    git clone [<repository_url>](https://github.com/ayam04/language-classifier)
    cd language-classifier
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables:
    Create a `.env` file in the root directory and add your HuggingFace API key:
    ```
    HF_API_KEY=your_huggingface_api_key
    ```

## Usage

1. **Run the FastAPI server**:
    ```bash
    python server.py
    ```

2. **Access the API**:
    Open your browser or use a tool like Postman to access `http://localhost:8080/docs` to interact with the API.

3. **Upload a Video**:
    Use the `/classify-english` endpoint to upload a video file and get the classification results.

## Example Request

### Endpoint

`POST /english-communication`

### Request

Upload a video file in the request body.

### Response

```json
{
  "Overall Performance": [
    7,
    "The candidate had a good understanding of the topics discussed and communicated effectively."
  ],
  "Fluency": [
    8,
    "The candidate spoke clearly and effectively with only minor pauses and hesitations."
  ],
  "Grammar and Syntax": [
    7,
    "The candidate had some errors in sentence structure and word usage."
  ],
  "Vocabulary and Word Choice": [
    6,
    "The candidate used a limited vocabulary and could benefit from expanding their word choice."
  ],
  "Pronunciation and Accent": [
    7,
    "The candidate had very clear pronunciation"
  ],
  "Comprehension and Responsiveness": [
    9,
    "The candidate demonstrated a strong understanding of the questions and provided detailed and accurate responses."
  ]
}
```

## File Structure

```
├── functions.py        # Contains functions for transcription and classification
├── server.py           # FastAPI server setup
├── requirements.txt    # Python package dependencies
├── .env                # Environment variables
```
