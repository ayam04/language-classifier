import os
import warnings
import boto3
from dotenv import load_dotenv
import whisperx

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_API_KEY")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
region_name = os.getenv("AWS_REGION")
bucket_name = os.getenv("BUCKET_NAME")

# Initialize S3 client
# s3 = boto3.client('s3', region_name=region_name)
print("a")

model = whisperx.load_model("base", device="cuda", compute_type="int8")
print("a")


print("All imports and initializations were successful!")
