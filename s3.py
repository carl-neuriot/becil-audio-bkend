import boto3, os
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)

BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

def upload_file_to_s3(file, filename: str):
    file_key = f"audio/{uuid4()}_{filename}"
    s3.upload_fileobj(file.file, BUCKET_NAME, file_key)
    return f"https://{BUCKET_NAME}.s3.amazonaws.com/{file_key}"
