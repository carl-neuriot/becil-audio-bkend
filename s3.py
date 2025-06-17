import boto3
import os
from uuid import uuid4
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
from botocore.exceptions import ClientError

load_dotenv()

BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)


def upload_file_to_s3(file, filename: str, s3_folder: str):
    file_key = f"{s3_folder}/{uuid4()}_{filename}"
    s3.upload_fileobj(file.file, BUCKET_NAME, file_key)
    return f"https://{BUCKET_NAME}.s3.amazonaws.com/{file_key}"


def download_file_from_s3(file_name: str, local_dir: str):
    """
    Downloads a file from 'broadcasts/{file_name}' in the S3 bucket
    to the specified local directory.
    """
    import os

    s3_key = f"broadcasts/{file_name}"
    local_path = os.path.join(local_dir, file_name)

    # Ensure the directory exists
    os.makedirs(local_dir, exist_ok=True)

    # Download the file
    s3.download_file(BUCKET_NAME, s3_key, local_path)
    return local_path


def stream_audio_from_s3(filename: str, s3_folder: str) -> StreamingResponse:
    s3_key = f"{s3_folder}/{filename}"
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        return StreamingResponse(
            obj["Body"],
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f'inline; filename="{filename}"',
                "Accept-Ranges": "bytes"
            }
        )
    except ClientError:
        raise FileNotFoundError(f"File '{filename}' not found in S3.")
