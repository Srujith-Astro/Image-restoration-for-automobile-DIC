# Install: pip install boto3
import boto3
from botocore.config import Config

# Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your environment
s3 = boto3.client(
    "s3",
    endpoint_url="https://s3.hf.co/srujith0101",
    config=Config(
        region_name="us-east-1",
        s3={"addressing_style": "path"},
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    ),
)

s3.download_file("car-parts-and-damage-dataset-bucket", "example.txt", "example.txt")