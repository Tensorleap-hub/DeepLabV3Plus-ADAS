import os
import json
from typing import Optional
from functools import lru_cache
from google.cloud import storage
from google.cloud.storage import Bucket
from google.oauth2 import service_account
from domain_gap.utils.config import CONFIG
from os.path import join

@lru_cache()
def _connect_to_gcs_and_return_bucket(bucket_name: str) -> Bucket:
    auth_secret_string = os.environ['AUTH_SECRET']

    # Check if the env var is a path to a file
    if os.path.isfile(auth_secret_string):
        credentials = service_account.Credentials.from_service_account_file(auth_secret_string)
    else:
        auth_secret = json.loads(auth_secret_string)
        credentials = service_account.Credentials.from_service_account_info(auth_secret)

    project = credentials.project_id
    gcs_client = storage.Client(project=project, credentials=credentials)
    return gcs_client.bucket(bucket_name)

def _download(cloud_file_path: str, local_file_path: Optional[str] = None, is_local=CONFIG['USE_LOCAL']) -> str:
    # if local_file_path is not specified saving in home dir
    if local_file_path is None:
        if is_local:
            local_file_path = join(CONFIG['LOCAL_BASE_PATH'], cloud_file_path)
        else:
            home_dir = os.getenv("HOME")
            local_file_path = os.path.join(home_dir, "Tensorleap", CONFIG['BUCKET_NAME'], cloud_file_path)

    # check if file already exists
    if os.path.exists(local_file_path):
        return local_file_path
    bucket = _connect_to_gcs_and_return_bucket(CONFIG['BUCKET_NAME'])
    dir_path = os.path.dirname(local_file_path)
    os.makedirs(dir_path, exist_ok=True)
    blob = bucket.blob(cloud_file_path)
    blob.download_to_filename(local_file_path)
    return local_file_path
