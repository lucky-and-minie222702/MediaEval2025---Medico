from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj = "local_file.txt",
    path_in_repo = "local_file.txt",
    repo_id = "trietbui/mediaeval-medico-submission",
)
