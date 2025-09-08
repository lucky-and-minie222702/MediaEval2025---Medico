from huggingface_hub import HfApi, create_repo

api = HfApi()
create_repo(
    name = "mediaeval-medico-submission",
    private = False,
    exist_ok = True,
)

api.upload_file(
    path_or_fileobj = "submission_task1.py",
    path_in_repo = "submission_task1.py",
    repo_id = "trietbui/mediaeval-medico-submission",
)
