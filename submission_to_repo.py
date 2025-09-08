from huggingface_hub import HfApi

api = HfApi()
api.create_repo(
    repo_id = "trietbui/mediaeval-medico-submissio",
    private = False,
    exist_ok = True,
)

api.upload_file(
    path_or_fileobj = "submission_task1.py",
    path_in_repo = "submission_task1.py",
    repo_id = "trietbui/mediaeval-medico-submission",
)
