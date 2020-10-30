import datetime
import platform
import subprocess
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def get_git_hash():
    stdout = subprocess.check_output(["git", "status", "--porcelain"])
    if len(stdout) > 0:
        print(
            "Git repo is not in a clean state. Commit your changes before running the experiment"
        )
        exit(1)

    git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    assert len(git_hash) > 0

    return git_hash


def is_remote():
    return platform.node() in ["cuda.cs.purdue.edu"]


def get_summary_writer(model_name, evaluation=False):
    if evaluation:
        if not is_remote():
            print('Do not run eval on a local machine. Use "cuda.cs.purdue.edu"')
            exit(1)

        git_hash = get_git_hash()

    log_dir = (Path(__file__).parent / '..').absolute()

    date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if evaluation:
        log_dir /= "runs"
        log_dir /= f"{date_str}-{git_hash}-{model_name}"
    else:
        log_dir /= "local_runs"
        log_dir /= f"{date_str}-{model_name}"

    writer = SummaryWriter(str(log_dir))
    return writer
