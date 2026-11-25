"""Entry point of FNAS-LCD's jobs on SDSC Expanse.

The two Tapis apps in tapis/ run this service's own image on Expanse, and
its entrypoint.sh hands them to this script. Each app fixes where and how
its jobs run, but whoever submits a job can add arguments to it, so this
accepts exactly the arguments FNAS-LCD sends and refuses anything else.

  train --dataset UPLOAD_JOB --plainnet_struct S --num_classes N --epochs E
        --batch_size_per_gpu B --workers_per_gpu W --lr_per_256 LR --warmup W
        [--max_iters M]
      Train one architecture with timm_trainer.py on a dataset the same
      Tapis user uploaded. The model and metrics go to the job's output
      folder.
  upload
      Unpack dataset.tar, which Tapis copied into the job's folder, into
      dataset/ there, and record which Tapis user uploaded it.
  delete UPLOAD_JOB
      Delete the folder of the upload job UPLOAD_JOB, if the same Tapis user
      uploaded it or is one of the site's admins (web/config.py).

Folders inside the container, bound by the apps:
  /workspace   this job's own folder on Expanse
  /datasets    the folder holding every upload job's folder
               (read-only for training jobs)
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tarfile
from pathlib import Path

# The site's settings, in the same image: its admins may delete any upload.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from web.config import ADMIN_USERS  # noqa: E402

WORKSPACE = Path("/workspace")
DATASETS = Path("/datasets")
TRAINER = Path(__file__).resolve().parent / "timm_trainer.py"
ARCH = Path(__file__).resolve().parents[1] / "model" / "Masternet.py"
ARCHIVE = "dataset.tar"
OWNER_FILE = "uploaded_by"
JOB_UUID = re.compile(r"^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}-[0-9]{3}$")
# Architecture strings from FNAS-LCD's search, e.g. SuperConvK3BNRELU(3,8,2,1)...
STRUCTURE = re.compile(r"^[A-Za-z0-9_(),.]{1,20000}$")


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr, flush=True)
    sys.exit(1)


def job_owner() -> str:
    # Set by Tapis for every job; a job request can't set or change it.
    owner = os.environ.get("_tapisJobOwner", "")
    if not owner:
        fail("Tapis didn't say which user this job belongs to (_tapisJobOwner).")
    return owner


def uploader(folder: Path) -> str:
    """The Tapis user who uploaded the dataset in an upload job's folder."""
    try:
        return (folder / OWNER_FILE).read_text().strip()
    except OSError:
        fail(f"{folder} isn't a finished FNAS-LCD upload")


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        fail(f"{self.prog}: {message}")


def _matching(pattern: re.Pattern, what: str):
    def check(value: str) -> str:
        if not pattern.match(value):
            raise argparse.ArgumentTypeError(f"not {what}: {value[:100]!r}")
        return value
    return check


def _number(kind, low, high):
    def check(value: str):
        try:
            number = kind(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a number: {value[:100]!r}") from None
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f"{value} is not between {low} and {high}")
        return number
    return check


def _refuse_repeats(argv: list[str]) -> None:
    # argparse would quietly keep the last of a repeated option.
    seen = set()
    for arg in argv:
        if arg.startswith("--"):
            name = arg.split("=", 1)[0]
            if name in seen:
                fail(f"{name} is given more than once")
            seen.add(name)


def train(argv: list[str]) -> None:
    parser = _Parser(prog="train", allow_abbrev=False, add_help=False)
    parser.add_argument("--dataset", required=True, type=_matching(JOB_UUID, "an upload job id"))
    parser.add_argument("--plainnet_struct", required=True,
                        type=_matching(STRUCTURE, "an architecture string"))
    parser.add_argument("--num_classes", required=True, type=_number(int, 1, 100_000))
    parser.add_argument("--epochs", required=True, type=_number(int, 1, 10_000))
    parser.add_argument("--batch_size_per_gpu", required=True, type=_number(int, 1, 4096))
    parser.add_argument("--workers_per_gpu", required=True, type=_number(int, 0, 64))
    parser.add_argument("--lr_per_256", required=True, type=_number(float, 0.0, 100.0))
    parser.add_argument("--warmup", required=True, type=_number(int, 0, 10_000))
    parser.add_argument("--max_iters", type=_number(int, 1, 1_000_000_000))
    _refuse_repeats(argv)
    args = parser.parse_args(argv)

    data_dir = DATASETS / args.dataset / "dataset"
    if not (data_dir / "train").is_dir():
        fail(f"no dataset with a train/ folder at {data_dir}")
    owner = job_owner()
    if uploader(DATASETS / args.dataset) != owner:
        fail(f"{owner} didn't upload the dataset {args.dataset}, so can't train with it")
    out = WORKSPACE / "output"
    out.mkdir(parents=True, exist_ok=True)

    # The same training settings as the earlier training app on Pitzer
    # (fnas-lcd-train 0.1.3).
    command = [
        sys.executable, str(TRAINER),
        "--model_type", "zen",
        "--arch", f"{ARCH}:MasterNet",
        "--dataset", "auto_imagefolder",
        "--plainnet_struct", args.plainnet_struct,
        "--num_classes", str(args.num_classes),
        "--data_dir", str(data_dir),
        "--save_dir", str(out),
        "--summary_file", str(out / "summary.csv"),
        "--completion_file", str(out / "train.done"),
        "--input_image_size", "224",
        "--epochs", str(args.epochs),
        "--batch_size", "256",
        "--batch_size_per_gpu", str(args.batch_size_per_gpu),
        "--workers_per_gpu", str(args.workers_per_gpu),
        "--dist_mode", "single",
        "--gpu", "0",
        "--optimizer", "sgd",
        "--lr_per_256", str(args.lr_per_256), "--target_lr_per_256", "0.0",
        "--lr_mode", "cosine", "--warmup", str(args.warmup),
        "--bn_momentum", "0.01", "--wd", "5e-4", "--weight_init", "xavier", "--nesterov",
        "--label_smoothing", "--random_erase", "--mixup", "--auto_augment",
        "--img_per_class", "-1", "--frac", "1", "--epfrac", "1", "--expid", "1",
        "--print_freq", "100", "--auto_resume",
        "--export",
    ]
    if args.max_iters is not None:
        command += ["--max_iters", str(args.max_iters)]
    print(f"Training on {data_dir} for {args.epochs} epoch(s); output in {out}", flush=True)
    os.execv(sys.executable, command)


def upload(argv: list[str]) -> None:
    if argv:
        fail("upload takes no arguments")
    owner = job_owner()
    archive = WORKSPACE / ARCHIVE
    target = WORKSPACE / "dataset"
    if not archive.is_file():
        fail(f"no {ARCHIVE} in the job's folder")
    if target.exists():
        fail(f"{target} already exists")

    partial = WORKSPACE / "dataset.partial"
    shutil.rmtree(partial, ignore_errors=True)
    # The service sends a plain tar of an already-checked dataset. The "data"
    # filter refuses anything that could land outside the target folder all
    # the same: absolute paths, "..", links, device files.
    with tarfile.open(archive, "r:") as tf:
        tf.extractall(partial, filter="data")
    if not (partial / "train").is_dir():
        fail("the archive has no train/ folder")
    partial.rename(target)
    (WORKSPACE / OWNER_FILE).write_text(owner + "\n")
    archive.unlink()

    classes = sum(1 for d in (target / "train").iterdir() if d.is_dir())
    files = sum(len(names) for _, _, names in os.walk(target))
    print(f"Unpacked {files} files ({classes} classes in train/) into {target}, "
          f"uploaded by {owner}", flush=True)


def delete(argv: list[str]) -> None:
    if len(argv) != 1 or not JOB_UUID.match(argv[0]):
        fail("usage: delete UPLOAD_JOB_ID")
    me = job_owner()
    folder = DATASETS / argv[0]
    if not folder.exists():
        # Already deleted, e.g. by SDSC's cleanup of old files.
        print(f"Nothing to delete: {folder} doesn't exist", flush=True)
        return
    if uploader(folder) != me and me not in ADMIN_USERS:
        fail(f"{me} didn't upload the dataset in {folder}, so can't delete it")
    shutil.rmtree(folder)
    print(f"Deleted {folder}", flush=True)


def main() -> None:
    # Expanse points TMPDIR at the job's folder on the node's local disk,
    # which doesn't exist inside the container; use the container's /tmp.
    if not os.path.isdir(os.environ.get("TMPDIR", "/tmp")):
        os.environ["TMPDIR"] = "/tmp"
    modes = {"train": train, "upload": upload, "delete": delete}
    if len(sys.argv) < 2 or sys.argv[1] not in modes:
        fail(f"the first argument must be one of: {', '.join(modes)}")
    modes[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
