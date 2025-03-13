"""This file gives the argparser.
"""
from math import floor
from multiprocessing import cpu_count
import torch

import argparse
parser = argparse.ArgumentParser(
    prog="EDM",
    description="""
EDM: Equivariant Diffusion for Molecule Generation in 3D

Implementation of key parts of Hoogeboom, Satorras, Vignac and Welling's (2022) paper titled "Equivariant Diffusion for Molecule Generation in 3D".

Created by MLMI students David Gailey, Katherine Jackson, Stella Tsiapali and James Yu for the MLMI 4 (Advanced Machine Learning) course.
    """.strip()
)

def _validate_args(args: argparse.Namespace):
    if args.pipeline == "valid" and args.checkpoint is None:
        raise argparse.ArgumentTypeError("--checkpoint must be set if --pipeline=='valid'")

parser.add_argument("--no-wandb", default=True, action="store_false", dest="use_wandb", help="specify if you do not want to use wandb (if not specified, we use wandb)")
parser.add_argument("--wandb-project", default="MLMI4 EDM", type=str, help="wandb project name")

parser.add_argument("--dataset", default="qm9", help="which dataset to train on, e.g. 'qm9', 'qm9_no_h'")
parser.add_argument("--noise-schedule", default="polynomial", type=str, help="which noising schedule to use", choices=["cosine", "polynomial"])
parser.add_argument("--use-resid", default=False, action="store_true", help="specify egnn learns residual of residual")
parser.add_argument("--tanh-range", default=15., type=float, help="if using tanh, what factor we should scale by after applying tanh")
parser.add_argument("--num-steps", default=1000, type=int, help="number of diffusion steps")
parser.add_argument("--batch-size", default=64, type=int, help="batch size")
parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
parser.add_argument("--no-clip_grad", default=True, action="store_false", dest="clip_grad", help="if specified, do not clip gradients (if not specified, we clip gradients)")
parser.add_argument("--max_grad_norm", default=8., type=float, help="maximum gradient norm to tolerate")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="torch device to use")

parser.add_argument("--hidden-d", default=256, type=int, help="EGNN hidden dimension")
parser.add_argument("--num-layers", default=9, type=int, help="EGNN layers")

parser.add_argument("--run-name", default="edm_run", type=str, help="the name of the run")
parser.add_argument("--out-dir", default="./checkpoints", type=str, help="output will be contained in the folder <out_dir>/<run_name>/")

parser.add_argument("--extension", default="vanilla", type=str, help="extension to use", choices=["vanilla", "variance"])
parser.add_argument("--use-non-masked", default=False, action="store_true", help="whether to use our non-masked architecture")
parser.add_argument("--pipeline", default="train", type=str, help="pipeline", choices=["train", "valid", "test", "demo"])
parser.add_argument("--checkpoint", default=None, type=str, help="if specified, load checkpoint located in this folder")
parser.add_argument("--no-restore-optim-state", default=True, action="store_false", dest="restore_optim_state", help="if specified, do not restore optim state from checkpoint (if not specified, then restores from optim.pth)")
parser.add_argument("--no-restore-scheduler-state", default=True, action="store_false", dest="restore_scheduler_state", help="if specified, do not restore scheduler state from checkpoint (if not specified, then restores from scheduler.pth)")

parser.add_argument("--ema-beta", default=0.99, type=float, help="beta factor to use when calculating ema_model: ema_model = beta * ema_model + (1 - beta) * current_model")
parser.add_argument("--scheduler-factor", default=0.5, type=float, help="specify the amount by which the scheduler decreases the lr upon reaching a plateau")
parser.add_argument("--scheduler-patience", default=10, type=int, help="specify how many epochs of non-improvement counts as a plateau")
parser.add_argument("--scheduler-threshold", default=0.01, type=float, help="specify scheduler relative improvement threshold")
parser.add_argument("--scheduler-min-lr", default=5e-6, type=float, help="specify minimum learning rate for scheduler")

parser.add_argument("--force-start-lr", default=None, type=float, help="if specified, force this learning rate upon checkpoint (no effect if not loading a checkpoint)")

default_dl_num_workers = floor(0.9 * cpu_count())
parser.add_argument("--dl-num-workers", default=default_dl_num_workers, type=int, help="set number of dataloader workers to use")
parser.add_argument("--dl-prefetch-factor", default=None if default_dl_num_workers == 0 else 4, type=int, help="dataloader prefetch factor")

parser.add_argument("--start-epoch", default=0, type=int, help="train epochs in [start-epoch, end-epoch) -- note this is only for bookkeeping and does not affect which model is loaded")
parser.add_argument("--end-epoch", default=1300, type=int, help="train epochs in [start-epoch, end-epoch)")

parser.add_argument("--data-dir", default="./data", type=str, help="directory in which datasets are stored")
parser.add_argument("--qm9-data-url", default="https://springernature.figshare.com/ndownloader/files/3195389", type=str, help="url from which to retrieve the raw xyz.tar.bz2 dataset")
parser.add_argument("--qm9-excluded-url", default="https://springernature.figshare.com/ndownloader/files/3195404", type=str, help="url from which to retrieve the excluded.txt file")
parser.add_argument("--qm9-atomref-url", default="https://springernature.figshare.com/ndownloader/files/3195395", type=str, help="url from which to retrieve the atomref.txt file")

parser.add_argument("--check-md5", action="store_true", default=False, help="Enable hash checking")

parser.add_argument("--qm9-raw-xyz-tar-md5", default="ad1ebd51ee7f5b3a6e32e974e5d54012", type=str, help="md5 hash of the raw qm9 tarball")
parser.add_argument("--qm9-raw-xyz-dir-md5", default="57fbe9a55b26af84d274f550a62a9225", type=str, help="md5 hash of the raw qm9 xyz directory")
parser.add_argument("--qm9-raw-excluded-txt-md5", default="a361887bacb427b8a0ce7903d92a53b4", help="md5 hash of the raw qm9 excluded.txt file")
parser.add_argument("--qm9-raw-spilts-npz-md5", default="a4ddef020412e8577429f359bd1d2c79", help="md5 hash of the raw qm9 splits.npz file")
parser.add_argument("--qm9-raw-atomref-txt-md5", default="2d30b2df8329d8fd805c0a4d158a0a0f", help="md5 hash of the raw qm9 atomref.txt file")
parser.add_argument("--qm9-raw-thermo-json-md5", default="f26b44bd4129e1556065f46c9b3c2efa", help="md5 hash of the raw qm9 splits.npz file")

parser.add_argument("--qm9-train-h-npz-md5", default="1983968ed7fecb60f12ddde58cea32c1", help="md5 hash of the processed qm9 train_h.npz file")
parser.add_argument("--qm9-train-no-h-npz-md5", default="917b45088735de67dbaf17cfcf0a7128", help="md5 hash of the processed qm9 train_no_h.npz file")
parser.add_argument("--qm9-valid-h-npz-md5", default="97f8e95f598a16ab51996e58b6e836da", help="md5 hash of the processed qm9 valid_h.npz file")
parser.add_argument("--qm9-valid-no-h-npz-md5", default="fe4a7d1605106be3e3970aaf90c9219c", help="md5 hash of the processed qm9 valid_no_h.npz file")
parser.add_argument("--qm9-test-h-npz-md5", default="c6933531acc8f4153bc18149470155b9", help="md5 hash of the processed qm9 test_h.npz file")
parser.add_argument("--qm9-test-no-h-npz-md5", default="522e5f0fe4087e26c0713f723bead062", help="md5 hash of the processed qm9 test_no_h.npz file")

args = parser.parse_args()

_validate_args(args)