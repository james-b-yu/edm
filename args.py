"""This file gives the argparser.
"""

import argparse
parser = argparse.ArgumentParser(
    prog="EDM",
    description="""
EDM: Equivariant Diffusion for Molecule Generation in 3D

Implementation of key parts of Hoogeboom, Satorras, Vignac and Welling's (2022) paper titled "Equivariant Diffusion for Molecule Generation in 3D".

Created by MLMI students David Gailey, Katherine Jackson, Stella Tsiapali and James Yu for the MLMI 4 (Advanced Machine Learning) course.
    """.strip()
)

parser.add_argument("--data-dir", default="./data", type=str, help="directory in which datasets are stored")
parser.add_argument("--qm9-data-url", default="https://springernature.figshare.com/ndownloader/files/3195389", type=str, help="url from which to retrieve the raw xyz.tar.bz2 dataset")
parser.add_argument("--qm9-excluded-url", default="https://springernature.figshare.com/ndownloader/files/3195404", type=str, help="url from which to retrieve the excluded.txt file")
parser.add_argument("--qm9-atomref-url", default="https://springernature.figshare.com/ndownloader/files/3195395", type=str, help="url from which to retrieve the atomref.txt file")
parser.add_argument("--qm9-raw-xyz-tar-md5", default="ad1ebd51ee7f5b3a6e32e974e5d54012", type=str, help="md5 hash of the raw qm9 tarball")
parser.add_argument("--qm9-raw-xyz-dir-md5", default="57fbe9a55b26af84d274f550a62a9225", type=str, help="md5 hash of the raw qm9 xyz directory")
parser.add_argument("--qm9-raw-excluded-txt-md5", default="a361887bacb427b8a0ce7903d92a53b4", help="md5 hash of the raw qm9 excluded.txt file")
parser.add_argument("--qm9-raw-spilts-npz-md5", default="a4ddef020412e8577429f359bd1d2c79", help="md5 hash of the raw qm9 splits.npz file")
parser.add_argument("--qm9-raw-atomref-txt-md5", default="2d30b2df8329d8fd805c0a4d158a0a0f", help="md5 hash of the raw qm9 atomref.txt file")
parser.add_argument("--qm9-raw-thermo-json-md5", default="f26b44bd4129e1556065f46c9b3c2efa", help="md5 hash of the raw qm9 splits.npz file")

parser.add_argument("--qm9-train-h-npz-md5", default="6b31c02c92243566f0cdeb5cd724b411", help="md5 hash of the processed qm9 train_h.npz file")
parser.add_argument("--qm9-train-no-h-npz-md5", default="09e27536af8744ee6c8833268eecaaba", help="md5 hash of the processed qm9 train_no_h.npz file")
parser.add_argument("--qm9-valid-h-npz-md5", default="78c859bbbd8bdb2005f5a8286a7236ab", help="md5 hash of the processed qm9 valid_h.npz file")
parser.add_argument("--qm9-valid-no-h-npz-md5", default="5024c4ee09fa6078cf9cfe5fd81314ea", help="md5 hash of the processed qm9 valid_no_h.npz file")
parser.add_argument("--qm9-test-h-npz-md5", default="ba06d1320c4a968159062d62e8a5a834", help="md5 hash of the processed qm9 test_h.npz file")
parser.add_argument("--qm9-test-no-h-npz-md5", default="5825239e6ffe391d3a7dc3d544eb83ae", help="md5 hash of the processed qm9 test_no_h.npz file")

args = parser.parse_args()