import hashlib
import os
import shutil
import sys
import tarfile
from tqdm.auto import tqdm
from urllib import request

sys.path.append("..")
from args import args

def urlretrieve(url: str, filename: str, desc: str|None=None):
    """Downloads a file using urlretrieve, showing a progress bar and optional description

    Args:
        url (str): url to download
        filename (str): path to file on disk to save at
        desc (str | None, optional): description. Defaults to None.
    """
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, leave=False, desc=desc) as t:
        def report_hook(_, block_size, total_size):
            if t.total is None:
                t.total = total_size
            t.update(block_size)
            
        request.urlretrieve(url=url, filename=filename, reporthook=report_hook)
        
def tar_extractall(tar_path: str, extract_path: str, desc: str|None=None):
    """Extracts a tarball, showing a progress bar and optional description

    Args:
        tar_path (str): path to the tarball
        extract_path (str): path to which we want to extract
    """
    assert tarfile.is_tarfile(tar_path)
    with tarfile.open(tar_path, "r:*") as tar:
        members = tar.getmembers() 
        total_files = len(members)

        with tqdm(total=total_files, unit="file", leave=False, desc=desc) as progress_bar:
            for member in members:
                tar.extract(member, path=extract_path)
                progress_bar.update(1) 
                
def check_hash_file(file_path: str, hash: str, hash_algo="md5"):
    """check whether the hash of a file is equal to a given hash. If hash checking is disabled via argv, always returns true

    Args:
        file_path (str):
        hash (str): 
        hash_algo (str, optional): Defaults to "md5".
    """
    if not args.check_md5:
        return True
    
    return hash_file(file_path, hash_algo) == hash

def check_hash_directory(directory: str, hash: str, desc: str|None=None, hash_algo="md5"):
    """Check the hash of a directory by hashing all files and combining their hashes, against a given hash string.

    Args:
        directory (str): _description_
        hash (str): _description_
        desc (str | None, optional): _description_. Defaults to None.
        hash_algo (str, optional): _description_. Defaults to "md5".
    """
    
    if not args.check_md5:
        return True
    
    return hash_directory(directory, hash_algo, desc) == hash
    

def hash_file(file_path, hash_algo="md5"):
    """Compute the hash checksum of a file using a specified hashing algorithm."""
    hasher = hashlib.new(hash_algo)  # Create a hash object with the specified algorithm

    # Open the file in binary mode
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):  # Read the file in 8KB chunks
            hasher.update(chunk)  # Update the hash with the file's contents

    return hasher.hexdigest()  # Return the hexadecimal hash

def hash_directory(directory: str, hash_algo="md5", desc: str|None=None):
    """Compute the hash of a directory by hashing all files and combining their hashes.

    Args:
        directory (str): directory to hash
        hash_algo (str, optional): hash algorithm. Defaults to "md5".
        desc (str | None, optional): progress bar description. Defaults to None.

    Returns:
        _type_: _description_
    """
    hasher = hashlib.new(hash_algo)
    files = []

    # Collect all file paths recursively
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            files.append(file_path)

    # Create a progress bar with tqdm
    with tqdm(total=len(files), unit="file", leave=False, desc=desc) as progress_bar:
        for file_path in files:
            # Hash file content
            with open(file_path, "rb") as f:
                while chunk := f.read(8192):  # Read in chunks
                    hasher.update(chunk)
            
            # Hash the relative file path to account for the structure
            relative_path = os.path.relpath(file_path, directory).encode()
            hasher.update(relative_path)

            progress_bar.update(1)  # Update the progress bar after each file

    return hasher.hexdigest()



def delete_folder(folder_path: str, desc: str|None=None):
    """delete a folder and display progress bar

    Args:
        folder_path (str):
        desc (str):
    """
    all_files = []
    
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for name in files:
            all_files.append(os.path.join(root, name))
        for name in dirs:
            all_files.append(os.path.join(root, name))

    # Create a progress bar
    with tqdm(total=len(all_files), unit="file", leave=False, desc=desc) as progress_bar:
        # Delete files
        for file_path in all_files:
            try:
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directories
                else:
                    os.remove(file_path)  # Remove files
            except Exception as e:
                print(f"Error deleting {file_path}: {e}", file=sys.stderr)
            
            progress_bar.update(1)  # Update progress bar

    # Remove the main folder (after everything inside is deleted)
    try:
        os.rmdir(folder_path)
    except Exception as e:
        print(f"Error deleting the folder {folder_path}: {e}", file=sys.stderr)