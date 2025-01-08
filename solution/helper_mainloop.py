import os
import pandas as pd

def ensure_folder_exists(file_path):
    """
    Ensure that the folder for the given file path exists. If it does not exist, create it.
    """
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    return

