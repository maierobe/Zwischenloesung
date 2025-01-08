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

def read_csv_file(path_csv):
    csv_dataframe = pd.read_csv(path_csv)
    #Zugreifen auf die einzelnen tasks mit z.B.: csv_dataframe.iloc[0], csv_dataframe.iloc[1], ...
    # --> in jedem element dann 'path' und 'gripper'
    # --> csv_dataframe.iloc[0]['path']
    # --> csv_dataframe.iloc[0]['gripper']
    return csv_dataframe

def save_results(path_results, parts_list, d):
    # save results as csv, visualisation plot and warnings/exceptions
    path_csv = path_results
    path_results_folder = os.path.dirname(path_results)
    

    # TODO: Diese funktion schreiben und in Main-Loop einbetten


    return