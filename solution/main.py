from pathlib import Path
from argparse import ArgumentParser
import sys
import time
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from rich.progress import track
import pandas as pd

from error_handling import CustomError, CustomWarningInfo  # custom functions in error_handling.py
import helper_positioning as hpos  # main functionality of positioning and scoring in helper_positioning.py
import helper_mainloop as hmain

import helper_load_model as hloadmodel
import helper_run_model as hrunmodel


#########################
import cv2

def compute_amazing_solution(
        part_image_path: Path, gripper_image_path: Path, results_file_path: Path, model
) -> tuple[float, float, float]:
    """Compute the solution for the given part and gripper images.

    :param part_image_path: Path to the part image
    :param gripper_image_path: Path to the gripper image
    :return: The x, y and angle of the gripper
    """

    try:
        # load image mask
        mask = hrunmodel.run_model(part_image_path, model)
        mask_array = np.array(mask)
        mask_array = cv2.bitwise_not(mask_array)


        ################################
        # TODO: Mask-Generierung von Robert/Johannes hier einfügen, statt masks von path zu laden
        ################################

        xpos, ypos, anglepos, warningInfo = hpos.main_get_position_and_visualization(mask_array, gripper_image_path,
                                                                                     part_image_path,
                                                                                     results_file_path)  # TODO: mask path durch image path ersetzen, da mask später nicht abgespeichert wird.

        # Handle warnings that (on purpose) didnt trigger a calculation termination, but shall be saved in the results folder
        # TODO: warning auch in Bild einfügen!!!!!!! -- alle warnings!

        # warningString, warningColor, exitCode = warningInfo.interpretWarning()
        # if warningString == '':
        #    warningMessage = ''
        # else:
        #    warningMessage = "WARNING: "+warningString
        exitCode = warningInfo.exitCode
        warningMessage = warningInfo.message
        warningInfo.getPrintWarningString(part_image_path,
                                          gripper_image_path)  # warningInfo already containes an interpreted error message etc.

        ################################
        # TODO: Abspeichern der Ergebnisse
        ################################

    except CustomError as e:
        # GLOBAL ERROR HANDLING FOR POSITIONING ALGORITHM:
        #
        # Error is only thrown, if calculation is aborted in any subfunction.
        #
        # Mild problems are handled with the full calculation of the best possible 
        # solution with a warning message as an annotation in the output file and console.
        #
        # In both cases, exit code 1 is returned for the main process to raise awareness
        # for problems in the calculation.
        #
        #
        print(f"\033[91mERROR: {e.message}\033[0m")  # rot: 91, gelb: 93
        exitCode = 1
        # ----------------
        ################################
        # TODO: alle Subfunktionen absichern, sodass diese bei Problemen einen CustomError raisen und bis hier weitergeben
        # TODO: im falle eines Errors mit exit code 1 statt 0 das Programm beenden --> was tun wenn nur einer von mehreren gripper-image-pairs einen Error schmeißt?
        ################################
        raise e
    except Exception as e:
        # Handle all other unexpected exceptions  
        print(f"\033[91mUnexpected error: part <{part_image_path}>, gripper <{gripper_image_path}>\033[0m")  # rot
        exitCode = 1
        raise e

    return xpos, ypos, anglepos, warningMessage, exitCode


def main():
    """The main function of your solution.

    Feel free to change it, as long as it maintains the same interface.
    """

    parser = ArgumentParser()
    parser.add_argument("input", help="input csv file")
    parser.add_argument("output", help="output csv file")
    args = parser.parse_args()

    # read the input csv file
    input_df = pd.read_csv(args.input) 

    # load model
    print('Loading Computer Vision Model...')
    model_path = "solution/RMS_cv_model.pth"
    model = hloadmodel.load_model(model_path)

    # compute the solution for each row
    results = []
    calculation_time_list = []
    main_exitCode = 0
    for _, row in track(
            input_df.iterrows(),
            description="Computing the solutions for each row",
            total=len(input_df),
    ):
        part_image_path = Path(row["part"])
        gripper_image_path = Path(row["gripper"])
        assert part_image_path.exists(), f"{part_image_path} does not exist"
        assert gripper_image_path.exists(), f"{gripper_image_path} does not exist"
        # TODO: wenn file nicht existiert, auch nicht weiter rechnen!!!

        # Zeitmessung starten
        start_time = time.time()

        try:
            x, y, angle, warningMessage, exitCode = compute_amazing_solution(part_image_path, gripper_image_path, args.output, model)  #TODO: Anpassen # output path is passed to save the visualization plot to the output dir

            results.append([str(part_image_path), str(gripper_image_path), x, y, angle, warningMessage])

            if exitCode == 1:
                main_exitCode = 1
        except:
            print('Skipped computation due to Error')
            main_exitCode = 1

        # Zeitmessung stoppen
        end_time = time.time()
        calculation_time_list.append(end_time - start_time)
        # Berechnungszeit ausgeben
        print(f"Gesamtzeit inkl. Ergebnisspeicherung < ENDE >: {end_time - start_time:.6f} Sekunden")

    # save the results to the output csv file
    output_df = pd.DataFrame(results, columns=["part", "gripper", "x", "y", "angle", "annotation"])
    # create output folder, if it doesnt exist
    hmain.ensure_folder_exists(args.output) #TODO: Anpassen
    output_df.to_csv(args.output, index=False)

    #plt.figure(1)
    #plt.bar(range(len(calculation_time_list)), calculation_time_list)
    #plt.xlabel('Gripper-Mask-Pair')
    #plt.ylabel('Calculation Time in sec.')
    #plt.show()

    sys.exit(main_exitCode)


if __name__ == "__main__":
    main()
