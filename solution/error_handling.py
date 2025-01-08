import numpy as np


class CustomError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)

class CustomWarningInfo:
    '''
    This class is used to store information about the interpreted position of the gripper relative to the part.
    For initialization, 
    - the number of gripper points
    - the number of gripper points near the edge
    - the information, wether the center of mass is too far of the gripper center
    is needed. Using this information, a warning is created, which contains a warining string, a color and an exit code, which should be
    retured as a result of the given warning.
    '''
    def __init__(self, num_of_grippers, num_of_grippers_near_edge_0_to_Xmm, is_over_edge, schwerpunkt_distanz_flag, min_distance_to_forbidden_area):
        self.num_grippers = num_of_grippers
        self.near_edge = num_of_grippers_near_edge_0_to_Xmm
        self.is_over_edge = is_over_edge
        #self.over_edge = num_of_grippers_over_edge
        self.is_schwerp_crit = schwerpunkt_distanz_flag
        self.near_edge_thres = min_distance_to_forbidden_area
        # initialize warning message etc.
        warningString, warningColor, exitCode = self.interpretWarning()
        if warningString == '':
            warningMessage = ''
        else:
            warningMessage = "WARNING: " + warningString
        self.raw_string = warningString
        self.message = warningMessage
        self.color = warningColor
        self.exitCode = exitCode

    def interpretWarning(self):
        if self.is_over_edge:
            # Some gripper are outside of the part - Warning of high importance (overrides less important warnings)
            exitCode = 1
            warningString = "There are some gripper points outside of the part. Check Position of gripper in the visualisation image (see output folder)!"
            warningColor = 'r'
            #if self.over_edge > np.ceil(self.num_grippers * 0.25).astype(int):
            #    # more than 25% of the grippers are outside
            #    warningString = "At least 25 percent of gripper points are outside of the part. Check Position of gripper in the visualisation image (see output folder)!"
            #    warningColor = "r"
            #else:
            #    # less than 25% of the grippers are outside
            #    warningString = "There are some gripper points outside of the part. Check Position of gripper in the visualisation image (see output folder)!"
            #    warningColor = 'r'
        else:
            # Warnings of lower importance (warning strings are concatenated)
            warningString = ""
            warningColor = ""
            exitCode = 0
            if self.near_edge > 0:
                # No grippers outside of part, but some near edge
                exitCode = 0
                edge_thres = self.near_edge_thres
                warningString += f"There are some gripper points near the edge (<{edge_thres} mm)."
                warningColor = 'y'
            if self.is_schwerp_crit:
                # No problems with gripper points near or over edge, but high distance between gripper center and part center of mass
                exitCode = 0
                if not warningString == "":
                    warningString += "   AND   "
                warningString += "Distance between gripper center and part center of mass is high. Check if gripper force can safely compensate the tilitng moment."
                warningColor = 'y'
           

        return warningString, warningColor, exitCode
    

    def getPrintWarningString(self, part_image_path, gripper_image_path):
        #warningString, warningColor, _ = self.interpretWarning()
        warningString = self.raw_string
        warningColor = self.color
        if warningColor == '':
            #empty warning
            return
        elif warningColor == "r":
            #red
            printString = f"\033[91mWARNING: part <{part_image_path}>, gripper <{gripper_image_path}>\n{warningString}\033[0m"
            print(printString)
        else:
            #yellow
            printString = f"\033[93mWARNING: part <{part_image_path}>, gripper <{gripper_image_path}>\n{warningString}\033[0m"
            print(printString)
        return
