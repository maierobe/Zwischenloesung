import os
from error_handling import CustomError, CustomWarningInfo # selbst definierte Klasse
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import xml.etree.ElementTree as ET
from svgpathtools import parse_path
import re
import cv2
import math
from pathlib import Path
from scipy.ndimage import center_of_mass
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import textwrap

import given_functions



def main_get_position_and_visualization(mask_np_array, gripper_path_win: Path, image_path_win: Path, results_file_path: Path):
    '''
    This is the main function of the positioning algorithm. It finds the optimal position of a given gripper in a given part mask.
    The calculated position (x, y, angle) is visualised in an image plot and saved in the output folder.
    Parameters, such as search_param_num_iter and min_distance_to_forbidden_area, which influence the positioning algorithm, are definded in this function.
    '''
    ######## PARAMETER #########
    min_distance_to_forbidden_area = 3 #
    static_cap_excl_gripper_radius = 5 # Positionen mit >Xmm Abstand vom Gripper zur Kante werden nur noch leicht betraft
    slight_gradient_multiplier = 0.15 # Dämpfung, mit der diese Positionen >Xmm reduziert werden
    is_forbidden_area_black = False #True for our masks, False for masks from wbk
    search_param_num_iter = 150000 #wähle 100 000 - 200 000. Bei schwachem Computer kleinere Werte wählen, sonst hohe Berechnungszeit.
    search_param_fine_multipl_angle = 1.5 # z.B. 1.5
    ############################
    debugging_bool = False
    
    try:
        #========== GRIPPER PREPROCESSING ==========

        #load gripper file and get radiii and center points
        gripper_path = str(gripper_path_win)
        image_path = str(image_path_win)
        if gripper_path_win.suffix == '.png':
            gripper_points_raw, radii, gripper_shape = load_gripper_from_png(gripper_path)
        elif gripper_path_win.suffix == '.svg':
            gripper_points_raw, radii, gripper_shape = load_gripper_from_svg(gripper_path, scale=1.0)
        else:
            raise CustomError("Gripper-file could not be opened. Make sure to provide the path to a .png or a .svg in your task.csv-Table!")
        
        #transform gripper points to new coordinate system with origin in the center
        gripper_center = ((gripper_shape[1] - 1)/2, (gripper_shape[0] - 1)/2) # (x,y) nicht gerundet!
        gripper_points = gripper_points_raw - gripper_center



        #========== IMAGE PREPROCESSING ==========

        # calculate the center of mass
        if is_forbidden_area_black:
            com = list(center_of_mass(mask_np_array == 255))
        else:
            com = list(center_of_mass(mask_np_array == 0))
        com[0], com[1] = com[1], com[0] #switch entries to get form com = [x,y]
        com = tuple(com)

        # check for image symmetry
        if gripper_path_win.suffix == '.png':
            is_gripper_symmetric = check_png_symmetry(gripper_path)
        else:
            is_gripper_symmetric = False

        #Distanzkarte berechnen
        dist_map = calculate_distance_map(mask_np_array, radii, is_forbidden_area_black, static_cap_excl_gripper_radius, slight_gradient_multiplier)
        #Distanzkarte filtern (bewusst auskommentiert. Erwies sich bei Validierung als nicht zielführend)
        #dist_map = filter_distance_map(dist_map, radii) 



        #========== POSITIONING ==========

        # Find the optimal gripper position
        height, width = dist_map.shape
        bounds = (0, width, 0, height)
        if is_gripper_symmetric:    # bei spiegelsymmetrischem Gripper nur 180 Grad Rotation prüfen --> ermöglicht kleinere Schrittweiten
            angle_to_check = 180
        else:
            angle_to_check = 360
        optimal_position, optimal_angle, optimal_score, debug_coarse_grid, debug_fine_area_outline = find_optimal_gripper_position(
            dist_map, gripper_points, radii, min_distance_to_forbidden_area, bounds, gripper_shape, com, search_param_num_iter, search_param_fine_multipl_angle, max_angle=angle_to_check
        )
        
        if optimal_angle < 0:
            optimal_angle = 360 + optimal_angle # handle case of negative optimal angle due to fine-search near 0 
        elif optimal_angle >= 360:
            optimal_angle = optimal_angle -360 # handle case of optimal angle >360 deg due to fine-search near 360 
        
        
        if debugging_bool:
            print("Optimal Position:", optimal_position)
            print("Optimal Angle (degrees):", optimal_angle)
            print("Optimal Score:", optimal_score)



        #========== ANALYSE CHOSEN POSITION ==========
        num_of_grippers, num_of_grippers_near_edge_0_to_Xmm, is_over_edge, schwerpunkt_distanz_flag = evaluate_and_analyse_position(
            dist_map, gripper_points, gripper_center, radii, optimal_position, optimal_angle, min_distance_to_forbidden_area, com,  gripper_path_win, mask_np_array)

        warningInfo = CustomWarningInfo(num_of_grippers, num_of_grippers_near_edge_0_to_Xmm, is_over_edge, schwerpunkt_distanz_flag, min_distance_to_forbidden_area)
        
        #========== PLOTTING RESULTS ==========
        results_folder_path = os.path.dirname(results_file_path) + "\\"
        if debugging_bool:
            fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6))
            # 1st plot: Distance map with chosen gripper position
            plt.figure(1)
            im1 = ax1.imshow(dist_map, cmap='hot')
            ax1.set_title("Distanzkarte")
            plt.colorbar(im1, ax=ax1)

            # Plot the gripper geometry with actual radii
            rotated_gripper_points = rotate_points(gripper_points, optimal_angle, origin=(0, 0)) + list(optimal_position)
            for point, radius in zip(rotated_gripper_points, radii):
                circle = patches.Circle((point[0], point[1]), radius=radius, color='black', fill=False)
                ax1.add_patch(circle)
                ax1.scatter(point[0], point[1], color='black', marker='x',s=10, )
            #draw gridlines that show the coarse search points
            xgrid = debug_coarse_grid[0]
            ygrid = debug_coarse_grid[1]
            for xg in np.arange(xgrid[0], xgrid[2] + xgrid[1], xgrid[1]):
                ax1.axvline(x=xg, color="blue", linestyle='--', linewidth=0.5)
            for yg in np.arange(ygrid[0], ygrid[2] + ygrid[1], ygrid[1]):
                ax1.axhline(y=yg, color="blue", linestyle='--', linewidth=0.5)
            #draw the area of the fine search 
            xrect = debug_fine_area_outline[0]
            yrect = debug_fine_area_outline[1]
            rectangle = patches.Rectangle((xrect[0], yrect[0]), xrect[1]-xrect[0], yrect[1]-yrect[0], linewidth=3, edgecolor="blue", facecolor='none')
            ax1.add_patch(rectangle)

            # 2nd plot: mask image with chosen gripper position and actual gripper geometry
            im = Image.fromarray(np.uint8(mask_np_array))
            ax2.imshow(im,cmap='gray')

            #im_gripper = cv2.imread(gripper_file_png)
            if gripper_path_win.suffix == '.png':
                im_gripper = Image.open(gripper_path)
                buffer = max(np.array(im_gripper).shape)
                im_gripper = ImageOps.expand(im_gripper, border=(buffer,buffer,np.ceil(optimal_position[0] + buffer).astype(int), np.ceil(optimal_position[1] + buffer).astype(int)), fill=(0,0,0,0))
                goal_point = (optimal_position[1]-gripper_center[1], optimal_position[0]-gripper_center[0])
                im_gripper = im_gripper.rotate(-optimal_angle, resample=Image.BICUBIC, center=(gripper_center[0] + buffer, gripper_center[1] + buffer))
                im_gripper = Image.fromarray(np.roll(np.array(im_gripper), (goal_point[0]-buffer, goal_point[1]-buffer), axis=(0,1)))
                im_gripper = im_gripper.crop((0,0,np.array(im).shape[1], np.array(im).shape[0]))

                ax2.imshow(im_gripper)
            # Plot the gripper geometry with actual radii
            rotated_gripper_points = rotate_points(gripper_points, optimal_angle, origin=(0, 0)) + list(optimal_position)
            #rotated_gripper_points = np.round(rotated_gripper_points).astype(int) # Runden für bessere Vergleichbarkeit mit rotiertem png
            for point, radius in zip(rotated_gripper_points, radii):
                circle = patches.Circle((point[0], point[1]), radius=radius, color='red', fill=False)
                ax2.add_patch(circle)
            

            # Highlight the optimal gripper center
            ax2.scatter(optimal_position[0], optimal_position[1], color='blue', s=100, label="Gripper Center")
            # Highlight the parts center of mass
            ax2.scatter(com[0], com[1], color='brown', label='Center of Mass')

            ax2.legend()
            ax2.set_title("Gripper Position")
            ax2.text(
                0.7, 0.05,
                #f"Time: {end_time - start_time:.2f} s\nX-Offset: {optimal_position[0]} pixel\nY-Offset: {optimal_position[1]} pixel\nAngle: {optimal_angle} deg",
                f"X-Offset: {optimal_position[0]} pixel\nY-Offset: {optimal_position[1]} pixel\nAngle: {optimal_angle} deg",
                color= "black",
                ha='left',
                va='bottom',
                transform=ax2.transAxes, # relative ax-coordinates
                bbox=dict(facecolor='darkseagreen', alpha=0.7, edgecolor='darkgreen'), # background box
            )

            plt.tight_layout()
            mask_name = image_path.split("\\")[-1][0:-4]
            gripper_name = gripper_path.split("\\")[-1][0:-4]
            save_path = results_folder_path + "result___" + mask_name + "___" + gripper_name + ".png"
            plt.savefig(save_path, format='png')
            plt.close()
            #plt.show()
        else:
            #fig, ax2 = plt.subplots(figsize=(6,6))
            if warningInfo.message == "":
                # Layout without warning message
                fig, ax2 = plt.subplots(figsize=(6,6))
            else:
                fig = plt.figure(figsize=(6, 6))
                gs = GridSpec(2, 1, height_ratios=[0.25, 0.85], figure = fig)
                ax2 = fig.add_subplot(gs[1])

                # add warning text
                if warningInfo.color == "r":
                    box_facecolor = "red"
                    box_edgecolor = "darkred"
                else:
                    box_facecolor = "yellow"
                    box_edgecolor = "#CCCC00"
                text_ax = fig.add_subplot(gs[0])
                text_ax.axis("off")
                wrapped_text = textwrap.fill(warningInfo.message, width=75)
                text_ax.text(
                    0.5, 0.5,  # Center of the box
                    #warningInfo.message,
                    wrapped_text,
                    ha="center",
                    va="center",
                    wrap=True,
                    bbox=dict(facecolor=box_facecolor, alpha=0.3, edgecolor=box_edgecolor, boxstyle="round,pad=0.5"),
                    fontsize=10,
                    color="black"
                )

            # mask image with chosen gripper position and actual gripper geometry
            #im = Image.fromarray(np.uint8(mask_np_array))
            im = Image.open(image_path)
            ax2.imshow(im,cmap='gray')

            #im_gripper = cv2.imread(gripper_file_png)
            if gripper_path_win.suffix == '.png':
                im_gripper = Image.open(gripper_path)
                buffer = max(np.array(im_gripper).shape)
                im_gripper = ImageOps.expand(im_gripper, border=(buffer,buffer,np.ceil(optimal_position[0] + buffer).astype(int), np.ceil(optimal_position[1] + buffer).astype(int)), fill=(0,0,0,0))
                goal_point = (optimal_position[1]-gripper_center[1], optimal_position[0]-gripper_center[0])
                im_gripper = im_gripper.rotate(-optimal_angle, resample=Image.BICUBIC, center=(gripper_center[0] + buffer, gripper_center[1] + buffer))
                goal_point = (int(goal_point[0]-buffer), int(goal_point[1])-buffer)
                im_gripper = Image.fromarray(np.roll(np.array(im_gripper), (goal_point), axis=(0,1)))
                im_gripper = im_gripper.crop((0,0,np.array(im).shape[1], np.array(im).shape[0]))

                ax2.imshow(im_gripper)
            # Plot the gripper geometry with actual radii
            rotated_gripper_points = rotate_points(gripper_points, optimal_angle, origin=(0, 0)) + list(optimal_position)+ [-1, 0]
            #rotated_gripper_points = np.round(rotated_gripper_points).astype(int) # Runden für bessere Vergleichbarkeit mit rotiertem png
            for point, radius in zip(rotated_gripper_points, radii):
                circle = patches.Circle((point[0], point[1]), radius=radius, color='red', fill=False)
                ax2.add_patch(circle)


            # Highlight the optimal gripper center
            ax2.scatter(optimal_position[0]-1, optimal_position[1], color='blue', s=100, label="Gripper Center")
            # Highlight the parts center of mass
            ax2.scatter(com[0], com[1], color='brown', label='Center of Mass')

            ax2.legend()
            ax2.set_title("Gripper Position")
            mask_name = image_path.split("\\")[-1][0:-4]
            gripper_name = gripper_path.split("\\")[-1][0:-4]
            ax2.text(
                0.7, 0.05,
                #f"Time: {end_time - start_time:.2f} s\nX-Offset: {optimal_position[0]} pixel\nY-Offset: {optimal_position[1]} pixel\nAngle: {optimal_angle} deg",
                f"Part: {mask_name}\nGripper: {gripper_name}\n\nX-Offset: {optimal_position[0]} pixel\nY-Offset: {optimal_position[1]} pixel\nAngle: {optimal_angle} deg",
                color= "black",
                ha='left',
                va='bottom',
                transform=ax2.transAxes, # relative ax-coordinates
                bbox=dict(facecolor='darkseagreen', alpha=0.7, edgecolor='darkgreen'), # background box
            )

            plt.tight_layout()
            save_path = results_folder_path + "result___" + mask_name + "___" + gripper_name + ".png"
            plt.savefig(save_path, format='png')
            plt.close()
            #plt.show()

    except CustomError as e:
        raise e
    except Exception as e:
        # Handle all other unexpected exceptions  
        raise e
        

    return optimal_position[0], optimal_position[1], optimal_angle, warningInfo


def change_angle_direction(angle):
    angle_out = -angle
    if angle_out < 0:
        angle_out = 360 + angle_out # handle case of negative optimal angle due to fine-search near 0 
    elif angle_out >= 360:
        angle_out = angle_out -360 # handle case of optimal angle >360 deg due to fine-search near 360 
    return angle_out



def load_gripper_from_png(gripper_png_path):
     # Step 1: Load the gripper image
    image = cv2.imread(gripper_png_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {gripper_png_path}")

    # Contour detection (should be more robust than cv, especially with these well defined circle images)
    # Preprocess the image (binary threshold)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # add padding, so contours on the edges can easily be identified
    padding = 10
    binary = np.pad(binary, pad_width=padding, mode='constant', constant_values=0)
    # Step 3: Detect circles using Contour detection
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circle_centers = []
    circle_radii = []
    # Process each contour
    for contour in contours:
        # Fit a minimum enclosing circle around the contour
        (x, y), radius = cv2.minEnclosingCircle(contour)
        circle_centers.append((int(x-padding), int(y-padding))) # dont forget to remove the manually added padding here
        circle_radii.append(int(radius))

    return np.array(circle_centers), np.array(circle_radii), image.shape

def load_gripper_from_svg(svg_path, scale=1.0):
    """
    Parse the SVG file and extract circles' centers and radii using bounding box approximation
    for each path element, ensuring only one center and radius per circle.
    """
    def strip_namespace(tag):
        """Remove namespace from an XML tag."""
        return tag.split('}')[-1] if '}' in tag else tag

    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Extract the width, height and transform attributes
    svg_width = float(root.attrib.get('width'))
    svg_height = float(root.attrib.get('height'))

    gripper_points = []
    radii = []

    # Iterate over each element in the SVG
    for elem in root.iter():
        if 'path' in elem.tag:  # Check for <path> elements
            d_attr = elem.attrib.get('d', None)  # Get the 'd' attribute
            if d_attr:
                path = parse_path(d_attr)  # Parse the path data

                # We want to extract only the bounding box of the path, not the segments
                bbox = path.bbox()  # Get the bounding box of the entire path
                if bbox:
                    # Calculate the center of the bounding box
                    cx = (bbox[0] + bbox[1]) / 2
                    cy = (bbox[2] + bbox[3]) / 2
                    # Calculate the radius as half the width (or height) of the bounding box
                    radius = (bbox[1] - bbox[0]) / 2

                    # Scale the points and radii
                    gripper_points.append(np.array([cx * scale, cy * scale]))
                    radii.append(radius * scale)
        elif strip_namespace(elem.tag) == 'g':
            if 'translate' in elem.attrib.get('transform'):
                transform = np.array(re.findall(r'-?\d+', elem.attrib.get('transform'))).astype(float)
                tx = transform[0]
                ty = transform[1]

    # Convert lists to numpy arrays for consistency
    gripper_points = np.array(gripper_points)
    for idx, point in enumerate(gripper_points):
        gripper_points[idx] = gripper_points[idx] + transform
    radii = np.array(radii)

    return gripper_points, radii, (svg_height, svg_width) # TO BE VALIDATED!!!


def check_png_symmetry(image_path):
    '''
    Checks a gripper png for symmetry to allow a more efficient search algorithm.
    '''
    image = Image.open(image_path).convert('L')
    image_array = np.array(image)
    flipped_image_array = np.flip(image_array, axis=1)
    # Symmetrie prüfen (Pixelweise Vergleich)
    difference = np.abs(image_array - flipped_image_array)
    symmetry_score = np.sum(difference) / image_array.size  # Durchschnittlicher Unterschied pro Pixel
    if symmetry_score < 0.5:
        return True
    else:
        return False


def rotate_points(points, angle, origin):
    # Umwandlung des Winkels von Grad in Bogenmaß
    angle_radians = np.deg2rad(angle)
    
    # Rotationsmatrix
    rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                                [np.sin(angle_radians), np.cos(angle_radians)]])
    
    # Verschiebe die Punkte so, dass der Mittelpunkt auf den Ursprung zeigt
    points_shifted = points - origin
    
    # Wende die Rotation an
    rotated_points = np.dot(points_shifted, rotation_matrix.T)
    
    # Verschiebe die Punkte zurück
    rotated_points += origin

    return rotated_points


def coarse_search(dist_map, gripper_points, radii, min_distance_to_forbidden_area, bounds, gripper_shape, part_center_of_mass, max_angle, search_area, coarse_step_size_x, coarse_step_size_y, coarse_step_size_angle ):
    """
    Perform the coarse search with a larger step size on a given grid.
    """
    gripper_center = (gripper_shape[0]/2, gripper_shape[1]/2) # nicht gerundet!

    best_score = -np.inf
    best_position = None
    best_angle = None

    # Coarse search over positions
    for x in range(search_area[0], search_area[1], coarse_step_size_x):
        for y in range(search_area[2], search_area[3], coarse_step_size_y):
            for angle in np.arange(0, max_angle, coarse_step_size_angle):
                score = evaluate_position(dist_map, gripper_points, gripper_center, radii, (x, y), angle, min_distance_to_forbidden_area, part_center_of_mass) 
                if score > best_score:
                    best_score = score
                    best_position = (x, y)
                    best_angle = angle

    return best_position, best_angle, best_score


def fine_search(dist_map, gripper_points, radii, min_distance_to_forbidden_area, gripper_shape, coarse_position, coarse_angle, part_center_of_mass, fine_search_area_half_x, fine_search_area_half_y, fine_search_area_half_angle):
    """
    Perform a finer search around the best coarse position and angle with a given search area.
    """
    gripper_center = (gripper_shape[0]/2, gripper_shape[1]/2) # nicht gerundet!

    best_score = -np.inf
    best_position = coarse_position
    best_angle = coarse_angle
    
    # Fine search over positions
    fine_step_size = 1
    
    for x in np.arange(coarse_position[0] - fine_step_size*fine_search_area_half_x, coarse_position[0] + fine_step_size*fine_search_area_half_x, fine_step_size):
        for y in np.arange(coarse_position[1] - fine_step_size*fine_search_area_half_y, coarse_position[1] + fine_step_size*fine_search_area_half_y, fine_step_size):
            for angle in np.arange(coarse_angle - fine_step_size*fine_search_area_half_angle, coarse_angle + fine_step_size*fine_search_area_half_angle, fine_step_size):
                score = evaluate_position(dist_map, gripper_points, gripper_center, radii, (x, y), angle, min_distance_to_forbidden_area, part_center_of_mass)
                #debug_save_score.append(score)
                if score > best_score:
                    best_score = score
                    best_position = (x, y)
                    best_angle = angle

    return best_position, best_angle, best_score

def decimal_search(dist_map, gripper_points, radii, min_distance_to_forbidden_area, gripper_shape, fine_position, fine_angle, part_center_of_mass):
    """
    Perform a precision search around the best fine position and angle.
    """
    gripper_center = (gripper_shape[0]/2, gripper_shape[1]/2) # nicht gerundet!

    best_score = -np.inf
    best_position = fine_position
    best_angle = fine_angle

    num = 0
    for x in np.arange(fine_position[0] - 0.5, fine_position[0] + 0.7, 0.2):
        for y in np.arange(fine_position[1] - 0.5, fine_position[1] + 0.7, 0.2):
            for angle in np.arange(fine_angle -0.5, fine_angle + 0.6, 0.1):
                score = evaluate_position(dist_map, gripper_points, gripper_center, radii, (x, y), angle, min_distance_to_forbidden_area, part_center_of_mass)
                num +=1
                if score > best_score:
                    best_score = score
                    best_position = (round(x, 1), round(y, 1))
                    best_angle = round(angle, 1)

    return best_position, best_angle, best_score


def evaluate_position(dist_map, gripper_points, gripper_center, radii, position, angle, min_distance_to_forbidden_area, part_center_of_mass, doPlotting=False):
    """
    Evaluate the score for a given position and angle by taking the distance to the nearest forbidden
    area and the distance between the gripper center and the part center of mass.
    
    ATTENTION: for analysis/interpretation of a given position use the more computationally
    intensive function evaluate_and_analyse_position()
    """
 
    # Rotate the gripper points based on the given angle
    rotated_gripper_points = rotate_points(gripper_points, angle, (0,0)) + [position[0], position[1]] # Rotation um Greifermitte (IMMER (0,0) !), dann Translation
    

    list_of_regular_distances = []

    # Calculate the score (in this case, the total distance from the points to the dist_map)
    score = 0
    multiplier_distance = 10/len(radii)
    multiplier_schwerp = 2
    damage_near_edge = 20
    damage_out_of_bound = 1000
    for point, radius in zip(rotated_gripper_points, radii):
       
        x, y = int(point[0]), int(point[1])
        # Avoid out-of-bounds checks
        if 0 <= x < dist_map.shape[1] and 0 <= y < dist_map.shape[0]:
            # Innerhalb des Bauteils
            if dist_map[y,x] > min_distance_to_forbidden_area + radius:
                # Hält gewünschten Abstand zu verbotenen Zonen ein
                list_of_regular_distances.append(dist_map[y, x] - radius)
                
            elif dist_map[y,x] >= radius:
                # Unterschreitet gewünschten Abstand zu verbotenen Zonen, aber immernoch >=0
                score -= damage_near_edge
                list_of_regular_distances.append(dist_map[y, x] - radius)

            else:
                # Greifer Ragt in verbotene Zone hinein
                score -= damage_out_of_bound*np.abs(dist_map[y, x] - radius)

        else:
            # Außerhalb des Bauteils
            score -= damage_out_of_bound*radius # worst case damage in this case
    
    try:
        score += 10*min(list_of_regular_distances)
    except:
        score += 0
    
    # Schwerpunktsabweichung:
    delta_xs = np.abs(np.mean(rotated_gripper_points[:,0]) - part_center_of_mass[0])  # x ist bei Greiferpunkten an Stelle null und bei Image an Stelle 1! Achtung!
    delta_ys = np.abs(np.mean(rotated_gripper_points[:,1]) - part_center_of_mass[1])
    score = score - multiplier_schwerp*np.linalg.norm([delta_xs, delta_ys])

    return score

def evaluate_and_analyse_position(dist_map, gripper_points, gripper_center, radii, position, angle, min_distance_to_forbidden_area, part_center_of_mass, gripper_image_path_win, mask_image):
    '''
    Analysis of a given position regarding the number of grippers overlapping with the forbidden area,
    the number of grippers near the edge and the distance between the gripper center and the
    part center of mass.
    '''
    # Initiate result variables
    num_of_grippers = len(radii)
    num_of_grippers_near_edge_0_to_Xmm = 0 # dependent on min_distance_to_forbidden_area
    num_of_grippers_over_edge = 0

    
    # Rotate the gripper points based on the given angle
    rotated_gripper_points = rotate_points(gripper_points, angle, (0,0)) + [position[0], position[1]] # Rotation um Greifermitte (IMMER (0,0) !), dann Translation
    list_of_regular_distances = []

    # Calculate the score (in this case, the total distance from the points to the dist_map)
    score = 0
    multiplier_schwerp = 2
    damage_near_edge = 20
    damage_out_of_bound = 1000
    for point, radius in zip(rotated_gripper_points, radii):
       
        x, y = int(point[0]), int(point[1])
        # Avoid out-of-bounds checks
        if 0 <= x < dist_map.shape[1] and 0 <= y < dist_map.shape[0]:
            # Innerhalb des Bauteils
            if dist_map[y,x] > min_distance_to_forbidden_area + radius:
                # Hält gewünschten Abstand zu verbotenen Zonen ein
                list_of_regular_distances.append(dist_map[y, x] - radius)
                
            elif dist_map[y,x] >= radius:
                # Unterschreitet gewünschten Abstand zu verbotenen Zonen, aber immernoch >=0
                score -= damage_near_edge
                list_of_regular_distances.append(dist_map[y, x] - radius)
                num_of_grippers_near_edge_0_to_Xmm += 1

            else:
                # Greifer Ragt in verbotene Zone hinein
                score -= damage_out_of_bound*np.abs(dist_map[y, x] - radius)
                num_of_grippers_over_edge += 1

        else:
            # Außerhalb des Bauteils
            score -= damage_out_of_bound*radius # worst case damage in this case
            num_of_grippers_over_edge += 1
    
    try:
        score += 10*min(list_of_regular_distances)
    except:
        score += 0
    
    # Schwerpunktsabweichung:
    delta_xs = np.abs(np.mean(rotated_gripper_points[:,0]) - part_center_of_mass[0])  # x ist bei Greiferpunkten an Stelle null und bei Image an Stelle 1! Achtung!
    delta_ys = np.abs(np.mean(rotated_gripper_points[:,1]) - part_center_of_mass[1])
    score = score - multiplier_schwerp*np.linalg.norm([delta_xs, delta_ys])

    # sehr hohe Abweichung von Bauteilschwerpunkt und Greifermittelpunkt (>50% der mittleren Bildabmessung)
    schwerpunkt_distanz_flag = 0
    cmp_size = np.mean(dist_map.shape)/2
    if np.linalg.norm([delta_xs, delta_ys]) > (0.4*cmp_size):
        schwerpunkt_distanz_flag = 1

    # Überschneidung von Grippern und Maske prüfen:
    if gripper_image_path_win.suffix == '.png':
        gripper_image = gripper_image = cv2.imread(
            str(gripper_image_path_win), cv2.IMREAD_UNCHANGED
        )
        _, is_over_edge = given_functions.overlay_gripper_on_part(
                gripper_image=gripper_image,
                part_image=mask_image,
                x=position[0],
                y=position[1],
                angle=change_angle_direction(angle),
            )
    else:
        # Wenn gripper nur als svg
        if num_of_grippers_over_edge > 0:
            is_over_edge = 1
        else:
            is_over_edge = 0

    return num_of_grippers, num_of_grippers_near_edge_0_to_Xmm, is_over_edge, schwerpunkt_distanz_flag

def find_optimal_gripper_position(dist_map, gripper_points, radii, min_distance_to_forbidden_area,  bounds, gripper_shape, part_center_of_mass, num_iterations, fine_multipl_angle, fine_step_size=1, max_angle=360):
    '''
    This function combines coarse-, fine- and decimal-search to find the best position
    of the gripper on the given binary mask.
    '''

    try:
        coarse_search_area, coarse_step_size_x, coarse_step_size_y, coarse_step_size_angle, fine_area_half_x, fine_area_half_y, fine_area_half_angle = calculate_coarse_and_fine_grid(
            bounds, gripper_shape, gripper_points, max_angle, fine_multipl_angle, num_iterations
            )

        # Step 1: Perform coarse search
        coarse_position, coarse_angle, best_coarse_score = coarse_search(dist_map, gripper_points, radii, min_distance_to_forbidden_area, bounds, gripper_shape, part_center_of_mass, max_angle, coarse_search_area, coarse_step_size_x, coarse_step_size_y, coarse_step_size_angle)
        
        # If no valid position was found during the coarse search, return None
        if coarse_position is None:
            print("No valid position found in coarse search!")
            return None

        # Step 2: Perform fine search around the best coarse result
        fine_position, fine_angle, best_fine_score = fine_search(dist_map, gripper_points, radii, min_distance_to_forbidden_area, gripper_shape, coarse_position, coarse_angle, part_center_of_mass, fine_area_half_x, fine_area_half_y, fine_area_half_angle)

        # Step 3: Perform decimal-precision search aroung the best fine result
        decimal_position, decimal_angle, best_decimal_score = decimal_search(dist_map, gripper_points, radii, min_distance_to_forbidden_area, gripper_shape, fine_position, fine_angle, part_center_of_mass)

        debug_coarse_grid = [[coarse_search_area[0], coarse_step_size_x, coarse_search_area[1]], [coarse_search_area[2], coarse_step_size_y, coarse_search_area[3]]] # [[x_start, x_step, x_end], [y...]]
        debug_fine_grid = [[coarse_position[0] - fine_area_half_x, coarse_position[0] + fine_area_half_x], [coarse_position[1] - fine_area_half_y, coarse_position[1] + fine_area_half_y]] # [[x_start, x_end], [y...]]
    
    except CustomError as e:
        raise e
    except Exception as e:
        # Handle all other unexpected exceptions  
        raise e

    return decimal_position, decimal_angle, best_fine_score, debug_coarse_grid, debug_fine_grid


def calculate_coarse_and_fine_grid(bounds, gripper_shape, gripper_points, max_angle, fine_angle_multiplier_fallback, num_iterations):
    '''
    Calculates the coarse and fine grid based on the desired computational effort (num_iterations)
    the part size and the gripper complexity. This enables a nearly constant calculation time for different
    gripper complexitys and part sizes.
    
    ---------------------------------------------------------------
    This code is based on the following mathematical considerations:
    Number_of_Iterations = g * (2c³ + fx + fy + fa)
        with g = Number of Gripper points
        c = Number of discretization points in coarse search (*2, as angles are checked with double resolution)
        fx, fy, fa = Width of the fine-search area in the x-direction, y-direction, angular dimension (due to search precision of 1px or 1deg)
   
    Relationship between coarse and fine-search grid:
    The fine grid should always cover a search area of +/- 1 times the step size of the coarse grid in all dimensions, with a precision of 1 pixel or 1 degree.

        fx = 2coarse_step_size_x = (2search_space[0]) / c
        fy = 2coarse_step_size_y = (2search_space[1]) / c
            where search space represents the search area, i.e., the image dimensions reduced by the minimum half gripper width.

        fa = fine_angle_multiplier*max_angle / c

    Computational effort with higher gripper complexity:
    Since the computational effort increases non-linearly with higher gripper complexity (g) due to calculations on intermediate loop levels, the influence of the gripper is dampened with:
    g = 0.4*num_g + 5
    
    Solving the resulting mathematical problem:
    The resulting mathematical problem is not always solvable (especially with low selected n_iter and large components).
    As a fallback solution, the following approach is implemented:

        Use 90% of the available iterations for the coarse search and 10% for the fine search.
    
    '''

    num_g = len(gripper_points) #number of grippers
    # Einfluss des Grippers dämpfen, denn in mittleren Schleifen finden ebenfalls Berechungen statt--> doppelte Gripperanzahl ist weniger als doppelte Berechnungszeit
    g = 0.4*num_g + 5

    min_gripper_dist_half = np.floor(min(gripper_shape)/2).astype(int)
    coarse_search_area = (bounds[0] + min_gripper_dist_half, bounds[1] - min_gripper_dist_half, 
                   bounds[2] + min_gripper_dist_half, bounds[3] - min_gripper_dist_half)
    search_shape = (coarse_search_area[1] - coarse_search_area[0], coarse_search_area[3] - coarse_search_area[2])
    
    if search_shape[0]<1 or search_shape[1]<1:
        # Fehler: Gripper zu groß für dieses Bauteil!
        raise CustomError("Gripper is too big for this part! Choose a smaller gripper!")


    # TODO: Herleitung und Berechnung hier einfügen

    fine_angle_multiplier_for_math_problem = 1
    Kc = 2*search_shape[0] * 2*search_shape[1] * max_angle*fine_angle_multiplier_for_math_problem

    #calculate coarse step amount for given num_iterations
    determinante_1 = (-num_iterations/g)**2 - 4*2*Kc
    if determinante_1 >= 0:
        determinante_2 = ((num_iterations/g) + math.sqrt(determinante_1))/(2*2) 
        c = determinante_2**(1/3)

        coarse_step_size_x = np.ceil(search_shape[0]/c).astype(int) # beachte, dass num_iterations wegen ceil() nicht ganz ausgenutzt wird! z.T. 30% weniger Punkte, als erlaubt wäre
        coarse_step_size_y = np.ceil(search_shape[1]/c).astype(int)
        coarse_step_size_angle = np.ceil(max_angle/(2*c)).astype(int) # doppelte Auflösung bei Winkel - max angle neu hier

        #define fine grid width 
        fine_area_half_x = coarse_step_size_x
        fine_area_half_y = coarse_step_size_y
        fine_area_half_angle = np.ceil(coarse_step_size_angle * fine_angle_multiplier_for_math_problem).astype(int)

    else:
        #num_iterations wurde zu klein gewählt und Problem ist nicht lösbar --> erhöhen von num_iterations auf minimum zum Teil mit hohen Rechenzeiten verbunden

        #Fallback-Lösung:
        # Aufteilung von num_iter auf 10% fine search und 90% coarse search. Güte der Lösung gefährdet, aber Rechenzeit hat Vorrang
        anteil_coarse = 0.9

        c = ((anteil_coarse*num_iterations)/(2*g))**(1/3)

        coarse_step_size_x = np.ceil(search_shape[0]/c).astype(int) # beachte, dass num_iterations wegen ceil() nicht ganz ausgenutzt wird! z.T. 30% weniger Punkte, als erlaubt wäre
        coarse_step_size_y = np.ceil(search_shape[1]/c).astype(int)
        coarse_step_size_angle = np.ceil(max_angle/(2*c)).astype(int) # doppelte Auflösung bei Winkel - max angle neu hier

        ratio = search_shape[0]/search_shape[1] # Verhältnis Suchbereich x/y
        fine_area_half_x = np.ceil((((1-anteil_coarse)*num_iterations)/(8*g*(1*(1/ratio)*fine_angle_multiplier_fallback*(0.5 + (1/(2*ratio))))))**(1/3)).astype(int) # hier kann ceil zu Erhöhung der Anzahl iterationen führen!
        fine_area_half_y = np.ceil(fine_area_half_x/ratio).astype(int)
        fine_area_half_angle = np.ceil(fine_angle_multiplier_fallback*((fine_area_half_x + fine_area_half_y)/2)).astype(int)

    

    return coarse_search_area, coarse_step_size_x, coarse_step_size_y, coarse_step_size_angle, fine_area_half_x, fine_area_half_y, fine_area_half_angle






def calculate_distance_map(image_array, gripper_radii, is_forbidden_area_black, static_cap_excl_gripper_radius,slight_gradient_multiplier ):
    """
    Berechnet die Distanzkarte eines Bildes basierend auf verbotenen Bereichen.
    Verbotene Bereiche sind schwarze/weiße Pixel, je nach is_forbidden_area_black-Variable.
    Zudem wird eine Skalierung von weit vom Rand entfernten Bildpunkten vorgenommen (siehe Readme).
    """
    # Prüfe, ob das Bild mehr als eine Farbebene hat (z. B. RGB).
    if image_array.ndim == 3:  # Shape (H, W, C)
        # Verbotene Bereiche: 
        #forbidden_mask = np.all(image_array == 0, axis=-1)
        if is_forbidden_area_black:
            forbidden_mask = np.all(image_array == [0,0,0], axis=-1)
        else:
            forbidden_mask = np.all(image_array == [255,255,255], axis=-1)
    else:  # Shape (H, W) - Graustufenbild
        if is_forbidden_area_black:
            forbidden_mask = image_array == 0
        else:
            forbidden_mask = image_array == 255
        

    # Füge die Bildränder als verbotenen Bereich hinzu
    forbidden_mask[0, :] = True  # Obere Kante
    forbidden_mask[-1, :] = True  # Untere Kante
    forbidden_mask[:, 0] = True  # Linke Kante
    forbidden_mask[:, -1] = True  # Rechte Kante
    
    # Invertiere die Maske: 1 für erlaubte Bereiche, 0 für verbotene Bereiche
    inverted_mask = ~forbidden_mask
    
    # Berechne die Distanzkarte (Entfernung zu den nächsten verbotenen Pixeln)
    distance_map = distance_transform_edt(inverted_mask)

    max_gripper_radius = max(gripper_radii)

    def transform_distance_map(map, static_cap, delta_factor=0.25):
        # Create a mask for values greater than the threshold (static_cap_excl_gripper_radius)
        mask_greater_than_threshold = map > static_cap
        
        # For values <= static_cap_excl_gripper_radius, leave them unchanged
        # For values > static_cap_excl_gripper_radius, apply the transformation
        result = np.where(mask_greater_than_threshold,
                        static_cap + (map - static_cap) * delta_factor, map)
        return result
    
    distance_map = transform_distance_map(distance_map, static_cap_excl_gripper_radius + max_gripper_radius, slight_gradient_multiplier)
  

    return distance_map


def filter_distance_map(distance_map, radii):
    # apply blur filter to the distance map with the average gripper radius
    
    radius = np.mean(radii)
    kernel_size = 2 * radius + 1
    y, x = np.ogrid[:kernel_size, :kernel_size]
    center = radius
    circular_mask = (x - center)**2 + (y - center)**2 <= radius**2
    circular_kernel = circular_mask.astype(np.float32)
    circular_kernel /= circular_kernel.sum()  # Normalize kernel
    filtered_map = cv2.filter2D(distance_map, -1, circular_kernel)

    #recreate clear edges of forbidden areas after blurring
    filtered_map[distance_map == 0] = 0

    return filtered_map
