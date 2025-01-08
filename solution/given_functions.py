import cvzone
import numpy as np
import cv2
#import matplotlib.pyplot as plt

def overlay_gripper_on_part(gripper_image, part_image, x, y, angle):
    """
    Overlay the gripper image onto the part image with specified translation and rotation.

    :param gripper_image: Input image of the gripper.
    :param part_image: Input image of the part mask.
    :param x: Horizontal offset (pixels) for gripper placement on the part.
    :param y: Vertical offset (pixels) for gripper placement on the part.
    :param angle: Rotation angle of the gripper in degrees.
    :return: Combined image with the gripper overlayed on the part., Boolean flag indicating if the gripper intersects with background.
    """
    part_image = np.stack((part_image, part_image, part_image), axis=-1)

    # Rotate the gripper image as first transformation step
    rotated_gripper = cvzone.rotateImage(
        gripper_image, angle=angle, scale=1, keepSize=False
    )

    # Calculate offsets for translation step
    # Dimensions of the gripper and part images
    gh, gw = rotated_gripper.shape[:2]

    # Pad the part image so that the gripper can be out of bounds, original is centered
    ph, pw = part_image.shape[:2]
    pad_top, pad_bottom = gh, gh
    pad_left, pad_right = gw, gw
    part_image_padded = (
        np.ones(
            (ph + (pad_top + pad_bottom), pw + (pad_left + pad_right), 3),
            dtype=np.uint8,
        )
        * 255
    )
    part_image_padded[
        (pad_top) : (pad_top + ph), (pad_left) : (pad_left + pw)
    ] = part_image

    # apply tranformation offset
    overlay_x = int(x - (gw / 2) + pad_left)
    overlay_y = int(y - (gh / 2) + pad_top)

    # Only for visualization
    part_image_copy = part_image_padded.copy()
    mask_combined = cvzone.overlayPNG(
        part_image_copy, rotated_gripper, [overlay_x, overlay_y]
    )

    ## detect intersections

    # make sure mask is 0,1 for instersection
    part_hsv = cv2.cvtColor(part_image_padded, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 20])
    upper_white = np.array([180, 255, 255])  # HSV (0-180, 0-255, 0-255)

    part_hole_mask = cv2.inRange(part_hsv, lower_white, upper_white)

    # Create a mask for the non-transparent regions of the gripper ,e.g the suction points
    alpha_channel = rotated_gripper[:, :, 3]  # Extract the alpha channel
    gripper_mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)[
        1
    ]  # Binary mask for visible regions

    # Align the visible mask with the part's coordinate system
    aligned_gripper_mask = np.zeros(part_image_padded.shape[:2], dtype=np.uint8)
    aligned_gripper_mask[
        overlay_y : overlay_y + gh, overlay_x : overlay_x + gw
    ] = gripper_mask

    # Check for intersection
    intersection = cv2.bitwise_and(aligned_gripper_mask, part_hole_mask)
    intersection_exists = np.any(intersection)

    ###
    #plt.figure(1)
    #plt.imshow(part_hole_mask, cmap='gray', interpolation='nearest')
    #plt.imshow(aligned_gripper_mask,cmap='jet', alpha=0.5, interpolation='nearest')
    #plt.show()
    ###

    return mask_combined, intersection_exists