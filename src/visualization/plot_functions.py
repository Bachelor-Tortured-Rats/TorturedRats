import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import colorsys
import random
from itertools import combinations
import cv2

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.2):
    """Apply the given mask to the image.
    """

    # The mask is drawn on the image for each color. Alpha controls the ratio between the existing image colors and the mask colors
    for c in range(3):
        image[:, :, c] = np.where(mask == 1, image[:, :, c] *
                                     (1 - alpha) + alpha * color[c] * 255,
                                     image[:, :, c])
    return image

def display_image(img, masks = None, colors = [(0, 0, 1)], alpha=0.2, display = False):

    # Check if there is a mask
    if not masks is None:

        # The image is converted to a 3-channel RGB image to allow for colored masks to be added.
        img = np.expand_dims(img, axis=2)
        img = np.dstack([img] * 3)

        # Ensure masks are  in a list
        if type(masks) != list:
            masks = [masks]

        # Make sure colors are in a list
        if type(colors) != list:
            colors = [colors]

        # If not all colors are defined, generate the remaining
        if len(colors) < len(masks):
            colors += random_colors(len(masks) - len(colors))

        # If there are multiple masks, make the intersections clear
        if len(masks) > 1:
            num_masks = len(masks)

            # Generate colors and masks for intersections by looping over every combination of masks
            for comb in combinations(range(num_masks), 2):

                # Find mask combination
                maskA = masks[comb[0]]
                maskB = masks[comb[1]]

                # Calculate the intersection
                intersect = maskA * maskB

                # Add intersection as a seperate mask
                masks.append(intersect)

                # Remove intersections from the other masks
                masks[comb[0]] = maskA * 1 - intersect * 1
                masks[comb[1]] = maskB * 1 - intersect * 1

                # Calculate the weighted average of the two masks
                intersect_color = tuple(np.array(colors[comb[0]]) * 0.5 + np.array(colors[comb[1]]) * 0.5)

                # Add the intersection color to the intersection mask
                colors.append(intersect_color)

        # Apply masks to image
        for mask, color in zip(masks, colors):
            img = apply_mask(img, mask, color, alpha)

        # Make sure all values are integers
        img = img.astype("int")

        # Display image if display = True
        if display:
            plt.imshow(img)

        # Else return the image
        else:
            return img

    # If no masks were added, the image is single channel gray scale
    else:

        # Display image if display = True
        if display:
            plt.imshow(img, cmap='gray')

        # Else return the image with no alterations
        else:
            return img

    plt.show()


def animate_CT(img_3D, angle = 1, masks = None, colors = [(0, 0, 1)], alpha=0.2, animation_interval = 50, filename="slices"):
    """
    Take a 3D image with corresponding masks and generate a .mp4 video going through every slice.
    img_3D: a 3D numpy array depicting a CT image.
    angle: integer (0, 1, 2) representing which angle the image is viewed from.
    masks: either a binary 3D numpy array or a list of such depicting segmentation masks for the CT image.
    colors: a 3-tuple or a list of such representing the (R, G, B) values of each mask color used. Each value is a float in [0, 1].
    alpha: a float in [0, 1] that adjusts the opacity of the masks
    animation_interval: an integer adjusting the number of miliseconds between each slice displayed.
    filename: a string for naming the .mp4 file that gets saved in the working directory
    """
    # Defines the indexing of the CT_image depending on the chosen view angle.
    dimension_choice = ["[i, :, :]",
                        "[:, i, :]",
                        "[:, :, i]"][angle]

    # Define the number of slices for the video
    num_slices = np.shape(img_3D)[angle]

    # Ensure masks are  in a list
    if type(masks) != list:
        masks = [masks]

    # Make a plot object for constructing the video
    fig, ax = plt.subplots()
    ims = []


    for i in range(num_slices):
        img = eval("img_3D" + dimension_choice)

        if not masks[0] is None:
            masks_2D = []
            for mask in masks:
                mask_2D = eval("mask" + dimension_choice)
                masks_2D.append(mask_2D)

        else:
            masks_2D = None

        im = ax.imshow(display_image(img, masks_2D, colors=colors.copy(), alpha=alpha, display=False), animated=True)
        text = ax.text(1.05, 0.5, f"{i}", transform=ax.transAxes, fontsize=30,
                       verticalalignment='bottom', horizontalalignment='left')
        ims.append([im, text])
        print(f"{i + 1} / {num_slices}")


    ani = animation.ArtistAnimation(fig, ims, interval=animation_interval, blit=True,
                                    repeat_delay=1000)

    ani.save(f"{filename}.mp4")



def displaySlice(CT_image, angle, slice_depth, mask = None, alpha = 0.6, colors = (0, 0, 1), display = True):
    """
    Cuts a 2D slice from a 3D image and displays it as a grayscale image.
    :numpy.ndarray CT_image: 3D image.
    :int angle: 0,1 or 2 represents which axis the image is viewed from.
    :int slice_depth: 0 to image depth. The index of the image along the angle axis.
    :numpy.ndarray mask: A binary array of the same dimensions as CT_image indicating
    the pixelwise segmentation
    :alpha float: A number between 0 and 1 indicating the strength of the mask color compared
    to the CT_image.
    :color tuple: A three tuple of 0 to 1 values indicating the RGB mixture of red, green and
    blue in the mask color.
    """

    # Rescale to 8-bit values
    CT_image = CT_image - np.min(CT_image)
    CT_image = CT_image / np.max(CT_image) * 255

    # Defines the indexing of the CT_image depending on the chosen view angle.
    dimension_choice = [f"[{slice_depth}, :, :]",
                        f"[:, {slice_depth}, :]",
                        f"[:, :, {slice_depth}]"][angle]

    # Redefines the CT_image as a 2D image corresponding to the selected slice and view.
    CT_image = eval("CT_image" + dimension_choice)

    if not mask is None:
        if type(mask) == list:
            new_mask = []
            for m in mask:
                new_mask.append(eval("m" + dimension_choice))
            
            mask = new_mask
        
        else:

            # If a mask is provided, the corresponding slice is made into the mask
            mask = eval("mask" + dimension_choice)

    image = display_image(CT_image, mask, colors=colors, alpha=alpha, display=display)
    
    if display == False:
        return image



def generate_dashed_lines(start_point, end_point, dash_length, gap_length):
    # Convert start and end points to NumPy arrays
    start_point = np.array(start_point)
    end_point = np.array(end_point)

    # Calculate the direction vector
    direction = end_point - start_point

    # Calculate the length of the direction vector
    length = np.linalg.norm(direction)

    # Normalize the direction vector
    direction_normalized = direction / length

    # Calculate the number of dashes required
    num_dashes = int(length / (dash_length + gap_length))

    # Calculate the step size for each dash
    step_size = (dash_length + gap_length) * direction_normalized

    # Generate the dashed lines
    dashed_lines = []
    for i in range(num_dashes):
        dash_start = start_point + i * step_size
        dash_end = dash_start + dash_length * direction_normalized
        dashed_lines.append((tuple(dash_start.astype(int)), tuple(dash_end.astype(int))))

    return dashed_lines

def plot_three_slices(CT_image,  dim_depths: list, mask=None, alpha = 0.6, colors = (0, 1, 0), thickness=1, dash_space = 4):

    dim_depths_openCV_version = [dim_depths[0], CT_image.shape[1] - dim_depths[1], dim_depths[2]]

    vertical_line_placements = {0: dim_depths[1], 1: dim_depths[0], 2: dim_depths[0]}
    horizontal_line_placements = {0: CT_image.shape[2] - dim_depths[2], 1: CT_image.shape[2] - dim_depths[2], 2: CT_image.shape[1] - dim_depths[1]}


    images = []

    for i in range(3):
        img = displaySlice(CT_image, i, dim_depths[i], mask, alpha = alpha, colors = colors, display = False)
        img = np.rot90(img)
        #img = np.flipud(img)
        
        
        start_point_vertical = (vertical_line_placements[i], 0)
        end_point_vertical = (vertical_line_placements[i], img.shape[0] + 10)
        
        start_point_horizontal = (0, horizontal_line_placements[i])
        end_point_horizontal = (img.shape[1], horizontal_line_placements[i])


        color = (0, 0, 255)  # Blue color

        img = img.astype(np.uint8)
        img = img.copy()

        vertical_dashes = generate_dashed_lines(start_point_vertical, end_point_vertical, dash_space, dash_space)
        horizontal_dashes = generate_dashed_lines(start_point_horizontal, end_point_horizontal, dash_space, dash_space)

        for vertical_stop, vertical_start in vertical_dashes:
            cv2.line(img, vertical_stop, vertical_start, color, thickness)

        for horizontal_start, horizontal_stop in horizontal_dashes:
            cv2.line(img, horizontal_start, horizontal_stop, color, thickness)

        images.append(img)
    
    return images



if __name__ == "__main__":
    from src.visualization.plot_functions import plot_three_slices
    import nibabel as nib
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import cv2
    import pandas as pd

    from monai.transforms import (
        AsDiscrete,
        AsDiscreted,
        EnsureChannelFirstd,
        Compose,
        CropForegroundd,
        LoadImaged,
        Orientationd,
        RandCropByPosNegLabeld,
        SaveImaged,
        ScaleIntensityRanged,
        Spacingd,
        Invertd,
        LabelFilterd,
        Rand3DElasticd,
        RandRotate90d,
        RandShiftIntensityd,
        RandZoomd
    )


    load_transform = Compose([LoadImaged(keys=["image", "label"])], Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")))#, Orientationd(keys=["image", "label"], axcodes="RAS")])

    # specify the path to the NIfTI file
    hepatic_path = '/dtu/3d-imaging-center/courses/02510/data/MSD/Task08_HepaticVessel/'

    # get the names of all files in the directory
    file_names = [f for f in os.listdir(hepatic_path + "imagesTr/") if not f.startswith('.')]

    fname = file_names[20]

    print(fname)

    load_dict = load_transform({"image": hepatic_path + "imagesTr/" + fname , "label": hepatic_path + "labelsTr/" + fname})

    CT_image = load_dict["image"]
    label = load_dict["label"]
    mask = CT_image <= 0#-57
    mask1 = CT_image >= 300


    dim_depths = [300, 200, 70]
    
    
    images = plot_three_slices(CT_image,  dim_depths, [mask, label, mask1], alpha = 0.6, colors = [])

    fig = plt.figure(figsize=(6,4))
    fig.subplots_adjust(bottom=0.025, left=0.025, top = 0.975, right=0.975)

    X = [ (2,2,1), (2,2,3), (2,2,(2,4))]
    titles = ['Sagittal plane', 'Coronal plane', 'Transverse plane']

    for i, (nrows, ncols, plot_number) in enumerate(X):
        sub = fig.add_subplot(nrows, ncols, plot_number)
        sub.imshow(images[i])
        sub.set_title(titles[i])
        sub.set_xticks([])
        sub.set_yticks([])


    plt.show()

    #animate_CT(img_3D, angle = 1, masks = img_3D < -57, colors = [(0, 0, 1)], alpha=0.2, animation_interval = 50, filename="reports/figures/02_Data/slices")