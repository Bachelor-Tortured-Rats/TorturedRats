import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import colorsys
import random
from itertools import combinations

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
                masks[comb[0]] = maskA - intersect
                masks[comb[1]] = maskB - intersect

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



def displaySlice(CT_image, angle, slice_depth, mask = None, alpha = 0.6, colors = (0, 0, 1)):
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
        # If a mask is provided, the corresponding slice is made into the mask
        mask = eval("mask" + dimension_choice)

    display_image(CT_image, mask, colors=colors, alpha=alpha, display=True)