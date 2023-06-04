from src.visualization.plot_functions import plot_three_slices, displaySlice
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import pandas as pd
plt.rcParams['figure.dpi'] = 300

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


load_transform = Compose([LoadImaged(keys=["image", "label"]), LabelFilterd(keys=["label"], applied_labels=[1])])#, Orientationd(keys=["image", "label"], axcodes="RAS")])

# specify the path to the NIfTI file
hepatic_path = '/dtu/3d-imaging-center/courses/02510/data/MSD/Task08_HepaticVessel/'

# get the names of all files in the directory
file_names = [f for f in os.listdir(hepatic_path + "imagesTr/") if not f.startswith('.')]

fname = file_names[10]

print(fname)

load_dict = load_transform({"image": hepatic_path + "imagesTr/" + fname , "label": hepatic_path + "labelsTr/" + fname})

    


PLOT_NAME = ["raw_hepatic_vessels", "depth_vs_spacing", "spacing_distribution", "intensity_histogram", "final_preprocessing", "high_low_res", "extra_hist"][-1]

if PLOT_NAME == "raw_hepatic_vessels":

    CT_image = load_dict["image"]
    mask  = load_dict["label"]

    #CT_image = np.rot90(CT_image, k=1, axes=(0, 1))
    #mask = np.rot90(mask, k=1, axes=(0, 1))

    dim_depths = [300, 200, 70]
    
    
    images = plot_three_slices(CT_image,  dim_depths, mask, alpha = 0.6, colors = [])

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

    plt.savefig("reports/figures/02_Data/raw_data_hepatic_vessels.png")

    plt.show()









if PLOT_NAME == "depth_vs_spacing":
    # Load CSV file into a pandas DataFrame
    df = pd.read_csv("reports/figures/Data_statistics/hepaticDataExploration.csv")

    # Extract the two variables of interest
    x = df["dim2"]
    y = df["pixdim2"]

    z = [x[i]* y[i] for i in range(len(x))]

    # Create a scatter plot
    plt.scatter(x, y, label='CT scans')
    x = np.linspace(24, 181, 1000)
    y_low = 165.3 / x
    y_high = 284.5 / x
    plt.plot(x, y_low, color='red', label='Boundaries for physical size of 90% of the data')
    plt.plot(x, y_high, color='red')
    plt.legend()
    plt.xlabel("Scan depth")
    plt.ylabel("Spacing")
    plt.title("Spacing vs. scan depth along superior-inferior axis")

    plt.savefig("reports/figures/02_Data/depth_vs_spacing.png")
    plt.show()









if PLOT_NAME == "spacing_distribution":
    df = pd.read_csv('reports/figures/Data_statistics/hepaticDataExploration.csv')

    # Generate random data for the histograms
    data1 = df["pixdim0"]
    data2 = df["pixdim2"]

    # Create a figure and axis object
    fig, ax = plt.subplots()
    
    #bins = np.linspace(min(data1.min(), data2.min()), max(data1.max(), data2.max()), 20)
    bins = np.arange(0, 8.25, 0.25)


    # Plot the red histogram
    ax.hist(data1, bins=bins, color='red', alpha=0.5, label='Spacing in the first two dimensions', density=True)

    # Plot the blue histogram
    ax.hist(data2, bins=bins, color='blue', alpha=0.5, label='Spacing along the superior-inferior axis', density=True)

    # Add a red vertical line at x=0
    ax.axvline(x=1, color='red', linestyle='--', linewidth=2, label='Resampling spacing in the first two dimensions')

    # Add a blue vertical line at x=2
    ax.axvline(x=2.5, color='blue', linestyle='--', linewidth=2, label='Resampling spacing along the superior-inferior axis')

    # Set the labels and title
    ax.set_xlabel('Spacing')
    ax.set_ylabel('Normalized frequency')
    ax.set_title('Distrubutions of scan spacings')

    # Add a legend
    ax.legend()

    plt.savefig("reports/figures/02_Data/spacing_distribution.png")

    # Show the plot
    plt.show()









if PLOT_NAME == "intensity_histogram":
    df = pd.read_csv('reports/figures/Data_statistics/hepaticDataExploration.csv')

    import ast
    df["img_hist"] = df["img_hist"].apply(ast.literal_eval)
    df["img_hist"] = df["img_hist"].apply(np.array)

    df["label_hist"] = df["label_hist"].apply(ast.literal_eval)
    df["label_hist"] = df["label_hist"].apply(np.array)


    all_intensities_hist = np.sum(df['img_hist'].values, axis=0)
    all_intensities_label = np.sum(df['label_hist'].values, axis=0)
    all_intensities_label[1024] = 0 #Remove all 0 intensity values
    initial_bin_edges = np.arange(-1024, 4544)
    print(np.mean(df['pixdim0']))
    """
    sumall = np.sum(all_intensities_label)
    q05 = sumall*0.05//100
    q95 = sumall*99//100

    mindst = 0
    størst = 0
    for i, y in enumerate(all_intensities_label):
        mindst += y
        if mindst >= q05:
            mindst = i -1024
            break
    
    for i, y in enumerate(all_intensities_label[::-1]):
        størst += y
        if størst >= q05:
            størst = 4543- i
            break
    

    print(mindst)
    print(størst)
    """

    bin_edges = []
    result = []
    result2 = []

    # Iterate over the array in groups of 10
    for i in range(0, len(all_intensities_hist), 50):
        # Sum the elements in the current group
        group_sum = sum(all_intensities_hist[i:i+50])
        group_sum2 = sum(all_intensities_label[i:i+50])
        # Append the group sum to the result array
        result.append(group_sum)
        result2.append(group_sum2)
        bin_edges.append(initial_bin_edges[i])



    """
    fig = plt.figure(figsize=(6,4))
    fig.subplots_adjust(bottom=0.025, left=0.025, top = 0.975, right=0.026)

    
    # Plot a
    subplot_a = fig.add_subplot(2, 3, (1,2))
    subplot_a.bar(bin_edges, result/np.sum(result), alpha=0.5, width=50, align='edge')
    subplot_a.set_xlabel('Bins')
    subplot_a.set_ylabel('Normalized frequency')
    subplot_a.set_title('a) Intensity distribution')

    # Plot b
    subplot_b = fig.add_subplot(2, 3, 3)
    subplot_b.bar(bin_edges[20:40], result[20:40], alpha=0.5, width=50, align='edge', label='All voxels')
    subplot_b.bar(bin_edges[20:40], result2[20:40], alpha=1, width=50, align='edge', color="red", label='Positive labels')
    subplot_b.set_xlabel('Bins')
    subplot_b.set_ylabel('Counts')
    subplot_b.set_title('b) Intensity counts')
    subplot_b.legend()
    subplot_b.set_box_aspect(1)

    # plot c
    subplot_c = fig.add_subplot(2, 3, (4,5))
    subplot_c.bar(bin_edges[:40], result[:40]/sum(result), alpha=0.5, width=50, align='edge', label= "All voxels")
    subplot_c.bar(bin_edges[:40], result2[:40]/sum(result2), alpha=0.5, width=50, align='edge', color="red", label="Positive labels")
    subplot_c.set_xlabel('Bins')
    subplot_c.set_ylabel('Normalized frequency')
    subplot_c.set_title('c) Distributions of voxel intenities')
    subplot_c.axvline(x=-58, color='yellow', linestyle='--', linewidth=2, label='Range boundaries')
    subplot_c.axvline(x=478, color='yellow', linestyle='--', linewidth=2)
    subplot_c.legend()
    #subplot_c.legend(fontsize="xx-small", loc="upper left")

    # plot d
    CT_image = load_dict["image"]
    mask  = load_dict["label"]
    mask = mask == 1
    

    img = displaySlice(CT_image, 2, 80, [mask, CT_image <= -58, CT_image >= 478], colors=[(1, 0, 0), (0, 0, 0.5), (0, 0.5, 0)] , alpha=1, display=False)
    img = np.rot90(img)

    subplot_d = fig.add_subplot(2, 3, 6)
    subplot_d.imshow(img)
    subplot_d.set_title('d) Transverse plane')
    subplot_d.set_xticks([])
    subplot_d.set_yticks([])
    """



    fig = plt.figure(figsize=(9,6))
    #fig.subplots_adjust(bottom=0.025, left=0.025, top = 0.975, right=0.026)

    
    # Plot a
    subplot_a = fig.add_subplot(2, 1, 1)
    subplot_a.bar(bin_edges[15:30], result[15:30], alpha=0.5, width=50, align='edge', label="All voxel intensitis")
    subplot_a.bar(bin_edges[15:30], result2[15:30], alpha=1, width=50, align='edge', color="red", label='Positive label intensities')
    subplot_a.set_xlabel('Bins')
    subplot_a.set_ylabel('Counts')
    subplot_a.legend()
    subplot_a.set_title('a) Voxel intensity counts')

    # plot c
    subplot_c = fig.add_subplot(2, 1, 2)
    subplot_c.bar(bin_edges[:40], result[:40]/sum(result), alpha=0.5, width=50, align='edge', label= "All voxels")
    subplot_c.bar(bin_edges[:40], result2[:40]/sum(result2), alpha=0.5, width=50, align='edge', color="red", label="Positive labels")
    subplot_c.set_xlabel('Bins')
    subplot_c.set_ylabel('Normalized frequency')
    subplot_c.set_title('b) Distributions of voxel intenities')
    subplot_c.axvline(x=-58, color='yellow', linestyle='--', linewidth=2, label='Range boundaries')
    subplot_c.axvline(x=478, color='yellow', linestyle='--', linewidth=2)
    subplot_c.legend()
    #subplot_c.legend(fontsize="xx-small", loc="upper left")



    plt.tight_layout()

    plt.savefig("reports/figures/02_Data/scaling_hepatic.png")

    plt.show()




if PLOT_NAME == "final_preprocessing":
    # specify the path to the NIfTI file
    hepatic_path = '/dtu/3d-imaging-center/courses/02510/data/MSD/Task08_HepaticVessel/'
    # get the names of all files in the directory
    file_names = [f for f in os.listdir(hepatic_path + "imagesTr/") if not f.startswith('.')]

    fname = file_names[37]

    def select_kidney(x):
        return x == 1

    prep_steps = [LoadImaged(keys=["image", "label"]), 
        EnsureChannelFirstd(keys=["image", "label"]), 
        Spacingd(keys=["image", "label"], pixdim=(1, 1, 1), mode=("bilinear", "nearest")), 
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-58,
            a_max=478,
            b_min=0.0,
            b_max=1.0,
            clip=True),
            CropForegroundd(keys=["image", "label"], select_fn=select_kidney, source_key="label", margin=20)
        ]

    all_dim_depths = [[int(512*0.75), int(512*0.6), int(54*0.75)], [int(400*0.75), int(400*0.6), int(266*0.75)], [int(400*0.75), int(400*0.6), int(266*0.75)], [146, 102, 125]]
    
    import torch
    fig = plt.figure()
    for i in range(4):
        load_transform = Compose(list(prep_steps[:2+i]))#, Orientationd(keys=["image", "label"], axcodes="RAS")])
        load_dict = load_transform({"image": hepatic_path + "imagesTr/" + fname , "label": hepatic_path + "labelsTr/" + fname})

        CT_image = torch.squeeze(load_dict["image"], dim=0)
        mask  = torch.squeeze(load_dict["label"], dim=0)
        print(CT_image.shape)
        print(mask.shape)
        print(load_dict['image_meta_dict']['pixdim'][1:4])
        dim_depths = all_dim_depths[i]
        
        
        images = plot_three_slices(CT_image,  dim_depths, mask, alpha = 0.6)

        #figsize=(6,4))
        #fig.subplots_adjust(bottom=0.025, left=0.025, top = 0.975, right=0.975)

        X = [ (4,3, 3*i + 1), (4,3, 3*i + 2), (4,3, 3*i + 3)]
        titles = ['Sagittal plane', 'Coronal plane', 'Transverse plane']

        for i, (nrows, ncols, plot_number) in enumerate(X):
            sub = fig.add_subplot(nrows, ncols, plot_number)
            sub.imshow(images[i])
            sub.set_xticks([])
            sub.set_yticks([])

    plt.savefig("reports/figures/02_Data/final_preprocessing.png")

    plt.show()





if PLOT_NAME == "high_low_res":
    df = pd.read_csv('reports/figures/Data_statistics/hepaticDataExploration.csv')

    low_res_idx = df["pixdim2"].idxmax()

    low_res_img = df.loc[low_res_idx]["filename"]

    high_res_idx = df["pixdim2"].idxmin()
    #median_res_idx = df[df["pixdim2"] == 5].index[0]

    high_res_img = df.loc[high_res_idx]["filename"]

    import torch

    prep_steps = [LoadImaged(keys=["image", "label"]), 
        EnsureChannelFirstd(keys=["image", "label"]), 
        Spacingd(keys=["image", "label"], pixdim=(1, 1, 1), mode=("bilinear", "nearest")), 
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-58,
            a_max=478,
            b_min=0.0,
            b_max=1.0,
            clip=True)
        ]

    hepatic_path = '/dtu/3d-imaging-center/courses/02510/data/MSD/Task08_HepaticVessel/'

    fname = high_res_img
    load_transform = Compose(prep_steps)#, Orientationd(keys=["image", "label"], axcodes="RAS")])
    load_dict = load_transform({"image": hepatic_path + "imagesTr/" + fname , "label": hepatic_path + "labelsTr/" + fname})


    CT_image = torch.squeeze(load_dict["image"])
    mask  = torch.squeeze(load_dict["label"])
    mask = mask == 1
    

    img = displaySlice(CT_image, 1, 250, mask, display=False)
    img1 = np.rot90(img)





    fname = low_res_img
    load_transform = Compose(prep_steps)#, Orientationd(keys=["image", "label"], axcodes="RAS")])
    load_dict = load_transform({"image": hepatic_path + "imagesTr/" + fname , "label": hepatic_path + "labelsTr/" + fname})


    CT_image = torch.squeeze(load_dict["image"])
    mask  = torch.squeeze(load_dict["label"])
    mask = mask == 1
    

    img = displaySlice(CT_image, 1, 230, mask, display=False)
    img2 = np.rot90(img)
    fig = plt.figure()

    sub = fig.add_subplot(1, 2, 1)
    sub.imshow(img1)
    sub.set_title("High resolution")
    sub.set_xticks([])
    sub.set_yticks([])

    img3 = img2[20:-44,:,:]
    sub = fig.add_subplot(1, 2, 2)
    sub.imshow(img3)
    sub.set_title("Low resolution")
    sub.set_xticks([])
    sub.set_yticks([])
    print(img1.shape, "  ", img3.shape)
    plt.savefig("reports/figures/02_Data/high_low_res.png")

    plt.show()




    



if PLOT_NAME == "extra_hist":
    df = pd.read_csv('reports/figures/Data_statistics/hepaticDataExploration.csv')

    import ast
    df["img_hist"] = df["img_hist"].apply(ast.literal_eval)
    df["img_hist"] = df["img_hist"].apply(np.array)

    df["label_hist"] = df["label_hist"].apply(ast.literal_eval)
    df["label_hist"] = df["label_hist"].apply(np.array)

    label_stuff = [np.sum(l*np.arange(-1024, 4544))/(np.sum(l)- l[1024]) for l in df["label_hist"]]
    #label_stuff2 = [np.sum(l*np.arange(-1024, 4544))/(np.sum(l)- l[1024]) for l in df["img_hist"]]

    bins = np.linspace(np.min(label_stuff),np.max(label_stuff), 30)
    hist, edges = np.histogram(label_stuff, bins = bins, density=True)
    x = (edges[1:] + edges[:-1])/2
    width = bins[1]-bins[0]
    fig, ax = plt.subplots()
    ax.bar(x,hist, width = .9*width, label="Normalized scan count")


    #plt.legend(loc="upper right")

    ax.set_xlabel('Average label intensity')
    ax.set_ylabel('Normalized frequency')
    ax.set_title('Distribution of average label intensity across scans')
    plt.savefig("reports/figures/02_Data/label_intensity.png")
    plt.show()
    #plt.hist(label_stuff)
    #plt.show()




