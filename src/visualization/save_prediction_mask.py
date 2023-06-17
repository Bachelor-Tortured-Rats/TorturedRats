import glob
from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscreted,
    Activationsd,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
)
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, Dataset, decollate_batch
import torch
import os

from tqdm import tqdm

from src.models.unet_model import load_unet


def select_kidney(x):
    return x == 1

test_org_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1, 1, 2.5), mode="bilinear"),
        CropForegroundd(keys=["image", "label"], select_fn=select_kidney, source_key="label", margin=20),
        CropForegroundd(keys=["image"], source_key="image"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=255,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        
    ]
)

test_org_transforms_rats = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        Orientationd(keys=["image",'mask'], axcodes="RAS"),
        Spacingd(keys=["image", "mask"], pixdim=(0.0226, 0.0226, 0.0226), mode=("bilinear", "nearest")),
        CropForegroundd(keys=["image"], source_key="mask"), 
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=255,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]
)


post_transforms = Compose(
    [
        Invertd(
            keys="pred",
            transform=test_org_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        #AsDiscreted(keys="pred", argmax=True, to_onehot=2),
        Activationsd(keys="pred", softmax=True), 
        AsDiscreted(keys="pred", argmax=True),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir='reports/save_prediction_mask', output_postfix="seg", resample=False),
    ]
)
post_transforms_rats = Compose(
    [
        Invertd(
            keys="pred",
            transform=test_org_transforms_rats,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        #AsDiscreted(keys="pred", argmax=True, to_onehot=2),
        Activationsd(keys="pred", softmax=True), 
        AsDiscreted(keys="pred", argmax=True),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir='reports/end/save_prediction_mask/16742264', output_postfix="seg", resample=False),
    ]
)

def save_prediction_masks(model, test_org_loader, post_transforms, device = 'cpu'):
    model.eval()

    with torch.no_grad():
        for test_data in tqdm(test_org_loader):
            test_inputs = test_data["image"]#.to(device)
            roi_size = (160, 160, 160)
            sw_batch_size = 8
            test_data["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model,sw_device=device,progress=True)

            test_data = [post_transforms(i) for i in decollate_batch(test_data)]

def save_model_prediction_mask(model, image_data, model_name,device = 'cpu'):
    roi_size = (160, 160, 160)
    sw_batch_size = 4
    
    post_transforms = Compose(
    [
        Invertd(
            keys="pred",
            transform=test_org_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        #AsDiscreted(keys="pred", argmax=True, to_onehot=2),
        Activationsd(keys="pred", softmax=True), 
        AsDiscreted(keys="pred", argmax=True),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=f'reports/save_prediction_mask/{model_name}', output_postfix="seg", resample=False),
    ]
    )
    
    model.eval()
    with torch.no_grad():
        image_data["pred"] = sliding_window_inference(image_data['image'], roi_size, sw_batch_size, model,sw_device=device,progress=True)
        image_data = [post_transforms(i) for i in decollate_batch(image_data)]

def save_models_prediction_masks(models_dict: dict, image_path, label_path):
    # image_data = test_org_transforms({'image': image_path, 'label': label_path})
    # image_data['image']= image_data['image'].unsqueeze(0)
    # image_data['label']= image_data['label'].unsqueeze(0)

    test_org_ds = Dataset([{'image': image_path, 'label': label_path}], transform=test_org_transforms)
    test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=1)

    for test_data in tqdm(test_org_loader):
        image_data = test_data
        break

    for model_name, model_path in models_dict.items():
        model = load_unet(model_path, device='cpu')[0]
        save_model_prediction_mask(model, image_data, model_name,device = 'cpu')
        
def main(model_load_path, data_type, num_images_to_test=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, params = load_unet(model_load_path, device=device)

    if data_type == "Hepatic":
        data_path = '/dtu/3d-imaging-center/courses/02510/data/MSD/Task08_HepaticVessel/'
        train_images = sorted(glob.glob(os.path.join(data_path, "imagesTr", "*.nii.gz")))
        train_labels = sorted(glob.glob(os.path.join(data_path, "labelsTr", "*.nii.gz")))

        data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)][:num_images_to_test]

        test_org_ds = Dataset(data=data_dicts, transform=test_org_transforms)
        test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=1)

        save_prediction_masks(model, test_org_loader, post_transforms)
    elif data_type == "rat37":
        train_images = ['/dtu/3d-imaging-center/projects/2020_QIM_22_rat_kidney/analysis/analysis_rat37/rat37_reorient.nii.gz']
        train_masks = ['/dtu/3d-imaging-center/projects/2020_QIM_22_rat_kidney/analysis/study_diabetic/aligned/rat37_aligned_rigid.nii']
        data_dicts = [{"image": image_name, "mask": train_mask} for image_name, train_mask in zip(train_images, train_masks)]

        print('loading data')
        test_org_ds = Dataset(data=data_dicts, transform=test_org_transforms_rats)
        test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=0)
        print('Finished loading data')

        save_prediction_masks(model, test_org_loader, post_transforms_rats, device)

if __name__ == "__main__":
    model_load_path_rat = "models/finetune-kfold/model_16742264.pth"
    model_load_path_rat_transfer = 'models/finetune-kfold/model_16803186-step500.pth'
    model_load_path= 'models/finetune-kfold/model_16694925.pth'
    data_save_path = 'reports/save_prediction_mask'

    model_load_path_rat_transfer_2 = 'models/finetune-kfold/model_16837355.pth'

    main('models/finetune-kfold/model_16742264.pth', data_type="rat37", num_images_to_test=3)


    # models_dict = {
    #     "3drpl_lp_0.02" : "models/finetune-kfold/model_16753065.pth", 
    #     "transfer_lp_0.02": "models/finetune-kfold/model_16761110.pth",
    #     "random_lp_0.02" : "models/finetune-kfold/model_16752015.pth",                 
    #     "3drpl_lp_1.0" : "models/finetune-kfold/model_16753090.pth",
    #     "transfer_lp_1.0": "models/finetune-kfold/model_16761137.pth",
    #     "random_lp_1.0" : "models/finetune-kfold/model_16752041.pth",
    #     }
    
    # print(type(models_dict))
    # save_models_prediction_masks(models_dict, '/zhome/a2/4/155672/Desktop/Bachelor/Task08_HepaticVessel/imagesTr/hepaticvessel_185.nii.gz', '/zhome/a2/4/155672/Desktop/Bachelor/Task08_HepaticVessel/labelsTr/hepaticvessel_185.nii.gz')