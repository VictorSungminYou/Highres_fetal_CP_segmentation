import numpy as np   
import nibabel as nib
import glob, os
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
import argparse
from tqdm import tqdm


def main(input_data_path, output_data_path, mask_file, GA_info = None, input_size = 192, split_method = 'train_valid_split', n_fold = 5):

    # Path of Dataset
    # You need to modify code to properly search MRI data and Segmentation label
    MR_list = np.asarray(sorted(glob.glob(input_data_path + '/Upsample_MR' + '/*.nii')))
    GT_list = np.asarray(sorted(glob.glob(input_data_path + '/Upsample_GT_LAS' + '/*.nii')))

    # Cross_validation is not implemented yet since we are not going to evaluate models performace this time.
    # Use train-test-split for now.
    # You may need to implement it later for the experiment for new model.

    mask_im = np.squeeze(nib.load(mask_file).get_fdata())
    mask_min = np.min(np.where(mask_im),axis=1)
    mask_max = np.max(np.where(mask_im),axis=1)
    repre_size = mask_max-mask_min
    repre_size[1] = input_size

    if split_method == 'train_valid_split':
        # Train-validation split
        output_dir = output_data_path + '/split'

        os.makedirs(output_dir+"/axi/train", exist_ok=True)
        os.makedirs(output_dir+"/axi/valid", exist_ok=True)
        os.makedirs(output_dir+"/cor/train", exist_ok=True)
        os.makedirs(output_dir+"/cor/valid", exist_ok=True)
        os.makedirs(output_dir+"/sag/train", exist_ok=True)
        os.makedirs(output_dir+"/sag/valid", exist_ok=True)
        
        MR_list_train, MR_list_valid, GT_list_train, GT_list_valid  = list(train_test_split(MR_list, GT_list, test_size=0.1, random_state=1))
        process_data(MR_list_train, GT_list_train, MR_list_valid, GT_list_valid, output_dir, input_size, mask_im, mask_min, mask_max, repre_size, flip=True)
        
    elif split_method == 'kf':
        # K-fold cross-validation
        output_dir = 'Highres_dataset/kf'
        
        for fold_idx in range(0,n_fold):
            os.makedirs(output_dir+"/{0}".fold_idx+"/axi/train", exist_ok=True)
            os.makedirs(output_dir+"/{0}".fold_idx+"/axi/valid", exist_ok=True)
            os.makedirs(output_dir+"/{0}".fold_idx+"/cor/train", exist_ok=True)
            os.makedirs(output_dir+"/{0}".fold_idx+"/cor/valid", exist_ok=True)
            os.makedirs(output_dir+"/{0}".fold_idx+"/sag/train", exist_ok=True)
            os.makedirs(output_dir+"/{0}".fold_idx+"/sag/valid", exist_ok=True)
        
        kf = KFold(n_splits=n_fold, random_state=1, shuffle=True)
        fold_info = list(kf.split(MR_list))
        ## The code to process_data for each fold need to be coded here.
        
    else:
        # Stratified K-fold (group with GA)
        fold_group = np.loadtxt(GA_info).astype(int)
        skf = StratifiedKFold(n_splits=n_fold, random_state=1, shuffle=True)
        fold_info = list(skf.split(MR_list, fold_group))
        ## The code to process_data for each fold need to be coded here.

def get_MR_data(img, label, mask_img, mask_min, mask_max):
    
    img = np.squeeze(nib.load(img).get_fdata()) * mask_img
    img = img[mask_min[0]:mask_max[0],mask_min[1]+2:mask_max[1]-1,mask_min[2]:mask_max[2]]
    # clip pixel less than 2% or over 98%
    loc = np.where(img<np.percentile(img,2))
    img[loc]=0
    loc = np.where(img>np.percentile(img,98))
    img[loc]=0
    loc = np.where(img)
    img[loc] = (img[loc] - np.mean(img[loc])) / np.std(img[loc])
    label = np.squeeze(nib.load(label).get_fdata())
    label = label[mask_min[0]:mask_max[0],mask_min[1]+2:mask_max[1]-1,mask_min[2]:mask_max[2]]
    return img, label

def axfliper(array,f=0):
    import numpy as np
    if f:
        array = array[:,:,::-1,:]
        array2 = np.concatenate((array[:,:,:,0,np.newaxis],array[:,:,:,2,np.newaxis],array[:,:,:,1,np.newaxis],
                                 array[:,:,:,4,np.newaxis],array[:,:,:,3,np.newaxis]),axis=-1)
        return array2
    else:
        array = array[:,:,::-1,:]
    return array

def cofliper(array,f=0):
    import numpy as np
    if f:
        array = array[:,::-1,:,:]
        array2 = np.concatenate((array[:,:,:,0,np.newaxis],array[:,:,:,2,np.newaxis],array[:,:,:,1,np.newaxis],
                                 array[:,:,:,4,np.newaxis],array[:,:,:,3,np.newaxis]),axis=-1)
        return array2
    else:
        array = array[:,::-1,:,:]
    return array


def preprocess_volume(MR_volume, GT_volume, input_size, view, mask_img, mask_min, mask_max, repre_size, flip):

    # For each input with designated view
    img, label = get_MR_data(MR_volume, GT_volume, mask_img, mask_min, mask_max)

    # Initialization
    if view == 'axi':
        X_image = np.zeros([repre_size[2], input_size, input_size,1])
        Y_label = np.zeros([repre_size[2], input_size, input_size,5])
    elif view == 'cor':
        X_image = np.zeros([repre_size[1], input_size, input_size,1])
        Y_label = np.zeros([repre_size[1], input_size, input_size,5])
    elif view == 'sag':
        X_image = np.zeros([repre_size[0], input_size, input_size,1])
        Y_label = np.zeros([repre_size[0], input_size, input_size,3])
    
    if view == 'axi':
        img2 = np.pad(img,((int(np.floor((input_size-img.shape[0])/2)),int(np.ceil((input_size-img.shape[0])/2))),
                        (int(np.floor((input_size-img.shape[1])/2)),int(np.ceil((input_size-img.shape[1])/2))),
                        (0,0)), 'constant')
        X_image[:repre_size[2],:,:,0]= np.swapaxes(img2,2,0)
        img2 = np.pad(label,((int(np.floor((input_size-img.shape[0])/2)),int(np.ceil((input_size-img.shape[0])/2))),
                            (int(np.floor((input_size-img.shape[1])/2)),int(np.ceil((input_size-img.shape[1])/2))),
                            (0,0)), 'constant')
        img2 = np.swapaxes(img2,2,0)
    elif view == 'cor':
        img2 = np.pad(img,((int(np.floor((input_size-img.shape[0])/2)),int(np.ceil((input_size-img.shape[0])/2))),
                        (0,0),
                        (int(np.floor((input_size-img.shape[2])/2)),int(np.ceil((input_size-img.shape[2])/2)))),'constant')
        X_image[:repre_size[1],:,:,0]= np.swapaxes(img2,1,0)
        img2 = np.pad(label,((int(np.floor((input_size-img.shape[0])/2)),int(np.ceil((input_size-img.shape[0])/2))),
                            (0,0),
                            (int(np.floor((input_size-img.shape[2])/2)),int(np.ceil((input_size-img.shape[2])/2)))),'constant')
        img2 = np.swapaxes(img2,1,0)
    elif view == 'sag':
        img2 = np.pad(img,((0,0),
                        (int(np.floor((input_size-img.shape[1])/2)),int(np.ceil((input_size-img.shape[1])/2))),
                        (int(np.floor((input_size-img.shape[2])/2)),int(np.ceil((input_size-img.shape[2])/2)))), 'constant')
        X_image[:repre_size[0],:,:,0]= img2
        img2 = np.pad(label,((0,0),
                            (int(np.floor((input_size-img.shape[1])/2)),int(np.ceil((input_size-img.shape[1])/2))),
                            (int(np.floor((input_size-img.shape[2])/2)),int(np.ceil((input_size-img.shape[2])/2))),), 'constant')

    if (view == 'axi') | (view == 'cor'):
        img3 = np.zeros_like(img2)
        back_loc = np.where(img2<0.5)
        left_in_loc = np.where((img2>160.5)&(img2<161.5))
        right_in_loc = np.where((img2>159.5)&(img2<160.5))
        left_plate_loc = np.where((img2>0.5)&(img2<1.5))
        right_plate_loc = np.where((img2>41.5)&(img2<42.5))
        img3[back_loc]=1
    elif view == 'sag':
        img3 = np.zeros_like(img2)
        back_loc = np.where(img<0.5)
        in_loc = np.where((img2>160.5)&(img2<161.5)|(img2>159.5)&(img2<160.5))
        plate_loc = np.where((img2>0.5)&(img2<1.5)|(img2>41.5)&(img2<42.5))
        img3[back_loc]=1

    if view == 'axi':
        Y_label[:repre_size[2],:,:,0]=img3
        img3[:]=0
        img3[left_in_loc]=1
        Y_label[:repre_size[2],:,:,1]=img3
        img3[:]=0
        img3[right_in_loc]=1
        Y_label[:repre_size[2],:,:,2]=img3
        img3[:]=0
        img3[left_plate_loc]=1
        Y_label[:repre_size[2],:,:,3]=img3
        img3[:]=0
        img3[right_plate_loc]=1
        Y_label[:repre_size[2],:,:,4]=img3
        img3[:]=0
    elif view == 'cor':
        Y_label[:repre_size[1],:,:,0]=img3
        img3[:]=0
        img3[left_in_loc]=1
        Y_label[:repre_size[1],:,:,1]=img3
        img3[:]=0
        img3[right_in_loc]=1
        Y_label[:repre_size[1],:,:,2]=img3
        img3[:]=0
        img3[left_plate_loc]=1
        Y_label[:repre_size[1],:,:,3]=img3
        img3[:]=0
        img3[right_plate_loc]=1
        Y_label[:repre_size[1],:,:,4]=img3
        img3[:]=0
    elif view == 'sag':
        Y_label[:repre_size[0],:,:,0]=img3
        img3[:]=0
        img3[in_loc]=1
        Y_label[:repre_size[0],:,:,1]=img3
        img3[:]=0
        img3[plate_loc]=1
        Y_label[:repre_size[0],:,:,2]=img3
        img3[:]=0

    if flip:
        if view == 'axi':
            X_image=np.concatenate((X_image,X_image[:,::-1,:,:]),axis=0)
            Y_label=np.concatenate((Y_label,Y_label[:,::-1,:,:]),axis=0)
            X_image=np.concatenate((X_image, axfliper(X_image)),axis=0)
            Y_label=np.concatenate((Y_label, axfliper(Y_label, 1)),axis=0)
        elif view == 'cor':
            X_image=np.concatenate((X_image,X_image[:,:,::-1,:]),axis=0)
            Y_label=np.concatenate((Y_label,Y_label[:,:,::-1,:]),axis=0)
            X_image=np.concatenate((X_image, cofliper(X_image)),axis=0)
            Y_label=np.concatenate((Y_label, cofliper(Y_label, 1)),axis=0)
        elif view == 'sag':
            X_image = np.concatenate((X_image, X_image[:,:,::-1,:], X_image[:,::-1,:,:]),axis=0)
            Y_label = np.concatenate((Y_label, Y_label[:,:,::-1,:], Y_label[:,::-1,:,:]),axis=0)
        
    return X_image, Y_label
  
def process_data(MR_list_train, GT_list_train, MR_list_valid, GT_list_valid, output_dir, input_size, mask_im, mask_min, mask_max, repre_size, flip):

    # Training data
    print("Processing training set")
    for (MR_volume, GT_label) in tqdm(zip(MR_list_train, GT_list_train), total=len(MR_list_train)):
        volume_name = os.path.splitext(os.path.basename(MR_volume))[0]
    
        # Axial plane preprocessing
        X_MR, Y_GT = preprocess_volume(MR_volume, GT_label, input_size=input_size, view='axi', mask_img=mask_im, mask_min=mask_min, mask_max=mask_max, repre_size=repre_size, flip=flip)
        
        #print("X_MR shape: ", X_MR.shape)
        #print("Y_GT shape: ", Y_GT.shape)
   
        for slice_idx in range(0, X_MR.shape[0]):
            
            save_name = output_dir + "/axi/train/" + volume_name + "_{:03}".format(slice_idx)
            
            np.save(save_name + "_MR", X_MR[slice_idx])
            np.save(save_name + "_GT", Y_GT[slice_idx])

        # Coronal plane preprocessing
        X_MR, Y_GT = preprocess_volume(MR_volume, GT_label, input_size=input_size, view='cor', mask_img=mask_im, mask_min=mask_min, mask_max=mask_max, repre_size=repre_size, flip=flip)
        
        for slice_idx in range(0, X_MR.shape[0]):
            
            save_name = output_dir + "/cor/train/" + volume_name + "_{:03}".format(slice_idx)
            
            np.save(save_name + "_MR", X_MR[slice_idx])
            np.save(save_name + "_GT", Y_GT[slice_idx])

        # Sagittal plane preprocessing
        X_MR, Y_GT = preprocess_volume(MR_volume, GT_label, input_size=input_size, view='sag', mask_img=mask_im, mask_min=mask_min, mask_max=mask_max, repre_size=repre_size, flip=flip)
        
        for slice_idx in range(0, X_MR.shape[0]):
            
            save_name = output_dir + "/sag/train/" + volume_name + "_{:03}".format(slice_idx)
            
            np.save(save_name + "_MR", X_MR[slice_idx])
            np.save(save_name + "_GT", Y_GT[slice_idx])

    # Validation data
    print("Processing validation set")
    for (MR_volume, GT_label) in tqdm(zip(MR_list_valid, GT_list_valid), total=len(MR_list_valid)):
        volume_name = os.path.splitext(os.path.basename(MR_volume))[0]
    
        # Axial plane preprocessing
        X_MR, Y_GT = preprocess_volume(MR_volume, GT_label, input_size=input_size, view='axi', mask_img=mask_im, mask_min=mask_min, mask_max=mask_max, repre_size=repre_size, flip=flip)
        
        for slice_idx in range(0, X_MR.shape[0]):
            
            save_name = output_dir + "/axi/valid/" + volume_name + "_{:03}".format(slice_idx)
            
            np.save(save_name + "_MR", X_MR[slice_idx])
            np.save(save_name + "_GT", Y_GT[slice_idx])

        # Coronal plane preprocessing
        X_MR, Y_GT = preprocess_volume(MR_volume, GT_label, input_size=input_size, view='cor', mask_img=mask_im, mask_min=mask_min, mask_max=mask_max, repre_size=repre_size, flip=flip)
        
        for slice_idx in range(0, X_MR.shape[0]):
            
            save_name = output_dir + "/cor/valid/" + volume_name + "_{:03}".format(slice_idx)
            
            np.save(save_name + "_MR", X_MR[slice_idx])
            np.save(save_name + "_GT", Y_GT[slice_idx])

        # Sagittal plane preprocessing
        X_MR, Y_GT = preprocess_volume(MR_volume, GT_label, input_size=input_size, view='sag', mask_img=mask_im, mask_min=mask_min, mask_max=mask_max, repre_size=repre_size, flip=flip)
        
        for slice_idx in range(0, X_MR.shape[0]):
            
            save_name = output_dir + "/sag/valid/" + volume_name + "_{:03}".format(slice_idx)
            
            np.save(save_name + "_MR", X_MR[slice_idx])
            np.save(save_name + "_GT", Y_GT[slice_idx])
            
if __name__ == "__main__":
    # Data preparation code for CP segmentation of high resolution recon.
    # Only code train-test-split (train:90%, validation:10%) is implemented yet.

    parser = argparse.ArgumentParser('   ==========   Data preparation code for fetal Attention U-Net for CP segmentation on high resolution recontruction script made by Sungmin You (11.10.23 ver.0)   ==========   ')
    parser.add_argument('-Input_dir', '--Input_data_dir',action='store',dest='input_dir',type=str, required=True, help='Input path for raw MRI files(*.nii)')
    parser.add_argument('-Output_dir', '--Output_data_dir',action='store',dest='output_dir',type=str, required=True, help='Output path for organized dataset(*.npy)')
    parser.add_argument('-mask', '--mask', action='store',dest='mask',type=str, default='down_mask-31_dil5.nii', help='mask file to make dictionary and intensity normalize')
    #parser.add_argument('-f', '--num_fold',action='store',dest='num_fold',default=10, type=int, help='number of fold for training')
    #parser.add_argument('-fi', '--stratified_info_file',action='store', dest='stratified_info',type=str, help='information for stratified fold')
    parser.add_argument('-is', '--input_shape',action='store', dest='input_shape',type=int, default=192, help='Input size')
    args = parser.parse_args()

    main(input_data_path = args.input_dir, output_data_path = args.output_dir, mask_file = args.mask, input_size = args.input_shape)

