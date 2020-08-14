# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 2020
@author: Yujin Oh (yujin.oh@kaist.ac.kr)
"""

import header 

# common
import torch
from torchvision.transforms import functional as TF
import random
import numpy as np

# dataset
from torch.utils.data import Dataset, DataLoader
import os
import glob
from PIL import Image

# add
import csv
import shutil


class MyTrainDataset(Dataset):

    def __init__(self, image_path, sampler):

        self.sample_arr = glob.glob(str(image_path) + "/*")
        self.sample_arr.sort()
        self.sample_arr = [self.sample_arr[x] for x in sampler]
        self.data_len = len(self.sample_arr)

        self.ids = []
        self.images = []
        self.masks = []

        if not os.path.isdir(image_path):
            raise RuntimeError('Dataset not found or corrupted. DIR : ' + image_path)

        for sample in self.sample_arr:
            self.ids.append(sample.replace(image_path, '').replace('.IMG', ''))
            self.images.append(sample)
            mask_list = []
            for x in header.dir_mask_path:
                mask_candidate = image_path.replace('/JSRT', x) + self.ids[-1] + '.gif'
                if(os.path.isfile(mask_candidate)):
                    mask_list.append(mask_candidate)
            self.masks.append(mask_list)
        

    def __len__(self):
        return self.data_len


    def __getitem__(self, index):

        # images
        images = np.asarray(np.reshape(np.fromfile(self.images[index], dtype='>u2', sep="", ), (header.orig_height, header.orig_width)))
        original_image_size = np.asarray(images.shape)

        # preprocessing
        images = pre_processing(images)
        
        # resize
        images = np.asarray(Image.fromarray(images).resize((header.resize_width, header.resize_height)))
        images = np.ma.masked_greater(images, (1<<header.rescale_bit)-1)
        images = np.ma.filled(images, (1<<header.rescale_bit)-1)
        images = np.ma.masked_less_equal(images, 0)
        images = np.ma.filled(images, 0)

        # masks
        masks = np.asarray([np.asarray(Image.open(x).resize((header.resize_width, header.resize_height))) for x in self.masks[index]])   
                                            
        # mask scaling
        masks = masks/255
        background_mask = np.expand_dims(0.5 * np.ones(masks.shape[-2:]), axis=0)

        # mask lists for networks
        masks_list = []
        masks_cat = np.concatenate((background_mask, masks), axis=0)
        masks_list = np.argmax(masks_cat, axis=0).astype("uint8") 

        return {'input':np.expand_dims(images, 0), 'masks':masks_list.astype("int64"), 'ids':self.ids[index], 'im_size':original_image_size}
        

    def get_id(self, index):

        return self.ids[index]


class MyTestDataset(Dataset):

    def __init__(self, image_path, sampler):

        self.sample_arr = glob.glob(str(image_path) + "/*")
        self.sample_arr.sort()
        self.sample_arr = [self.sample_arr[x] for x in sampler]

        # filter out nodule cases
        sample_nn = []
        for i in self.sample_arr:
            if(i.find('JPCNN')>0):
                sample_nn.append(i)
        self.sample_arr = sample_nn

        self.data_len = len(self.sample_arr)

        self.ids = []
        self.images = []
        self.masks = []
        
        if not os.path.isdir(image_path):
            raise RuntimeError('Dataset not found or corrupted. DIR : ' + image_path)

        for sample in self.sample_arr:
            self.ids.append(sample.replace(image_path, '').replace('.IMG', ''))
            self.images.append(sample)
            mask_list = []
            for x in header.dir_mask_path:
                mask_candidate = image_path.replace('/JSRT', x) + self.ids[-1] + '.gif'
                if(os.path.isfile(mask_candidate)):
                    mask_list.append(mask_candidate)
            self.masks.append(mask_list)


    def __len__(self):
        return self.data_len


    def __getitem__(self, index):

        # images
        images = np.asarray(np.reshape(np.fromfile(self.images[index], dtype='>u2', sep="", ), (header.orig_height, header.orig_width)))
        original_image_size = np.array(images.shape)

        # preprocessing
        images = pre_processing(images)
        images = images.astype('float32')

        # resize
        images = np.asarray(Image.fromarray(images).resize((header.resize_width, header.resize_height))) 

        # mask
        masks = np.asarray([np.asarray(Image.open(x).resize(original_image_size)) for x in self.masks[index]]).astype("int64")    

        # mask scaling
        masks = masks/255

        # mask lists for networks
        masks_list = masks

        return {'input':np.expand_dims(images, 0), 'masks':masks_list.astype("int64"), 'ids':self.ids[index], 'im_size':original_image_size}


    def get_id(self, index):
        return self.ids[index]


    def get_original(self, index):

        images = np.asarray(np.reshape(np.fromfile(self.images[index], dtype='>u2', sep="", ), (header.orig_height, header.orig_width)))
        images = pre_processing(images)

        return images


class MyInferenceClass(Dataset):

    def __init__(self, tag):

        image_path = header.dir_data_root + tag
        self.images = glob.glob(image_path + '/*.png')
        self.images.extend(glob.glob(image_path + '/*.jpeg'))
        self.images.extend(glob.glob(image_path + '/*.jpg'))
        self.images.extend(glob.glob(image_path + '/*.gif'))

        self.images.sort()
        self.data_len = len(self.images)
        self.ids = []

        if not os.path.isdir(image_path):
            raise RuntimeError('Dataset not found or corrupted. DIR : ' + image_path)

        for sample in self.images:
            self.ids.append(sample.replace(image_path, ''))


    def __len__(self):
        return self.data_len


    def __getitem__(self, index):

        # load image
        images_original = np.asarray(Image.open(self.images[index]).convert("L"))

        # crop blank area
        line_center = images_original[int(images_original.shape[0]/2):,int(images_original.shape[1]/2)]
        if(line_center.min() == 0):
            images = images_original[:int(images_original.shape[0]/2)+np.where(line_center==0)[0][0],:]
        else:
            images = images_original
        original_image_size = np.asarray(images.shape)

        # preprocessing
        images = pre_processing(images, flag_jsrt=0)
        images = images.astype('float32')

        # resize
        images = np.asarray(Image.fromarray(images).resize((header.resize_width, header.resize_height)))

        return {'input':np.expand_dims(images, 0), 'ids':self.ids[index], 'im_size':original_image_size}
    

    def get_original(self, index, flag=True):

        # load image
        images = np.asarray(Image.open(self.images[index]).convert("L"))

        # crop blank area
        line_center = images[int(images.shape[0]/2):,int(images.shape[1]/2)]
        if (line_center.min() == 0):
            images = images[:int(images.shape[0]/2)+np.where(line_center==0)[0][0],:]

        # preprocessing
        images = pre_processing(images, flag_jsrt=0)

        return images


class MyInferenceClassChest14(Dataset):

    def __init__(self):

        self.images = []

        image_path = '../../../../../F/chestxray14/'
        metadata = '../../../../../F/chestxray14/Data_Entry_2017.csv'
        f = open(metadata)
        read = csv.reader(f)
        for line in read:
            isfile = glob.glob(image_path + 'images_001/*/' + line[0])
            # if (len(isfile) == 0):
            #     isfile = glob.glob(image_path + '/images_002/*/' + line[0])
            if (line[1].find('No Finding')>=0):
                if (len(isfile)):
                    self.images.append(isfile[0])
        f.close()

        self.images.sort()
        total_len = len(self.images)
        seed = 407
        random.seed(seed)
        list_division = random.sample(range(0, total_len), 829)
        self.images = [self.images[k] for k in list_division]

        if not os.path.isdir(image_path):
            raise RuntimeError('Dataset not found or corrupted. DIR : ' + image_path)

        self.ids = []   
        for sample in self.images:
            self.ids.append('/' + sample.split('/')[-1])
            shutil.copy(sample, header.dir_save + 'Normal/')

        self.data_len = len(self.images)


    def __len__(self):
        return self.data_len


    def __getitem__(self, index):

        # load image
        images_original = np.asarray(Image.open(self.images[index]).convert("L"))

        # crop blank area
        line_center = images_original[int(images_original.shape[0]/2):,int(images_original.shape[1]/2)]
        if(line_center.min() == 0):
            images = images_original[:int(images_original.shape[0]/2)+np.where(line_center==0)[0][0],:]
        else:
            images = images_original
        original_image_size = np.asarray(images.shape)

        # preprocessing
        images = pre_processing(images, flag_jsrt=0)
        images = images.astype('float32')

        # resize
        images = np.asarray(Image.fromarray(images).resize((header.resize_width, header.resize_height)))

        return {'input':np.expand_dims(images, 0), 'ids':self.ids[index], 'im_size':original_image_size}
    

    def get_original(self, index, flag=True):

        # load image
        images = np.asarray(Image.open(self.images[index]).convert("L"))

        # crop blank area
        line_center = images[int(images.shape[0]/2):,int(images.shape[1]/2)]
        if (line_center.min() == 0):
            images = images[:int(images.shape[0]/2)+np.where(line_center==0)[0][0],:]

        # preprocessing
        images = pre_processing(images, flag_jsrt=0)

        return images


class MyInferenceCohen(Dataset):

    def __init__(self, tag):

        self.images = []

        image_path = '../../../../../F/covid-chestxray-dataset/images/'
        metadata = '../../../../../F/covid-chestxray-dataset/metadata.csv'
        f = open(metadata)
        read = csv.reader(f)
        for line in read:
            if (line[4].find(tag)>=0):
                if (line[19].find('X-ray')>=0):
                    if (line[18].find('AP')>=0) | (line[18].find('PA')>=0):
                        isfile = glob.glob(image_path + line[23])
                        if (len(isfile)):
                            self.images.append(isfile[0])
                            shutil.copy(isfile[0], header.dir_save + tag + '/')
        f.close()

        # outlier
        # ajr.20.23034.pdf-003
        # aqaa062i0002-a
        # aqaa062i0002-b

        self.images.sort()
        if not os.path.isdir(image_path):
            raise RuntimeError('Dataset not found or corrupted. DIR : ' + image_path)

        self.ids = []   
        for sample in self.images:
            self.ids.append('/' + sample.split('/')[-1])

        self.data_len = len(self.images)


    def __len__(self):
        return self.data_len


    def __getitem__(self, index):

        # load image
        images_original = np.asarray(Image.open(self.images[index]).convert("L"))

        # crop blank area
        line_center = images_original[int(images_original.shape[0]/2):,int(images_original.shape[1]/2)]
        if(line_center.min() == 0):
            images = images_original[:int(images_original.shape[0]/2)+np.where(line_center==0)[0][0],:]
        else:
            images = images_original
        original_image_size = np.asarray(images.shape)

        # preprocessing
        images = pre_processing(images, flag_jsrt=0)
        images = images.astype('float32')

        # resize
        images = np.asarray(Image.fromarray(images).resize((header.resize_width, header.resize_height)))

        return {'input':np.expand_dims(images, 0), 'ids':self.ids[index], 'im_size':original_image_size}
    

    def get_original(self, index, flag=True):

        # load image
        images = np.asarray(Image.open(self.images[index]).convert("L"))

        # crop blank area
        line_center = images[int(images.shape[0]/2):,int(images.shape[1]/2)]
        if (line_center.min() == 0):
            images = images[:int(images.shape[0]/2)+np.where(line_center==0)[0][0],:]

        # preprocessing
        images = pre_processing(images, flag_jsrt=0)

        return images


class MyInferenceClassBIMCV(Dataset):

    def __init__(self):

        self.images = []
        self.ids = []

        dir_csv = '../../../../../F/COVID_BIMCV/COVID_BIMCV/V1.0/derivatives/labels/labels_covid19_posi.tsv'
        image_path = '../../../../../F/COVID_BIMCV/COVID_BIMCV/V1.0/'

        if not os.path.isdir(image_path):
            raise RuntimeError('Dataset not found or corrupted. DIR : ' + image_path)

        f = open(dir_csv)
        read = csv.reader(f)
    
        for line in read:

            if (line[0].find('COVID')>0):

                line = line[0].split('\t')

                # file name - pa
                image = glob.glob(image_path + '*/' + line[1] + '/' + line[2] + '/*/*pa_*.png')
                if (image.__len__()>0):
                    self.images.append(image[0])

                # ap
                image = glob.glob(image_path + '*/' + line[1] + '/' + line[2] + '/*/*ap_*.png')
                if (image.__len__()>0):
                    self.images.append(image[0])

        f.close()

        # outlier (13-)
        outlier = ['sub-S03542_ses-E07201_run-1_bp-chest_vp-pa_dx', 'sub-S03573_ses-E07272_run-1_bp-chest_vp-pa_cr', 
                    'sub-S03634_ses-E08888_run-1_bp-chest_vp-ap_cr', 'sub-S03751_ses-E07576_run-1_bp-chest_vp-ap_cr', 
                    'sub-S03810_ses-E07675_run-1_bp-chest_vp-pa_cr', 'sub-S03852_ses-E07775_run-1_bp-chest_vp-ap_cr', 
                    'sub-S03897_ses-E07941_run-1_bp-chest_vp-ap_cr', 'sub-S03967_ses-E08122_run-1_bp-chest_vp-ap_dx', 
                    'sub-S03996_ses-E08157_run-1_bp-chest_vp-ap_cr', 'sub-S04347_ses-E08656_run-1_bp-chest_vp-pa_cr',
                    'sub-S03168_ses-E06915_run-1_bp-chest_vp-ap_dx', 'sub-S03232_ses-E07749_run-1_bp-chest_vp-ap_cr', 
                    'sub-S03258_ses-E06425_run-1_bp-chest_vp-ap_dx', 'sub-S03404_ses-E06752_run-1_bp-chest_vp-pa_cr']
        for k in outlier:
            find_flag = np.asarray([l.find(k) for l in self.images])
            if (np.sum(find_flag>0) > 0):
                idx_flag = np.argmax(find_flag)
                self.images.pop(idx_flag)
            
        for sample in self.images:
            shutil.copy(sample, header.dir_save + 'COVID-19/')
            self.ids.append('/' + sample.split('/')[-1])

        self.data_len = len(self.images)
                    

    def __len__(self):
        return self.data_len


    def __getitem__(self, index):

        # load image
        images_original = np.asarray(Image.open(self.images[index]))

        # # crop blank area
        # line_center = images_original[int(images_original.shape[0]/2):,int(images_original.shape[1]/2)]
        # if(line_center.min() == 0):
        #     images = images_original[:int(images_original.shape[0]/2)+np.where(line_center==0)[0][0],:]
        # else:
        images = images_original
        original_image_size = np.asarray(images.shape)

        # invert
        flag_invert = 0
        line_top = np.mean(images[0,:])
        if (line_top > 10000):
            flag_invert = 10

        # preprocessing
        images = pre_processing(images, flag_jsrt=flag_invert)
        images = images.astype('float32')

        # resize
        images = np.asarray(Image.fromarray(images).resize((header.resize_width, header.resize_height)))

        return {'input':np.expand_dims(images, 0), 'ids':self.ids[index], 'im_size':original_image_size}
    

    def get_original(self, index, flag=True):

        # load image
        images = np.asarray(Image.open(self.images[index]))

        # # crop blank area
        # line_center = images[int(images.shape[0]/2):,int(images.shape[1]/2)]
        # if (line_center.min() == 0):
        #     images = images[:int(images.shape[0]/2)+np.where(line_center==0)[0][0],:]

        # invert
        flag_invert = 0
        line_top = np.mean(images[0,:])
        print('%d : %s' % (line_top, self.ids[index]))
        if (line_top > 10000):
            flag_invert = 10

        # preprocessing
        images = pre_processing(images, flag_jsrt=flag_invert)

        return images


def pre_processing(images, flag_jsrt = 10):

    # histogram
    num_out_bit = 1<<header.rescale_bit
    num_bin = images.max()+1

    # histogram specification, gamma correction
    hist, bins = np.histogram(images.flatten(), num_bin, [0, num_bin])
    cdf = hist_specification(hist, num_out_bit, images.min(), num_bin, flag_jsrt)
    images = cdf[images].astype('float32')

    return images


def hist_specification(hist, bit_output, min_roi, max_roi, flag_jsrt):

    cdf = hist.cumsum()
    cdf = np.ma.masked_equal(cdf, 0)

    # hist sum of low & high
    hist_low = np.sum(hist[:min_roi+1]) + flag_jsrt
    hist_high = cdf.max() - np.sum(hist[max_roi:])

    # cdf mask
    cdf_m = np.ma.masked_outside(cdf, hist_low, hist_high)

    # build cdf_modified
    if not (flag_jsrt):
        cdf_m = (cdf_m - cdf_m.min())*(bit_output-1) / (cdf_m.max() - cdf_m.min())
    else:
        cdf_m = (bit_output-1) - (cdf_m - cdf_m.min())*(bit_output-1) / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m.astype('float32'), 0)

    # gamma correction
    cdf = pow(cdf/(bit_output-1), header.gamma) * (bit_output-1)

    return cdf


def one_hot(x, class_count):

    return torch.eye(class_count)[:, x]


def create_folder(directory):

    if not os.path.exists(directory):
        os.makedirs(directory)


def get_size_id(idx, size, case_id, dir_label):

    original_size_w_h = (size[idx][1].item(), size[idx][0].item())
    case_id = case_id[idx]
    dir_results = [case_id + case_id + '_' + j + '.png' for j in dir_label]

    return original_size_w_h, case_id, dir_results


def split_dataset(len_dataset):

    # set parameter
    offset_split_train = int(np.floor(header.train_split * len_dataset))
    offset_split_valid = int(np.floor(header.valid_split * len_dataset))
    indices = list(range(len_dataset))

    # shuffle
    np.random.seed(407)
    np.random.shuffle(indices)

    # set samplers
    train_sampler = indices[:offset_split_train]
    valid_sampler = indices[offset_split_train:offset_split_valid]
    test_sampler = indices[offset_split_train:] #[offset_split_valid:]

    return train_sampler, valid_sampler, test_sampler
