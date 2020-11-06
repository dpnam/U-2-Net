import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name, pred_mask, d_dir, thresold = 0.9):
    # Define Rescale
    RESCALE = 255

    # get mask
    mask = pred_mask
    mask = mask.squeeze()
    mask_np = mask.cpu().data.numpy()

    # get orgin image
    input_image = load_img(image_name)
    img_name = image_name.split(os.sep)[-1]
    input_np = img_to_array(input_image)
    input_np /= RESCALE

    # resize mask
    mask_image = Image.fromarray(mask_np*RESCALE).convert('RGB')
    mask_image = mask_image.resize((input_np.shape[1], input_np.shape[0]), resample=Image.BILINEAR)

    # back to mask numpy
    mask_np = img_to_array(mask_image)
    mask_np /= RESCALE

    # filter with thresold
    mask_np[mask_np > thresold] = 1
    mask_np[mask_np <= thresold] = 0

    # convert the rbg image to an rgba image and set the zero values to transparent
    a_layer_init = np.ones(shape = (mask_np.shape[0], mask_np.shape[1], 1))
    mul_layer = np.expand_dims(mask_np[:,:,0], axis=2)
    a_layer = mul_layer*a_layer_init
    rgba_mask = np.append(mask_np, a_layer, axis=2)


    # since the output image is rgba, convert this also to rgba, but with no transparency
    a_layer = np.ones(shape = (input_np.shape[0], input_np.shape[1], 1))
    rgba_input = np.append(input_np, a_layer, axis=2)

    # ouput image
    output_np = (rgba_input*rgba_mask)
    output_image = Image.fromarray((output_np*RESCALE).astype('uint8'), 'RGBA')


    # save output
    output_name = d_dir + img_name.split(".")[0] + '.png'
    output_image .save(output_name)


def main():

    # --------- 1. get image path and name ---------
    model_name='u2netp'# fixed as u2netp



    image_dir = os.path.join(os.getcwd(), 'images') # 'images' directory
    result_dir = os.path.join(os.getcwd(), 'results/') # 'results' directory
    model_dir = os.path.join(os.getcwd(), model_name + '.pth') # path to u2netp pretrained weights

    img_name_list = glob.glob(image_dir + os.sep + '*')
    # print(img_name_list)

    result_name_dir = glob.glob(result_dir + os.sep + '*')

    # --------- 2. dataloader ---------
    #1. dataloader
    salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                   lbl_name_list = [],
                                   transform=transforms.Compose([RescaleT(320),
                                   ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset,
                                   batch_size=1,
                                   shuffle=False,
                                   num_workers=1)

    # --------- 3. model define ---------
    net = U2NETP(3,1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    for i, data in enumerate(salobj_dataloader):
        # get name image and check processed image
        img_name = img_name_list[i].split(os.sep)[-1]
        expected_result_name = result_dir + img_name.split(".")[0] +'.png'
        if expected_result_name in result_name_dir:
            print('[ERROR] - output: '+ img_name.split(".")[0] +'.png' + ' already exists in ' + result_dir)
            continue

        # process imge
        print("inferencing:", img_name)

        input_image = data['image']
        input_image = input_image.type(torch.FloatTensor)

        if torch.cuda.is_available():
            input_image = Variable(input_image.cuda())
        else:
            input_image = Variable(input_image)

        d1,d2,d3,d4,d5,d6,d7= net(input_image)

        # normalization
        pred_mask = d1[:,0,:,:]
        pred_mask = normPRED(pred_mask)
        
        # save results to results folder
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)
        save_output(img_name_list[i], pred_mask, result_dir, 0.9)

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()
