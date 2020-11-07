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
import cv2

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
    # output_np[output_np == 0] = 1
    output_image = Image.fromarray((output_np*RESCALE).astype('uint8'), 'RGBA')


    # save output
    output_name = d_dir + '/rv_bg_' + img_name.split(".")[0] + '.png'
    output_image .save(output_name)


def main():

    # --------- 1. get image path and name ---------
    model_name='u2netp'# fixed as u2netp

    video_dir = os.path.join(os.getcwd(), 'videos') # 'videos' directory
    result_dir = os.path.join(os.getcwd(), 'results/') # 'results' directory
    model_dir = os.path.join(os.getcwd(), model_name + '.pth') # path to u2netp pretrained weights

    video_name_list = glob.glob(video_dir + os.sep + '*')
    # print(video_name_list)

    result_name_dir = glob.glob(result_dir + os.sep + '*')

    # --------- 2. model define ---------
    net = U2NETP(3,1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 3. inference for each video ---------
    for video_name in video_name_list:
        # check processed video
        expected_result_name = result_dir + 'rmBG_' + video_name.split(os.sep)[-1].split(".")[0] + '.mp4'
        if expected_result_name in result_name_dir:
            print('[ERROR] -output: '+ video_name.split(os.sep)[-1].split(".")[0] + '.mp4' + ' already exists in ' + result_dir)
            continue

        print("inferencing:", video_name)


        # get info video
        video = cv2.VideoCapture(video_name)
        # frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        success, frame = video.read()
        height, width, layers = frame.shape

        print("Size video: width = {0}, height = {1}".format(width, height))


        # create video result
        result = cv2.VideoWriter(expected_result_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        count = 0

        # process video
        while success:
            # write 'frame.png' in video folder
            frame_name = video_dir + '/frame.png'
            cv2.imwrite(frame_name, frame)

            # dataloader for frame.png
            salobj_dataset = SalObjDataset(img_name_list = [frame_name],
                                           lbl_name_list = [],
                                           transform=transforms.Compose([RescaleT(320),
                                           ToTensorLab(flag=0)]))
            salobj_dataloader = DataLoader(salobj_dataset,
                                           batch_size=1,
                                           shuffle=False,
                                           num_workers=1)

            # predict mask and remove background for frame
            for i, data in enumerate(salobj_dataloader):
                input_frame = data['image']
                input_frame = input_frame.type(torch.FloatTensor)

                if torch.cuda.is_available():
                    input_frame = Variable(input_frame.cuda())
                else:
                    input_frame = Variable(input_frame)

                d1,d2,d3,d4,d5,d6,d7= net(input_frame)

                # normalization'
                pred_mask = d1[:,0,:,:]
                pred_mask = normPRED(pred_mask)
        
                save_output(frame_name, pred_mask, video_dir, 0.9)

                del d1,d2,d3,d4,d5,d6,d7

            # read result
            result_frame_name = video_dir + '/rv_bg_' + 'frame.png'
            result_frame = cv2.imread(result_frame_name, cv2.IMREAD_UNCHANGED)
            if result_frame.shape[2] == 3:
                result_frame = cv2.cvtColor(result_frame, cv2.COLOR_RGB2RGBA)

            if count < 10:
                print("-> Size frame: width = {0}, height = {1}".format(frame.shape[1], frame.shape[0]))
                print("-> Size result_frame: width = {0}, height = {1}".format(result_frame.shape[1], result_frame.shape[0]))

            count += 1

            # write frame to result video
            result.write(result_frame)

            # delete current frame and result_frame
            os.remove(frame_name)
            os.remove(result_frame_name)

            # next frame image
            success, frame = video.read()


        video.release()
        result.release()


if __name__ == "__main__":
    main()
