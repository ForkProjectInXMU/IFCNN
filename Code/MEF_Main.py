from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
from utils.myTransforms import denorm, norms, detransformcv2
from utils.myDatasets import ImageSequence
import os
import cv2
import time
import torch
from model import myIFCNN

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

fuse_scheme = 2
model_name = 'IFCNN-MEAN'


# load pretrained model
model = myIFCNN(fuse_scheme=fuse_scheme)
model.load_state_dict(torch.load('snapshots/' + model_name + '.pth'))
model.eval()
model = model.cuda()


dataset = 'ME'
is_save = True
is_gray = False
is_folder = True
toggle = True
is_save_Y = False
mean = [0, 0, 0]
std = [1, 1, 1]
begin_time = time.time()
root = 'datasets/MEDataset/'

for subdir, dirs, files in os.walk(root):
    if toggle:
        toggle = False
    else:
        # Load the sequential images in each subfolder
        paths = [subdir]
        seq_loader = ImageSequence(is_folder, 'YCbCr', transforms.Compose([
            transforms.ToTensor()]), *paths)
        imgs = seq_loader.get_imseq()
        # print(imgs)
        # input()

        # separate the image channels
        NUM = len(imgs)
        c, h, w = imgs[0].size()
        Cbs = torch.zeros(NUM, h, w)
        Crs = torch.zeros(NUM, h, w)
        Ys = []
        for idx, img in enumerate(imgs):
            # print(img)
            Cbs[idx, :, :] = img[1]
            Crs[idx, :, :] = img[2]
            Ys.append(img[0].unsqueeze_(0).unsqueeze_(0).repeat(1, 3, 1, 1))  # Y

        # Fuse the color channels (Cb and Cr) of the image sequence
        Cbs *= 255
        Crs *= 255
        Cb128 = abs(Cbs - 128);
        Cr128 = abs(Crs - 128);
        CbNew = sum((Cbs * Cb128) / (sum(Cb128).repeat(NUM, 1, 1)));
        CrNew = sum((Crs * Cr128) / (sum(Cr128).repeat(NUM, 1, 1)));
        CbNew[torch.isnan(CbNew)] = 128
        CrNew[torch.isnan(CrNew)] = 128

        # Fuse the Y channel of the image sequence
        imgs = norms(mean, std, *Ys)  # normalize the Y channels
        with torch.no_grad():
            vimgs = []
            for idx, img in enumerate(imgs):
                vimgs.append(Variable(img.cuda()))
            vres = model(*vimgs)

        # Enhance the Y channel using CLAHE
        img = detransformcv2(vres[0], mean, std)  # denormalize the fused Y channel
        y = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # generate the single y channel

        y = y / 255  # initial enhancement
        y = y * 235 + (1 - y) * 16;
        y = y.astype('uint8')

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # clahe enhancement
        cy = clahe.apply(y)

        # Merge the YCbCr channels back and covert to RGB color space
        cyrb = np.zeros([h, w, c]).astype('uint8')
        cyrb[:, :, 0] = cy
        cyrb[:, :, 1] = CrNew
        cyrb[:, :, 2] = CbNew
        rgb = cv2.cvtColor(cyrb, cv2.COLOR_YCrCb2RGB)

        # Save the fused image
        img = Image.fromarray(rgb)
        filename = subdir.split('/')[-1]
        # filename = model_name + '-' + dataset + '-' + filename  # y channels are fused by IFCNN, cr and cb are weighted fused

        if is_save:
            if is_gray:
                img.convert('L').save('results/' + filename + '.png', format='PNG', compress_level=0)
            else:
                img.save('results/' + filename + '.png', format='PNG', compress_level=0)

# when evluating time costs, remember to stop writing images by setting is_save = False
proc_time = time.time() - begin_time
print('Total processing time of {} dataset: {:.3}s'.format(dataset, proc_time))

