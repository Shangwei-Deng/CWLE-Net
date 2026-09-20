import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import *
import time
from collections import OrderedDict

from model.UT4 import MN as GaussDet

import os.path
import torch
import torch.nn.functional as F
import torchvision
from utils0 import find_alexnet_layer, find_vgg_layer, find_resnet_layer, find_densenet_layer, find_squeezenet_layer,visualize_cam
from torchvision import transforms
from utils import *
import cv2
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import PIL
from PIL import Image

from skimage import measure

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD test")
parser.add_argument('--ROC_thr', type=int, default=10, help='num')

parser.add_argument("--model_names", default=['GaussDet'], type=list,
                    help="model_name: 'ACM', 'Ours01', 'DNANet', 'ISNet', 'ACMNet', 'Ours01', 'ISTDU-Net', 'U-Net', 'RISTDnet'")


parser.add_argument("--dataset_names", default=['NUDT-SIRST'], type=list,  # 数据集名称
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea', 'Image'")

parser.add_argument("--pth_dirs", default=['/home/x3090/work/dsw/LFNet/log/NUDT-SIRST/best/GaussDet_255_best.pth.tar'], type=list) #nuaa 700

parser.add_argument("--dataset_dir", default='/home/x3090/work/dsw/LFNet/Dataset', type=str,
                    help="train_dataset_dir")

parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--save_img", default=True, type=bool, help="save image of or not")  # 是否保存

parser.add_argument("--save_img_dir", type=str, default='/home/x3090/work/dsw/LFNet/test_result/heatmap/D1_w_o/', help="path of saved image")
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 2,3. use -1 for CPU')

global opt
opt = parser.parse_args(args=[])



class Net(nn.Module):
    def __init__(self, model_name, mode):
        super(Net, self).__init__()
        self.model_name = model_name
        # ************************************************loss*************************************************#
        self.cal_loss = nn.BCELoss(size_average=True)
        if model_name == 'GaussDet':
            if mode == 'train':
                self.model = GaussDet(1, 1, mode='train', deepsuper=True)

                if torch.cuda.device_count() > 1:
                    print("Let's use ", torch.cuda.device_count(), " GPUs!")
                    self.model = nn.DataParallel(self.model, device_ids=[0, 1])
                self.model = self.model.to(device)
            else:
                self.model = GaussDet(1, 1, mode='test', deepsuper=True)
                if torch.cuda.device_count() > 1:
                    print("Let's use ", torch.cuda.device_count(), " GPUs!")
                    self.model = nn.DataParallel(self.model, device_ids=[0, 1])

                self.model = self.model.to(device)

    def forward(self, img):
        return self.model(img)

    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)

        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = sum(a)
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

class GradCAM(object):

    def __init__(self, model_dict, h, w, verbose=False, stage='stage2d'):
        self.model_arch = model_dict['arch']
        self.h = h
        self.w = w

        self.gradients = dict()
        self.activations = dict()
        new_stage = stage
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        self.target_layer = getattr(self.model_arch.model, new_stage)

        target_layer = self.target_layer

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        if verbose:
            try:
                input_size = model_dict['input_size']
            except KeyError:
                print("please specify size of input image in model_dict. e.g. {'input_size':(224, 224)}")
                pass
            else:
                device = 'cuda' if next(self.model_arch.parameters()).is_cuda else 'cpu'
                self.model_arch(torch.zeros(1, 3, *(input_size), device=device))
                print('saliency_map size :', self.activations['value'].shape[2:])


    def forward(self, image, mask, retain_graph=False):

        image = Variable(image.cuda())
        pred = self.model_arch(image)

        pred = pred[:, :, :self.h, :self.w].cuda()
        gt_mask = mask[:, :, :self.h, :self.w].cuda()


        loss = self.model_arch.loss(pred, gt_mask)
        self.model_arch.zero_grad()
        loss.backward()


        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        #alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(self.h, self.w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data


        self.gradients.clear()
        self.activations.clear()

        return saliency_map

    def __call__(self, image, mask, retain_graph=False):
        return self.forward(image, mask, retain_graph)


def Test():
    if opt.test_dataset_name == 'NUAA-SIRST':
        test_set = TestSetLoader_NUAA(opt.dataset_dir, opt.test_dataset_name, opt.test_dataset_name,
                                      img_norm_cfg=opt.img_norm_cfg)

    elif opt.test_dataset_name == 'NUDT-SIRST':
        test_set = TestSetLoader_NUDT(opt.dataset_dir, opt.test_dataset_name, opt.test_dataset_name,
                                      img_norm_cfg=opt.img_norm_cfg)

    elif opt.test_dataset_name == 'IRSTD-1K':
        test_set = TestSetLoader_IRSTD_1K(opt.dataset_dir, opt.test_dataset_name, opt.test_dataset_name,
                                          img_norm_cfg=opt.img_norm_cfg)

    else:
        raise NotImplementedError
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)



    net = Net(model_name='GaussDet', mode='test').cuda()
    ckpt = torch.load(pth_dir, map_location=device)
    original_keys = ckpt['state_dict'].keys()
    # print("Original keys:", original_keys)

    new_state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        # if k.startswith('module'):
        #     print(111111)
        name = 'model.'+ k[13:]  # 去掉 'module.' 前缀
        # else:
        #     name = k  # 直接使用原键名
        new_state_dict[name] = v

    new_keys = new_state_dict.keys()
    # print("New keys:", new_keys)


    net.load_state_dict(new_state_dict)
    net.eval()
    model_dict = dict(arch=net)

#--------------------------------------------------------------------------

    stage = 'stage1d'

# --------------------------------------------------------------------------
    tbar = tqdm(test_loader)
    for idx_iter, (img, gt_mask, size, img_dir) in enumerate(tbar):


            gradcam = GradCAM(model_dict, h=size[0], w=size[1], stage=stage)
            mask = gradcam(img, gt_mask)




            heatmap, cam_result = visualize_cam(mask, img)
            heatmap = heatmap.permute(1, 2, 0).cpu().numpy()  # Convert to [H, W, C] shape
            cam_result = cam_result.permute(1, 2, 0).cpu().numpy()  # Convert to numpy array

            heatmap = np.uint8(255 * heatmap)  # Convert to uint8 for OpenCV
            cam_result = np.uint8(255 * cam_result)  # Convert to uint8 for OpenCV

            # heatmap = np.clip(heatmap, 0,255)
            # cam_result = np.clip(cam_result,0,255)
    # save img
            if opt.save_img == True:
                if not os.path.exists(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name):
                    os.makedirs(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name)

                heatmap_save = opt.save_img_dir + opt.test_dataset_name+ '/'+ opt.model_name + '/'+img_dir[0]+'_heatmap.png'
                cam_save = opt.save_img_dir + opt.test_dataset_name+ '/'+ opt.model_name + '/'+img_dir[0]+'_cam.png'
                cv2.imwrite(heatmap_save, heatmap)
                cv2.imwrite(cam_save, cam_result)




if __name__ == '__main__':
        for model_name in opt.model_names:
            for dataset_name in opt.dataset_names:
                for pth_dir in opt.pth_dirs:
                    # if dataset_name in pth_dir and model_name in pth_dir:
                    opt.test_dataset_name = dataset_name
                    opt.model_name = model_name
                    opt.train_dataset_name = dataset_name
                    print(pth_dir)
                    print(opt.test_dataset_name)
                    opt.pth_dir = pth_dir
                    # opt.pth_dir = opt.save_log + pth_dir
                    Test()
                    print('\n')

