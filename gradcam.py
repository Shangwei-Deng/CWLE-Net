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
from other_models.UIU.model.uiunet import UIUNET
import PIL
from PIL import Image

pth_dir = '/home/x3090/work/dsw/UIU/log/best/UIU_380_best_NUDT.pth.tar'
img_path = '/home/x3090/work/dsw/UIU/images/image/000316.png'
mask_path = '/home/x3090/work/dsw/UIU/images/mask/000316.png'
save_path = '/home/x3090/work/dsw/UIU/images/saves'
dataname = 'IRDST-real'

class GradCAM(object):

    def __init__(self, model_dict, h, w, verbose=False):
        self.model_arch = model_dict['arch']
        self.h = h
        self.w = w

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        target_layer = self.model_arch.model.stage2d

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

        return saliency_map

    def __call__(self, image, mask, retain_graph=False):
        return self.forward(image, mask, retain_graph)

class Net(nn.Module):
    def __init__(self, model_name, mode):
        super(Net, self).__init__()
        self.model_name = model_name
        # ************************************************loss*************************************************#
        self.cal_loss = nn.BCELoss(size_average=True)
        if model_name == 'UIU':
            if mode == 'train':
                self.model = UIUNET(1, 1, mode='train', deepsuper=True)
            else:
                self.model = UIUNET(1, 1, mode='test', deepsuper=True)
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

########设置网络#######

net = Net(model_name='UIU', mode='test').cuda()
net.eval()
model_dict = dict(arch=net)



ckpt = torch.load(pth_dir)
net.load_state_dict(ckpt['state_dict'])
img = Image.open(img_path).convert('I')
mask = Image.open(mask_path)
img_norm_cfg = get_img_norm_cfg('NUDT-SIRST', None)
img = Normalized(np.array(img, dtype=np.float32), img_norm_cfg)
mask = np.array(mask, dtype=np.float32) / 255.0

if len(mask.shape) > 2:
    mask = mask[:, :, 0]

h, w = img.shape
img = PadImg(img)
mask = PadImg(mask)
img, mask = img[np.newaxis, :], mask[np.newaxis, :]
img, mask = img[np.newaxis, :], mask[np.newaxis, :]
img = torch.from_numpy(np.ascontiguousarray(img))
mask = torch.from_numpy(np.ascontiguousarray(mask))

if img.size() != mask.size():
    print('111')

gradcam = GradCAM(model_dict, h=h, w=w)
mask = gradcam(img,mask)

heatmap, cam_result = visualize_cam(mask, img)

heatmap = heatmap.permute(1, 2, 0).cpu().numpy()  # Convert to [H, W, C] shape
cam_result = cam_result.permute(1, 2, 0).cpu().numpy()  # Convert to numpy array

heatmap = np.uint8(255 * heatmap)  # Convert to uint8 for OpenCV
cam_result = np.uint8(255 * cam_result)  # Convert to uint8 for OpenCV


cv2.imshow('Heatmap', heatmap)  # Show heatmap
cv2.imshow('CAM Result', cam_result)  # Show GradCAM result
heatmap_save = os.path.join(save_path, 'heatmap.png')
cam_save = os.path.join(save_path, 'cam.png')

# cv2.imwrite(heatmap_save, heatmap)
# cv2.imwrite(cam_save,cam_result)
cv2.waitKey(0)  # Wait for key press to close windows
# cv2.destroyAllWindows()  # Close all OpenCV windows



