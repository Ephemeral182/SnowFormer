import argparse
import os
from os.path import exists, join as join_paths
import torch
import numpy as np
from torchvision.transforms import functional as FF
from metrics import *
import warnings
from torchvision.utils import save_image,make_grid
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloader import *

from SnowFormer import *
from PIL import Image
warnings.filterwarnings("ignore")
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--tile', type=int, default=256, help='Tile size, None for no tile during testing (testing as a whole)')
parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8') 
parser.add_argument('--dataset_type', type=str, default='CSD', help='CSD/SRRS/Snow100K') 
parser.add_argument('--dataset_CSD', type=str, default='/home/PublicDataset/CSD/Test', help='path of CSD dataset') 
parser.add_argument('--dataset_SRRS', type=str, default='/home/PublicDataset/SRRS/SRRS-2021/', help='path of SRRS dataset') 
parser.add_argument('--dataset_Snow100K', type=str, default='/home/PublicDataset/Snow100K/media/jdway/GameSSD/overlapping/test/test/', help='path of Snow100k dataset') 
parser.add_argument('--savepath', type=str, default='./out/', help='path of output image') 
parser.add_argument('--model_path', type=str, default='/mnt/csx/SnowFormer/SnowFormer/SnowFormer_CSD.pth', help='path of SnowFormer checkpoint') 

opt = parser.parse_args()
if opt.dataset_type == 'CSD':
    snow_test = DataLoader(dataset=CSD_Dataset(opt.dataset_CSD,train=False,size=256,rand_inpaint=False,rand_augment=None),batch_size=1,shuffle=False,num_workers=4)
if opt.dataset_type == 'SRRS':
    snow_test = DataLoader(dataset=SRRS_Dataset(opt.dataset_SRRS,train=False,size=256,rand_inpaint=False,rand_augment=None),batch_size=1,shuffle=False,num_workers=4)
if opt.dataset_type == 'Snow100K':
    snow_test = DataLoader(dataset=Snow100K_Dataset(opt.dataset_Snow100K,train=False,size=256,rand_inpaint=False,rand_augment=None),batch_size=1,shuffle=False,num_workers=4)


netG_1 = Transformer().cuda()

if __name__ == '__main__':   

    ssims = []
    psnrs = []
    rmses = []
    
    g1ckpt1 = opt.model_path
    ckpt = torch.load(g1ckpt1)
    netG_1.load_state_dict(ckpt)

    savepath_dataset = os.path.join(opt.savepath,opt.dataset_type)
    if not os.path.exists(savepath_dataset):
        os.makedirs(savepath_dataset)
    loop = tqdm(enumerate(snow_test),total=len(snow_test))

    for idx,(haze,clean,name) in loop:
        
        with torch.no_grad():
                
                haze = haze.cuda();clean = clean.cuda()

                b, c, h, w = haze.size()

                tile = min(opt.tile, h, w)
                tile_overlap = opt.tile_overlap
                sf = opt.scale

                stride = tile - tile_overlap
                h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
                w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
                E1 = torch.zeros(b, c, h*sf, w*sf).type_as(haze)
                W1 = torch.zeros_like(E1)

                for h_idx in h_idx_list:
                    for w_idx in w_idx_list:
                        in_patch = haze[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                        out_patch1 = netG_1(in_patch)
                        out_patch_mask1 = torch.ones_like(out_patch1)
                        E1[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch1)
                        W1[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask1)
                dehaze = E1.div_(W1)

                save_image(dehaze,os.path.join(savepath_dataset,'%s.png'%(name)),normalize=False)


                ssim1=SSIM(dehaze,clean).item()
                psnr1=PSNR(dehaze,clean)

                ssims.append(ssim1)
                psnrs.append(psnr1)

                print('Generated images %04d of %04d' % (idx+1, len(snow_test)))
                print('ssim:',(ssim1))
                print('psnr:',(psnr1))

        ssim = np.mean(ssims)
        psnr = np.mean(psnrs)
        print('ssim_avg:',ssim)
        print('psnr_avg:',psnr)
 