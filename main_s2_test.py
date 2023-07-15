
import argparse, os, glob, torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
from models import vgg13bn_unet256, UShapedNet
from dataloader import *
from torchvision.utils import save_image
from tools import CopyFiles, get_normal_255, N_SFS2CM, mkdirss

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", type=str)
    parser.add_argument("--iter", type=int, default=300000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgSize", type=int, default=256)
    # parser.add_argument("--p2ckpt", type=str, default=None)
    parser.add_argument("--p2ckpt", type=str, default="./exp/p2_ckpt_150.pt")
    parser.add_argument("--p1ckpt", type=str, default="./exp/p1_ckpt_150.pkl")

    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--exp_name", type=str, default="RR")
    parser.add_argument("--gpuID", type=int, default=1)


    args = parser.parse_args()
    device = "cuda:" + str(args.gpuID)
    args.start_iter = 0

    FNR2R = vgg13bn_unet256().to(device)
    g_optim = optim.Adam(list(FNR2R.parameters()), lr=args.lr, betas=(0.9, 0.99))

    print("load p1 model:", args.p1ckpt)
    PreTrainModel = UShapedNet(inputdim=1).to(device)
    ckptModel = torch.load(args.p1ckpt, map_location=lambda storage, loc: storage)
    PreTrainModel.load_state_dict(ckptModel)
    PreTrainModel.eval()

    if args.p2ckpt is not None:
        print("load p2 model:", args.p2ckpt)
        p2ckpt = torch.load(args.p2ckpt, map_location=lambda storage, loc: storage)
        p2ckpt_name = os.path.basename(args.p2ckpt)
        FNR2R.load_state_dict(p2ckpt["model"])
    FNR2R.eval()
    pathd = './inputs/TestImagesList.csv'
    train_dataset, _ = getTrainDataP2(csvPath=pathd, imgSize=args.imgSize, validation_split=0)
    CelebaLowtest_dl  = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=8)

    svg_celebatest  = './results/'
    with torch.no_grad():
        for iter_idx, batch_data in enumerate(CelebaLowtest_dl):
            input_img, index = batch_data
            input_img = input_img.to(device)
            b, c, w, h = input_img.shape
            Ex_normal = N_SFS2CM(F.normalize(PreTrainModel(input_img.mean(1).unsqueeze(1))))
            Re_Normal = FNR2R(input_img, Ex_normal) 
            Re_Normal = F.normalize(Re_Normal)

            for ii in range(b):
                tpName = index[ii][-9:]
                save_image(input_img[ii], svg_celebatest + tpName.replace('.png', '_input.png'), nrow=1, normalize=True)
                save_image(get_normal_255(Re_Normal)[ii], svg_celebatest + tpName.replace('.png', '_re.png'), nrow=1, normalize=True)
                save_image(get_normal_255(Ex_normal)[ii], svg_celebatest + tpName.replace('.png', '_ex.png'), nrow=1, normalize=True)

               
        print("Done!")

