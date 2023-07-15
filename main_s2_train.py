
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

def train(args, train_dl, PreTrainModel, FNR2R, g_optim, device):
    train_loader = sample_data(train_dl)
    pbar = range(args.iter + 1)
    pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)
    CosLoss = nn.CosineSimilarity(dim=1, eps=1e-6).to(device)

    savepath = '/home/xteam/PaperCode/MM_IJCV/TIP/06_results/TIP_P2/' + args.exp_name
    weight_path = savepath + '/'
    logPathcodes = weight_path + 'codes' + '/'
    imgsPath = weight_path + 'imgs' + '/'
    mkdirss(imgsPath)
    expPath = weight_path + 'exp' + '/'
    mkdirss(expPath)

    #backup codes
    src_dir = './'
    src_file_list = glob.glob(src_dir + '*')                   
    for srcfile in src_file_list:
        CopyFiles(srcfile, logPathcodes)                   
    print('copy codes have done!!!')
    
    FNR2R.train()
    requires_grad(FNR2R, True)
    for idx in pbar:
        i = idx + args.start_iter
        if i > args.iter:
            print("Done!")
            break

        face, index = next(train_loader)
        b,c,w,h = face.shape
        face = face.to(device) 
        
        with torch.no_grad():
            Ex_normal = N_SFS2CM(F.normalize(PreTrainModel(face.mean(1).unsqueeze(1))))
        
        Re_Normal = FNR2R(face, Ex_normal) # [-1, 1]
        Re_Normal = F.normalize(Re_Normal)
        recon_F = (1 - CosLoss(Re_Normal, Ex_normal).mean())

        # print('******************************************************')
        LossTotal =  recon_F
        # print('******************************************************')

        g_optim.zero_grad()
        LossTotal.backward()
        g_optim.step()
        
        pbar.set_description((f"p2 i:{i:6d}; reconF:{recon_F.item():.4f}; "))

        if i % 200 == 0:
            sampleImgs = torch.cat([face, get_normal_255(Ex_normal), get_normal_255(Re_Normal)], 0)
            save_image(sampleImgs, imgsPath + str(i) + '_vis.png', nrow=b, normalize=True)
        if i % 5000 == 0 and i!=0:
            torch.save({"model": FNR2R.state_dict()}, f"%s/{str(i).zfill(6)}.pt"%(expPath))
            
        if i>350010:
            break 


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", type=str)
    parser.add_argument("--iter", type=int, default=300000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgSize", type=int, default=256)
    # parser.add_argument("--p2ckpt", type=str, default=None)
    parser.add_argument("--p2ckpt", type=str, default="/home/xteam/PaperCode/MM_IJCV/TIP/06_results/TIP_P2/RR/exp/p2_ckpt_150.pt")
    parser.add_argument("--p1ckpt", type=str, default="/home/xteam/PaperCode/MM_IJCV/TIP/06_results/TIP_P2/RR/exp/p1_ckpt_150.pkl")

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

    pathd = '/home/xteam/PaperCode/data_zoo/NormalPredict/23_ph_300w_ffhq.csv'
    train_dataset, _ = getTrainDataP2(csvPath=pathd, imgSize=args.imgSize, validation_split=0)
    train_dl  = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=8)
        
    train(args, train_dl, PreTrainModel, FNR2R, g_optim, device)

