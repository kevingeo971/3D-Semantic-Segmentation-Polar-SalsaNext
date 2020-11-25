#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from network.BEV_Unet import BEV_Unet
from network.ptBEV import ptBEVnet
from dataloader.dataset import collate_fn_BEV,SemKITTI,SemKITTI_label_name,spherical_dataset,voxel_dataset
from network.lovasz_losses import lovasz_softmax
from SalsaNext import SalsaNext
from SalsaNext_Circular import SalsaNext_Circular
import logging  
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.enabled = False
# logger.info("imported")
# exit()
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(message)s', 
                    handlers=[
                        logging.FileHandler("exp_runtime.log"),
                        logging.StreamHandler()
                    ]) 
  
#Creating an object 
logger=logging.getLogger() 
  
#Setting the threshold of logger to DEBUG 
# logger.setLevel(logging.DEBUG) 
#ignore weird np warning
import warnings
warnings.filterwarnings("ignore")

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count=np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label)+1)
    hist=hist[unique_label,:]
    hist=hist[:,unique_label]
    return hist

def SemKITTI2train(label):
    if isinstance(label, list):
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)

def SemKITTI2train_single(label):
    return label - 1 # uint8 trick

def main(args):
    data_path = args.data_dir
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    check_iter = args.check_iter
    model_save_path = args.model_save_path
    compression_model = args.grid_size[2]
    grid_size = args.grid_size
    pytorch_device = torch.device('cuda:0')
    # pytorch_device = torch.device('cpu')
    model = args.model
    logger.debug( " \n\nModel : " + model + "\n\n") 
    if model == 'polar':
        fea_dim = 9
        circular_padding = True
    elif model == 'traditional':
        fea_dim = 7
        circular_padding = False

    #prepare miou fun
    unique_label=np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str=[SemKITTI_label_name[x] for x in unique_label+1]

    #prepare model
    # my_BEV_model=BEV_Unet(n_class=len(unique_label), n_height = compression_model, input_batch_norm = True, dropout = 0.5, circular_padding = circular_padding)
    # if model == 'traditional':
    #     sn_model = SalsaNext(19*32)
    # elif model == 'polar':
    #     logger.debug("Cicular SalsaNext")
    #     sn_model = SalsaNext_Circular(19*32)
    sn_model=BEV_Unet(n_class=len(unique_label), n_height = compression_model, input_batch_norm = True, dropout = 0.5, circular_padding = circular_padding)
    print("BEV Unet")
    my_model = ptBEVnet(sn_model, pt_model = 'pointnet', grid_size =  grid_size, fea_dim = fea_dim, max_pt_per_encode = 256,
                            out_pt_fea_dim = 512, kernal_size = 1, pt_selection = 'random', fea_compre = compression_model)
    # if os.path.exists(model_save_path):
    #     my_model.load_state_dict(torch.load(model_save_path))
    
    my_model.to(pytorch_device)

    optimizer = optim.Adam(my_model.parameters())
    loss_fun = torch.nn.CrossEntropyLoss(ignore_index=255)

    #prepare dataset
    train_pt_dataset = SemKITTI(data_path + '/sequences/', imageset = 'train', return_ref = True)
    val_pt_dataset = SemKITTI(data_path + '/sequences/', imageset = 'val', return_ref = True)
    if model == 'polar':
        logger.debug("Polar Dataloader")
        train_dataset=spherical_dataset(train_pt_dataset, grid_size = grid_size, flip_aug = True, ignore_label = 0,rotate_aug = True, fixed_volume_space = True)
        val_dataset=spherical_dataset(val_pt_dataset, grid_size = grid_size, ignore_label = 0, fixed_volume_space = True)
    elif model == 'traditional':
        train_dataset=voxel_dataset(train_pt_dataset, grid_size = grid_size, flip_aug = True, ignore_label = 0,rotate_aug = True, fixed_volume_space = True)
        val_dataset=voxel_dataset(val_pt_dataset, grid_size = grid_size, ignore_label = 0, fixed_volume_space = True)
    train_dataset_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                    batch_size = train_batch_size,
                                                    collate_fn = collate_fn_BEV,
                                                    shuffle = True,
                                                    num_workers = 4)
    val_dataset_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                    batch_size = val_batch_size,
                                                    collate_fn = collate_fn_BEV,
                                                    shuffle = False,
                                                    num_workers = 4)

    # training
    epoch=0
    best_val_miou=0
    start_training=True
    my_model.train()
    global_iter = 0
    exce_counter = 0

    logger.debug("\n\n #### Epoch : " + str(epoch+1) + "\n\n")
    loss_list=[]
    pbar = tqdm(total=len(train_dataset_loader))
    on_l = []
    tt_t = []
    print("val_dataset_loader : ", len(val_dataset_loader ))
    with torch.no_grad():
        for i_iter_val,(_,val_vox_label,val_grid,val_pt_labs,val_pt_fea) in tqdm(enumerate(val_dataset_loader)):
            if i_iter_val >= 2000:
                break
            
            start_time  = time.time() 
            val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in val_pt_fea]
            val_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in val_grid]
            #val_label_tensor=val_vox_label.type(torch.LongTensor).to(pytorch_device)
            start_time_network = time.time()
            predict_labels = my_model(val_pt_fea_ten, val_grid_ten)

            curr_time = time.time()
            time_taken = curr_time - start_time
            only_network = curr_time - start_time_network
            #print("time_taken : ", time_taken, "\n Only network : ", only_network)
            on_l.append( only_network )
            tt_t.append( time_taken )

    print("time_taken : ", sum(tt_t) / len(tt_t), "\n Only network : ", sum(on_l) / len(on_l))
    # print(tt_t, on_l)
                
                
        

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--data_dir', default='../dataset')
    parser.add_argument('-p', '--model_save_path', default='./SemKITTI_PolarSeg_Exp_Runtime.pt')
    parser.add_argument('-m', '--model', choices=['polar','traditional'], default='polar', help='training model: polar or traditional (default: polar)')
    parser.add_argument('-s', '--grid_size', nargs='+', type=int, default = [480,360,32], help='grid size of BEV representation (default: [480,360,32])')
    parser.add_argument('--train_batch_size', type=int, default=2, help='batch size for training (default: 2)')
    parser.add_argument('--val_batch_size', type=int, default=1, help='batch size for validation (default: 2)')
    parser.add_argument('--check_iter', type=int, default=4000, help='validation interval (default: 4000)')
    
    args = parser.parse_args()
    if not len(args.grid_size) == 3:
        raise Exception('Invalid grid size! Grid size should have 3 dimensions.')

    logger.debug(' '.join(sys.argv))
    logger.debug(args)
    main(args)