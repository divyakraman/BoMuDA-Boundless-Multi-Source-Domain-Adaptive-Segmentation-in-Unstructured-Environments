import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
import random

from model.source3_concat import *
from utils.loss import CrossEntropy2d
from dataset.idd_dataset import iddDataSet

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DRN_D-38'
BATCH_SIZE = 2
ITER_SIZE = 1
NUM_WORKERS = 1
DATA_DIRECTORY = '/datasets/idd'  
DATA_LIST_PATH = './dataset/idd_list/train_images.txt'
INPUT_SIZE = '840,488' 
IGNORE_LABEL = 255
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 30000
NUM_STEPS_STOP = 30000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
#RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = './snapshots/idd_multi3source_combinedbddbase_iteration3/'
WEIGHT_DECAY = 0.0005
#THRESHOLD = 0.9 

pseudoWeight = 1.0


SET = 'train'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    #parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
    #                    help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    #parser.add_argument("--threshold", type=str, default=THRESHOLD,
    #                    help="multi-source labels threshold")

    parser.add_argument("--pseudoWeight", type=str, default=pseudoWeight,
                        help="pseudo weight")
    
    
    return parser.parse_args()


args = get_arguments()

def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    #label = Variable(label.long()).cuda(gpu)
    label = torch.from_numpy(label)
    label = Variable(label).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)

    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def main():
    """Create the model and start the training."""

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    cudnn.enabled = True
    gpu = args.gpu

    source1 = torch.load('snapshots/cs2idd_multi_drnd38/CS_50000.pth') 
    source2 = torch.load('snapshots/bddbase_multi3source_refinedbddfromiter1labels_drnd38_iteration2/IDD_30000.pth')
    source3 = torch.load('snapshots/gta2idd_multi_drnd38/GTA_50000.pth')

     

    # Create network
    #model = Net3()
    model = torch.load('snapshots/idd_multi3source_combinedbddbase_iteration1/IDD_30000.pth')

    model.train()
    model.cuda(args.gpu)

    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader = data.DataLoader(
        iddDataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                    crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    # implement model.optim_parameters(args) to handle different models' lr setting

    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')
    
   
    for i_iter in range(0, args.num_steps):

        loss_seg_value1 = 0
        
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        
        for sub_i in range(args.iter_size):
           
            _, batch = trainloader_iter.__next__()
            images, labels = batch
            del labels
            images = Variable(images).cuda(args.gpu)

            pred_intermediate1, pred1 = source1(images)
            pred_intermediate2, pred2 = source2(images)
            pred_intermediate3, pred3 = source3(images)

            #Pseudo labels generation
            pred = pred2.detach()
            pred = pred.cpu()
            pred = pred.numpy()
            pred_labels = np.argmax(pred,1)
            pseudo_labels = pred_labels
            
            
            
            
            pred_3source = torch.cat((pred1,pred2,pred3),1)
            pred_final = model(pred_3source)
            
            loss_seg1 = args.pseudoWeight * loss_calc(pred_final, pseudo_labels, args.gpu)
            loss = loss_seg1

            
            # proper normalization
            loss = loss / args.iter_size
            loss.backward()
            loss_seg_value1 += loss_seg1.data.cpu().numpy() / args.iter_size
            
        optimizer.step()
        
        torch.cuda.empty_cache()
        
        print('exp = {}'.format(args.snapshot_dir))
        print(
        'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f} '.format(
            i_iter, args.num_steps, loss_seg_value1))

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model, osp.join(args.snapshot_dir, 'IDD_' + str(args.num_steps_stop) + '.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model, osp.join(args.snapshot_dir, 'IDD_' + str(i_iter) + '.pth'))
        
        if i_iter % 100 == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model, osp.join(args.snapshot_dir, 'IDD_' + str(args.num_steps_stop) + '.pth'))
                    

if __name__ == '__main__':
    main()

