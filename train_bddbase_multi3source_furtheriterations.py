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

from model.drn38_multi import *
from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d
from dataset.idd_dataset import iddDataSet
from dataset.bdd_source import BDDDataSet

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DRN_D-38'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 1
DATA_DIRECTORY = '/datasets/bdd/bdd100k/seg' 
DATA_LIST_PATH = './dataset/bdd_list/train_images.txt'
INPUT_SIZE = '1280,720'
DATA_DIRECTORY_TARGET = '/datasets/idd' 
DATA_LIST_PATH_TARGET = './dataset/idd_list/train_images.txt'
IGNORE_LABEL = 255
INPUT_SIZE_TARGET = '840,488' 
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
SNAPSHOT_DIR = './snapshots/save_folder/'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001
GAN = 'LS'

lambdaKL = 1.0
csWeight = 0.28
bddWeight = 0.411
gtaWeight = 0.3
pseudoWeight = 1.0

TARGET = 'idd'
SET = 'train'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
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
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
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
    parser.add_argument("--gan", type=str, default=GAN,
                        help="choose the GAN objective.")

    parser.add_argument("--lambdaKL", type=str, default=lambdaKL,
                        help="KL divergence weight")
    parser.add_argument("--csWeight", type=str, default=csWeight,
                        help="Cityscapes weight")
    parser.add_argument("--bddWeight", type=str, default=bddWeight,
                        help="BDD weight")
    parser.add_argument("--gtaWeight", type=str, default=gtaWeight,
                        help="GTA weight")
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
    #label = torch.from_numpy(label)
    #label = Variable(label).cuda(gpu)
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)

    return criterion(pred, label)

def loss_calc_target_pseudo(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = torch.from_numpy(label)
    label = Variable(label).cuda(gpu)
    #label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)

    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


def main():
    """Create the model and start the training."""

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True
    gpu = args.gpu

    # Create network
    #model = DRNSeg()
    #model = torch.load('snapshots/bdd2idd_multi_drnd38/BDD_125000.pth')
    model = torch.load('snapshots/bddbase_multi3source_refinedbddfromiter0labels_drnd38_iteration1/IDD_30000.pth')

    model.train()
    model.cuda(args.gpu)

    cudnn.benchmark = True

    #KL loss
    kl_loss = nn.KLDivLoss()
    kl_loss = kl_loss.cuda(gpu)

    #Load 3 source information
    source1 = torch.load('snapshots/cs2idd_multi_drnd38/CS_50000.pth') 
    source2 = torch.load('snapshots/bddbase_multi3source_refinedbddfromiter1labels_drnd38_iteration2/IDD_30000.pth')
    source3 = torch.load('snapshots/gta2idd_multi_drnd38/GTA_50000.pth')

    
    source1_D2 = torch.load('snapshots/cs2idd_multi_drnd38/CS_50000_D2.pth')
    source2_D2 = torch.load('snapshots/bdd2idd_multi_drnd38/BDD_90000_D2.pth')
    source3_D2 = torch.load('snapshots/gta2idd_multi_drnd38/GTA_50000_D2.pth')
    #3 source information load done
    


    # init D
    #model_D1 = FCDiscriminator(num_classes=args.num_classes)
    #model_D2 = FCDiscriminator(num_classes=args.num_classes)
    
    model_D1 = torch.load('snapshots/bddbase_multi3source_refinedbddfromiter0labels_drnd38_iteration1/IDD_30000_D1.pth')
    model_D2 = torch.load('snapshots/bddbase_multi3source_refinedbddfromiter0labels_drnd38_iteration1/IDD_30000_D2.pth')


    model_D1.train()
    model_D1.cuda(args.gpu)

    model_D2.train()
    model_D2.cuda(args.gpu)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader = data.DataLoader(
        BDDDataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                    crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(iddDataSet(args.data_dir_target, args.data_list_target,
                                                     max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                     crop_size=input_size_target,
                                                     scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
                                                     set=args.set),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)


    targetloader_iter = enumerate(targetloader)

    # implement model.optim_parameters(args) to handle different models' lr setting

    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()

    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D2.zero_grad()

    if args.gan == 'Vanilla':
        bce_loss = torch.nn.BCEWithLogitsLoss()
    elif args.gan == 'LS':
        bce_loss = torch.nn.MSELoss()

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear')

    # labels for adversarial training
    source_label = 0
    target_label = 1

    for i_iter in range(0, args.num_steps):

        loss_seg_value1 = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0

        loss_seg_value2 = 0
        loss_adv_target_value2 = 0
        loss_D_value2 = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D1.zero_grad()
        optimizer_D2.zero_grad()
        adjust_learning_rate_D(optimizer_D1, i_iter)
        adjust_learning_rate_D(optimizer_D2, i_iter)

        for sub_i in range(args.iter_size):

            # train G

            # don't accumulate grads in D
            for param in model_D1.parameters():
                param.requires_grad = False

            for param in model_D2.parameters():
                param.requires_grad = False

            # train with source

            _, batch = trainloader_iter.__next__()
            images, labels = batch
            images = Variable(images).cuda(args.gpu)

            pred1, pred2 = model(images)
            #pred1 = interp(pred1)
            #pred2 = interp(pred2)

            loss_seg1 = loss_calc(pred1, labels, args.gpu)
            loss_seg2 = loss_calc(pred2, labels, args.gpu)
            loss = loss_seg2 + args.lambda_seg * loss_seg1

            # proper normalization
            loss = loss / args.iter_size
            loss.backward()
            loss_seg_value1 += loss_seg1.data.cpu().numpy() / args.iter_size
            loss_seg_value2 += loss_seg2.data.cpu().numpy() / args.iter_size

            # train with target

            _, batch = targetloader_iter.__next__()
            images, labels = batch
            images = Variable(images).cuda(args.gpu)

            #Generate pseudo labels
            pred_intermediate1, pred1 = source1(images)
            pred_intermediate2, pred2 = source2(images)
            pred_intermediate3, pred3 = source3(images)

            pred_D2_source1 = source1_D2(prob_2_entropy(F.softmax(pred1)))
            pred_D2_source2 = source2_D2(prob_2_entropy(F.softmax(pred2)))
            pred_D2_source3 = source3_D2(prob_2_entropy(F.softmax(pred3)))

        
            loss_adv_target_source1 = bce_loss(pred_D2_source1,
                                        Variable(torch.FloatTensor(pred_D2_source1.data.size()).fill_(source_label)).cuda(
                                            args.gpu))
            loss_adv_target_source2 = bce_loss(pred_D2_source2,
                                        Variable(torch.FloatTensor(pred_D2_source2.data.size()).fill_(source_label)).cuda(
                                            args.gpu))
            loss_adv_target_source3 = bce_loss(pred_D2_source3,
                                        Variable(torch.FloatTensor(pred_D2_source3.data.size()).fill_(source_label)).cuda(
                                            args.gpu))

            loss_adv_target_source1 = loss_adv_target_source1.detach()
            loss_adv_target_source1 = loss_adv_target_source1.cpu()
            loss_adv_target_source1 = loss_adv_target_source1.numpy()
            loss_adv_target_source2 = loss_adv_target_source2.detach()
            loss_adv_target_source2 = loss_adv_target_source2.cpu()
            loss_adv_target_source2 = loss_adv_target_source2.numpy()
            loss_adv_target_source3 = loss_adv_target_source3.detach()
            loss_adv_target_source3 = loss_adv_target_source3.cpu()
            loss_adv_target_source3 = loss_adv_target_source3.numpy()
            
    
            #BCE lower the better
            if(loss_adv_target_source1<loss_adv_target_source2 and loss_adv_target_source1<loss_adv_target_source3):
                pred = pred1.detach()
                pred = pred.cpu()
                pred = pred.numpy()
                pred_labels = np.argmax(pred,1)
                pseudo_labels = pred_labels
            elif(loss_adv_target_source2<loss_adv_target_source1 and loss_adv_target_source2<loss_adv_target_source3):
                pred = pred2.detach()
                pred = pred.cpu()
                pred = pred.numpy()
                pred_labels = np.argmax(pred,1)
                pseudo_labels = pred_labels
            else:
                pred = pred3.detach()
                pred = pred.cpu()
                pred = pred.numpy()
                pred_labels = np.argmax(pred,1)
                pseudo_labels = pred_labels

            #Pseudo labels generation done

            pred_target1, pred_target2 = model(images)
            pred_target1 = interp_target(pred_target1)
            pred_target2 = interp_target(pred_target2)

            D_out1 = model_D1(prob_2_entropy(F.softmax(pred_target1)))
            D_out2 = model_D2(prob_2_entropy(F.softmax(pred_target2)))

            loss_adv_target1 = bce_loss(D_out1,
                                       Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda(
                                           args.gpu))

            loss_adv_target2 = bce_loss(D_out2,
                                        Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda(
                                            args.gpu))

            loss = args.lambda_adv_target1 * loss_adv_target1 + args.lambda_adv_target2 * loss_adv_target2
            #loss = loss / args.iter_size
            #loss.backward()
            loss_adv_target_value1 += loss_adv_target1.data.cpu().numpy() / args.iter_size
            loss_adv_target_value2 += loss_adv_target2.data.cpu().numpy() / args.iter_size

            #Add target pseudo labels cross entropy loss
            loss_seg_target = args.pseudoWeight * loss_calc_target_pseudo(pred_target2, pseudo_labels, args.gpu)
            loss = loss + loss_seg_target

            #Add KL divergence
            loss_kl1_source1 = kl_loss(pred_target1, pred_intermediate1)
            loss_kl2_source1 = kl_loss(pred_target2, pred1)
            loss = loss + args.lambdaKL * args.csWeight * (loss_kl2_source1 + args.lambda_seg * loss_kl1_source1)

            loss_kl1_source2 = kl_loss(pred_target1, pred_intermediate2)
            loss_kl2_source2 = kl_loss(pred_target2, pred2)
            loss = loss + args.lambdaKL * args.bddWeight * (loss_kl2_source2 + args.lambda_seg * loss_kl1_source2)

            loss_kl1_source3 = kl_loss(pred_target1, pred_intermediate3)
            loss_kl2_source3 = kl_loss(pred_target2, pred3)
            loss = loss + args.lambdaKL * args.gtaWeight * (loss_kl2_source3 + args.lambda_seg * loss_kl1_source3)



            # proper normalization
            loss = loss / args.iter_size
            loss.backward()

            #Pseudo labels cross entropy loss added
            

            # train D

            # bring back requires_grad
            for param in model_D1.parameters():
                param.requires_grad = True

            for param in model_D2.parameters():
                param.requires_grad = True

            # train with source
            pred1 = pred1.detach()
            pred2 = pred2.detach()

            D_out1 = model_D1(prob_2_entropy(F.softmax(pred1)))
            D_out2 = model_D2(prob_2_entropy(F.softmax(pred2)))

            loss_D1 = bce_loss(D_out1,
                              Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda(args.gpu))

            loss_D2 = bce_loss(D_out2,
                               Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda(args.gpu))

            loss_D1 = loss_D1 / args.iter_size / 2
            loss_D2 = loss_D2 / args.iter_size / 2

            loss_D1.backward()
            loss_D2.backward()

            loss_D_value1 += loss_D1.data.cpu().numpy()
            loss_D_value2 += loss_D2.data.cpu().numpy()

            # train with target
            pred_target1 = pred_target1.detach()
            pred_target2 = pred_target2.detach()

            D_out1 = model_D1(prob_2_entropy(F.softmax(pred_target1)))
            D_out2 = model_D2(prob_2_entropy(F.softmax(pred_target2)))

            loss_D1 = bce_loss(D_out1,
                              Variable(torch.FloatTensor(D_out1.data.size()).fill_(target_label)).cuda(args.gpu))

            loss_D2 = bce_loss(D_out2,
                               Variable(torch.FloatTensor(D_out2.data.size()).fill_(target_label)).cuda(args.gpu))

            loss_D1 = loss_D1 / args.iter_size / 2
            loss_D2 = loss_D2 / args.iter_size / 2

            loss_D1.backward()
            loss_D2.backward()

            loss_D_value1 += loss_D1.data.cpu().numpy()
            loss_D_value2 += loss_D2.data.cpu().numpy()

        optimizer.step()
        optimizer_D1.step()
        optimizer_D2.step()

        torch.cuda.empty_cache()
        
        print('exp = {}'.format(args.snapshot_dir))
        print(
        'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f} loss_seg2 = {3:.3f} loss_adv1 = {4:.3f}, loss_adv2 = {5:.3f} loss_D1 = {6:.3f} loss_D2 = {7:.3f}'.format(
            i_iter, args.num_steps, loss_seg_value1, loss_seg_value2, loss_adv_target_value1, loss_adv_target_value2, loss_D_value1, loss_D_value2))

        if i_iter % 100 == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model, osp.join(args.snapshot_dir, 'IDD_' + str(args.num_steps_stop) + '.pth'))
            torch.save(model_D1, osp.join(args.snapshot_dir, 'IDD_' + str(args.num_steps_stop) + '_D1.pth'))
            torch.save(model_D2, osp.join(args.snapshot_dir, 'IDD_' + str(args.num_steps_stop) + '_D2.pth'))


        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model, osp.join(args.snapshot_dir, 'IDD_' + str(i_iter) + '.pth'))
            torch.save(model_D1, osp.join(args.snapshot_dir, 'IDD_' + str(i_iter) + '_D1.pth'))
            torch.save(model_D2, osp.join(args.snapshot_dir, 'IDD_' + str(i_iter) + '_D2.pth'))


if __name__ == '__main__':
    main()


