import argparse
import os
import sys
import random
import math
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter # https://github.com/lanpa/tensorboard-pytorch
import pcpnet
from dataset import PointcloudPatchDataset, RandomPointcloudPatchSampler, SequentialShapeRandomPointcloudPatchSampler
from dsac import DSAC, WDSAC, MSDSAC, MoEDSAC
from ngran import ngran
import lhhngran
import numpy as np
import time
import utils
def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    parser.add_argument('--name', type=str, default='k256_s007_nostd_sumd_pt32_pl32_num_c', help='training run name')
    parser.add_argument('--patch_radius', type=float, default=[0.07], nargs='+', help='patch radius in multiples of the shape\'s bounding box diagonal, multiple values for multi-scale.')
    #parser.add_argument('--patch_radius', type=float, default=[0.05, 0.03, 0.02], nargs='+', help='patch radius in multiples of the shape\'s bounding box diagonal, multiple values for multi-scale.')
    parser.add_argument('--knn', type=int, default=True, help='k nearest neighbors.')
    parser.add_argument('--decoder', type=str, default='PointPredNet', help='PointPredNet, PointGenNet')
    parser.add_argument('--use_mask', type=int, default=False, help='use point mask')
    parser.add_argument('--share_pts_stn', type=int, default=True, help='')

    #parser.add_argument('--gpu_idx', type=str, default='1,2,3', help='set < 0 to use CPU')
    parser.add_argument('--gpu_idx', type=int, default=3, help='set < 0 to use CPU')
    parser.add_argument('--refine', type=str, default='../../data/dsacmodels/k256_s007_nostd_sumd_pt32_pl32_num_model.pth', help='refine model at this patch')
    parser.add_argument('--expert_refine', type=str, default=['../../data/dsacmodels/k256_s007_nostd_sumd_pt32_pl32_num_model.pth',
        '../../data/dsacmodels/k256_s007_nostd_sumd_pt32_pl32_num_model.pth',
        '../../data/dsacmodels/k256_s007_nostd_sumd_pt32_pl32_num_model.pth'], help='refine model at this patch')

    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    #parser.add_argument('--point_tuple', type=int, default=1, help='use n-tuples of points as input instead of single points')
    parser.add_argument('--points_per_patch', type=int, default=[256], nargs='+', help='max. number of points per patch')
    #parser.add_argument('--points_per_patch', type=int, default=[512, 256, 128], nargs='+', help='max. number of points per patch')
    parser.add_argument('--in_points_dim', '-ipdim', type=int, default=3, help='3 for position, 6 for position + normal, ')

    parser.add_argument('--desc', type=str, default='training for single-scale normal estimation.', help='description')
    parser.add_argument('--indir', type=str, default='/data/pclouds', help='input folder (point clouds)')
    parser.add_argument('--indir2', type=str, default='../../data/results/s003_nostd_sumd_pt32_pl32_dist_c', help='input folder (point clouds)')
    parser.add_argument('--outdir', type=str, default='../../data/dsacmodels', help='output folder (trained models)')
    parser.add_argument('--logdir', type=str, default='../../data/dsaclogs', help='training log folder')
    # parser.add_argument('--trainset', type=str, default='trainingset_whitenoise.txt', help='training set file name')
    # parser.add_argument('--testset', type=str, default='validationset_whitenoise.txt', help='test set file name')
    parser.add_argument('--trainset', type=str, default='validationset_no_noise.txt', help='training set file name')
    parser.add_argument('--testset', type=str, default='validationset_no_noise.txt', help='test set file name')
    parser.add_argument('--saveinterval', type=int, default='10', help='save model each n epochs')

    # training parameters 
    parser.add_argument('--patch_point_count_std', type=float, default=0.0, help='standard deviation of the number of points in a patch')
    parser.add_argument('--patches_per_shape', type=int, default=1000, help='number of patches sampled from each shape in an epoch')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=100, help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--seed', type=int, default=3627474, help='manual seed')
    parser.add_argument('--training_order', type=str, default='random', help='order in which the training patches are presented:\n'
                        'random: fully random over the entire dataset (the set of all patches is permuted)\n'
                        'random_shape_consecutive: random over the entire dataset, but patches of a shape remain consecutive (shapes and patches inside a shape are permuted)')
    parser.add_argument('--identical_epochs', type=int, default=False, help='use same patches in each epoch, mainly for debugging')


    parser.add_argument('--opti', type=str, default='SGD', help='optimizer, SGD or Adam')
    # lr = 0.0001 & momentum = 0.9 for SGD in PCPNet; lr = 0.001 for Adam in 3dcoded,
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='gradient descent momentum')
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')

    
    #parser.add_argument('--use_pca', type=int, default=False, help='Give both inputs and ground truth in local PCA coordinate frame')
    parser.add_argument('--normal_loss', type=str, default='ms_euclidean', help='Normal loss type:\n'
                        'ms_euclidean: mean square euclidean distance\n'
                        'ms_oneminuscos: mean square 1-cos(angle error)')
    # model hyperparameters
    parser.add_argument('--outputs', type=str, nargs='+', default=['unoriented_normals'], help='outputs of the network, a list with elements of:\n'
                        'unoriented_normals: unoriented (flip-invariant) point normals\n'
                        'oriented_normals: oriented point normals\n'
                        'max_curvature: maximum curvature\n'
                        'min_curvature: mininum curvature')
    parser.add_argument('--use_point_stn', type=int, default=True, help='use point spatial transformer')  
    parser.add_argument('--use_feat_stn', type=int, default=True, help='use feature spatial transformer')
    parser.add_argument('--sym_op', type=str, default='sumd', help='symmetry operation: max, sum, sumd (sum_dist), aved (ave_dist) ')

    ##### RANSAC hyperparameters
    parser.add_argument('--generate_points_num', '-gpnum', type=int, default=32, help='number of points output form the net')
    parser.add_argument('--generate_points_dim', '-gpdim', type=int, default=3, help='dim of points output form the net: 4 for pts + weight')
    parser.add_argument('--hypotheses', '-hyps', type=int, default=32, help='number of planes hypotheses sampled for each patch')
    
    # two parameters if score with sum of error distance: 
    #   p[0] for sigma^2 of gaussian used in the soft inlier count. 
    # three parameters if socre with inliner count:
    #   p[0] for threshold used in the soft inlier count, 
    #   p[1] for scaling factor within the sigmoid of the soft inlier count'
    # common parameter: p[end] for scaling factor for the soft inlier scores (controls the peakiness of the hypothesis distribution
    # parser.add_argument('--inlier_params', '-ip', type=float, default=[0.01, 0.5], help='RANSAC scorer with inlier distance') 
    parser.add_argument('--inlier_params', '-ip', type=float, default=[0.1, 100, 0.5], help='RANSAC scorer with inlier count') 

    return parser.parse_args()

opt = parse_arguments()
device = 0
train_dataset = PointcloudPatchDataset(
        root=opt.indir,
        root_in=opt.indir2,
        shape_list_filename=opt.trainset,
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        dim_pts = opt.in_points_dim,
        knn = opt.knn,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        cache_capacity=opt.cache_capacity)
if opt.training_order == 'random':
        train_datasampler = RandomPointcloudPatchSampler(
            train_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
elif opt.training_order == 'random_shape_consecutive':
    train_datasampler = SequentialShapeRandomPointcloudPatchSampler(
        train_dataset,
        patches_per_shape=opt.patches_per_shape,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs)
else:
    raise ValueError('Unknown training order: %s' % (opt.training_order))

train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_datasampler,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))

test_dataset = PointcloudPatchDataset(
        root=opt.indir,
        root_in=opt.indir2,
        shape_list_filename=opt.testset,
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        dim_pts = opt.in_points_dim,
        knn = opt.knn,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        cache_capacity=opt.cache_capacity)
if opt.training_order == 'random':
        test_datasampler = RandomPointcloudPatchSampler(
            test_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
elif opt.training_order == 'random_shape_consecutive':
    test_datasampler = SequentialShapeRandomPointcloudPatchSampler(
            test_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs_=opt.identical_epochs)
else:
    raise ValueError('Un_known training order: %s' % (opt.training_order))

test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=test_datasampler,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))

model = ngran(points_per_patch=opt.points_per_patch[0],  dim_pts = opt.in_points_dim,
              num_gpts = opt.points_per_patch[0],
              dim_gpts=1,                           
              use_mask=False,  sym_op='max', ith = 0, 
              use_point_stn=opt.use_point_stn, use_feat_stn=opt.use_feat_stn,               
              device=0)
      
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

start_time = time.time()
iteration = 0
for epoch in range(500):
    LOSS = 0
    
    train_batchind = -1
    train_fraction_done = 0.0
    train_enum = enumerate(train_dataloader, 0)

    test_batchind = -1
    test_fraction_done = 0.0
    test_enum = enumerate(test_dataloader, 0)

    for train_batchind, data in train_enum:
            # set to training mode
        model.train()

        points = data[0]#这时的point是64*512*3的类型
        target = data[1]
        mask = data[2]
        dist = data[3]

        points = points.transpose(2, 1)
        points = points.to(device)          
        # target = target.to(device)
        # mask = mask.to(device)
        # dist = dist.to(device)
        log_probs = model(points)

       
        # log_probs = model(points)
        probs = torch.exp(log_probs).cpu()
        Pts = points.cpu()
        Pts = torch.unsqueeze(Pts,3)

        # this tensor will contain the gradients for the entire batch
        log_probs_grad = torch.zeros(log_probs.size())
        normalout = torch.zeros(points.size(0),1, 3)
        avg_loss = 0
        GTloss = 0
        # loop over batch
        for b in range(points.size(0)):

            # we sample multiple times per input and keep the gradients and losse in the following lists
            log_prob_grads = []
            losses = []
            

            # loop over samples for approximating the expected loss
            for s in range(5):

                # gradient tensor of the current sample
                # when running NG-RANSAC, this tensor will indicate which correspondences have been samples
                # this is multiplied with the loss of the sample to yield the gradients for log-probabilities
                gradients = torch.zeros(probs[b].size())

                # inlier mask of the best model

                out_N = torch.zeros(3)

                # random seed used in C++ (would be initialized in each call with the same seed if not provided from outside)
                rand_seed = random.randint(0, 10000)
                # ngransac.find_essential_mat(correspondences[b], probs[b], rand_seed, opt.hyps, opt.threshold, E, inliers, gradients)
                Bestscore = lhhngran.normal_estimate(Pts[b], probs[b], out_N ,gradients, rand_seed, 16, 0, 0.1, 100.0)
                Gt = target[b][0]
                GTloss += torch.min((out_N-Gt).pow(2).sum(0), (out_N+Gt).pow(2).sum(0))
 



                # choose the user-defined training signal
                loss = -Bestscore

                log_prob_grads.append(gradients)
                losses.append(loss)

            # calculate the gradients of the expected loss
            normalout[b][0] = out_N
            baseline = sum(losses) / len(losses)  # expected loss
            for i, l in enumerate(losses):  # substract baseline for each sample to reduce gradient variance
                log_probs_grad[b] += log_prob_grads[i] * (l - baseline) / 5

            avg_loss += baseline

        avg_loss /= points.size(0)
        



        # update model
        torch.autograd.backward((log_probs), (log_probs_grad.to(device)))
        optimizer.step()
        optimizer.zero_grad()

        # print("Iteration: ", iteration, "Loss: ", avg_loss, "Gtloss: ", GTloss)


        iteration += 1
        LOSS += GTloss
    end_time = time.time()
    print('epoch: %d, batchsize: %d, cost time: %s, LOSS: %s' % (epoch, 32, end_time - start_time, LOSS))
