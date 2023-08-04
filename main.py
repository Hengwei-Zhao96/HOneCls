#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/5/29 20:17
# @Author : Hw-Zhao
# @Site : 
# @File : main.py
# @Software: PyCharm
import argparse
import importlib
import logging
import os
from tqdm import tqdm
import torch
import numpy as np
from model.freeocnet import FreeOCNet
from utils import basic_logging, classmap_2_RGBmap, ScalarRecorder, set_random_seed, get_cfg_dataloader, all_metric


def Argparse():
    parser = argparse.ArgumentParser(
        description='FGOCC')
    parser.add_argument('-p', '--prior', type=float, default=0.3769, help='Class prior')
    parser.add_argument('-c', '--cls', type=int, default=4, help='Detected class')
    parser.add_argument('-d', '--dataset', type=str, default='HongHu', help='Dataset')
    parser.add_argument('-g', '--gpu_id', type=str, default='0', help='GPU_ID')
    parser.add_argument('-r', '--risk', type=str, default="OneClassRiskEstimator", help='Risk Estimation')
    parser.add_argument('-m', '--model', type=str, default='FreeOCNet', help='Model')
    parser.add_argument('-f', '--focal_weight', type=float, default='0.1', help='Focal Weight')
    parser.add_argument('-w', '--class_weight', type=float, default='0.3', help='Class Weight')

    return parser.parse_args()


def fcn_evaluate_fn(model, test_dataloader, out_fig_config, cls, path, device):
    # start = time.time()

    model.eval()
    f1 = 0
    # start = time.time()
    with torch.no_grad():
        for (im, positive_test_mask, negative_test_mask) in test_dataloader:
            im = im.to(device)
            positive_test_mask = positive_test_mask.squeeze()
            negative_test_mask = negative_test_mask.squeeze()
            pred_pro = torch.sigmoid(model(im)).squeeze().cpu()
            # end = time.time()
            # print("Time:%f" % (end - start))
            pred_class = torch.where(pred_pro > 0.5, 1, 0)

            cls_fig = classmap_2_RGBmap(
                pred_class[:out_fig_config['image_size'][0], :out_fig_config['image_size'][1]].numpy(),
                palette=out_fig_config['palette'], cls=cls)
            cls_fig.save(path[0])

            np.save(path[1], pred_pro[:out_fig_config['image_size'][0], :out_fig_config['image_size'][1]].numpy())

            mask = (positive_test_mask + negative_test_mask).bool()

            label = positive_test_mask
            target = torch.masked_select(label.view(-1), mask.view(-1)).numpy()
            pred_class = torch.masked_select(pred_class.view(-1).cpu(), mask.view(-1)).numpy()
            pred_pro = torch.masked_select(pred_pro.view(-1).cpu(), mask.view(-1)).numpy()

            auc, fpr, tpr, threshold, pre, rec, f1 = all_metric(pred_pro, pred_class, target)
    # end = time.time()
    # print("Time:%f" % (end - start))

    return auc, fpr, tpr, threshold, pre, rec, f1


if __name__ == '__main__':
    args = Argparse()
    # set_random_seed(2333)  # Fixed random seed
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    config, DataLoader = get_cfg_dataloader(dataset=args.dataset)

    if args.risk == "OneClassRiskEstimator":
        Loss = importlib.import_module('risk_estimation.one_class_risk_estimator').OneClassRiskEstimator
    elif args.risk == "absNegative":
        Loss = importlib.import_module('risk_estimation.absPU').absNegative

    config['risk_estimation']['class_prior'] = args.prior
    config['data']['train']['params']['cls'] = args.cls
    config['data']['test']['params']['cls'] = args.cls
    config['risk_estimation']['focal_weight'] = args.focal_weight
    config['risk_estimation']['class_weight'] = args.class_weight

    # Config log file
    extra_name = 'normal'

    folder_name = os.path.join(args.dataset,
                               'positive-samples_' + str(config['data']['train']['params'][
                                                             'num_positive_train_samples']) + '_sub-minibatch_' + str(
                                   config['data']['train']['params']['sub_minibatch']) + '_ratio_' + str(
                                   config['data']['train']['params']['ratio']),
                               'class_weight_' + str(
                                   config['risk_estimation']['class_weight']) + '_focal_weight_' + str(
                                   config['risk_estimation']['focal_weight']),
                               'warm_up_epoch_' + str(config['risk_estimation']['warm_up_epoch']) + '_loss_' +
                               config['risk_estimation']['loss'],
                               args.model,
                               extra_name
                               )

    save_path = basic_logging(
        os.path.join('log', args.risk, folder_name,
                     str(config['data']['train']['params']['cls'])))
    print("The save path is:", save_path)

    dataloader = DataLoader(config=config['data']['train']['params'])
    test_dataloader = DataLoader(config=config['data']['test']['params'])

    if args.model == 'FreeOCNet':
        model = FreeOCNet(config['model']['params']).to(DEVICE)

    if args.risk == "OneClassRiskEstimator":
        loss_function = Loss(prior=config['risk_estimation']['class_prior'],
                             class_weight=config['risk_estimation']['class_weight'],
                             warm_up_epoch=config['risk_estimation']['warm_up_epoch'],
                             focal_weight=config['risk_estimation']['focal_weight'],
                             loss=config['risk_estimation']['loss'])
    elif args.risk == "absNegative":
        loss_function = Loss(prior=config['risk_estimation']['class_prior'])

    if config['optimizer']['type'] == 'SGD':
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    momentum=config['optimizer']['params']['momentum'],
                                    weight_decay=config['optimizer']['params']['weight_decay'],
                                    lr=config['learning_rate']['params']['base_lr'])
    elif config['optimizer']['type'] == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     weight_decay=config['optimizer']['params']['weight_decay'],
                                     lr=config['learning_rate']['params']['base_lr'])
    else:
        NotImplemented

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                       gamma=config['learning_rate']['params']['power'])

    f1_recorder = ScalarRecorder()
    loss_recorder = ScalarRecorder()
    p_loss_recorder = ScalarRecorder()
    n_loss_recorder = ScalarRecorder()
    p_n_loss_recorder = ScalarRecorder()
    u_n_loss_recorder = ScalarRecorder()

    bar = tqdm(list(range(config['learning_rate']['params']['max_iters'])), ncols=180)
    for i in bar:
        training_loss = 0.0
        training_p_loss = 0.0
        training_n_loss = 0.0
        training_p_n_loss = 0.0
        training_u_n_loss = 0.0
        num = 0
        model.train()
        for (data, positive_train_mask, unlabeled_train_mask) in dataloader:
            data = data.to(DEVICE)
            positive_train_mask = positive_train_mask.to(DEVICE)
            unlabeled_train_mask = unlabeled_train_mask.to(DEVICE)

            target = model(data)

            loss, estimated_p_loss, estimated_n_loss, estimated_u_n_loss, estimated_p_n_loss = loss_function(target,
                                                                                                             positive_train_mask,
                                                                                                             unlabeled_train_mask,
                                                                                                             epoch=i)
            loss_recorder.update_gradient(loss.item())
            p_loss_recorder.update_gradient(estimated_p_loss.item())
            n_loss_recorder.update_gradient(estimated_n_loss.item())
            p_n_loss_recorder.update_gradient(estimated_p_n_loss.item())
            u_n_loss_recorder.update_gradient(estimated_u_n_loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            training_loss += loss.item()
            training_p_loss += estimated_p_loss.item()
            training_n_loss += estimated_n_loss.item()
            training_u_n_loss += estimated_u_n_loss.item()
            training_p_n_loss += estimated_p_n_loss.item()
            num += 1
        scheduler.step()

        path_1 = os.path.join(save_path, str(i + 1) + '.png')
        path_2 = os.path.join(save_path, 'probaility.npy')
        # if i < 995:
        #     continue
        auc, fpr, tpr, threshold, pre, rec, f1 = fcn_evaluate_fn(model,
                                                                 test_dataloader=test_dataloader,
                                                                 out_fig_config=config['out_config']['params'],
                                                                 cls=config['data']['train']['params']['cls'],
                                                                 device=DEVICE,
                                                                 path=(path_1, path_2))
        f1_recorder.update_gradient(f1)

        auc_roc = {}
        auc_roc['fpr'] = fpr
        auc_roc['tpr'] = tpr
        auc_roc['threshold'] = threshold
        auc_roc['auc'] = auc

        np.save(os.path.join(save_path, 'auc_roc.npy'), auc_roc)

        bar.set_description(
            'loss: %.4f,p_loss: %.4f,n_loss: %.4f,un_loss: %.4f,pn_loss: %.4f,AUC:%.6f, Precision:%.6f,Recall:%6f,'
            'F1: %.6f' % (training_loss / num,
                          training_p_loss / num,
                          training_n_loss / num,
                          training_u_n_loss / num,
                          training_p_n_loss / num,
                          auc,
                          pre,
                          rec,
                          f1))
        logging.info(
            "{} epoch, Training loss {:.4f}, Training p_loss {:.4f}, Training n_loss {:.4f},Training un_loss {:.4f},"
            "Training pn_loss {:.4f},AUC {:.6f}, Precision {:.6f}, Recall {:.6f}, F1 {:.6f}".format(
                i + 1,
                training_loss / num,
                training_p_loss / num,
                training_n_loss / num,
                training_u_n_loss / num,
                training_p_n_loss / num,
                auc,
                pre,
                rec,
                f1))

    loss_recorder.save_scalar_npy('loss_npy', save_path)
    loss_recorder.save_lineplot_fig('Loss', 'loss', save_path)
    p_loss_recorder.save_scalar_npy('p_loss_npy', save_path)
    p_loss_recorder.save_lineplot_fig('Estimated Positive Loss', 'p_loss', save_path)
    n_loss_recorder.save_scalar_npy('n_loss_npy', save_path)
    n_loss_recorder.save_lineplot_fig('Estimated Negative Loss', 'n_loss', save_path)
    p_n_loss_recorder.save_scalar_npy('p_n_loss_npy', save_path)
    p_n_loss_recorder.save_lineplot_fig('Estimated Pn Loss', 'p_n_loss', save_path)
    u_n_loss_recorder.save_scalar_npy('u_n_loss_npy', save_path)
    u_n_loss_recorder.save_lineplot_fig('Estimated Un Loss', 'u_n_loss', save_path)
    f1_recorder.save_scalar_npy('f1_npy', save_path)
    f1_recorder.save_lineplot_fig('F1-score', 'f1-score', save_path)
