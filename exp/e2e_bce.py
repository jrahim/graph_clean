import os, shutil, warnings, cv2, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader
from core_dl.train_params import TrainParameters
from core_dl.base_train_box import BaseTrainBox
from core_dl.torch_vision_ext import *
from models.encoder import Dense_Corr
from math import ceil
from torchvision import transforms
from random import randint
from models.cleaner import Cleaner

import graph_utils.utils as graph_util
from core_3dv.quaternion import *


class DenseCorrTrainBox(BaseTrainBox):

    def __init__(self, train_params: TrainParameters,
                 log_dir=None,
                 ckpt_path_dict=None,  # {'ckpt': ckpt_path (optional)}
                 bs_enc = 12):

        assert ckpt_path_dict is not None
        assert len(train_params.DEV_IDS) > 1

        super(DenseCorrTrainBox, self).__init__(train_params, log_dir,
                                                     checkpoint_path=ckpt_path_dict[
                                                         'ckpt'] if 'ckpt' in ckpt_path_dict else None,
                                                     comment_msg=train_params.NAME_TAG,
                                                     load_optimizer=True)
        # self.rsz = [480, 640]
        self.rsz = [360, 480]
        self.preprocess = transforms.Compose([
            transforms.Resize(self.rsz),
            transforms.ToTensor()
        ])
        self.toPIL = transforms.ToPILImage(mode='RGB')
        self.rotimg = transforms.RandomRotation((90,90))
        self.bs_enc = bs_enc


    def _set_loss_func(self):
        self.criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    def _set_optimizer(self):
        # config the optimizer
        super(DenseCorrTrainBox, self)._set_optimizer()
        self.optimizer = torch.optim.Adam([
            {'params': self.dense_corr_net.parameters(), 'lr': 2.0e-4},
            {'params': self.cleaner.parameters(), 'lr': 2.0e-3},
        ], lr=self.train_params.START_LR)

    def _set_network(self):
        super(DenseCorrTrainBox, self)._set_network()
        # with torch.cuda.device(self.dev_ids[0]):
        self.dense_corr_net = Dense_Corr(feat_model="resnet34", devs=self.dev_ids)
        self.cleaner = Cleaner(256, 256+4)

    def _load_network_from_ckpt(self, checkpoint_dict):
        # load network from checkpoint, ignore the instance if not found in dict.
        super(DenseCorrTrainBox, self)._load_network_from_ckpt(checkpoint_dict)
        with torch.cuda.device(self.dev_ids[0]):
            if 'dense_corr' in checkpoint_dict:
                self.dense_corr_net.load_state_dict(checkpoint_dict['dense_corr'])
                self.dense_corr_net.cuda()

    def _save_checkpoint_dict(self, checkpoint_dict: dict):
        # save the instance when save_check_point was activated in training loop
        super(DenseCorrTrainBox, self)._save_checkpoint_dict(checkpoint_dict)
        checkpoint_dict['dense_corr'] = self.dense_corr_net.state_dict()

    """ Train Routines -------------------------------------------------------------------------------------------------
        """

    def _prepare_train(self):
        self.dense_corr_net.train()
        self.dense_corr_net.feat_model.eval()

    def _train_feed(self, train_sample, cur_train_epoch, cur_train_itr, eval_flag=False) -> dict():

        super(DenseCorrTrainBox, self)._train_feed(train_sample, cur_train_epoch, cur_train_itr)

        _, _, _, img_lists, _, _, _, out_graph, _, _, _, e_node_idx, edge_label, _, _, e_rel_Rt, _, _, _ = train_sample

        # Preprocess images
        img_lists2 = torch.ones(len(img_lists), 3, self.rsz[0], self.rsz[1])
        for i in range(len(img_lists)):
            if img_lists[i][0].size(1) > img_lists[i][0].size(2):
                # img_lists2[i] = self.preprocess(self.rotimg(self.toPIL(img_lists[i][0])))
                img_lists2[i] = F.interpolate(torch.flip(img_lists[i][0].permute(0, 2, 1), [1]).unsqueeze(0), self.rsz)[0]
            else:
                # img_lists2[i] = self.preprocess(self.toPIL(img_lists[i][0]))
                img_lists2[i] = F.interpolate(img_lists[i], self.rsz)[0]

        # img_lists2 = img_lists2.to(cur_dev)
        # edge_label = edge_label.to(cur_dev)
        # Preprocess edge-node indexing, divide into batches
        e_n_idx = np.array(e_node_idx)

        n = ceil(len(e_n_idx) / self.bs_enc)

        N = out_graph.shape[1]
        E = edge_label.shape[1]

        n_feats = torch.zeros(size=[N, 256]).cuda().to(1)
        e_feats = torch.zeros(size=[E, 256]).cuda().to(1)

        for i in range(n):
            end = i*self.bs_enc+self.bs_enc if i*self.bs_enc+self.bs_enc <= len(e_n_idx) else len(e_n_idx)
            sel_idx = e_n_idx[i*self.bs_enc:end].transpose()
            r, c = sel_idx[0], sel_idx[1]
            sel = r.tolist()+c.tolist()
            imgs = img_lists2[sel]
            _, e_enc, n_enc = self.dense_corr_net(imgs)
            n_enc_mv = n_enc.to(1)
            e_enc_mv = e_enc.to(1)
            for idx in range(len(sel)):
                n_feats[sel[idx]] = n_enc_mv[idx]
            e_feats[i*self.bs_enc:end] = e_enc_mv
            # print("done", i)

        # build bi-direct
        bi_e_node_idx, bi_e_rel_Rt = graph_util.bi_direct_edge(e_node_idx, e_rel_Rt)
        bi_e_rel_q = rot2quaternion(bi_e_rel_Rt).to(1).detach()
        bi_e_feat = torch.cat([e_feats, e_feats], dim=0)
        bi_e_node_idx = bi_e_node_idx.permute(1, 0).long().detach().to(1)
        bi_e_label = torch.cat([edge_label, edge_label], dim=0).to(1).float()
        input_edge_feat = torch.cat([bi_e_feat.view(2 * E, -1), bi_e_rel_q.view(2 * E, -1)], dim=-1)

        ori_w, _, _ = self.cleaner.forward(n_feats.view(N, -1), input_edge_feat.view(2 * E, -1), bi_e_node_idx)

        # loss_w =

        self.optimizer.step()
        print("loss", np.mean(losses))
        return {'Loss(Train)/batch_loss': np.mean(losses)}

    """ Validation Routines --------------------------------------------------------------------------------------------
        """

    def _prepare_eval(self):
        self.dense_corr_net.eval()

    def _valid_loop(self, valid_loader, cur_train_epoch, cur_train_itr):
        losses = []

        for valid_batch_idx, valid_sample in enumerate(valid_loader):
            _, _, _, img_lists, _, _, _, _, _, _, _, e_node_idx, edge_label, _, _, _, _, _, _ = valid_sample

            bs = 8
            ineach = bs // 2

            e_l_np = np.array(edge_label).reshape(edge_label.size(1))
            inliers = np.argwhere(e_l_np == 1).reshape(-1)
            outliers = np.argwhere(e_l_np == 0).reshape(-1)
            numin = len(inliers)
            numout = len(outliers)
            torun = min(numin - ineach, numout)
            if torun > 0:
                anchor_idxs = []
                for i in range(ineach):
                    r = randint(0, len(inliers) - 1)
                    while r in anchor_idxs:
                        r = randint(0, len(inliers) - 1)
                    anchor_idxs.append(r)

                # Preprocess images
                img_lists2 = torch.ones(len(img_lists), 3, self.rsz[0], self.rsz[1])
                for i in range(len(img_lists)):
                    if img_lists[i][0].size(1) > img_lists[i][0].size(2):
                        # img_lists2[i] = self.preprocess(self.rotimg(self.toPIL(img_lists[i][0])))
                        img_lists2[i] = F.interpolate(torch.flip(img_lists[i][0].permute(0,2,1), [1]).unsqueeze(0), self.rsz)[0]
                    else:
                        # img_lists2[i] = self.preprocess(self.toPIL(img_lists[i][0]))
                        img_lists2[i] = F.interpolate(img_lists[i], self.rsz)[0]

                # img_lists2 = img_lists2.to(cur_dev)
                # edge_label = edge_label.to(cur_dev)
                # Preprocess edge-node indexing, divide into batches
                e_n_idx = np.array(e_node_idx).transpose()
                ra, ca = e_n_idx[0, anchor_idxs], e_n_idx[1, anchor_idxs]

                e_n_idx = np.delete(e_n_idx, inliers[anchor_idxs], axis=1)
                e_l_np = np.delete(e_l_np, inliers[anchor_idxs])

                inliers = np.argwhere(e_l_np == 1).reshape(-1)
                outliers = np.argwhere(e_l_np == 0).reshape(-1)

                extra = ineach
                n = ceil(torun / ineach)
                for i in range(n):
                    if i == n - 1 and not torun % ineach == 0:
                        extra = torun % ineach
                        ra = ra[:extra]
                        ca = ca[:extra]
                    # elif i == n - 1 and torun % ineach == 0:
                    #     break
                    ri, ro = e_n_idx[0, inliers[i * ineach:i * ineach + extra]], e_n_idx[
                        0, outliers[i * ineach:i * ineach + extra]]
                    ci, co = e_n_idx[1, inliers[i * ineach:i * ineach + extra]], e_n_idx[
                        1, outliers[i * ineach:i * ineach + extra]]
                    sel = ra.tolist() + ri.tolist() + ro.tolist() + ca.tolist() + ci.tolist() + co.tolist()
                    imgs = img_lists2[sel]
                    _, sim = self.dense_corr_net(imgs)
                    anchor = sim[0:extra].reshape(extra, -1)
                    positive = sim[extra:2 * extra].reshape(extra, -1)
                    negative = sim[2 * extra:3 * extra].reshape(extra, -1)
                    loss = self.criterion(anchor, positive, negative)
                    losses.append(loss.item())
        print("loss", np.mean(losses))
        return {'Loss(Valid)/batch_loss': np.mean(losses)}