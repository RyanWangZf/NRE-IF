# -*- coding: utf-8 -*-

from .BasicModule import BasicModule
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pdb

class PCNN_IF(BasicModule):
    '''
    Adapted from Zeng 2015 DS PCNN.
    '''
    def __init__(self, opt):
        super(PCNN_IF, self).__init__()

        self.opt = opt

        self.model_name = 'PCNN_IF'

        self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)
        self.pos1_embs = nn.Embedding(self.opt.pos_size, self.opt.pos_dim)
        self.pos2_embs = nn.Embedding(self.opt.pos_size, self.opt.pos_dim)

        feature_dim = self.opt.word_dim + self.opt.pos_dim * 2

        # for more filter size
        self.convs = nn.ModuleList([nn.Conv2d(1, self.opt.filters_num, (k, feature_dim), 
            padding=(int(k / 2), 0)) for k in self.opt.filters])

        all_filter_num = self.opt.filters_num * len(self.opt.filters)

        if self.opt.use_pcnn:
            all_filter_num = all_filter_num * 3
            masks = torch.FloatTensor(([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))
            if self.opt.use_gpu:
                masks = masks.cuda()
            self.mask_embedding = nn.Embedding(4, 3)
            self.mask_embedding.weight.data.copy_(masks)
            self.mask_embedding.weight.requires_grad = False

        self.linear = nn.Linear(all_filter_num, self.opt.rel_num)
        self.dropout = nn.Dropout(self.opt.drop_out)

        self.init_model_weight()
        self.init_word_emb()

    def init_model_weight(self):
        '''
        use xavier to init
        '''
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0.0)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)

    def init_word_emb(self):

        def p_2norm(path):
            v = torch.from_numpy(np.load(path))
            if self.opt.norm_emb:
                v = torch.div(v, v.norm(2, 1).unsqueeze(1))
                v[v != v] = 0.0
            return v

        w2v = p_2norm(self.opt.w2v_path)
        p1_2v = p_2norm(self.opt.p1_2v_path)
        p2_2v = p_2norm(self.opt.p2_2v_path)

        if self.opt.use_gpu:
            self.word_embs.weight.data.copy_(w2v.cuda())
            self.pos1_embs.weight.data.copy_(p1_2v.cuda())
            self.pos2_embs.weight.data.copy_(p2_2v.cuda())
        else:
            self.pos1_embs.weight.data.copy_(p1_2v)
            self.pos2_embs.weight.data.copy_(p2_2v)
            self.word_embs.weight.data.copy_(w2v)

    def mask_piece_pooling(self, x, mask):
        '''
        refer: https://github.com/thunlp/OpenNRE
        A fast piecewise pooling using mask
        '''
        x = x.unsqueeze(-1).permute(0, 2, 1, -1)
        masks = self.mask_embedding(mask).unsqueeze(-2) * 100
        x = masks.float() + x
        if self.opt.use_gpu:
            x = torch.max(x, 1)[0] - torch.FloatTensor([100]).cuda()
        else:
            x = torch.max(x, 1)[0] - torch.FloatTensor([100])
        x = x.view(-1, x.size(1) * x.size(2))
        return x

    def piece_max_pooling(self, x, insPool):
        '''
        old version piecewise
        '''
        split_batch_x = torch.split(x, 1, 0)
        split_pool = torch.split(insPool, 1, 0)
        batch_res = []
        for i in range(len(split_pool)):
            ins = split_batch_x[i].squeeze()  # all_filter_num * max_len
            pool = split_pool[i].squeeze().data    # 2
            seg_1 = ins[:, :pool[0]].max(1)[0].unsqueeze(1)          # all_filter_num * 1
            seg_2 = ins[:, pool[0]: pool[1]].max(1)[0].unsqueeze(1)  # all_filter_num * 1
            seg_3 = ins[:, pool[1]:].max(1)[0].unsqueeze(1)
            piece_max_pool = torch.cat([seg_1, seg_2, seg_3], 1).view(1, -1)    # 1 * 3all_filter_num
            batch_res.append(piece_max_pool)

        out = torch.cat(batch_res, 0)
        assert out.size(1) == 3 * self.opt.filters_num
        return out

    def forward(self, x, train=False, hidden=False):

        insEnt, _, insX, insPFs, insPool, insMasks = x
        insPF1, insPF2 = [i.squeeze(1) for i in torch.split(insPFs, 1, 1)]

        word_emb = self.word_embs(insX) # 3, 82, 50
        pf1_emb = self.pos1_embs(insPF1) # 3, 82, 5
        pf2_emb = self.pos2_embs(insPF2) # 3, 82, 5

        x = torch.cat([word_emb, pf1_emb, pf2_emb], 2)
        x = x.unsqueeze(1) # 3, 1, 82, 60
        x = self.dropout(x)

        x = [conv(x).squeeze(3) for conv in self.convs] # 3,230, 82
        if self.opt.use_pcnn:
            # insMasks 3, 82
            x = [self.mask_piece_pooling(i, insMasks) for i in x] # 2, 690
        else:
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        
        h = torch.cat(x, 1) # 2, 690

        # exctation
        h = h.tanh()

        out = self.dropout(h)
        out = self.linear(out) # 2, 27
        
        if hidden:
            return out, h
        else:
            return out

    def loss(self, x, label, reduction="mean"):

        def _sharpen(x, T):
            temp = x**(1/T)
            return temp / temp.sum(1, keepdim=True)

        batch_embs = self.get_bags_feature(x)
        h = torch.cat(batch_embs, 0)
                
        out = self.dropout(h)
        out = self.linear(h)
        
        out = F.softmax(out, 1)
        num_class = out.shape[1]
        label_one_hot = F.one_hot(label, num_classes=num_class).float()

        loss = -torch.sum(label_one_hot * torch.log(out+1e-10), 1)

        # out = _sharpen(out, 2)
        # loss = torch.mean(label_one_hot * torch.log((label_one_hot+1e-10)/(out+1e-10)), 1)

        if reduction == "mean":
            loss = torch.mean(loss)

        return loss
        
    def inference(self, x, hidden=False):
        """Inputs, a list of a batch bags,
        Outputs prediction for all instances.
        """
        batch_embs = self.get_bags_feature(x)
        h = torch.cat(batch_embs, 0)
        out = self.dropout(h)
        out = self.linear(h)
        if hidden:
            return out, h
        else:
            return out

    def get_bags_feature(self, bags):
        '''
        get all bags embedding in one batch before Attention
        '''
        bags_feature = []
        for bag in bags:
            if self.opt.use_gpu:
                data = map(lambda x: Variable(torch.LongTensor(x).cuda()), bag)
            else:
                data = map(lambda x: Variable(torch.LongTensor(x)), bag)
            
            # get all instances embedding in one bag
            bag_embs = self.get_ins_emb(data)
            bags_feature.append(bag_embs)

        return bags_feature

    def get_ins_emb(self, x):
        '''
        x: all instance in a Bag
        '''
        insEnt, _, insX, insPFs, insPool, mask = x
        insPF1, insPF2 = [i.squeeze(1) for i in torch.split(insPFs, 1, 1)]

        word_emb = self.word_embs(insX)
        pf1_emb = self.pos1_embs(insPF1)
        pf2_emb = self.pos2_embs(insPF2)

        x = torch.cat([word_emb, pf1_emb, pf2_emb], 2)                          # insNum * 1 * maxLen * (word_dim + 2pos_dim)
        x = x.unsqueeze(1)                                                      # insNum * 1 * maxLen * (word_dim + 2pos_dim)
        x = [conv(x).squeeze(3) for conv in self.convs]
        x = [self.mask_piece_pooling(i, mask) for i in x]
        x = torch.cat(x, 1).tanh()
        return x

    def forward_emb(self, x_h):        
        # excitation
        out = x_h.tanh()

        out = self.dropout(out)
        out = self.linear(out)
        return out


    def fit(self, batch_data, labels, s_test):
        self.train()

        s_test = [s_.view(-1,1) for s_ in s_test]

        bag_pred_list = []

        for idx, bag in enumerate(batch_data):
            insNum = bag[1]
            label = labels[idx]
            selected_ins_id = [0]

            if insNum > 1:
                self.batch_size = insNum
                if self.opt.use_gpu:
                    data = map(lambda x: torch.LongTensor(x).cuda(), bag)
                else:
                    data = map(lambda x: torch.LongTensor(x), bag)

                batch_label = label.repeat(insNum)

                if self.opt.use_gpu:
                    batch_label = batch_label.cuda()

                # cal influence function
                pred, x_h = self.forward(data, hidden=True)
                pred = F.softmax(pred, 1)
                num_class = pred.shape[1]
                y_oh = F.one_hot(batch_label, num_classes=num_class).float()

                phi = self._cal_influence(s_test, pred, y_oh, x_h)

                if insNum <= 3:
                    selected_ins_id = [np.argmin(phi)]

                else:
                    prob_pi = self._cal_sampling_prob(phi, sigmoid_k=10)
                    selected_ins_id = sampling(prob_pi, 0.1)

                # use instance max pooling after subsampling
                x_h_sub = x_h[selected_ins_id]
                # instance max pooling
                x_h_max = torch.max(x_h_sub, 0)[0].view(1,-1)

                pred_sub = self.forward_emb(x_h_max)
                bag_pred_list.append(pred_sub)

            else:
                if self.opt.use_gpu:
                    data = map(lambda x: torch.LongTensor(x).cuda(), bag)
                else:
                    data = map(lambda x: torch.LongTensor(x), bag)

                pred = self.forward(data)
                bag_pred_list.append(pred)

        # cat
        batch_pred = F.softmax(torch.cat(bag_pred_list), 1)

        num_class = batch_pred.shape[1]
        label_one_hot = F.one_hot(labels, num_classes=num_class).float()
        loss = torch.sum(-label_one_hot * torch.log(batch_pred+1e-10), 1)
        loss = torch.mean(loss)

        return loss

    def _cal_influence(self, s_test, pred, y_oh, x_h):
        self.eval()

        # get grad theta
        num_class = pred.shape[1]
        diff_pred = pred - y_oh
        x_h = torch.unsqueeze(x_h, 1)
        partial_J_theta = x_h * torch.unsqueeze(diff_pred, 2)
        partial_J_theta = partial_J_theta.view(-1, partial_J_theta.shape[1] * partial_J_theta.shape[2]).detach()

        # get grad bias
        if self.opt.use_gpu:
            partial_J_b = torch.mm(diff_pred, torch.eye(num_class).cuda())
        else:
            partial_J_b = torch.mm(diff_pred, torch.eye(num_class))

        # get the IF
        predicted_loss_diff = -torch.mm(partial_J_theta, s_test[0]) \
         -torch.mm(partial_J_b, s_test[1])

        predicted_loss_diff = predicted_loss_diff.view(-1).detach().cpu().numpy()

        self.train()
        return predicted_loss_diff

    def _cal_sampling_prob(self, phi, sigmoid_k=10):
        # standardize
        # remove outliers
        upper_bound = phi.mean() + phi.std()
        sub_phi = phi[phi < upper_bound]
        phi_std = phi - sub_phi.mean()
        a_param = sigmoid_k / (sub_phi.max() - sub_phi.min())
        prob_pi = 1 / (1 + np.exp(a_param*phi_std))
        prob_pi[phi >= upper_bound] = 0
        return prob_pi
    

def sampling(prob_pi, ratio):
    num_sample = prob_pi.shape[0]
    all_idx = np.arange(num_sample)
    obj_sample_size = int(np.ceil(ratio * num_sample))
    sb_idx = None
    iteration = 0
    while True:
        rand_prob = np.random.rand(num_sample)
        iter_idx = all_idx[rand_prob < prob_pi]
        if sb_idx is None:
            sb_idx = iter_idx
        else:
            new_idx = np.setdiff1d(iter_idx, sb_idx)
            diff_size = obj_sample_size - sb_idx.shape[0]
            if new_idx.shape[0] < diff_size:
                sb_idx = np.union1d(iter_idx, sb_idx)
            else:
                new_idx = np.random.choice(new_idx, diff_size, replace=False)
                sb_idx = np.union1d(sb_idx, new_idx)
        iteration += 1
        if sb_idx.shape[0] >= obj_sample_size:
            sb_idx = np.random.choice(sb_idx,obj_sample_size,replace=False)
            return sb_idx

        if iteration > 100:
            diff_size = obj_sample_size - sb_idx.shape[0]
            leave_idx = np.setdiff1d(all_idx, sb_idx)
            # left samples are sorted by their IF
            # leave_idx = leave_idx[np.argsort(prob_pi[leave_idx])[-diff_size:]]
            leave_idx = np.random.choice(leave_idx,diff_size,replace=False)
            sb_idx = np.union1d(sb_idx, leave_idx)
            return sb_idx
