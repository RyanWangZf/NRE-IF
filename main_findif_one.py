# -*- coding: utf-8 -*-

from config import opt
import models
import dataset
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from utils import save_pr, now, eval_metric
from torch.autograd import grad

import pdb

def collate_fn(batch):
    data, label = zip(*batch)
    return data, label

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(**kwargs):

    setup_seed(opt.seed)

    # kwargs.update({'model': 'PCNN_ONE'})
    kwargs.update({'model': 'PCNN_IF'})    
    opt.parse(kwargs)

    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    # torch.manual_seed(opt.seed)
    model = getattr(models, 'PCNN_IF')(opt)
    if opt.use_gpu:
        # torch.cuda.manual_seed_all(opt.seed)
        model.cuda()
        # parallel
        #  model = nn.DataParallel(model)

    # loading data
    DataModel = getattr(dataset, opt.data + 'Data')
    train_data = DataModel(opt.data_root, "train")
    train_data_loader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=collate_fn)

    val_data = DataModel(opt.data_root, "val")
    val_data_loader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers, collate_fn=collate_fn)

    test_data = DataModel(opt.data_root, "test")
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, collate_fn=collate_fn)
    print('train data: {}; test data: {}'.format(len(train_data), len(test_data)))

    params = get_model_param_dict(model)
    theta = [params["linear.weight"], params["linear.bias"]]

    print("Start Training.")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), rho=0.95, eps=1e-6, weight_decay=opt.weight_decay)

    max_pre = -1.0
    max_rec = -1.0
    for epoch in range(opt.num_epochs):
        total_loss = 0
        for idx, (data, label_set) in enumerate(train_data_loader):
            label = [l[0] for l in label_set]

            if opt.use_gpu:
                label = torch.LongTensor(label).cuda()
            else:
                label = torch.LongTensor(label)

            if epoch == 0:
                # first epoch, we use PCNN-ONE strategy
                sub_data = select_instance_ONE(model, data, label)
            else:
                sub_data, sub_label = select_influential_instance(model, data, label, s_test)

            model.batch_size = opt.batch_size

            optimizer.zero_grad()
            if epoch == 0:
                out = model(sub_data, train=True)
                loss = criterion(out, label)
            else:
                loss = model.loss(sub_data, sub_label)
                
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # evaluate
        true_y, pred_y, pred_p = predict(model, test_data_loader)
        all_pre, all_rec, fp_res = eval_metric(true_y, pred_y, pred_p)

        last_pre, last_rec = all_pre[-1], all_rec[-1]
        if last_pre > 0.1 and last_rec > 0.1:
            save_pr(opt.result_dir, model.model_name, epoch, all_pre, all_rec, fp_res, opt=opt.print_opt)
            print('{} Epoch {} save pr'.format(now(), epoch + 1))
            if last_pre > max_pre and last_rec > max_rec:
                print("save model")
                max_pre = last_pre
                max_rec = last_rec
                model.save(opt.print_opt)

        print('{} Epoch {}/{}: train loss: {}; test precision: {}, test recall {}'.format(now(), epoch + 1, opt.num_epochs, total_loss, last_pre, last_rec))

        # in each epoch, update s_test
        s_test = cal_inverse_hvp_lissa(model, val_data_loader, train_data_loader, theta, tol=1e-3, verbose=True)



def predict(model, test_data_loader):

    model.eval()

    pred_y = []
    true_y = []
    pred_p = []
    for idx, (data, labels) in enumerate(test_data_loader):
        true_y.extend(labels)

        for bag in data:
            insNum = bag[1]
            model.batch_size = insNum
            if opt.use_gpu:
                data = map(lambda x: torch.LongTensor(x).cuda(), bag)
            else:
                data = map(lambda x: torch.LongTensor(x), bag)

            out = model(data)
            out = F.softmax(out, 1)
            max_ins_prob, max_ins_label = map(lambda x: x.data.cpu().numpy(), torch.max(out, 1))
            tmp_prob = -1.0
            tmp_NA_prob = -1.0
            pred_label = 0
            pos_flag = False

            for i in range(insNum):
                if pos_flag and max_ins_label[i] < 1:
                    continue
                else:
                    if max_ins_label[i] > 0:
                        pos_flag = True
                        if max_ins_prob[i] > tmp_prob:
                            pred_label = max_ins_label[i]
                            tmp_prob = max_ins_prob[i]
                    else:
                        if max_ins_prob[i] > tmp_NA_prob:
                            tmp_NA_prob = max_ins_prob[i]

            if pos_flag:
                pred_p.append(tmp_prob)
            else:
                pred_p.append(tmp_NA_prob)

            pred_y.append(pred_label)

    size = len(test_data_loader.dataset)
    assert len(pred_y) == size and len(true_y) == size

    model.train()
    return true_y, pred_y, pred_p

def select_influential_instance(model, batch_data, labels, s_test):
    model.eval()
    s_test = [s_.view(-1,1) for s_ in s_test]

    sub_bag_list = []
    sub_label_list = []

    for idx, bag in enumerate(batch_data):
        select_ent = []
        select_num = []
        select_sen = []
        select_pf = []
        select_pool = []
        select_mask = []

        insNum = bag[1]
        label = labels[idx]
        selected_ins_id = [0]

        if insNum > 1:
            model.batch_size = insNum
            if model.opt.use_gpu:
                data = map(lambda x: torch.LongTensor(x).cuda(), bag)
            else:
                data = map(lambda x: torch.LongTensor(x), bag)

            batch_label = label.repeat(insNum)

            if model.opt.use_gpu:
                batch_label = batch_label.cuda()

            phi = cal_influence_function(s_test, model, data, batch_label)

            # select the minimum phi sample
            selected_ins_id = [np.argmin(phi)]


        # collect all together
        for j in selected_ins_id:
            # entity word's id
            select_ent = bag[0]
            # instance number in this bag
            select_num = len(selected_ins_id)
            # sentence
            select_sen.append(bag[2][j])
            # position feature
            select_pf.append(bag[3][j])
            # pool
            select_pool.append(bag[4][j])
            # piece-wise mask
            select_mask.append(bag[5][j])

        sub_label_list.append(label.repeat(select_num))

        sub_bag_list.append([select_ent, select_num, select_sen, select_pf, select_pool, select_mask])

    data = sub_bag_list

    sub_labels = torch.cat(sub_label_list)

    if model.opt.use_gpu:
        sub_labels = sub_labels.cuda()

    # if model.opt.use_gpu:
    #     data = map(lambda x: torch.LongTensor(x).cuda(), [select_ent, select_num, select_sen, select_pf, select_pool, select_mask])
    # else:
    #     data = map(lambda x: torch.LongTensor(x), [select_ent, select_num, select_sen, select_pf, select_pool, select_mask])

    model.train()
    return data, sub_labels

def cal_sampling_prob(phi, sigmoid_k=10):
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
        
def get_model_param_dict(model):
    params = {}
    for name,param in model.named_parameters():
        params[name] = param

    return params

def hvp(y, w, v):
    """Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.

    Arguments:
        y: scalar/tensor, for example the output of the loss function
        w: list of torch tensors, tensors over which the Hessian
            should be constructed
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian

    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.

    Raises:
        ValueError: `y` and `w` have a different length."""
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem.detach())

    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True)

    return return_grads

def cal_inverse_hvp_lissa(model, 
    val_data_loader,
    train_data_loader,
    theta,
    damp=0.01,
    scale=25.0,
    tol=1e-3,
    recursion_depth=1000,
    verbose=False,
    ):

    def _compute_diff(h0, h1):
        assert len(h0) == len(h1)
        diff_ratio = [1e8] * len(h0)
        for i in range(len(h0)):
            h0_ = h0[i].detach().cpu().numpy()
            h1_ = h1[i].detach().cpu().numpy()
            norm_0 = np.linalg.norm(h0_) 
            norm_1 = np.linalg.norm(h1_)
            abs_diff = abs(norm_0 - norm_1)
            diff_ratio[i] = abs_diff / norm_0

        return max(diff_ratio)

    model.eval()

    # get grad theta on val data
    fp_loss_list = []
    for idx,(data, label_set) in enumerate(val_data_loader):
        # label_set: 0: NA relation, 1: has relation
        labels = torch.LongTensor(np.concatenate(label_set,0))
        pred = model.inference(data)

        pred_prob = F.softmax(pred, 1)
        _, max_ins_label = map(lambda x: x.data.cpu().numpy(), torch.max(pred_prob, 1))
        pred_label = torch.zeros(len(labels))
        pred_label[max_ins_label != 0] = 1
        mask = pred_label.long() ^ labels
        mask = mask.float()

        if model.opt.use_gpu:
            mask = mask.cuda()

        pred_max_ins_prob = torch.max(pred_prob, 1)[0]
        loss = - mask * torch.log(1 - pred_max_ins_prob)

        fp_loss_list.append(loss)

    fp_loss_list = torch.cat(fp_loss_list)

    # fp_loss_avg = torch.mean(fp_loss_list)
    fp_loss_sum = torch.sum(fp_loss_list)

    grads_val = list(grad(fp_loss_sum, theta, create_graph=True))
    grads_val = [g.detach() for g in grads_val]

    # start recurssively update the estimate
    h_estimate = grads_val.copy()
    xent_loss_func = nn.CrossEntropyLoss()

    for i in range(recursion_depth):
        h_estimate_last = h_estimate
        # randomly select a batch from train data
        for data, label_set in train_data_loader:
            label = []
            for j in range(len(data)):
                insNum = data[j][1]
                label.append([label_set[j][0]]*insNum)

            label = torch.LongTensor(np.concatenate(label))
            if model.opt.use_gpu:
                label = torch.LongTensor(label).cuda()
            else:
                label = torch.LongTensor(label)

            pred = model.inference(data)
            loss = xent_loss_func(pred, label)

            hv = hvp(loss, theta, h_estimate)
            h_estimate = [ _v + (1 - damp) * _h_e - _hv.detach() / scale for _v, _h_e, _hv in zip(grads_val, h_estimate, hv)]
            diff_ratio = _compute_diff(h_estimate, h_estimate_last)
            break

        if i % 10 == 0:
            if verbose:
                print("[LISSA]: epoch: {}, diff: {:.4f}".format(i, diff_ratio))

        # average diff to stop iteration
        if diff_ratio <= tol:
            print("[LISSA]: Reach tolerance in epoch {}.".format(int(i)))
            break

    return h_estimate

def one_hot_transform(y, num_class=10):
    one_hot_y = nn.functional.one_hot(y, num_classes=num_class)
    return one_hot_y.float()

def cal_influence_function(s_test, model, x, y):
    # do forward and get num class
    model.eval()
    pred, x_h = model(x, hidden=True)
    pred = nn.functional.softmax(pred, 1)
    num_class = pred.shape[1]
    y_oh = one_hot_transform(y, num_class)

    # get grad theta
    diff_pred = pred - y_oh
    x_h = torch.unsqueeze(x_h, 1)
    partial_J_theta = x_h * torch.unsqueeze(diff_pred, 2)
    partial_J_theta = partial_J_theta.view(-1, partial_J_theta.shape[1] * partial_J_theta.shape[2]).detach()

    # get grad bias
    if model.opt.use_gpu:
        partial_J_b = torch.mm(diff_pred, torch.eye(num_class).cuda())
    else:
        partial_J_b = torch.mm(diff_pred, torch.eye(num_class))

    # get the IF
    predicted_loss_diff = -torch.mm(partial_J_theta, s_test[0]) \
     -torch.mm(partial_J_b, s_test[1])

    predicted_loss_diff = predicted_loss_diff.view(-1).detach().cpu().numpy()

    return predicted_loss_diff

def select_instance_ONE(model, batch_data, labels):
    model.eval()
    select_ent = []
    select_num = []
    select_sen = []
    select_pf = []
    select_pool = []
    select_mask = []
    for idx, bag in enumerate(batch_data):
        insNum = bag[1] # num of instances in this bag
        label = labels[idx] # labels are a batch of bags' labels, batch_size is 128
        max_ins_id = 0
        if insNum > 1:
            model.batch_size = insNum
            if opt.use_gpu:
                data = map(lambda x: torch.LongTensor(x).cuda(), bag)
            else:
                data = map(lambda x: torch.LongTensor(x), bag)

            out = model(data) # ?, 27

            #  max_ins_id = torch.max(torch.max(out, 1)[0], 0)[1]
            max_ins_id = torch.max(out[:, label], 0)[1] # select index of the largest instance

            if opt.use_gpu:
                #  max_ins_id = max_ins_id.data.cpu().numpy()[0]
                max_ins_id = max_ins_id.item()
            else:
                max_ins_id = max_ins_id.data.numpy()
                pass  

        max_sen = bag[2][max_ins_id] # sentence word
        max_pf = bag[3][max_ins_id] # position features
        max_pool = bag[4][max_ins_id] # entity's postion in this sentence
        max_mask = bag[5][max_ins_id] # mask for piece pooling in [1,2,3]

        select_ent.append(bag[0]) # entity word's id
        select_num.append(bag[1]) # instance number in this bag
        select_sen.append(max_sen)
        select_pf.append(max_pf)
        select_pool.append(max_pool)
        select_mask.append(max_mask)


    if opt.use_gpu:
        data = map(lambda x: torch.LongTensor(x).cuda(), [select_ent, select_num, select_sen, select_pf, select_pool, select_mask])
    else:
        data = map(lambda x: torch.LongTensor(x), [select_ent, select_num, select_sen, select_pf, select_pool, select_mask])

    model.train()
    return data


if __name__ == '__main__':
    import fire
    fire.Fire()

