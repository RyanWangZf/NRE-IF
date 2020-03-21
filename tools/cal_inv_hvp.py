# -*- coding: utf-8 -*-
import numpy as np
import torch
import os
import pdb
import torch.nn as nn
from torch.utils.data import DataLoader
from config import opt
import models
import dataset
import torch.nn.functional as F
from torch.autograd import grad

def get_model_param_dict(model):
    params = {}
    for name,param in model.named_parameters():
        params[name] = param

    return params

def collate_fn(batch):
    data, label = zip(*batch)
    return data, label

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_s_test(s_test):
    s_test_dict = {kk:vv.detach().cpu().numpy() for kk,vv in enumerate(s_test)}
    np.save("s_test", s_test_dict)

def go(**kwargs):

    setup_seed(opt.seed)

    # kwargs.update({'model': 'PCNN_ONE'})
    kwargs.update({'model': 'PCNN_IF'})

    kwargs.update({"load_model_path":"./checkpoints/PCNN_IF_DEF.pth"})
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
    val_data_loader = DataLoader(val_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=collate_fn)

    print('train data: {}'.format(len(train_data)))

    params = get_model_param_dict(model)
    theta = [params["linear.weight"], params["linear.bias"]]
    
    # load model params
    model.load(opt.load_model_path, opt.use_gpu)
    
    s_test = cal_inverse_hvp_lissa(model, val_data_loader, train_data_loader, theta, tol=1e-3, verbose=True)

    save_s_test(s_test)
    print("Done.")


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
    fp_loss_avg = torch.mean(fp_loss_list)

    grads_val = list(grad(fp_loss_avg, theta, create_graph=True))
    
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

if __name__ == '__main__':
    import fire
    fire.Fire()
