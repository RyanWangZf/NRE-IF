# -*- coding: utf-8 -*-
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import pdb

from config import opt
import models
import dataset


def collate_fn(batch):
    data, label = zip(*batch)
    return data, label

def load_rel(root_path="./dataset/NYT"):
    filename = os.path.join(root_path,"extract_cpp/relation2id.txt")
    f_in = open(filename, "r")
    rel_list = []
    for line in f_in.readlines():
        rel_list.append(line.split()[0])

    f_in.close()
    rel2id = {j: i for i, j in enumerate(rel_list)}
    id2rel = {i: j for i, j in enumerate(rel_list)}

    return id2rel, rel2id

def load_word2id(root_path="./dataset/NYT"):
    w2id_path = os.path.join(root_path, "word2id.npy")
    id2w_path = os.path.join(root_path, "id2word.npy")
    
    word2id = np.load(w2id_path, allow_pickle=True).item()
    id2word = np.load(id2w_path, allow_pickle=True).item()
    return word2id, id2word

def load_ent2id(root_path="./dataset/NYT"):
    ent2id_path = os.path.join(root_path, "ent2id.npy")
    id2ent_path = os.path.join(root_path, "id2ent.npy")
    ent2id = np.load(ent2id_path, allow_pickle=True).item()
    id2ent = np.load(id2ent_path, allow_pickle=True).item()
    return id2ent, ent2id

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def go(**kwargs):
    # set one or att
    kwargs.update({"model":"PCNN_ATT"})
    kwargs.update({"load_model_path":"./checkpoints/PCNN_ATT_ATT_5.pth"})

    # kwargs.update({"model":"PCNN_ONE"})
    # kwargs.update({"load_model_path":"./checkpoints/PCNN_ONE_ONE_7.pth"})

    opt.parse(kwargs)

    # load id to word dict
    word2id, id2word = load_word2id("./dataset/NYT")
    id2ent, ent2id = load_ent2id("./dataset/NYT")

    # blank placeholder
    id2word[0] = "\t"
    
    id2rel, rel2id = load_rel("./dataset/NYT")

    selected_rel = opt.rel

    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    # load model
    setup_seed(opt.seed)

    model = getattr(models, kwargs["model"])(opt)

    if opt.use_gpu:
        model.cuda()

    model.load(opt.load_model_path, opt.use_gpu)

    # start loop in train loader
    DataModel = getattr(dataset, opt.data + 'Data')
    train_data = DataModel(opt.data_root, "train")
    train_data_loader = DataLoader(train_data, 
        opt.batch_size, 
        shuffle=False, 
        num_workers=opt.num_workers, 
        collate_fn=collate_fn)

    print('train data: {}'.format(len(train_data)))

    demo_prefix = "./demo_{}".format(kwargs["model"])
    if not os.path.exists(demo_prefix):
        os.mkdir(demo_prefix)

    model.eval()

    for idx, (batch_data, label_set) in enumerate(train_data_loader):
        labels = [l[0] for l in label_set]
        if opt.use_gpu:
            labels = torch.LongTensor(labels).cuda()
        else:
            labels = torch.LongTensor(labels)
    
        for i, data in enumerate(batch_data):
            insNum = data[1]
            label = labels[i]

            if label.item() == 0:
                continue

            if label.item() == selected_rel and insNum > 0:
            
                print(data[0])
                # get entity
                ent1, ent2 = [id2ent[x] for x in data[0]]

                # get the relation
                rel = id2rel[label.item()]
                                
                # make prediction
                pred, att_score = model.visualize([data])
                pred = pred.t()
                
                # get the raw sentence
                sent_list = []
                for j in range(insNum):
                    sent = [id2word[se] for se in data[2][j]]
                    sent_list.append(" ".join(sent).strip())

                # save this bag and its influence
                save_name = "{}_{}_{}.csv".format(ent1,ent2,insNum)
                save_name = os.path.join(demo_prefix, save_name)
                
                pred_label = torch.max(pred, 1)[1].cpu().detach().numpy().tolist()
                pred_label_ent = [id2rel[pp] for pp in pred_label]
                pred_prob = torch.max(pred, 1)[0].cpu().detach().numpy().tolist()

                att_score = F.softmax(att_score, 1)
                
                att_target = att_score[:,label].detach().cpu().numpy().tolist()
                
                att_max = torch.max(att_score,1)[0].detach().cpu().numpy().tolist()
                
                att_min = torch.min(att_score,1)[0].detach().cpu().numpy().tolist()
                
                df = pd.DataFrame({
                    "ent1": [ent1] * insNum,
                    "ent2": [ent2] * insNum,
                    "rel": [rel] * insNum,
                    "score":att_target,
                    "max_score":att_max,
                    "min_score":att_min,
                    "pred_prob":[pred_prob[0]]*insNum,
                    "pred_label":[pred_label_ent[0]]*insNum,
                    "sentence":sent_list,
                    })
                
                # sort by att score
                df = df.sort_values(by="score").reset_index(drop=True)
                df.to_csv(save_name, encoding="utf-8")

if __name__ == "__main__":
    import fire
    fire.Fire()
