# -*- coding: utf-8 -*-
import numpy as np
import torch
import os
import pdb
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

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

def load_ent2id(root_path="./dataset/NYT"):
    ent2id_path = os.path.join(root_path, "ent2id.npy")
    id2ent_path = os.path.join(root_path, "id2ent.npy")
    ent2id = np.load(ent2id_path, allow_pickle=True).item()
    id2ent = np.load(id2ent_path, allow_pickle=True).item()
    return id2ent, ent2id


def load_ent2word(root_path="./dataset/NYT"):
    filename = os.path.join(root_path,"ent2word.npy")
    ent2word = np.load(filename, allow_pickle=True).item()
    return ent2word

def load_w2v(root_path="./dataset/NYT"):
    '''
    reading from vec.bin
    add two extra tokens:
        : UNK for unkown tokens
    '''
    wordlist = []

    w2v_path = os.path.join(root_path,"vector.txt")

    f = open(w2v_path)
    # dim = int(f.readline().split()[1])
    # f = f.readlines()

    vecs = []
    for line in f:
        line = line.strip('\n').split()
        vec = list(map(float, line[1].split(',')[:-1]))
        vecs.append(vec)
        wordlist.append(line[0])

    #  wordlist.append('UNK')
    #  vecs.append(np.random.uniform(low=-0.5, high=0.5, size=dim))
    word2id = {j: i for i, j in enumerate(wordlist)}
    id2word = {i: j for i, j in enumerate(wordlist)}

    return np.array(vecs, dtype=np.float32), word2id, id2word

def save_s_test(s_test):
    s_test_dict = {kk:vv.detach().cpu().numpy() for kk,vv in enumerate(s_test)}
    np.save("s_test", s_test_dict)

def load_s_test(filename="s_test.npy"):
    stest = np.load(filename,allow_pickle=True).item()
    s_test_list = []
    for k in stest.keys():
        s_test_list.append(stest[k])
    return s_test_list

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def go(**kwargs):
    kwargs.update({'model': 'PCNN_IF'})

    kwargs.update({"load_model_path":"./checkpoints/PCNN_IF_DEF.pth"})

    opt.parse(kwargs)
     
    # load id to word dict
    w2v, word2id, id2word = load_w2v("./dataset/NYT")
    id2ent, ent2id = load_ent2id("./dataset/NYT")
    
    # blank placeholder
    id2word[0] = "\t"
    
    id2rel, rel2id = load_rel("./dataset/NYT")

    selected_rel = opt.rel

    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    # load s_test
    s_test_ar = load_s_test("s_test.npy")
    
    if opt.use_gpu:
        s_test = [torch.Tensor(st).cuda() for st in s_test_ar]
    else:
        s_test = [torch.Tensor(st) for st in s_test_ar]

    # load model
    setup_seed(opt.seed)

    model = getattr(models, 'PCNN_IF')(opt)

    if opt.use_gpu:
        # torch.cuda.manual_seed_all(opt.seed)
        model.cuda()

    # model.load("./checkpoints/PCNN_IF_DEF.pth", opt.use_gpu)
    model.load(opt.load_model_path, opt.use_gpu)

    # start loop in train loader
    DataModel = getattr(dataset, opt.data + 'Data')
    train_data = DataModel(opt.data_root, "train")
    train_data_loader = DataLoader(train_data, opt.batch_size, 
        shuffle=False, 
        num_workers=opt.num_workers, 
        collate_fn=collate_fn)

    print('train data: {}'.format(len(train_data)))

    demo_prefix = "./demo"

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
                
                # compute the influence in this bag
                phi, pred = model.cal_ins_influence(s_test, data, label)

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
                
                df = pd.DataFrame({
                    "ent1": [ent1] * insNum,
                    "ent2": [ent2] * insNum,
                    "rel": [rel] * insNum,
                    "influence":phi,
                    "pred_prob":pred_prob,
                    "pred_label":pred_label_ent,
                    "sentence":sent_list,
                    })

                # sort by influence
                df = df.sort_values(by="influence").reset_index(drop=True)
                df.to_csv(save_name, encoding="utf-8")


if __name__ == '__main__':
    import fire
    fire.Fire()
