import torch
import numpy as np
from config import Config
import random
import dgl
import json

def cosine_similarity_nbyn(a, b):
    '''
    a shape: [num_item_1, embedding_dim]
    b shape: [num_item_2, embedding_dim]
    return sim_matrix: [num_item_1, num_item_2]
    '''
    a = a / torch.clamp(a.norm(dim=-1, keepdim=True, p=2), min=1e-10)
    b = b / torch.clamp(b.norm(dim=-1, keepdim=True, p=2), min=1e-10)
    if b.shape[0] * b.shape[1] > 20000 * 128:
        return cosine_similarity_nbyn_batched(a, b)
    return torch.mm(a, b.t())

def cosine_similarity_nbyn_batched(a, b):
    '''
    a shape: [num_item_1, embedding_dim]
    b shape: [num_item_2, embedding_dim]
    return sim_matrix: [num_item_1, num_item_2]
    '''
    batch_size = 512
    data_num = b.shape[0]
    b = b.t()
    sim_matrix = []
    for i in range(0, data_num, batch_size):
        sim_matrix.append(torch.mm(a, b[:, i:i+batch_size]).cpu())
    sim_matrix = torch.cat(sim_matrix, dim=1)
    return sim_matrix

def negative_sample(pos_ids, data_range, nega_sample_num):
    # Output shape = (data_len, negative_sample_num)
    nega_ids_arrays = np.random.randint(low=0, high=data_range - 1, size=(len(pos_ids), nega_sample_num))
    for idx, pos_id in enumerate(pos_ids):
        for j in range(nega_sample_num):
            if nega_ids_arrays[idx][j] >= pos_id:
                nega_ids_arrays[idx][j] += 1
    assert nega_ids_arrays.shape == (len(pos_ids), nega_sample_num), print(nega_ids_arrays.shape)
    return nega_ids_arrays


class LoadFile():
    def __init__(self,data_set,lang,nega_num,nega_freq,Config):
        self.device = Config.device
        self.path = data_set + '/'
        self.sr_language,self.tg_language = lang

        self.se_sr_weight = self.load_semantic_weight(self.sr_language)
        self.se_tg_weight = self.load_semantic_weight(self.tg_language)

        self.sr_triples,self.sr_rel_num,self.sr_ent_num = self.triples_load(lang[0])
        self.tg_triples,self.tg_rel_num,self.tg_ent_num = self.triples_load(lang[1])

        self.sr_graph = self.construct_graph(self.sr_triples,self.sr_rel_num,self.sr_ent_num)
        self.tg_graph = self.construct_graph(self.tg_triples,self.tg_rel_num,self.tg_ent_num)

        seeds,train_entity_seeds,test_entity_seeds,valid_entity_seeds = self.load_seed()
        rel_seeds,train_rel_seeds,test_rel_seeds, valid_rel_seeds = self.rel_seed_load()
        self.entity_seeds = seeds  # The entity seeds in the original order
        self.rel_seeds = rel_seeds

        # train_ent_seeds shape = [length, 2]
        train_sr_ent_seeds_ori, train_tg_ent_seeds_ori = zip(*train_entity_seeds)
        train_sr_rel_seeds_ori, train_tg_rel_seeds_ori = zip(*train_rel_seeds)

        self.train_sr_ent_seeds_ori = np.asarray(train_sr_ent_seeds_ori)
        self.train_tg_ent_seeds_ori = np.asarray(train_tg_ent_seeds_ori)
        self.train_sr_rel_seeds_ori = np.asarray(train_sr_rel_seeds_ori)
        self.train_tg_rel_seeds_ori = np.asarray(train_tg_rel_seeds_ori)

        # valid_ent_seeds shape = [length]
        valid_sr_ent_seeds, valid_tg_ent_seeds = zip(*valid_entity_seeds)
        valid_sr_rel_seeds, valid_tg_rel_seeds = zip(*valid_rel_seeds)
        self.valid_sr_ent_seeds = np.asarray(valid_sr_ent_seeds)
        self.valid_tg_ent_seeds = np.asarray(valid_tg_ent_seeds)
        self.valid_sr_rel_seeds = np.asarray(valid_sr_rel_seeds)
        self.valid_tg_rel_seeds = np.asarray(valid_tg_rel_seeds)

        test_sr_ent_seeds, test_tg_ent_seeds = zip(*test_entity_seeds)
        test_sr_rel_seeds, test_tg_rel_seeds = zip(*test_rel_seeds)
        self.test_sr_ent_seeds = np.asarray(test_sr_ent_seeds)
        self.test_tg_ent_seeds = np.asarray(test_tg_ent_seeds)
        self.test_sr_rel_seeds = np.asarray(test_sr_rel_seeds)
        self.test_tg_rel_seeds = np.asarray(test_tg_rel_seeds)
        self.device = Config.device
        self.negative_sample()
        self.negative_sample_rel()
        self.to_torch()

    def load_semantic_weight(self,lang):
        with open(self.path+f"{lang}_vectorList.json") as f:
            weight = json.load(f)
        return weight

    def update_negative_sample(self, sr_nega_sample, tg_nega_sample):
        # nega sample shape = (data_len, negative_sample_num)
        assert sr_nega_sample.shape == (len(self.train_sr_ent_seeds_ori), Config.nega_sample_num)
        assert tg_nega_sample.shape == (len(self.train_tg_ent_seeds_ori), Config.nega_sample_num)

        if not (hasattr(self, "sr_posi_sample") and hasattr(self, "tg_posi_sample")):
            sr_posi_sample = np.tile(self.train_sr_ent_seeds_ori.reshape((-1, 1)), (1, Config.nega_sample_num))
            tg_posi_sample = np.tile(self.train_tg_ent_seeds_ori.reshape((-1, 1)), (1, Config.nega_sample_num))
            self.sr_posi_sample = torch.from_numpy(sr_posi_sample.reshape((-1, 1))).to(self.device)
            self.tg_posi_sample = torch.from_numpy(tg_posi_sample.reshape((-1, 1))).to(self.device)

        sr_nega_sample = sr_nega_sample.reshape((-1, 1))
        tg_nega_sample = tg_nega_sample.reshape((-1, 1))
        self.train_sr_ent_seeds = torch.cat((self.sr_posi_sample, sr_nega_sample), dim=1)
        self.train_tg_ent_seeds = torch.cat((self.tg_posi_sample, tg_nega_sample), dim=1)
        self.train_sr_ent_seeds = self.train_sr_ent_seeds.type(torch.long)
        self.train_tg_ent_seeds = self.train_tg_ent_seeds.type(torch.long)

    def to_torch(self):
        self.valid_sr_ent_seeds = torch.from_numpy(self.valid_sr_ent_seeds).to(self.device)
        self.valid_tg_ent_seeds = torch.from_numpy(self.valid_tg_ent_seeds).to(self.device)
        self.valid_sr_rel_seeds = torch.from_numpy(self.valid_sr_rel_seeds).to(self.device)
        self.valid_tg_rel_seeds = torch.from_numpy(self.valid_tg_rel_seeds).to(self.device)

    def negative_sample(self):
        # Randomly negative sample
        sr_nega_sample = negative_sample(self.train_sr_ent_seeds_ori, self.sr_ent_num, Config.nega_sample_num)
        tg_nega_sample = negative_sample(self.train_tg_ent_seeds_ori, self.tg_ent_num, Config.nega_sample_num)
        sr_nega_sample = torch.from_numpy(sr_nega_sample).to(self.device)
        tg_nega_sample = torch.from_numpy(tg_nega_sample).to(self.device)

        self.update_negative_sample(sr_nega_sample, tg_nega_sample)

    def negative_sample_rel(self):
        sr_nega_sample = negative_sample(self.train_sr_rel_seeds_ori, self.sr_rel_num, Config.nega_sample_num)
        tg_nega_sample = negative_sample(self.train_tg_rel_seeds_ori, self.tg_rel_num, Config.nega_sample_num)
        sr_nega_sample = torch.from_numpy(sr_nega_sample).to(self.device)
        tg_nega_sample = torch.from_numpy(tg_nega_sample).to(self.device)

        self.update_negative_sample_rel(sr_nega_sample, tg_nega_sample)

    def update_negative_sample_rel(self, sr_nega_sample, tg_nega_sample):
        # nega sample shape = (data_len, negative_sample_num)
        assert sr_nega_sample.shape == (len(self.train_sr_rel_seeds_ori), Config.nega_sample_num)
        assert tg_nega_sample.shape == (len(self.train_tg_rel_seeds_ori), Config.nega_sample_num)

        if not (hasattr(self, "sr_rel_posi_sample") and hasattr(self, "tg_rel_posi_sample")):
            sr_posi_sample = np.tile(self.train_sr_rel_seeds_ori.reshape((-1, 1)), (1, Config.nega_sample_num))
            tg_posi_sample = np.tile(self.train_tg_rel_seeds_ori.reshape((-1, 1)), (1, Config.nega_sample_num))
            self.sr_rel_posi_sample = torch.from_numpy(sr_posi_sample.reshape((-1, 1))).to(self.device)
            self.tg_rel_posi_sample = torch.from_numpy(tg_posi_sample.reshape((-1, 1))).to(self.device)

        sr_nega_sample = sr_nega_sample.reshape((-1, 1))
        tg_nega_sample = tg_nega_sample.reshape((-1, 1))
        self.train_sr_rel_seeds = torch.cat((self.sr_rel_posi_sample, sr_nega_sample), dim=1)
        self.train_tg_rel_seeds = torch.cat((self.tg_rel_posi_sample, tg_nega_sample), dim=1)
        self.train_sr_rel_seeds = self.train_sr_rel_seeds.type(torch.long)
        self.train_tg_rel_seeds = self.train_tg_rel_seeds.type(torch.long)


    def load_seed(self):
        with open(self.path + 'entity_seeds.txt','r',encoding='utf8') as f:
            seed = [tuple(int(i) for i in line.strip().split('\t')) for line in f]
        seed_num = len(seed)
        random.shuffle(seed)
        train_entities_seeds = seed[:int(seed_num * Config.train_seeds_ratio)]
        test_entities_seeds = seed[int(seed_num * Config.train_seeds_ratio):]
        valid_entities_seeds = test_entities_seeds
        return seed,train_entities_seeds,test_entities_seeds,valid_entities_seeds

    def triples_load(self,lang):
        ma_tmp = []
        with open(f'{self.path}triples_{lang}.txt','r',encoding='utf8') as f:
            # tmp = [tuple(int(i) for i in line.strip().split('\t')) for line in f]
            for line in f:
                line = line.strip().split('\t')
                ma_tmp.append((int(line[0]),int(line[2]),int(line[1])))
        with open(f'{self.path}id2relation_{lang}.txt','r',encoding='utf8') as f:
            tmp_rel = [i for i,_ in enumerate(f)]
            count = len(tmp_rel)
        with open(f'{self.path}entity2id_{lang}.txt','r',encoding='utf8') as f:
            tmp_ent = [i for i,_ in enumerate(f)]
            ent_count = len(tmp_ent)
        return ma_tmp,count,ent_count

    def rel_seed_load(self):
        with open(self.path + 'relation_seeds.txt','r',encoding='utf8') as f:
            seed = [tuple(int(i) for i in line.strip().split('\t')) for line in f]
        seed_num = len(seed)
        random.shuffle(seed)
        train_rel_seeds = seed[:int(seed_num * Config.train_seeds_ratio)]
        test_rel_seeds = seed[int(seed_num * Config.train_seeds_ratio):]
        valid_rel_seeds = test_rel_seeds
        return seed,train_rel_seeds,test_rel_seeds,valid_rel_seeds

    def construct_graph(self,triples,num_rel,num_nodes):
        tmp = []
        triples = np.array(triples,dtype=np.int64).transpose()
        g = dgl.DGLGraph()
        g.add_nodes(num_nodes)
        # +++++++++++++++++++++++++
        src, rel, dst = triples
        inv_rel = rel + num_rel
        src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
        rel = np.concatenate((rel, inv_rel))
        # +++++++++++++++++++++++++
        node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
        g.ndata.update({'n_id': node_id})
        g.add_edges(src, dst)
        g.edata['e_label'] = torch.from_numpy(rel).view(-1, 1)
        g.add_edges(g.nodes(), g.nodes(),
                    {'e_label': torch.ones(g.number_of_nodes(), 1, dtype=torch.long) * 2 * num_rel})
        t,h = g.all_edges('uv')
        g.edata['h'] = h
        g.edata['t'] = t
        return g.to(self.device)
        # return g