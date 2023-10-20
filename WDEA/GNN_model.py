from tty import CFLAG
import torch
import torch.nn as nn
from torch.functional import F
from config import Config
from utils import cosine_similarity_nbyn
from WDEA.KGEncoder import WDGCNKGEncoder
# from GCN import GCN,GAT
import json
from config import Config as cf

def newloaddata(path,is_json=True):
    with open(path,'r',encoding='utf8') as f:
        if is_json:
            return json.load(f)
        else:
            return [tuple(i.strip().split('\t')) for i in f.readlines()]

class GNNChannel(nn.Module):
    def __init__(self,ent_num_sr,ent_num_tg,in_dim,hid_dim,out_dim,sr_garph,tg_graph,sr_rel_num,tg_rel_num):
        super(GNNChannel, self).__init__()
        self.gnn = GCNChannel(ent_num_sr,ent_num_tg,in_dim,hid_dim,out_dim,Config.device,sr_garph,tg_graph,sr_rel_num, tg_rel_num)

    def forward(self,sr_seed,tg_seed,sr_rel_seed,tg_rel_seed):
        sr_seed_hid,\
        tg_seed_hid,\
        sr_rel_hid,\
        tg_rel_hid,\
        sr_ent_hid,\
        tg_ent_hid = self.gnn.forward(sr_seed,tg_seed,sr_rel_seed,tg_rel_seed)
        
        return sr_seed_hid, tg_seed_hid, sr_rel_hid, tg_rel_hid, sr_ent_hid, tg_ent_hid

    def predict(self, sr_ent_seeds, tg_ent_seeds,sr_rel_seeds,tg_rel_seeds):  # 计算相似度
        with torch.no_grad():
            sr_seed_hid, tg_seed_hid, _, _,_,_ = self.forward(sr_ent_seeds, tg_ent_seeds,sr_rel_seeds,tg_rel_seeds)
            sim = - cosine_similarity_nbyn(sr_seed_hid, tg_seed_hid)
        return sim

    def negative_sample(self, sr_ent_seeds, tg_ent_seeds,sr_rel_seeds,tg_rel_seeds):
        with torch.no_grad():
            sr_seed_hid, tg_seed_hid,_,_, sr_ent_hid, tg_ent_hid = self.forward(sr_ent_seeds, tg_ent_seeds,sr_rel_seeds,tg_rel_seeds)
            sim_sr = - cosine_similarity_nbyn(sr_seed_hid, sr_ent_hid)
            sim_tg = - cosine_similarity_nbyn(tg_seed_hid, tg_ent_hid)
        return sim_sr.to(cf.device), sim_tg.to(cf.device)

class GCNChannel(nn.Module):
    def __init__(self,ent_num_sr,ent_num_tg,in_dim,hid_dim,out_dim,device,sr_graph,tg_graph,sr_rel_num,tg_rel_num):
        super(GCNChannel, self).__init__()
        embedding_weight = torch.zeros((ent_num_sr + ent_num_tg, in_dim), dtype=torch.float, device=device)
        nn.init.xavier_uniform_(embedding_weight)
        relation_weight = torch.zeros((sr_rel_num * 2 + tg_rel_num * 2 + 2, in_dim), dtype=torch.float, device=device)
        nn.init.xavier_uniform_(relation_weight)
        # self.trans_line = nn.Linear(128 + 256, 256).to(device)
        self.feats_sr = nn.Parameter(embedding_weight[:ent_num_sr], requires_grad=True)
        self.feats_tg = nn.Parameter(embedding_weight[ent_num_sr:], requires_grad=True)
        self.feats_sr_rel = nn.Parameter(embedding_weight[:sr_rel_num * 2 + 1], requires_grad=True)
        self.feats_tg_rel = nn.Parameter(embedding_weight[tg_rel_num * 2 + 1:], requires_grad=True)
        # self.rel_get()
        self.linear = nn.Linear(in_dim,in_dim).to(Config.device)
        nn.init.xavier_normal_(tensor=self.linear.weight.data, gain=1.414)

        self.gcn = WDGCNKGEncoder(num_layers=2,
                    in_ent_dim=Config.in_dim,
                    in_rel_dim=Config.in_dim,
                    topk=Config.topk,
                    num_heads=Config.num_head,
                    alpha=Config.alpha,
                    hidden_dim=Config.hid_dim,
                    hop_num=Config.hop_num,
                    input_drop=0,
                    feat_drop=0,
                    attn_drop=0,
                    topk_type='local',
                    edge_drop=0,
                    negative_slope=0.2)
        self.sr_graph = sr_graph
        self.tg_graph = tg_graph


    def forward(self,sr_ent_seeds,tg_ent_seeds,sr_rel_seeds,tg_rel_seeds):
        sr_rel_hids = self.feats_sr_rel
        tg_rel_hids = self.feats_tg_rel

        sr_ent_hids, sr_rel_hids = self.gcn(self.sr_graph, self.feats_sr,self.feats_sr_rel)
        tg_ent_hids, tg_rel_hids = self.gcn(self.tg_graph, self.feats_tg,self.feats_tg_rel)
        sr_ent_hids = F.normalize(sr_ent_hids)
        tg_ent_hids = F.normalize(tg_ent_hids)
        sr_rel_hids = F.normalize(sr_rel_hids)
        tg_rel_hids = F.normalize(tg_rel_hids)
        try:
            if sr_ent_seeds.dtype is torch.int32:
                sr_ent_seeds = sr_ent_seeds.type(torch.long)
                tg_ent_seeds = tg_ent_seeds.type(torch.long)
                sr_rel_seeds = sr_rel_seeds.type(torch.long)
                tg_rel_seeds = tg_rel_seeds.type(torch.long)
        except:
            pass
        sr_ent_seed_hid = sr_ent_hids[sr_ent_seeds]
        tg_ent_seed_hid = tg_ent_hids[tg_ent_seeds]
        sr_rel_seed_hid = sr_rel_hids[sr_rel_seeds]
        tg_rel_seed_hid = tg_rel_hids[tg_rel_seeds]
        return sr_ent_seed_hid, tg_ent_seed_hid, sr_rel_seed_hid, tg_rel_seed_hid,sr_ent_hids,tg_ent_hids

    def rel_get(self):
        sr_dir = f'../bin/T100k/zh_vi/running_temp/zh_rel_feats.pth'
        tg_dir = f'../bin/T100k/zh_vi/running_temp/vi_rel_feats.pth'
        # linear = torch.nn.Linear(768,128).to(self.device)
        print('发现关系特征记录')
        print(f'读取:\n{sr_dir}\n{tg_dir}')
        sr_rel_hids = torch.load(sr_dir)
        tg_rel_hids = torch.load(tg_dir)
        print(f'源KG——关系个数： {sr_rel_hids.shape[0]} | 嵌入维度： {sr_rel_hids.shape[1]}')
        print(f'目标KG——关系个数： {tg_rel_hids.shape[0]} | 嵌入维度： {tg_rel_hids.shape[1]}')
        sr_ent_rel, tg_ent_rel = self.get_relation_count()
        feats_sr = self.cat_rel_and_ent(sr_rel_hids, self.feats_sr, sr_ent_rel)
        feats_tg = self.cat_rel_and_ent(tg_rel_hids, self.feats_tg, tg_ent_rel)
        self.feats_sr = nn.Parameter(feats_sr.detach(), requires_grad=True)
        self.feats_tg = nn.Parameter(feats_tg.detach(), requires_grad=True)

    def cat_rel_and_ent(self,rel,ent_feat,ent):
        out = torch.tensor([],requires_grad=False)
        for i in range(len(ent)):
            tmp_ent_feat = ent_feat[int(i)].detach().reshape((1,-1))
            j = ent[str(i)]
            tmp_rel_feat = rel.index_select(0,torch.LongTensor([int(k) for k in j]).to('cuda'))
            tmp_rel_feat = torch.mean(tmp_rel_feat,dim=0,keepdim=True)
            tmp_ent_feat = torch.cat((tmp_ent_feat, tmp_rel_feat), dim=1) #(1,256)
            tmp_ent_feat = F.normalize(self.trans_line(tmp_ent_feat))
            if out.numel() == 0:
                out = tmp_ent_feat.clone()
            else:
                out = torch.cat((out,tmp_ent_feat),dim=0)
        return out.detach()

    def get_relation_count(self):
        srpath = f'../bin/T100k/zh_vi/triples_zh.txt'
        tgpath = f'../bin/T100k/zh_vi/triples_vi.txt'
        sr_trip = newloaddata(srpath, False)
        tg_trip = newloaddata(tgpath, False)

        def counting_rel(kg):
            tmp = {}
            for i in kg:
                head = i[0]
                tail = i[1]
                rel = i[2]
                if head not in tmp:
                    tmp[head] = set([rel])
                else:
                    tmp[head].add(rel)
                if tail not in tmp:
                    tmp[tail] = set([rel])
                else:
                    tmp[tail].add(rel)
            return tmp

        sr_rel = counting_rel(sr_trip)
        tg_rel = counting_rel(tg_trip)
        return sr_rel, tg_rel



