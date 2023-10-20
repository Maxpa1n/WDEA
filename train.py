import os.path

import torch
import torch.nn as nn
import random
import numpy as np
from config import Config
from torch.utils.tensorboard import SummaryWriter
from utils import LoadFile
from WDEA.GNN_model import GNNChannel

def sort_and_keep_indices(matrix, device):
    batch_size = 512
    data_len = matrix.shape[0]
    sim_matrix = []
    indice_list = []
    for i in range(0, data_len, batch_size):
        batch = matrix[i:i + batch_size]
        batch = torch.from_numpy(batch).to(device)
        sorted_batch, indices = torch.sort(batch, dim=-1)
        sorted_batch = sorted_batch[:, :500].cpu()
        indices = indices[:, :500].cpu()
        sim_matrix.append(sorted_batch)
        indice_list.append(indices)
    sim_matrix = torch.cat(sim_matrix, dim=0).numpy()
    indice_array = torch.cat(indice_list, dim=0).numpy()
    sim = np.concatenate([np.expand_dims(sim_matrix, 0), np.expand_dims(indice_array, 0)], axis=0)
    return sim

def get_hits(sim, top_k=(1, 10,50,100), device='cpu'):
    if isinstance(sim, np.ndarray):
        sim = torch.from_numpy(sim)
    top_lr, mr_lr, mrr_lr = topk(sim, top_k, device=device)
    top_rl, mr_rl, mrr_rl = topk(sim.t(), top_k, device=device)
    # return Hits@10
    return top_lr, top_rl, mr_lr, mr_rl, mrr_lr, mrr_rl


def topk(sim, top_k=(1, 10), device='cpu'):
    # Sim shape = [num_ent, num_ent]
    assert sim.shape[0] == sim.shape[1]
    test_num = sim.shape[0]
    batched = True
    if sim.shape[0] * sim.shape[1] < 20000 * 128:
        batched = False
        sim = sim.to(device)

    def _opti_topk(sim):
        sorted_arg = torch.argsort(sim)
        true_pos = torch.arange(test_num, device=device).reshape((-1, 1))
        locate = sorted_arg - true_pos
        del sorted_arg, true_pos
        locate = torch.nonzero(locate == 0)
        cols = locate[:, 1]  # Cols are ranks
        cols = cols.float()
        top_x = [0.0] * len(top_k)
        for i, k in enumerate(top_k):
            top_x[i] = float(torch.sum(cols < k)) / test_num * 100
        mr = float(torch.sum(cols + 1)) / test_num
        mrr = float(torch.sum(1.0 / (cols + 1))) / test_num * 100
        return top_x, mr, mrr

    def _opti_topk_batched(sim):
        mr = 0.0
        mrr = 0.0
        top_x = [0.0] * len(top_k)
        batch_size = 1024
        for i in range(0, test_num, batch_size):
            batch_sim = sim[i:i + batch_size].to(device)
            sorted_arg = torch.argsort(batch_sim)
            true_pos = torch.arange(
                batch_sim.shape[0]).reshape((-1, 1)).to(device) + i
            locate = sorted_arg - true_pos
            del sorted_arg, true_pos
            locate = torch.nonzero(locate == 0,)
            cols = locate[:, 1]  # Cols are ranks
            cols = cols.float()
            mr += float(torch.sum(cols + 1))
            mrr += float(torch.sum(1.0 / (cols + 1)))
            for i, k in enumerate(top_k):
                top_x[i] += float(torch.sum(cols < k))
        mr = mr / test_num
        mrr = mrr / test_num * 100
        for i in range(len(top_x)):
            top_x[i] = top_x[i] / test_num * 100
        return top_x, mr, mrr

    with torch.no_grad():
        if not batched:
            return _opti_topk(sim)
        return _opti_topk_batched(sim)

def get_nearest_neighbor(sim, nega_sample_num=25):
    # Sim do not have to be a square matrix
    # Let us assume sim is a numpy array
    ranks = torch.argsort(sim, dim=1)
    ranks = ranks[:, 1:nega_sample_num + 1]
    return ranks

class AlignLoss(nn.Module):
    def __init__(self, margin, p=2, reduction='mean'):
        super(AlignLoss, self).__init__()
        self.p = p
        self.criterion = nn.TripletMarginLoss(margin, p=p, reduction=reduction)

    def forward(self, repre_sr, repre_tg,rel_sr,rel_tg):
        '''
        score shape: [batch_size, 2, embedding_dim]
        '''
        # distance = torch.abs(score).sum(dim=-1) * self.re_scale
        sr_true = repre_sr[:, 0, :]
        sr_nega = repre_sr[:, 1, :]
        tg_true = repre_tg[:, 0, :]
        tg_nega = repre_tg[:, 1, :]
        loss_ent = self.criterion(torch.cat((sr_true, tg_true), dim=0), torch.cat((tg_true, sr_true), dim=0),
                                  torch.cat((tg_nega, sr_nega), dim=0))
        sr_rel_true = rel_sr[:, 0, :]
        sr_rel_nega = rel_sr[:, 1, :]
        tg_rel_true = rel_tg[:, 0, :]
        tg_rel_nega = rel_tg[:, 1, :]
        loss_rel = self.criterion(torch.cat((sr_rel_true, tg_rel_true), dim=0),
                                  torch.cat((tg_rel_true, sr_rel_true), dim=0),
                                  torch.cat((tg_rel_nega, sr_rel_nega), dim=0))
        return loss_ent + Config.rel_loss * loss_rel


class Train_Model(object):
    def __init__(self,dataset,lang):
        self.loaddata = LoadFile(dataset,lang,Config.nega_sample_num,Config.nega_sample_freq,Config)
        self.sr_graph = self.loaddata.sr_graph
        self.tg_graph = self.loaddata.tg_graph
        self.sr_rel_num = self.loaddata.sr_rel_num
        self.tg_rel_num = self.loaddata.tg_rel_num

    def train(self,device):
        self.gnnchannel = GNNChannel(self.loaddata.sr_ent_num,self.loaddata.tg_ent_num,Config.in_dim,Config.hid_dim,Config.out_dim,self.sr_graph,self.tg_graph,self.sr_rel_num,self.tg_rel_num)
        self.gnnchannel.to(device)
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.gnnchannel.parameters()),lr=Config.lr,weight_decay=Config.l2,betas=(0.9,0.99))
        criterion = AlignLoss(Config.margin_gamma)
        best_hit_at_1 = 0
        best_epoch_num = 0
        for epoch in range(1,Config.epoch_num + 1):
            self.gnnchannel.train()
            optimizer.zero_grad()
            sr_seed_hid,tg_seed_hid,\
            sr_rel_seed_hid,\
            tg_rel_seed_hid,\
            _,_ = self.gnnchannel.forward(self.loaddata.train_sr_ent_seeds,\
                                        self.loaddata.train_tg_ent_seeds,\
                                        self.loaddata.train_sr_rel_seeds,\
                                        self.loaddata.train_tg_rel_seeds)
            loss = criterion(sr_seed_hid,tg_seed_hid,sr_rel_seed_hid,tg_rel_seed_hid)
            # writer.add_scalar('Loss',loss,epoch)
            loss.backward()
            optimizer.step()
            if epoch % Config.nega_sample_freq == 0:
                self.negative_sample()
            hits1, h10 = self.evaluate(epoch,self.gnnchannel,device=Config.device)
            # print(f'best hits@1 is {best_hit_at_1}, hits@10 is {h10} in epoch {best_epoch_num}.')
            if hits1 > best_hit_at_1:
                best_hit_at_1 = hits1
                best_epoch_num = epoch
                print(f'best hits@1 is {best_hit_at_1}, hits@10 is {h10} in epoch {best_epoch_num}.')

    def evaluate(self, epoch_num, info_gnn, device='cpu'):
        info_gnn.eval()
        sim = info_gnn.predict(self.loaddata.valid_sr_ent_seeds, self.loaddata.valid_tg_ent_seeds, self.loaddata.valid_sr_rel_seeds,self.loaddata.valid_tg_rel_seeds)
        top_lr, top_rl, mr_lr, mr_rl, mrr_lr, mrr_rl = get_hits(sim, device=device)
        hit_at_1 = (top_lr[0] + top_rl[0]) / 2
        hit_at_10 = (top_lr[1] + top_rl[1]) / 2
        mrr = (mrr_lr + mrr_rl) / 2
        # writer.add_scalar('hits@1',hit_at_1,epoch_num)
        # writer.add_scalar('hits@10',hit_at_10,epoch_num)
        # writer.add_scalar('MRR',mrr,epoch_num)
        return hit_at_1,hit_at_10

    def negative_sample(self, ):
        sim_sr, sim_tg = self.gnnchannel.negative_sample(self.loaddata.train_sr_ent_seeds_ori,
                                                          self.loaddata.train_tg_ent_seeds_ori,
                                                         self.loaddata.train_sr_rel_seeds_ori,
                                                         self.loaddata.train_tg_rel_seeds_ori)
        sr_nns = get_nearest_neighbor(sim_sr, Config.nega_sample_num)
        tg_nns = get_nearest_neighbor(sim_tg, Config.nega_sample_num)
        self.loaddata.update_negative_sample(sr_nns, tg_nns)

    def negative_sample_rel(self):
        sim_sr, sim_tg = self.gnnchannel.negative_sample(self.loaddata.train_sr_rel_seeds_ori,
                                                         self.loaddata.train_tg_rel_seeds_ori,
                                                         self.loaddata.train_sr_rel_seeds_ori,
                                                         self.loaddata.train_tg_rel_seeds_ori)
        sr_nns = get_nearest_neighbor(sim_sr, Config.nega_sample_num)
        tg_nns = get_nearest_neighbor(sim_tg, Config.nega_sample_num)
        self.loaddata.update_negative_sample_rel(sr_nns, tg_nns)

    def save_sim_matrix(self, device,log_path):
        # Get the similarity matrix of the current model
        self.log_path = log_path
        self.gnnchannel.eval()
        sim_train = self.gnnchannel.predict(self.loaddata.train_sr_ent_seeds_ori,
                                             self.loaddata.train_tg_ent_seeds_ori,self.loaddata.train_sr_rel_seeds_ori,self.loaddata.train_tg_rel_seeds_ori)
        sim_valid = self.gnnchannel.predict(self.loaddata.valid_sr_ent_seeds,
                                             self.loaddata.valid_tg_ent_seeds,self.loaddata.train_sr_rel_seeds_ori,self.loaddata.train_tg_rel_seeds_ori)
        sim_test = self.gnnchannel.predict(self.loaddata.test_sr_ent_seeds, self.loaddata.test_tg_ent_seeds,self.loaddata.train_sr_rel_seeds_ori,self.loaddata.train_tg_rel_seeds_ori)
        seed = self.loaddata.test_sr_ent_seeds
        seed = np.stack((self.loaddata.test_tg_ent_seeds,seed))
        get_hits(sim_test, device=device)
        sim_train = sim_train.cpu().numpy()
        sim_valid = sim_valid.cpu().numpy()
        sim_test = sim_test.cpu().numpy()
        if not os.path.exists(f'{self.log_path}/log/{Config.data_set}/{Config.lang[0]}_{Config.lang[1]}/'):
            os.mkdir(f'{self.log_path}/log/{Config.data_set}/{Config.lang[0]}_{Config.lang[1]}/')

        def save_sim(sim, comment):
            np.save(str(f'{self.log_path}/log/{Config.data_set}/{Config.lang[0]}_{Config.lang[1]}/%s_sim.npy' % comment), sim)

        save_sim(sim_train, 'train')
        save_sim(sim_valid, 'valid')
        save_sim(sim_test, 'test')
        save_sim(seed,'test_seed')

if __name__ == '__main__':
    seed_value = 199603
    config = Config()
    print(config)
    # writer = SummaryWriter(f"./testlogs/{Config.data_set}/{Config.lang[0]}_{Config.lang[1]}/{Config.model}_lr_{Config.lr}_l2_{Config.l2}_alp_{Config.alpha}_1")
    train_model = Train_Model(Config.data_set,Config.lang)
    train_model.train(device=Config.device)
    # train_model.save_sim_matrix(Config.device)


