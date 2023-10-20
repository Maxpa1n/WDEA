import torch.nn as nn
import torch
from WDEA.KGConv import WDGCNKGlayer
from dgl import DGLGraph
import numpy as np


class WDGCNKGEncoder(nn.Module):
    def __init__(self,
                 num_layers: int,
                 in_ent_dim: int,
                 in_rel_dim: int,
                 topk: int,
                 num_heads: int,
                 alpha: float,
                 hidden_dim: int,
                 hop_num: int,
                 input_drop: float,
                 feat_drop: float,
                 attn_drop: float,
                 topk_type: str,
                 edge_drop: float,
                 negative_slope: float):
                 # ,ntriples: int):
                 
        super(WDGCNKGEncoder, self).__init__()
        self.num_layers = num_layers
        self.trans_layers = nn.ModuleList()
        self.hop_num = hop_num
        self.top_k = topk
        self.alpha = alpha
        self.edge_drop = edge_drop
        # self.ntriples = ntriples
        self.feat_drop_out = nn.Dropout(feat_drop)

        self.trans_layers.append(WDGCNKGlayer(in_ent_feats=in_ent_dim,\
                                            num_heads=num_heads,\
                                            in_rel_feats=in_rel_dim,\
                                            out_feats=hidden_dim,\
                                            feat_drop=feat_drop,\
                                            hop_num=self.hop_num,\
                                            top_k=self.top_k,\
                                            alpha=self.alpha,\
                                            attn_drop=attn_drop,\
                                            topk_type=topk_type,\
                                            negative_slope=negative_slope,\
                                            input_drop=input_drop))
        for l in range(1, num_layers):
            self.trans_layers.append(WDGCNKGlayer(in_ent_feats=hidden_dim,  num_heads=num_heads, in_rel_feats=in_rel_dim, out_feats=hidden_dim, hop_num=self.hop_num,
                                                       top_k=self.top_k, alpha=self.alpha,
                                                       feat_drop=feat_drop, topk_type=topk_type, attn_drop=attn_drop, negative_slope=negative_slope, input_drop=input_drop))
        self.fc_rel = nn.Linear(in_rel_dim, hidden_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if isinstance(self.fc_rel, nn.Linear):
            nn.init.xavier_normal_(self.fc_rel.weight.data, gain=1.414)

    def forward(self, graph: DGLGraph, entity_embedder, rel_embedder, mask_edge_ids=None):
        h = entity_embedder
        h_r = rel_embedder
        # drop_edges_ids = self.get_drop_edge_pair_ids(number_triples=number_triples)
        drop_edges_ids = None
        try:
            drop_edges_ids = drop_edges_ids.type(torch.int64)
        except:
            pass
        for l in range(self.num_layers):
            h = self.trans_layers[l](graph, h, h_r, mask_edge_ids, drop_edges_ids)
        h_r = self.fc_rel(h_r)
        return h,h_r

    def get_drop_edge_pair_ids(self, number_triples):
        drop_edge_num = int(number_triples * self.edge_drop)
        if self.training:
            if drop_edge_num > 0:
                drop_edges_ids = np.random.choice(number_triples, drop_edge_num, replace=False)
                inv_drop_edges_ids = drop_edges_ids + number_triples
                drop_edges_ids = np.concatenate([drop_edges_ids, inv_drop_edges_ids])
                drop_edges_ids = torch.from_numpy(drop_edges_ids)
            else:
                drop_edges_ids = None
        else:
            drop_edges_ids = None
        return drop_edges_ids