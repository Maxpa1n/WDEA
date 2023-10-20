class Config():
    data_set = '[DATAPATH]'
    lang = ['[SRCLAN]','[TGTLAN]']
    lr = 0.0005
    l2 = 1e-8
    epoch_num = 100
    drop_out = 0.0
    margin_gamma = 2.0
    train_seeds_ratio = 0.3
    nega_sample_freq = 3
    nega_sample_num = 100
    device =  "cuda:5" 
    in_dim = 512
    hid_dim = 256
    out_dim = 512
    max_steps = 400
    alpha = 0.001
    topk = 10
    num_head = 60
    hop_num = 3
    rel_loss = 0.7


    def __str__(self):
        attrs = []
        for key, value in self.__class__.__dict__.items():
            if not key.startswith("__"):
                attrs.append(f"{key} = {value}")
        return "\n".join(attrs)
