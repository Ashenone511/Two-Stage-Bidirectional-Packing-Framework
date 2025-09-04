import torch


class DefaultConfig(object):
    gpu = True
    train_size = 512 * 64
    batch_size = 64
    nof_epoch = 500
    actor_lr = 1e-5
    critic_lr = 2e-5
    nof_item = 20
    W = 100
    L = 100
    hidden_size = 128
    inner_hidden_size = 128
    Lambda = 0.95
    gamma = 0.99
    C = 10.
    ngae = 5
    device = torch.device("cuda:0")
    BP = True
    TS = True
    ISAB = True
    work_path = '.'
    inducing_num = 30
    use_tensorboard = True
