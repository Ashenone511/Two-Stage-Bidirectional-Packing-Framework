import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import DefaultConfig
from Data import ItemDataset
from ActorCritic import A2C
from tensorboardX import SummaryWriter
import numpy as np
import time


def train():
    ac = A2C()
    for epoch in range(opt.nof_epoch):
        iterator = tqdm(dataloader, unit='Batch')
        for i_batch, sample_batched in enumerate(iterator):
            iterator.set_description('Batch %i/%i' % (epoch + 1, opt.nof_epoch))
            train_batch = Variable(sample_batched.to(opt.device))

            state_item = train_batch
            state_bin = Variable(torch.zeros([opt.batch_size, min(opt.L, opt.W), min(opt.L, opt.W)], device=opt.device))
            state_packed_item = torch.tensor([], device=opt.device)
            t = 0
            show = torch.tensor([], device=opt.device)
            while True:

                out1, out2 = ac.Actor.forward(state_bin, state_item, state_packed_item)
                if opt.TS:
                    state_bin1, state_item1, state_packed_item1, io_prob1, io1, p_prob1, p1, plot1 = out1
                else:
                    state_bin1, state_item1, state_packed_item1, i_prob1, i1, o_prob1, o1, p_prob1, p1, plot1 = out1
                if opt.BP:
                    if opt.TS:
                        state_bin2, state_item2, state_packed_item2, io_prob2, io2, p_prob2, p2, plot2 = out2
                    else:
                        state_bin2, state_item2, state_packed_item2, i_prob2, i2, o_prob2, o2, p_prob2, p2, plot2 = out2

                    n1, n2 = state_item1.size(1), state_item2.size(1)
                    m1, m2 = state_packed_item1.size(1), state_packed_item2.size(1)
                    value_idx = ac.cal_value12(state_bin1, state_item1, state_bin2, state_item2)
                    plot = torch.where(
                      value_idx.view(-1, 1).expand(-1, 6),
                      plot1,
                      plot2
                      )
                    show = torch.cat((show, plot[0].unsqueeze(0)), dim=0)
                    state_bin = torch.where(
                      value_idx.view(-1, 1, 1).expand(-1, min(opt.L, opt.W), min(opt.L, opt.W)),
                      state_bin1,
                      state_bin2
                      )
                    state_item = torch.where(
                      value_idx.view(-1, 1, 1).expand(-1, n1, 3),
                      state_item1,
                      state_item2
                      )
                    state_packed_item = torch.where(
                      value_idx.view(-1, 1, 1).expand(-1, m1, 3),
                      state_packed_item1,
                      state_packed_item2
                      )
                    if opt.TS:
                        io_prob = torch.where(value_idx, io_prob1, io_prob2)
                        io = torch.where(value_idx, io1, io2)
                    else:
                        i_prob = torch.where(value_idx, i_prob1, i_prob2)
                        i = torch.where(value_idx, i1, i2)
                        o_prob = torch.where(value_idx, o_prob1, o_prob2)
                        o = torch.where(value_idx, o1, o2)

                    p_prob = torch.where(value_idx, p_prob1, p_prob2)
                    p = torch.where(value_idx, p1, p2)

                else:
                    state_bin = state_bin1
                    state_item = state_item1
                    p_prob, p = p_prob1, p1
                    if opt.TS:
                        io_prob, io = io_prob1, io1
                    else:
                        i_prob, i = i_prob1, i1
                        o_prob, o = o_prob1, o1
                    state_packed_item = state_packed_item1

                ac.set_value1(state_bin, state_item)
                ac.store_position(p_prob, p)
                if opt.TS:
                    ac.store_index_orientation(io_prob, io)
                else:
                    ac.store_index(i_prob, i)
                    ac.store_orientation(o_prob, o)
                ac.set_g(state_bin, state_packed_item)

                if opt.BP:
                    state_bin = torch.where(
                      value_idx.view(-1, 1, 1).expand(-1, min(opt.L, opt.W), min(opt.L, opt.W)),
                      state_bin,
                      state_bin.transpose(1, 2)
                      )


                t += 1

                if ac.value.size(0) == opt.ngae or t >= opt.nof_item:
                    ac.set_advantage1(t >= opt.nof_item)
                    ac.learn()

                if t >= opt.nof_item:
                    # print(show)
                    ac.set_utilization(state_bin, state_packed_item)
                    iterator.write(str(ac.util.mean().item()))
                    ac.g = torch.tensor([], device=opt.device)
                    break
        util = evaluate(ac, epoch)
        print(f'Evaluate utilization {util}')
        torch.save(ac.Actor.state_dict(), opt.work_path + f'/actor_{opt.L}_{opt.W}_{opt.nof_item}_{int(opt.TS)}{int(opt.ISAB)}{int(opt.BP)}.pth')
        torch.save(ac.Critic.state_dict(), opt.work_path + f'/critic_{opt.L}_{opt.W}_{opt.nof_item}_{int(opt.TS)}{int(opt.ISAB)}{int(opt.BP)}.pth')


def evaluate(ac, step):
    sample_batched = next(iter(dataloader_eva))
    U = []
    for i in range(opt.batch_size):
        val_batch = Variable(sample_batched[i].unsqueeze(0).repeat(16, 1, 1))
        state_item = val_batch.cuda()
        state_bin = Variable(torch.zeros([16, min(opt.L, opt.W), min(opt.L, opt.W)])).cuda()
        state_packed_item = torch.tensor([]).cuda()

        t = 0
        while True:
            out1, out2 = ac.Actor.forward(state_bin, state_item, state_packed_item)
            if opt.TS:
                state_bin1, state_item1, state_packed_item1, io_prob1, io1, p_prob1, p1, _ = out1
            else:
                state_bin1, state_item1, state_packed_item1, i_prob1, i1, o_prob1, o1, p_prob1, p1, _ = out1

            if opt.BP:
                if opt.TS:
                    state_bin2, state_item2, state_packed_item2, io_prob2, io2, p_prob2, p2, _ = out2
                else:
                    state_bin2, state_item2, state_packed_item2, i_prob2, i2, o_prob2, o2, p_prob2, p2, _ = out2
                n1, n2 = state_item1.size(1), state_item2.size(1)
                m1, m2 = state_packed_item1.size(1), state_packed_item2.size(1)
                value_idx = ac.cal_value12(state_bin1, state_item1, state_bin2, state_item2)
                state_bin = torch.where(
                    value_idx.view(-1, 1, 1).expand(-1, min(opt.L, opt.W), min(opt.L, opt.W)),
                    state_bin1,
                    state_bin2
                )
                state_item = torch.where(
                    value_idx.view(-1, 1, 1).expand(-1, n1, 3),
                    state_item1,
                    state_item2
                )
                state_packed_item = torch.where(
                    value_idx.view(-1, 1, 1).expand(-1, m1, 3),
                    state_packed_item1,
                    state_packed_item2
                )
                state_bin = torch.where(
                    value_idx.view(-1, 1, 1).expand(-1, min(opt.L, opt.W), min(opt.L, opt.W)),
                    state_bin,
                    state_bin.transpose(1, 2)
                )
            else:
                state_bin = state_bin1
                state_item = state_item1
                state_packed_item = state_packed_item1

            t += 1

            if t >= opt.nof_item:
                ac.set_utilization(state_bin, state_packed_item)
                util = ac.util.max().item()
                U.append(util)
                ac.g = torch.tensor([], device=opt.device)
                break
    if opt.use_tensorboard:
        writer.add_scalar(f'utilization/TS{int(opt.TS)}_ISAB{int(opt.ISAB)}_BP{int(opt.BP)}', np.mean(U), step)
    return np.mean(U)


if __name__ == '__main__':
    opt = DefaultConfig()

    dataset = ItemDataset(opt.train_size, opt.nof_item)

    dataloader = DataLoader(dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=0)

    data_eva = ItemDataset(opt.train_size, opt.nof_item)

    dataloader_eva = DataLoader(data_eva,
                                batch_size=opt.batch_size,
                                shuffle=True,
                                num_workers=0)

    if opt.use_tensorboard:
        writer = SummaryWriter(log_dir=opt.work_path + f'/logs_{opt.L}_{opt.W}_{opt.nof_item}/')
    train()

