from Net import *
import torch
import torch.nn as nn
from config import DefaultConfig
from Util import TransformDecoder
import torch.optim as optim
opt = DefaultConfig()


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.bin_encoder = BinEncoder()
        self.item_encoder = ItemEncoder(opt.hidden_size)
        if opt.TS:
            self.io_decoder = IndexOrientationDecoder(opt.hidden_size)
        else:
            self.i_decoder = IndexDecoder(opt.hidden_size)
            self.o_decoder = OrientationDecoder(opt.hidden_size)
        self.p_decoder = PositionDecoder(opt.hidden_size)
        self.softmax = nn.Softmax(-1)

    def forward(self, state_bin, state_item, state_packed_item):
        """
        :param state_bin: bin state
        :param state_item: unpacked items state
        :param state_packed_item: packed items state
        :return:
            new_state_bin1/2: two new bin states resulting from two directional packing
            new_state_item1/2: two new unpacked item states resulting from two directional packing
            new_state_packed_item1/2: two new packed item states resulting from two directional packing
            io_prob1/2: two index-orientation probabilities
            io1/2: two index-orientation selections
            p_prob1/2: two position probabilities
            p1/2: two position selections
            state_bin_last1/2: two last bin states
            state_item_last: last item state
        """
        if not opt.BP:
            h_i, state_item_allr = self.item_encoder.forward(state_item)
            h_b = self.bin_encoder.forward(state_bin.unsqueeze(1))
            if opt.TS:
                io_prob = self.io_decoder.forward(h_b, h_i)
                io = torch.multinomial(io_prob, 1, replacement=False).squeeze(1)
                i = (io / 6).long()
                o = io % 6
                io_prob = torch.gather(io_prob, 1, io.unsqueeze(1).long()).squeeze()

                h_i_selected, item_selected_r = self.select_item(h_i, state_item_allr, io)
                h_i_leftover = self.gen_h_i_leftover(h_i, i)

                p_prob = self.p_decoder.forward(item_selected_r, h_i_selected, h_i_leftover, h_b)
                p = torch.multinomial(p_prob, 1, replacement=False).squeeze(1)
                p_prob = torch.gather(p_prob, 1, p.unsqueeze(1).long()).squeeze()

                x, y = self.gen_xy(state_bin, item_selected_r, p)

                new_state_bin, new_state_item, new_state_packed_item, z = self.update_state(state_bin, state_item,
                                                                                                state_packed_item, i, o,
                                                                                                x, y,
                                                                                                item_selected_r)

                plot = torch.cat((x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1), item_selected_r), dim=1)
                out = [new_state_bin, new_state_item, new_state_packed_item, io_prob, io, p_prob, p, plot]
                return out, None
            else:
                i_prob = self.i_decoder.forward(h_b, h_i)
                i = torch.multinomial(i_prob, 1, replacement=False).squeeze(1)
                i_prob = torch.gather(i_prob, 1, i.unsqueeze(1).long()).squeeze()
                _, item_selected = self.select_item(h_i, state_item, i)
                h_i_leftover = self.gen_h_i_leftover(h_i, i)
                o_prob = self.o_decoder.forward(h_b, item_selected, h_i_leftover)
                o = torch.multinomial(o_prob, 1, replacement=False).squeeze(1)
                o_prob = torch.gather(o_prob, 1, o.unsqueeze(1).long()).squeeze()
                io = i * 6 + o
                item_selected_r = torch.gather(state_item_allr, 1, io.view(-1, 1, 1).repeat(1, 1, 3)).squeeze(1)
                p_prob = self.p_decoder.forward(item_selected_r, None, h_i_leftover, h_b)
                p = torch.multinomial(p_prob, 1, replacement=False).squeeze(1)
                p_prob = torch.gather(p_prob, 1, p.unsqueeze(1).long()).squeeze()

                x, y = self.gen_xy(state_bin, item_selected_r, p)
                new_state_bin, new_state_item, new_state_packed_item, z = self.update_state(state_bin, state_item,
                                                                                            state_packed_item, i, o,
                                                                                            x, y,
                                                                                            item_selected_r)
                plot = torch.cat((x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1), item_selected_r), dim=1)
                out = [new_state_bin, new_state_item, new_state_packed_item, i_prob, i, o_prob, o, p_prob, p, plot]
                return out, None

        if opt.L == opt.W:
            h_i, state_item_allr = self.item_encoder.forward(state_item)
            h_b1 = self.bin_encoder.forward(state_bin.unsqueeze(1))
            h_b2 = self.bin_encoder.forward(state_bin.transpose(1, 2).unsqueeze(1))
            if opt.TS:
                io_prob1 = self.io_decoder.forward(h_b1, h_i)
                io1 = torch.multinomial(io_prob1, 1, replacement=False).squeeze(1)
                i1 = (io1 / 6).long()
                o1 = io1 % 6
                io_prob1 = torch.gather(io_prob1, 1, io1.unsqueeze(1).long()).squeeze()

                io_prob2 = self.io_decoder.forward(h_b2, h_i)
                io2 = torch.multinomial(io_prob2, 1, replacement=False).squeeze(1)
                i2 = (io2 / 6).long()
                o2 = io2 % 6
                io_prob2 = torch.gather(io_prob2, 1, io2.unsqueeze(1).long()).squeeze()

                h_i_selected1, item_selected_r1 = self.select_item(h_i, state_item_allr, io1)
                h_i_selected2, item_selected_r2 = self.select_item(h_i, state_item_allr, io2)

                h_i_leftover1 = self.gen_h_i_leftover(h_i, i1)
                h_i_leftover2 = self.gen_h_i_leftover(h_i, i2)

                p_prob1 = self.p_decoder.forward(item_selected_r1, h_i_selected1, h_i_leftover1, h_b1)
                p1 = torch.multinomial(p_prob1, 1, replacement=False).squeeze(1)
                p_prob1 = torch.gather(p_prob1, 1, p1.unsqueeze(1).long()).squeeze()
                p_prob2 = self.p_decoder.forward(item_selected_r2, h_i_selected2, h_i_leftover2, h_b2)
                p2 = torch.multinomial(p_prob2, 1, replacement=False).squeeze(1)
                p_prob2 = torch.gather(p_prob2, 1, p2.unsqueeze(1).long()).squeeze()

                x1, y1 = self.gen_xy(state_bin, item_selected_r1, p1)
                x2, y2 = self.gen_xy(state_bin.transpose(1, 2), item_selected_r2, p2)

                new_state_bin1, new_state_item1, new_state_packed_item1, z1 = self.update_state(state_bin, state_item,
                                                                                                 state_packed_item, i1, o1, x1, y1,
                                                                                                 item_selected_r1)
                new_state_bin2, new_state_item2, new_state_packed_item2, z2 = self.update_state(state_bin.transpose(1, 2), state_item,
                                                                                                 state_packed_item, i2, o2, x2, y2,
                                                                                                 item_selected_r2)
                plot1 = torch.cat((x1.unsqueeze(1), y1.unsqueeze(1), z1.unsqueeze(1), item_selected_r1), dim=1)
                plot2 = torch.cat((y2.unsqueeze(1), x2.unsqueeze(1), z2.unsqueeze(1), item_selected_r2[:,[1,0,2]]), dim=1)
                out1 = [new_state_bin1, new_state_item1, new_state_packed_item1, io_prob1, io1, p_prob1, p1, plot1]
                out2 = [new_state_bin2, new_state_item2, new_state_packed_item2, io_prob2, io2, p_prob2, p2, plot2]
                return out1, out2

            else:
                i_prob1 = self.i_decoder.forward(h_b1, h_i)
                i1 = torch.multinomial(i_prob1, 1, replacement=False).squeeze(1)
                i_prob1 = torch.gather(i_prob1, 1, i1.unsqueeze(1).long()).squeeze()
                _, item_selected1 = self.select_item(h_i, state_item, i1)
                h_i_leftover1 = self.gen_h_i_leftover(h_i, i1)
                o_prob1 = self.o_decoder.forward(h_b1, item_selected1, h_i_leftover1)
                o1 = torch.multinomial(o_prob1, 1, replacement=False).squeeze(1)
                o_prob1 = torch.gather(o_prob1, 1, o1.unsqueeze(1).long()).squeeze()
                io1 = i1 * 6 + o1
                item_selected_r1 = torch.gather(state_item_allr, 1, io1.view(-1, 1, 1).repeat(1, 1, 3)).squeeze(1)
                p_prob1 = self.p_decoder.forward(item_selected_r1, None, h_i_leftover1, h_b1)
                p1 = torch.multinomial(p_prob1, 1, replacement=False).squeeze(1)
                p_prob1 = torch.gather(p_prob1, 1, p1.unsqueeze(1).long()).squeeze()

                x1, y1 = self.gen_xy(state_bin, item_selected_r1, p1)
                new_state_bin1, new_state_item1, new_state_packed_item1, z1 = self.update_state(state_bin, state_item,
                                                                                            state_packed_item, i1, o1,
                                                                                            x1, y1,
                                                                                            item_selected_r1)

                plot1 = torch.cat((x1.unsqueeze(1), y1.unsqueeze(1), z1.unsqueeze(1), item_selected_r1), dim=1)
                out1 = [new_state_bin1, new_state_item1, new_state_packed_item1, i_prob1, i1, o_prob1, o1, p_prob1, p1, plot1]

                i_prob2 = self.i_decoder.forward(h_b2, h_i)
                i2 = torch.multinomial(i_prob2, 1, replacement=False).squeeze(1)
                i_prob2 = torch.gather(i_prob2, 1, i2.unsqueeze(1).long()).squeeze()
                _, item_selected2 = self.select_item(h_i, state_item, i2)
                h_i_leftover2 = self.gen_h_i_leftover(h_i, i2)
                o_prob2 = self.o_decoder.forward(h_b2, item_selected2, h_i_leftover2)
                o2 = torch.multinomial(o_prob2, 1, replacement=False).squeeze(1)
                o_prob2 = torch.gather(o_prob2, 1, o2.unsqueeze(1).long()).squeeze()
                io2 = i2 * 6 + o2
                item_selected_r2 = torch.gather(state_item_allr, 1, io2.view(-1, 1, 1).repeat(1, 1, 3)).squeeze(1)
                p_prob2 = self.p_decoder.forward(item_selected_r2, None, h_i_leftover2, h_b2)
                p2 = torch.multinomial(p_prob2, 1, replacement=False).squeeze(1)
                p_prob2 = torch.gather(p_prob2, 1, p2.unsqueeze(1).long()).squeeze()

                x2, y2 = self.gen_xy(state_bin.transpose(1, 2), item_selected_r2, p2)
                new_state_bin2, new_state_item2, new_state_packed_item2, z2 = self.update_state(state_bin.transpose(1, 2), state_item,
                                                                                                state_packed_item, i2,
                                                                                                o2,
                                                                                                x2, y2,
                                                                                                item_selected_r2)
                plot2 = torch.cat((y2.unsqueeze(1), x2.unsqueeze(1), z2.unsqueeze(1), item_selected_r2[:, [1, 0, 2]]), dim=1)
                out2 = [new_state_bin2, new_state_item2, new_state_packed_item2, i_prob2, i2, o_prob2, o2, p_prob2, p2,
                        plot2]
                return out1, out2

        elif opt.L == 400 and opt.W == 200:
            h_i1, state_item_allr1 = self.item_encoder.forward(state_item, istranspose=False)
            h_i2, state_item_allr2 = self.item_encoder.forward(state_item, istranspose=True)

            h_b1 = self.bin_encoder.forward(state_bin.unsqueeze(1))
            h_b2 = self.bin_encoder.forward(state_bin.transpose(1, 2).unsqueeze(1))
            io_prob1 = self.io_decoder.forward(h_b1, h_i1)
            io1 = torch.multinomial(io_prob1, 1, replacement=False).squeeze(1)
            i1 = (io1 / 6).long()
            o1 = io1 % 6
            io_prob1 = torch.gather(io_prob1, 1, io1.unsqueeze(1).long()).squeeze()

            io_prob2 = self.io_decoder.forward(h_b2, h_i2)
            io2 = torch.multinomial(io_prob2, 1, replacement=False).squeeze(1)
            i2 = (io2 / 6).long()
            o2 = io2 % 6
            io_prob2 = torch.gather(io_prob2, 1, io2.unsqueeze(1).long()).squeeze()

            h_i_selected1, item_selected_r1 = self.select_item(h_i1, state_item_allr1, io1)
            h_i_selected2, item_selected_r2 = self.select_item(h_i2, state_item_allr2, io2)

            h_i_leftover1 = self.gen_h_i_leftover(h_i1, i1)
            h_i_leftover2 = self.gen_h_i_leftover(h_i2, i2)

            p_prob1 = self.p_decoder.forward(item_selected_r1, h_i_selected1, h_i_leftover1, h_b1, istranspose=False)
            p1 = torch.multinomial(p_prob1, 1, replacement=False).squeeze(1)
            p_prob1 = torch.gather(p_prob1, 1, p1.unsqueeze(1).long()).squeeze()
            p_prob2 = self.p_decoder.forward(item_selected_r2, h_i_selected2, h_i_leftover2, h_b2, istranspose=True)
            p2 = torch.multinomial(p_prob2, 1, replacement=False).squeeze(1)
            p_prob2 = torch.gather(p_prob2, 1, p2.unsqueeze(1).long()).squeeze()

            x1, y1 = self.gen_xy(state_bin, item_selected_r1, p1)
            x2, y2 = self.gen_xy(state_bin.transpose(1, 2), item_selected_r2, p2)


            new_state_bin1, new_state_item1, new_state_packed_item1, z1 = self.update_state(state_bin, state_item,
                                                                                            state_packed_item, i1, o1, x1,
                                                                                            y1,
                                                                                            item_selected_r1)
            new_state_bin2, new_state_item2, new_state_packed_item2, z2 = self.update_state(state_bin.transpose(1, 2),
                                                                                            state_item,
                                                                                            state_packed_item, i2, o2, x2,
                                                                                            y2,
                                                                                            item_selected_r2)
            plot1 = torch.cat((x1.unsqueeze(1) * 2, y1.unsqueeze(1), z1.unsqueeze(1), new_state_packed_item1[:, -1]), dim=1)
            plot2 = torch.cat((y2.unsqueeze(1) * 2, x2.unsqueeze(1), z2.unsqueeze(1), new_state_packed_item2[:, -1, [1, 0, 2]]), dim=1)
            # plot1 = [x1 * 2, y1, z1, new_state_packed_item1[:, -1]]
            # plot2 = [y2 * 2, x2, z2, new_state_packed_item2[:, -1, [1, 0, 2]]]
            out1 = [new_state_bin1, new_state_item1, new_state_packed_item1, io_prob1, io1, p_prob1, p1, plot1]
            out2 = [new_state_bin2, new_state_item2, new_state_packed_item2, io_prob2, io2, p_prob2, p2, plot2]
            return out1, out2
    
    @staticmethod
    def select_item(h_i, state_item_allr, io):
        """
        select item and h_i from all unpacked items and feature sequence
        """
        h_i_selected = torch.gather(h_i, 1, io.view(-1, 1, 1).repeat(1, 1, opt.hidden_size)).squeeze(1)
        item_selected_r = torch.gather(state_item_allr, 1, io.view(-1, 1, 1).repeat(1, 1, 3)).squeeze(1)
        return h_i_selected, item_selected_r

    @staticmethod
    def gen_h_i_leftover(h_i, idx):
        """
        generate leftover feature sequence
        """
        bs = h_i.size(0)
        if opt.TS:
            io_idx = torch.tensor([0, 1, 2, 3, 4, 5], device=opt.device).unsqueeze(0).repeat(bs, 1) + \
                     idx.unsqueeze(1) * 6
            io_idx = io_idx.unsqueeze(2).repeat(1, 1, opt.hidden_size)
            h = h_i.scatter(1, io_idx, (torch.zeros([bs, 6, opt.hidden_size], device=opt.device)))
            idx_leftover = torch.nonzero(h)[:, 1].reshape(bs, h_i.size(1) - 6, opt.hidden_size)
            h_i_leftover = torch.gather(h_i, 1, idx_leftover)
        else:
            i_idx = idx.view(-1, 1, 1).repeat(1, 1, opt.hidden_size)
            h = h_i.scatter(1, i_idx, (torch.zeros([bs, 1, opt.hidden_size], device=opt.device)))
            idx_leftover = torch.nonzero(h)[:, 1].reshape(bs, h_i.size(1) - 1, opt.hidden_size)
            h_i_leftover = torch.gather(h_i, 1, idx_leftover)
        return h_i_leftover

    def gen_xy(self, state_bin, item_selected_r, p):
        """
        generate position x and y
        """
        bs = state_bin.size(0)
        y = torch.zeros([bs])
        bl = item_selected_r[:, 0].long()
        bw = item_selected_r[:, 1].long()

        x = p
        idx = torch.arange(min(opt.L, opt.W), device=opt.device).unsqueeze(0).repeat(opt.W, 1).unsqueeze(0).repeat(bs, 1, 1)
        view = torch.where((idx - x.unsqueeze(1).unsqueeze(2) >= 0) * (
                    idx - x.unsqueeze(1).unsqueeze(2) < bl.unsqueeze(1).unsqueeze(2)),
                           state_bin,
                           torch.zeros([bs, min(opt.L, opt.W), min(opt.L, opt.W)], device=opt.device))
        view_max = torch.max(view, dim=2)[0]
        idx = torch.arange(min(opt.L, opt.W), device=opt.device).unsqueeze(0).repeat(bs, 1)
        z = float('inf') * torch.ones([bs], device=opt.device)

        view_max = view_max.cpu()
        idx = idx.cpu()
        z = z.cpu()
        bw = bw.cpu()
        for i in range(opt.W):
            view_window = torch.where((idx - i * torch.ones(bs, opt.W) >= 0) * (
                        idx - i * torch.ones(bs, opt.W) < bw.unsqueeze(1).repeat(1, opt.W)),
                                      view_max,
                                      torch.zeros([bs, opt.W]))
            view_window = torch.where(
                i * torch.ones(bs, opt.W) + bw.unsqueeze(1).repeat(1, opt.W) <= opt.W,
                view_window,
                float('inf') * torch.ones([bs, opt.W]))

            view_window_max = torch.max(view_window, dim=1)[0]
            y = torch.where(view_window_max < z, i * torch.ones([bs]), y)
            z = torch.where(view_window_max < z, view_window_max, z)
        return x, y.to(opt.device)

    def update_state(self, state_bin, state_item, state_packed_item, i, o, x, y, item_selected_r):
        """
        update bin state and item state
        """
        bs = state_bin.size(0)
        new_state_bin = state_bin.clone()
        new_state_packed_item = state_packed_item.clone()

        x1 = x.long()
        y1 = y.long()
        lwh = item_selected_r

        view_x = torch.arange(0, min(opt.L, opt.W), device=opt.device).repeat(min(opt.L, opt.W), 1).repeat(bs, 1, 1)
        view_y = torch.arange(0, min(opt.L, opt.W), device=opt.device).repeat(min(opt.L, opt.W), 1).transpose(0, 1).repeat(bs, 1, 1)

        view_x_min = x1.view(-1, 1, 1).expand(-1, min(opt.L, opt.W), min(opt.L, opt.W))
        view_x_max = (x1 + lwh[:, 0]).view(-1, 1, 1).expand(-1, min(opt.L, opt.W), min(opt.L, opt.W))

        view_y_min = y1.view(-1, 1, 1).expand(-1, min(opt.L, opt.W), min(opt.L, opt.W))
        view_y_max = (y1 + lwh[:, 1]).view(-1, 1, 1).expand(-1, min(opt.L, opt.W), min(opt.L, opt.W))

        view_h = lwh[:, 2].view(-1, 1, 1).expand(-1, min(opt.L, opt.W), min(opt.L, opt.W))

        view_state_ = torch.where(
            (view_x >= view_x_min) * (view_x < view_x_max) * (view_y >= view_y_min) * (view_y < view_y_max), state_bin,
            torch.zeros([bs, min(opt.L, opt.W), min(opt.L, opt.W)], device=opt.device))

        bin_state_max = torch.max(torch.max(view_state_, 2)[0], 1)[0].view(-1, 1, 1)
        view_selected_item = torch.where(
            (view_x >= view_x_min) * (view_x < view_x_max) * (view_y >= view_y_min) * (view_y < view_y_max),
            view_h + bin_state_max, torch.zeros([bs, min(opt.L, opt.W), min(opt.L, opt.W)], device=opt.device))
        new_state_bin = torch.max(new_state_bin, view_selected_item)

        new_state_item = state_item.scatter(1, i.unsqueeze(1).unsqueeze(2).repeat(1, 1, 3),
                                           torch.zeros([bs, 1, 3], device=opt.device))

        idx = torch.nonzero(new_state_item)[:, 1].reshape(bs, state_item.size(1) - 1, 3)

        new_state_item = torch.gather(new_state_item, 1, idx)

        select_item = torch.gather(state_item, 1, i.unsqueeze(1).unsqueeze(1).repeat(1, 1, 3))
        sequence = [0, 1, 2, 0, 2, 1, 1, 0, 2, 1, 2, 0, 2, 0, 1, 2, 1, 0]
        inx_allr = torch.tensor(sequence, dtype=torch.long, device=opt.device).reshape(6, 3)
        inx_allr = inx_allr.repeat(1, 1).unsqueeze(0).repeat(bs, 1, 1)
        select_item_allr = select_item.repeat(1, 1, 6).reshape(bs, 6, 3)
        select_item_allr = torch.gather(select_item_allr, 2, inx_allr)
        select_item_r = torch.gather(select_item_allr, 1, o.unsqueeze(1).unsqueeze(1).repeat(1, 1, 3))
        new_state_packed_item = torch.cat((new_state_packed_item, select_item_r), dim=1)
        
        return new_state_bin, new_state_item, new_state_packed_item, bin_state_max.squeeze()


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.transform = TransformDecoder(opt.hidden_size)
        self.item_encoder = ItemEncoder(opt.hidden_size)
        self.bin_encoder = BinEncoder()
        self.head = nn.Sequential(
            nn.Linear(opt.hidden_size, opt.hidden_size),
            nn.ReLU(),
            nn.Linear(opt.hidden_size, 1))

    def forward(self, state_bin, state_item, istranspose):
        """
        :param state_bin: bin state
        :param state_item: unpacked item state
        :return: value
        """
        h_i, _ = self.item_encoder.forward(state_item, istranspose)
        h_b = self.bin_encoder.forward(state_bin.unsqueeze(1))
        l = self.transform.forward(h_b.unsqueeze(1), h_i, h_i).squeeze()
        value = self.head(l).squeeze(1)
        return value


class A2C(nn.Module):
    def __init__(self):
        super(A2C, self).__init__()
        self.batch_size = opt.batch_size
        self.Actor = Actor().to(opt.device)
        self.Critic = Critic().to(opt.device)
        self.optimizer = optim.Adam([
            {'params': self.Actor.parameters(), 'lr': opt.actor_lr},
            {'params': self.Critic.parameters(), 'lr': opt.critic_lr},
        ])

        if opt.TS:
            self.io = torch.tensor([], device=opt.device)
            self.io_prob = torch.tensor([], device=opt.device)
        else:
            self.i = torch.tensor([], device=opt.device)
            self.i_prob = torch.tensor([], device=opt.device)
            self.o = torch.tensor([], device=opt.device)
            self.o_prob = torch.tensor([], device=opt.device)

        self.p = torch.tensor([], device=opt.device)
        self.p_prob = torch.tensor([], device=opt.device)

        self.g = torch.tensor([], device=opt.device)
        self.As = torch.empty([opt.batch_size, 0], device=opt.device)

        self.util = torch.tensor([], device=opt.device)
        self.value = torch.tensor([], device=opt.device)
        self.value1 = torch.tensor([], device=opt.device)

        self.critic_losses = torch.tensor([], device=opt.device)
        self.mse = nn.MSELoss(reduction='mean')
        self.opt = opt

    def set_utilization(self, state_bin, state_packed_item):
        V = torch.sum(torch.prod(state_packed_item, 2), dim=1)
        H = torch.max(torch.max(state_bin, dim=2)[0], dim=1)[0]
        self.util = torch.div(V, opt.L * opt.W * H)

    def set_g(self, state_bin, state_packed_item):
        V = torch.sum(torch.prod(state_packed_item, 2), dim=1)
        H = torch.max(torch.max(state_bin, dim=2)[0], dim=1)[0]
        g = (opt.L * opt.W * H - V) / (opt.L * opt.W * max(opt.L, opt.W))
        self.g = torch.cat((self.g, g.unsqueeze(0)), dim=0)

    def cal_value12(self, state_bin1, state_item1, state_bin2, state_item2):
        value1 = self.Critic.forward(state_bin1, state_item1, istranspose=False).detach()
        value2 = self.Critic.forward(state_bin2, state_item2, istranspose=True).detach()
        value_idx = value1 > value2
        return value_idx

    def set_value(self, value):
        self.value = torch.cat((self.value, value.unsqueeze(0)), dim=0)

    def set_value1(self, state_bin, state_item):
        value = self.Critic.forward(state_bin, state_item, False)
        self.value = torch.cat((self.value, value.unsqueeze(0)), dim=0)

    def store_index_orientation(self, s_prob, s):
        self.io_prob = torch.cat((self.io_prob, s_prob.unsqueeze(1)), dim=1)
        self.io = torch.cat((self.io, s.unsqueeze(1)), dim=1)

    def store_index(self, i_prob, i):
        self.i_prob = torch.cat((self.i_prob, i_prob.unsqueeze(1)), dim=1)
        self.i = torch.cat((self.i, i.unsqueeze(1)), dim=1)

    def store_orientation(self, o_prob, o):
        self.o_prob = torch.cat((self.o_prob, o_prob.unsqueeze(1)), dim=1)
        self.o = torch.cat((self.o, o.unsqueeze(1)), dim=1)

    def store_position(self, p_prob, p):
        self.p_prob = torch.cat((self.p_prob, p_prob.unsqueeze(1)), dim=1)
        self.p = torch.cat((self.p, p.unsqueeze(1)), dim=1)

    def set_advantage(self, end=False):
        """
        calculate advantage and critic_loss
        """
        r = self.g.clone()
        for i in reversed(range(r.size(0))):
            r[i] = (r[i - 1] - r[i]) if i > 0 else -r[i]

        j = r.size(0) - 2 if not end else r.size(0) - 1
        if end:
            self.value = torch.cat((self.value, self.value1.unsqueeze(0)), dim=0)

        for i in reversed(range(self.value.size(0))):
            if j == opt.nof_item - 1:
                v_i1 = torch.zeros(opt.batch_size, device=opt.device)
            elif i < self.value.size(0) - 1:
                v_i1 = self.value[i + 1]
            else:
                v_i1 = self.value1

            v_i = self.value[i]
            delta = r[j].detach() + opt.gamma * v_i1.detach() - v_i
            j -= 1

            self.A = self.A * opt.gamma * opt.Lambda + delta
            self.critic_loss = self.critic_loss + self.mse(v_i.detach(), self.A + v_i.detach())

    def set_advantage1(self, end=False):
        A = torch.zeros(opt.batch_size, device=opt.device)
        r = self.g.clone()
        for i in reversed(range(r.size(0))):
            r[i] = (r[i - 1] - r[i]) if i > 0 else -r[i]
        j = r.size(0) - 1
        for i in reversed(range(self.value.size(0) - 1)):
            if end and i == self.opt.ngae - 2:
                v_i1 = torch.zeros(self.opt.batch_size, device=opt.device)
            else:
                v_i1 = self.value[i + 1]
            v_i = self.value[i]
            delta = r[j].detach() + self.opt.gamma * v_i1.detach() - v_i
            j -= 1
            A = A * self.opt.gamma * self.opt.Lambda + delta
            self.As = torch.cat((A.unsqueeze(1), self.As), dim=1)
            a = self.mse(v_i.detach(), A + v_i.detach())
            self.critic_losses = torch.cat((self.mse(v_i.detach(), A + v_i.detach()).unsqueeze(0), self.critic_losses), dim=0)

    def learn(self):

        selected_log_p_prob = torch.log(self.p_prob + 1e-8)
        selected_p_prob = self.p_prob

        adv = self.As.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        if self.opt.TS:
            selected_log_io_prob = torch.log(self.io_prob + 1e-8)
            selected_io_prob = self.io_prob
            actor_loss = torch.mul(selected_log_io_prob[:, 1:] + selected_log_p_prob[:, 1:], -adv)
            ent_loss = -(selected_io_prob * selected_log_io_prob + selected_p_prob * selected_log_p_prob).mean()
        else:
            selected_log_i_prob = torch.log(self.i_prob + 1e-8)
            selected_i_prob = self.i_prob
            selected_log_o_prob = torch.log(self.o_prob + 1e-8)
            selected_o_prob = self.o_prob
            actor_loss = torch.mul(selected_log_i_prob[:, 1:] + selected_log_o_prob[:, 1:] + selected_log_p_prob[:, 1:], -adv)
            ent_loss = -(selected_i_prob * selected_log_i_prob + selected_o_prob * selected_log_o_prob + selected_p_prob * selected_log_p_prob).mean()


        actor_loss = actor_loss.mean()  # batch
        critic_loss = self.critic_losses.sum()

        loss = actor_loss + critic_loss + 0.001 * ent_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.p = torch.tensor([], device=opt.device)
        self.p_prob = torch.tensor([], device=opt.device)
        self.As = torch.empty([opt.batch_size, 0], device=opt.device)
        if opt.TS:
            self.io = torch.tensor([], device=opt.device)
            self.io_prob = torch.tensor([], device=opt.device)
        else:
            self.i = torch.tensor([], device=opt.device)
            self.i_prob = torch.tensor([], device=opt.device)
            self.o = torch.tensor([], device=opt.device)
            self.o_prob = torch.tensor([], device=opt.device)
        self.critic_losses = torch.tensor([], device=opt.device)
        self.value = torch.tensor([], device=opt.device)
        self.value1 = torch.tensor([], device=opt.device)
