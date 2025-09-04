import torch
import torch.nn as nn
from config import DefaultConfig
from Util import *

opt = DefaultConfig()


class BinEncoder(nn.Module):
    def __init__(self):
        super(BinEncoder, self).__init__()
        if opt.L == 100 and opt.W == 100:
            self.Conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
            self.Conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.FC = nn.Linear(64 * 11 * 11, 128)
            self.LayerNorm1 = nn.LayerNorm([32, 24, 24])
            self.LayerNorm2 = nn.LayerNorm([64, 11, 11])
            self.LayerNorm3 = nn.LayerNorm([128])
            self.relu = nn.ReLU()

        elif opt.L == 200 and opt.W == 200:
            self.Conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=2)
            self.Conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.Conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
            self.FC = nn.Linear(64 * 11 * 11, 128)
            self.LayerNorm1 = nn.LayerNorm([32, 50, 50])
            self.LayerNorm2 = nn.LayerNorm([64, 24, 24])
            self.LayerNorm3 = nn.LayerNorm([64, 11, 11])
            self.LayerNorm4 = nn.LayerNorm([128])
            self.relu = nn.ReLU()
        elif opt.L == 400 and opt.W == 200:
            self.Conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=2)
            self.Conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.Conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
            self.FC = nn.Linear(64 * 11 * 11, 128)
            self.LayerNorm1 = nn.LayerNorm([32, 50, 50])
            self.LayerNorm2 = nn.LayerNorm([64, 24, 24])
            self.LayerNorm3 = nn.LayerNorm([64, 11, 11])
            self.LayerNorm4 = nn.LayerNorm([128])
            self.relu = nn.ReLU()

    def forward(self, state_bin):
        """
        :param state_bin: bin state
        :return h_b: bin feature vector
        """
        if opt.L == 100 and opt.W == 100:
            c1 = self.relu(self.LayerNorm1(self.Conv1(state_bin)))
            c2 = self.relu(self.LayerNorm2(self.Conv2(c1)))
            l2 = torch.flatten(c2, 1)
            h_b = self.relu(self.LayerNorm3(self.FC(l2)))
        elif opt.L == 200 and opt.W == 200:
            c1 = self.relu(self.LayerNorm1(self.Conv1(state_bin)))
            c2 = self.relu(self.LayerNorm2(self.Conv2(c1)))
            c3 = self.relu(self.LayerNorm3(self.Conv3(c2)))
            l2 = torch.flatten(c3, 1)
            h_b = self.relu(self.LayerNorm4(self.FC(l2)))
        elif opt.L == 400 and opt.W == 200:
            c1 = self.relu(self.LayerNorm1(self.Conv1(state_bin)))
            c2 = self.relu(self.LayerNorm2(self.Conv2(c1)))
            c3 = self.relu(self.LayerNorm3(self.Conv3(c2)))
            l2 = torch.flatten(c3, 1)
            h_b = self.relu(self.LayerNorm4(self.FC(l2)))
        return h_b


class ItemEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(ItemEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.ItemEmbedding = nn.Linear(3, hidden_size)
        self.Transformer = TransformEncoder(hidden_size)
        self.trans_layer_1_1 = TransformerLayer(hidden_size)
        self.trans_layer_1_2 = TransformerLayer(hidden_size)
        self.trans_layer_2_1 = TransformerLayer(hidden_size)
        self.trans_layer_2_2 = TransformerLayer(hidden_size)
        self.I1 = nn.Parameter(torch.Tensor(1, opt.inducing_num, hidden_size))
        self.I2 = nn.Parameter(torch.Tensor(1, opt.inducing_num, hidden_size))
        nn.init.xavier_uniform_(self.I1)
        nn.init.xavier_uniform_(self.I2)

    def forward(self, state_item, istranspose=False):
        """
        :param state_item: unpacked items state
        :return h_i: item feature sequence
        :return state_item_allr: all rotated unpacked items state
        """
        bs = state_item.size(0)
        if opt.TS:
            sequence = [0, 1, 2, 0, 2, 1, 1, 0, 2, 1, 2, 0, 2, 0, 1, 2, 1, 0]
            length = state_item.size(1)

            inx_allr = torch.tensor(sequence, dtype=torch.long, device=opt.device).reshape(6, 3)
            inx_allr = inx_allr.repeat(length, 1).unsqueeze(0).repeat(bs, 1, 1)

            state_item_allr = state_item.repeat(1, 1, 6).reshape(bs, 6 * length, 3)
            state_item_allr = torch.gather(state_item_allr, 2, inx_allr)
            if opt.L > opt.W:
                if not istranspose:
                    state_item_allr[:, :, 0] = torch.ceil(state_item_allr[:, :, 0] / (opt.L/opt.W))

                else:
                    state_item_allr[:, :, 1] = torch.ceil(state_item_allr[:, :, 1] / (opt.L/opt.W))

            attn_mask = torch.eye(length, device=opt.device).repeat(1, 6).reshape(6 * length, length)
            attn_mask = attn_mask.transpose(0, 1).repeat(1, 6).reshape(6 * length, 6 * length) - \
                        torch.eye(6 * length, device=opt.device)
            attn_mask = attn_mask * (-1e8)

            attn_mask = attn_mask.unsqueeze(0).repeat(bs * 8, 1, 1)

            if state_item.size(1) != 0:
                h_i = self.ItemEmbedding(state_item_allr)
                if opt.ISAB:
                    h_i = self.ISAB(h_i)
                else:
                    h_i = self.Transformer.forward(h_i, attn_mask)
            else:
                h_i = torch.zeros([bs, 0, opt.hidden_size], device=opt.device)
            return h_i, state_item_allr

        else:
            if state_item.size(1) != 0:
                h_i = self.ItemEmbedding(state_item)
                h_i = self.Transformer.forward(h_i, None)
            else:
                h_i = torch.zeros([bs, 0, opt.hidden_size], device=opt.device)

            sequence = [0, 1, 2, 0, 2, 1, 1, 0, 2, 1, 2, 0, 2, 0, 1, 2, 1, 0]
            length = state_item.size(1)

            inx_allr = torch.tensor(sequence, dtype=torch.long, device=opt.device).reshape(6, 3)
            inx_allr = inx_allr.repeat(length, 1).unsqueeze(0).repeat(bs, 1, 1)

            state_item_allr = state_item.repeat(1, 1, 6).reshape(bs, 6 * length, 3)
            state_item_allr = torch.gather(state_item_allr, 2, inx_allr)
            return h_i, state_item_allr

    def ISAB(self, h_i):
        bs = h_i.size(0)
        I1 = self.I1.repeat(bs, 1, 1)
        I2 = self.I2.repeat(bs, 1, 1)
        kv = self.trans_layer_1_1.forward(I1, h_i, h_i)
        h_i = self.trans_layer_1_2.forward(h_i, kv, kv)
        kv = self.trans_layer_2_1.forward(I2, h_i, h_i)
        h_i = self.trans_layer_2_2.forward(h_i, kv, kv)
        return h_i


class IndexDecoder(nn.Module):
    def __init__(self, hidden_size):
        super(IndexDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.Transformer = TransformDecoder(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(opt.hidden_size, opt.hidden_size),
            nn.LayerNorm(opt.hidden_size),
            nn.ReLU(),
            nn.Linear(opt.hidden_size, opt.hidden_size),
            nn.LayerNorm(opt.hidden_size),
            nn.ReLU(),
            nn.Linear(opt.hidden_size, 1)
        )
        self.softmax = nn.Softmax(1)
        self.tanh = nn.Tanh()
        self.C = opt.C

    def forward(self, h_b, h_i):
        """
        :param h_b: bin feature vector
        :param h_i: unpacked items feature sequence
        :return i_prob: probability distribution over the index selection actions
        """
        l = self.Transformer.forward(h_i, h_b.unsqueeze(1), h_b.unsqueeze(1))
        i_logits = self.head(l).squeeze(2)
        i_logits = opt.C * self.tanh(i_logits)
        i_prob = self.softmax(i_logits)
        return i_prob


class OrientationDecoder(nn.Module):
    def __init__(self, hidden_size):
        super(OrientationDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.Transformer = TransformDecoder(hidden_size)
        self.embedding = nn.Linear(3, opt.hidden_size)
        self.head = nn.Sequential(
            nn.Linear(opt.hidden_size, opt.hidden_size),
            nn.LayerNorm(opt.hidden_size),
            nn.ReLU(),
            nn.Linear(opt.hidden_size, opt.hidden_size),
            nn.LayerNorm(opt.hidden_size),
            nn.ReLU(),
            nn.Linear(opt.hidden_size, 6)
        )
        self.softmax = nn.Softmax(1)
        self.tanh = nn.Tanh()
        self.C = opt.C

    def forward(self, h_b, item_sel, h_i_leftover):
        """
        :param h_b: bin feature vector
        :param item_sel: selected item
        :param h_i_leftover: leftover item sequence
        :return o_prob: probability distribution over the orientation selection actions
        """
        h_i_sel = self.embedding(item_sel)
        h = h_b + h_i_sel
        l = self.Transformer.forward(h.unsqueeze(1), h_i_leftover, h_i_leftover).squeeze(1)
        o_logits = self.head(l)
        o_logits = opt.C * self.tanh(o_logits)
        o_prob = self.softmax(o_logits)
        return o_prob


class IndexOrientationDecoder(nn.Module):
    def __init__(self, hidden_size):
        super(IndexOrientationDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.Transformer = TransformDecoder(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(opt.hidden_size, opt.hidden_size),
            nn.LayerNorm(opt.hidden_size),
            nn.ReLU(),
            nn.Linear(opt.hidden_size, opt.hidden_size),
            nn.LayerNorm(opt.hidden_size),
            nn.ReLU(),
            nn.Linear(opt.hidden_size, 1)
        )
        self.softmax = nn.Softmax(1)
        self.tanh = nn.Tanh()
        self.C = opt.C

    def forward(self, h_b, h_i):
        """
        :param h_b: bin feature vector
        :param h_i: unpacked items feature sequence
        :return io_prob: probability distribution over the index-orientation selection actions
        """
        l = self.Transformer.forward(h_i, h_b.unsqueeze(1), h_b.unsqueeze(1))
        io_logits = self.head(l).squeeze(2)
        io_logits = opt.C * self.tanh(io_logits)
        io_prob = self.softmax(io_logits)
        return io_prob


class PositionDecoder(nn.Module):
    def __init__(self, hidden_size):
        super(PositionDecoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.LayerNorm(4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, 4 * hidden_size),
            nn.LayerNorm(4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, min(opt.L,opt.W))
        )
        self.embedding = nn.Linear(3, opt.hidden_size)
        self.Transformer = TransformDecoder(hidden_size)
        self.softmax = nn.Softmax(1)
        self.tanh = nn.Tanh()
        self.C = opt.C

    def forward(self, item_selected_r, h_i_selected, h_i_leftover, h_b, istranspose=False):
        """
        :param item_selected_r: selected item state
        :param h_i_selected: selected item feature
        :param h_i_leftover: leftover item sequence
        :param h_b: bin feature sequence
        :return p_prob:  probability distribution over the position selection actions
        """
        if opt.TS:
            q = h_i_selected + h_b
        else:
            q = self.embedding(item_selected_r) + h_b
        h = self.Transformer.forward(q.unsqueeze(1), h_i_leftover, h_i_leftover).squeeze(1)
        l = self.layer(h)
        p_logits = self.C * self.tanh(l)
        p_logits = self.mask_infeasible(p_logits, item_selected_r, istranspose)
        p_prob = self.softmax(p_logits)
        return p_prob

    @staticmethod
    def mask_infeasible(p_logits, item_selected_r, istranspose):
        """
        Remove impossible position selections
        """
        bs = p_logits.size(0)
        bl = torch.gather(item_selected_r, 1, torch.zeros([bs, 1], dtype=torch.long, device=opt.device)).squeeze(1)
        _inf = -float('inf') * torch.ones([bs, min(opt.L,opt.W)], device=opt.device)
        l_map = (torch.arange(min(opt.L,opt.W), 0, -1, device=opt.device).repeat(bs, 1).float() - bl.unsqueeze(1))
        p_logits = torch.where(l_map >= 0, p_logits, _inf)
        # if istranspose and opt.L == 400 and opt.W == 200:
        #     mask = torch.arange(0, opt.L, device=opt.device).long().repeat(opt.batch_size, 1)
        #     p_logits = torch.where(mask % 2 == 0, p_logits, _inf)
        return p_logits
