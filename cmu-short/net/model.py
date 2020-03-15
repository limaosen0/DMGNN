import torch
import torch.nn as nn
import torch.nn.functional as F

from net.utils.graph import Graph_J, Graph_P, Graph_B
from net.utils.module import *
from net.utils.operation import PartLocalInform, BodyLocalInform


class Model(nn.Module):
    
    def __init__(self, n_in_enc, n_hid_enc, n_in_dec, n_hid_dec, graph_args_j, graph_args_p, graph_args_b, fusion_layer, cross_w, **kwargs):
        super().__init__()

        self.encoder_pos = Encoder(n_in_enc, graph_args_j, graph_args_p, graph_args_b, True, fusion_layer, cross_w, **kwargs)
        self.encoder_vel = Encoder(n_in_enc, graph_args_j, graph_args_p, graph_args_b, True, fusion_layer, cross_w, **kwargs)
        self.encoder_acl = Encoder(n_in_enc, graph_args_j, graph_args_p, graph_args_b, True, fusion_layer, cross_w, **kwargs)
        self.decoder = Decoder(n_in_dec, n_hid_dec, graph_args_j, **kwargs)
        self.linear = nn.Linear(n_hid_enc*3, n_hid_dec)
        self.relu = nn.ReLU()

    def forward(self, enc_p, enc_v, enc_a, dec_curr, dec_prev, dec_prev2,
                t, relrec_s1, relsend_s1, relrec_s2, relsend_s2, relrec_s3, relsend_s3, lamda_p=1):
        hidd_p = self.encoder_pos(enc_p, relrec_s1, relsend_s1, relrec_s2, relsend_s2, relrec_s3, relsend_s3, lamda_p)
        hidd_v = self.encoder_vel(enc_v, relrec_s1, relsend_s1, relrec_s2, relsend_s2, relrec_s3, relsend_s3, lamda_p)
        hidd_a = self.encoder_acl(enc_a, relrec_s1, relsend_s1, relrec_s2, relsend_s2, relrec_s3, relsend_s3, lamda_p)
        hidden = self.linear(torch.cat((hidd_p, hidd_v, hidd_a), dim=1).permute(0,2,1)).permute(0,2,1)
        hidden = self.relu(hidden)
        pred = self.decoder(dec_curr, dec_prev, dec_prev2, hidden, t)
        return pred
        

class Encoder(nn.Module):

    def __init__(self, n_in_enc, graph_args_j, graph_args_p, graph_args_b, edge_weighting, fusion_layer, cross_w, **kwargs):
        super().__init__()

        self.graph_j = Graph_J(**graph_args_j)
        self.graph_p = Graph_P(**graph_args_p)
        self.graph_b = Graph_B(**graph_args_b)
        A_j = torch.tensor(self.graph_j.A_j, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_j', A_j)
        A_p = torch.tensor(self.graph_p.A_p, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_p', A_p)
        A_b = torch.tensor(self.graph_b.A_b, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_b', A_b)

        t_ksize, s_ksize_1, s_ksize_2, s_ksize_3 = 5, self.A_j.size(0), self.A_p.size(0), self.A_b.size(0)
        ksize_1 = (t_ksize, s_ksize_1)
        ksize_2 = (t_ksize, s_ksize_2)
        ksize_3 = (t_ksize, s_ksize_3)
        
        self.s2_init = AveargeJoint()
        self.s3_init = AveargePart()
        self.s1_l1 = St_gcn(n_in_enc, 32, ksize_1, stride=1, residual=False, **kwargs)
        self.s1_l2 = St_gcn(32, 64, ksize_1, stride=2, **kwargs)
        self.s1_l3 = St_gcn(64, 128, ksize_1, stride=2, **kwargs)
        self.s1_l4 = St_gcn(128, 256, ksize_1, stride=2, **kwargs)
        self.s1_l5 = St_gcn(256, 256, ksize_1, stride=1, **kwargs)
        self.s2_l1 = St_gcn(n_in_enc, 32, ksize_2, stride=1, residual=False, **kwargs)
        self.s2_l2 = St_gcn(32, 64, ksize_2, stride=2, **kwargs)
        self.s2_l3 = St_gcn(64, 128, ksize_2, stride=2, **kwargs)
        self.s2_l4 = St_gcn(128, 256, ksize_2, stride=2, **kwargs)
        self.s3_l1 = St_gcn(n_in_enc, 32, ksize_3, stride=1, residual=False, **kwargs)
        self.s3_l2 = St_gcn(32, 64, ksize_3, stride=2, **kwargs)
        self.s3_l3 = St_gcn(64, 128, ksize_3, stride=2, **kwargs)
        self.s3_l4 = St_gcn(128, 256, ksize_3, stride=2, **kwargs)
        self.s2_back = PartLocalInform()
        self.s3_back = BodyLocalInform()
        self.fusion_layer = fusion_layer
        self.cross_w = cross_w

        if self.fusion_layer == 0:
            pass
        elif self.fusion_layer == 1:
            self.j2p_1 = S1_to_S2(n_j1=32, n_j2=(800, 256), n_p1=32, n_p2=(800, 256), t_kernel=5, t_stride=(1,2), t_padding=2)
            self.p2j_1 = S2_to_S1(n_p1=32, n_p2=(800, 256), n_j1=32, n_j2=(800, 256), t_kernel=5, t_stride=(1,2), t_padding=2)
            self.p2b_1 = S2_to_S3(n_p1=32, n_p2=(800, 256), n_b1=32, n_b2=(800, 256), t_kernel=5, t_stride=(1,2), t_padding=2)
            self.b2p_1 = S3_to_S2(n_b1=32, n_b2=(800, 256), n_p1=32, n_p2=(800, 256), t_kernel=5, t_stride=(1,2), t_padding=2)
        elif self.fusion_layer == 2:
            self.j2p_1 = S1_to_S2(n_j1=32, n_j2=(800, 256), n_p1=32, n_p2=(800, 256), t_kernel=5, t_stride=(1,2), t_padding=2)
            self.p2j_1 = S2_to_S1(n_p1=32, n_p2=(800, 256), n_j1=32, n_j2=(800, 256), t_kernel=5, t_stride=(1,2), t_padding=2)
            self.p2b_1 = S2_to_S3(n_p1=32, n_p2=(800, 256), n_b1=32, n_b2=(800, 256), t_kernel=5, t_stride=(1,2), t_padding=2)
            self.b2p_1 = S3_to_S2(n_b1=32, n_b2=(800, 256), n_p1=32, n_p2=(800, 256), t_kernel=5, t_stride=(1,2), t_padding=2)
            self.j2p_2 = S1_to_S2(n_j1=64, n_j2=(832, 256), n_p1=64, n_p2=(832, 256), t_kernel=5, t_stride=(1,2), t_padding=2)
            self.p2j_2 = S2_to_S1(n_p1=64, n_p2=(832, 256), n_j1=64, n_j2=(832, 256), t_kernel=5, t_stride=(1,2), t_padding=2)
            self.p2b_2 = S2_to_S3(n_p1=64, n_p2=(832, 256), n_b1=64, n_b2=(832, 256), t_kernel=5, t_stride=(1,2), t_padding=2)
            self.b2p_2 = S3_to_S2(n_b1=64, n_b2=(832, 256), n_p1=64, n_p2=(832, 256), t_kernel=5, t_stride=(1,2), t_padding=2)
        else:
            raise ValueError('No Such Fusion Architecture')

        if edge_weighting:
            self.emul_s1 = nn.ParameterList([nn.Parameter(torch.ones(self.A_j.size())) for i in range(5)])
            self.eadd_s1 = nn.ParameterList([nn.Parameter(torch.zeros(self.A_j.size())) for i in range(5)])
            self.emul_s2 = nn.ParameterList([nn.Parameter(torch.ones(self.A_p.size())) for i in range(4)])
            self.eadd_s2 = nn.ParameterList([nn.Parameter(torch.zeros(self.A_p.size())) for i in range(4)])
            self.emul_s3 = nn.ParameterList([nn.Parameter(torch.ones(self.A_b.size())) for i in range(4)])
            self.eadd_s3 = nn.ParameterList([nn.Parameter(torch.zeros(self.A_b.size())) for i in range(4)])
        else:
            self.emul_s1 = [1]*0
            self.eadd_s1 = nn.ParameterList([nn.Parameter(torch.zeros(self.A_j.size())) for i in range(5)])
            self.emul_s2 = [1]*4
            self.eadd_s2 = nn.ParameterList([nn.Parameter(torch.zeros(self.A_p.size())) for i in range(4)])
            self.emul_s3 = [1]*4
            self.eadd_s3 = nn.ParameterList([nn.Parameter(torch.zeros(self.A_b.size())) for i in range(4)])

    def fuse_operation(self, x1, x2, x3, w):
        x = x1 + w * (x2+x3)
        return x

    def forward(self, x, relrec_s1, relsend_s1, relrec_s2, relsend_s2, relrec_s3, relsend_s3, lamda_p):
        N, T, D = x.size()                                                               # N = 64(batch-size), T = 49, D = 66
        V = self.A_j.size()[1]                                                             # V = 21
        x = x.contiguous().view(N,T,V,-1)                                                # [N, T, V, d] = [64, 49, 21, 3]
        x = x.permute(0, 3, 1, 2).contiguous()                                           # [N, d, T, V] = [64, 3, 49, 21]

        x_s1_0, x_s2_0, x_s3_0 = x, self.s2_init(x), self.s3_init(x)

        if self.fusion_layer == 0:
            x_s1_1 = self.s1_l1(x_s1_0, self.A_j*self.emul_s1[0]+self.eadd_s1[0])
            x_s2_1 = self.s2_l1(x_s2_0, self.A_p*self.emul_s2[0]+self.eadd_s2[0])
            x_s3_1 = self.s3_l1(x_s3_0, self.A_b*self.emul_s3[0]+self.eadd_s3[0])

            x_s1_2 = self.s1_l2(x_s1_1, self.A_j*self.emul_s1[1]+self.eadd_s1[1])
            x_s2_2 = self.s2_l2(x_s2_1, self.A_p*self.emul_s2[1]+self.eadd_s2[1])
            x_s3_2 = self.s3_l2(x_s3_1, self.A_b*self.emul_s3[1]+self.eadd_s3[1])

            x_s1_3 = self.s1_l3(x_s1_2, self.A_j*self.emul_s1[2]+self.eadd_s1[2])
            x_s2_3 = self.s2_l3(x_s2_2, self.A_p*self.emul_s2[2]+self.eadd_s2[2])
            x_s3_3 = self.s3_l3(x_s3_2, self.A_b*self.emul_s3[2]+self.eadd_s3[2])

            x_s1_4 = self.s1_l4(x_s1_3, self.A_j*self.emul_s1[3]+self.eadd_s1[3])
            x_s2_4 = self.s2_l4(x_s2_3, self.A_p*self.emul_s2[3]+self.eadd_s2[3])
            x_s3_4 = self.s3_l4(x_s3_3, self.A_b*self.emul_s3[3]+self.eadd_s3[3])

        elif self.fusion_layer == 1:
            x_s1_1 = self.s1_l1(x_s1_0, self.A_j*self.emul_s1[0]+self.eadd_s1[0])
            x_s2_1 = self.s2_l1(x_s2_0, self.A_p*self.emul_s2[0]+self.eadd_s2[0])
            x_s3_1 = self.s3_l1(x_s3_0, self.A_b*self.emul_s3[0]+self.eadd_s3[0])

            c12_1 = self.j2p_1(x_s1_1, x_s2_1, relrec_s1, relsend_s1, relrec_s2, relsend_s2)
            r12_1 = self.p2j_1(x_s2_1, x_s1_1, relrec_s2, relsend_s2, relrec_s1, relsend_s1)
            c23_1 = self.p2b_1(x_s2_1, x_s3_1, relrec_s2, relsend_s2, relrec_s3, relsend_s3)
            r23_1 = self.b2p_1(x_s3_1, x_s2_1, relrec_s3, relsend_s3, relrec_s2, relsend_s2)
            x_s1_1 = self.fuse_operation(x_s1_1, r12_1, 0,     self.cross_w)
            x_s2_1 = self.fuse_operation(x_s2_1, c12_1, r23_1, self.cross_w)
            x_s3_1 = self.fuse_operation(x_s3_1, c23_1, 0,     self.cross_w)

            x_s1_2 = self.s1_l2(x_s1_1, self.A_j*self.emul_s1[1]+self.eadd_s1[1])
            x_s2_2 = self.s2_l2(x_s2_1, self.A_p*self.emul_s2[1]+self.eadd_s2[1])
            x_s3_2 = self.s3_l2(x_s3_1, self.A_b*self.emul_s3[1]+self.eadd_s3[1])

            x_s1_3 = self.s1_l3(x_s1_2, self.A_j*self.emul_s1[2]+self.eadd_s1[2])
            x_s2_3 = self.s2_l3(x_s2_2, self.A_p*self.emul_s2[2]+self.eadd_s2[2])
            x_s3_3 = self.s3_l3(x_s3_2, self.A_b*self.emul_s3[2]+self.eadd_s3[2])

            x_s1_4 = self.s1_l4(x_s1_3, self.A_j*self.emul_s1[3]+self.eadd_s1[3])
            x_s2_4 = self.s2_l4(x_s2_3, self.A_p*self.emul_s2[3]+self.eadd_s2[3])
            x_s3_4 = self.s3_l4(x_s3_3, self.A_b*self.emul_s3[3]+self.eadd_s3[3])

        elif self.fusion_layer == 2:
            x_s1_1 = self.s1_l1(x_s1_0, self.A_j*self.emul_s1[0]+self.eadd_s1[0])
            x_s2_1 = self.s2_l1(x_s2_0, self.A_p*self.emul_s2[0]+self.eadd_s2[0])
            x_s3_1 = self.s3_l1(x_s3_0, self.A_b*self.emul_s3[0]+self.eadd_s3[0])

            c12_1 = self.j2p_1(x_s1_1, x_s2_1, relrec_s1, relsend_s1, relrec_s2, relsend_s2)
            r12_1 = self.p2j_1(x_s2_1, x_s1_1, relrec_s2, relsend_s2, relrec_s1, relsend_s1)
            c23_1 = self.p2b_1(x_s2_1, x_s3_1, relrec_s2, relsend_s2, relrec_s3, relsend_s3)
            r23_1 = self.b2p_1(x_s3_1, x_s2_1, relrec_s3, relsend_s3, relrec_s2, relsend_s2)
            x_s1_1 = self.fuse_operation(x_s1_1, r12_1, 0,     self.cross_w)
            x_s2_1 = self.fuse_operation(x_s2_1, c12_1, r23_1, self.cross_w)
            x_s3_1 = self.fuse_operation(x_s3_1, c23_1, 0,     self.cross_w)

            x_s1_2 = self.s1_l2(x_s1_1, self.A_j*self.emul_s1[1]+self.eadd_s1[1])
            x_s2_2 = self.s2_l2(x_s2_1, self.A_p*self.emul_s2[1]+self.eadd_s2[1])
            x_s3_2 = self.s3_l2(x_s3_1, self.A_b*self.emul_s3[1]+self.eadd_s3[1])

            c12_2 = self.j2p_2(x_s1_2, x_s2_2, relrec_s1, relsend_s1, relrec_s2, relsend_s2)
            r12_2 = self.p2j_2(x_s2_2, x_s1_2, relrec_s2, relsend_s2, relrec_s1, relsend_s1)
            c23_2 = self.p2b_2(x_s2_2, x_s3_2, relrec_s2, relsend_s2, relrec_s3, relsend_s3)
            r23_2 = self.b2p_2(x_s3_2, x_s2_2, relrec_s3, relsend_s3, relrec_s2, relsend_s2)
            x_s1_2 = self.fuse_operation(x_s1_2, r12_2, 0,     self.cross_w)
            x_s2_2 = self.fuse_operation(x_s2_2, c12_2, r23_2, self.cross_w)
            x_s3_2 = self.fuse_operation(x_s3_2, c23_2, 0,     self.cross_w)

            x_s1_3 = self.s1_l3(x_s1_2, self.A_j*self.emul_s1[2]+self.eadd_s1[2])
            x_s2_3 = self.s2_l3(x_s2_2, self.A_p*self.emul_s2[2]+self.eadd_s2[2])
            x_s3_3 = self.s3_l3(x_s3_2, self.A_b*self.emul_s3[2]+self.eadd_s3[2])

            x_s1_4 = self.s1_l4(x_s1_3, self.A_j*self.emul_s1[3]+self.eadd_s1[3])
            x_s2_4 = self.s2_l4(x_s2_3, self.A_p*self.emul_s2[3]+self.eadd_s2[3])
            x_s3_4 = self.s3_l4(x_s3_3, self.A_b*self.emul_s3[3]+self.eadd_s3[3])

        else:
            raise ValueError('No Such Fusion Architecture')

        x_s21 = self.s2_back(x_s2_4)
        x_s31 = self.s3_back(x_s3_4)
        x_s1_5 = x_s1_4+lamda_p*x_s21+lamda_p*x_s31
        x_out = torch.mean(self.s1_l5(x_s1_5, self.A_j*self.emul_s1[4]+self.eadd_s1[4]), dim=2)

        return x_out


class Decoder(nn.Module):
   
    def __init__(self, n_in_dec, n_hid_dec, graph_args_j, edge_weighting=True, dropout=0.3, **kwargs):
        super().__init__()

        self.graph_j = Graph_J(**graph_args_j)
        A_j = torch.tensor(self.graph_j.A_j, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_j', A_j)
        k_num, self.V = self.A_j.size(0), self.A_j.size(1)
        if edge_weighting:
            self.emul = nn.Parameter(torch.ones(self.A_j.size()))
            self.eadd = nn.Parameter(torch.ones(self.A_j.size()))
        else:
            self.emul = 1
            self.eadd = nn.Parameter(torch.ones(self.A_j.size()))

        self.msg_in = DecodeGcn(n_hid_dec, n_hid_dec, k_num)

        self.input_r = nn.Linear(n_in_dec, n_hid_dec, bias=True)
        self.input_i = nn.Linear(n_in_dec, n_hid_dec, bias=True)
        self.input_n = nn.Linear(n_in_dec, n_hid_dec, bias=True)

        self.hidden_r = nn.Linear(n_hid_dec, n_hid_dec, bias=False)
        self.hidden_i = nn.Linear(n_hid_dec, n_hid_dec, bias=False)
        self.hidden_h = nn.Linear(n_hid_dec, n_hid_dec, bias=False)

        self.out_fc1 = nn.Linear(n_hid_dec, n_hid_dec)
        self.out_fc2 = nn.Linear(n_hid_dec, n_hid_dec)
        self.out_fc3 = nn.Linear(n_hid_dec, 3)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

        self.mask = torch.ones(78).cuda().detach()
        self.zero_idx = torch.tensor([5, 11, 17, 23, 45, 49, 50,54, 55, 63, 67, 68 ,72, 73]).cuda().detach()
        self.mask[self.zero_idx] = 0.

    def step_forward(self, x, hidden, step):                     # inputs: [64, 21, 3]; hidden: [64, 256, 21]=N, hid, V
        N, V, d = x.size()
        if step<10:
            msg = self.msg_in(hidden, self.A_j*self.emul+self.eadd)    # msg: [64, 256, 21]=N, hid, V
        else:
            msg = hidden
        msg, hidden = msg.permute(0, 2, 1), hidden.permute(0, 2, 1)             # msg: [64, 21, 256]=N, V, hid, hidden: [64, 21, 256]

        r = torch.sigmoid(self.input_r(x) + self.hidden_r(msg))            # r: [64, 21, 256]=N, V, hid
        z = torch.sigmoid(self.input_i(x) + self.hidden_i(msg))            # z: [64, 21, 256]=N, V, hid
        n = torch.tanh(self.input_n(x) + r*self.hidden_h(msg))             # n: [64, 21, 256]=N, V, hid
        hidden = (1-z)*n + z*hidden                                             # hidden: [64, 21, 256]
        pred = hidden.new_zeros((N, V, 3))

        hidd = hidden
        hidd = self.dropout1(self.leaky_relu(self.out_fc1(hidd)))
        hidd = self.dropout2(self.leaky_relu(self.out_fc2(hidd)))
        pred = self.out_fc3(hidd)
        pred_ = x[:,:,:3] + pred                         # pred_: [64, 21, 3]
        hidden = hidden.permute(0, 2, 1)                                        # hidden: [64, 256, 21] for next convolution
        return pred_, hidden, pred                                              # pred: [64, 21, 3], hidden: [64, 256, 21]
        
    def forward(self, inputs, inputs_previous, inputs_previous2, hidden, t):      # inputs:[64, 1, 63];  hidden:[64, 256, 21]
        pred_all = []
        res_all = []

        N, T, D = inputs.size()
        inputs = inputs.contiguous().view(N, T, self.V, -1)                        # [64, 1, 21, 3]
        inputs_previous = inputs_previous.contiguous().view(N, T, self.V, -1)
        inputs_previous2 = inputs_previous2.contiguous().view(N, T, self.V, -1)
        self.mask = self.mask.view(self.V, 3)

        for step in range(0, t):
            if step < 1:
                ins_p = inputs[:, 0, :, :]                                         # ins_p: [64, 21, 3]
                ins_v = (inputs_previous-inputs_previous2)[:, 0, :, :]                          # ins_v: [64, 21, 3]
                ins_a = ins_p-inputs_previous[:, 0, :, :]-ins_v
                ins_v_dec = (inputs-inputs_previous)[:, 0, :, :]
            elif step==1:
                ins_p = pred_all[step-1]                                           # ins_p: [64, 21, 3]
                ins_v = (inputs-inputs_previous)[:, 0, :, :]                                   # ins_v: [64, 21, 3]
                ins_a = ins_p-inputs[:, 0, :, :]-ins_v # ins_v-(inputs-inputs_previous)[:, 0, :, :]
                ins_v_dec = pred_all[step-1]-inputs[:, 0, :, :]
            elif step==2:
                ins_p = pred_all[step-1]
                ins_v = pred_all[step-2]-inputs[:, 0, :, :]
                ins_a = ins_p-pred_all[step-2]-ins_v # ins_v-(pred_all[step-2]-inputs[:, 0, :, :])
                ins_v_dec = pred_all[step-1]-pred_all[step-2]
            else:
                ins_p = pred_all[step-1]
                ins_v = pred_all[step-2]-pred_all[step-3]
                ins_a = ins_p-pred_all[step-2]-ins_v # ins_v-(pred_all[step-2]-pred_all[step-3])
                ins_v_dec = pred_all[step-1]-pred_all[step-2]
            n = torch.randn(ins_p.size()).cuda()*0.0005
            ins = torch.cat((ins_p+n, ins_v, ins_a), dim=-1)
            pred_, hidden, res_ = self.step_forward(ins, hidden, step)
            pred_all.append(pred_)                                                 # [t, 64, 21, 3]
            res_all.append(res_)

        preds = torch.stack(pred_all, dim=1)                                       # [64, t, 21, 3]
        reses = torch.stack(res_all, dim=1)
        preds = preds * self.mask
       
        return preds.transpose(1, 2).contiguous()      # [64, 21, t, 3]
