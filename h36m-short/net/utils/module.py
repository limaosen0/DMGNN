import torch
import torch.nn as nn
import torch.nn.functional as F

from net.utils.operation import *


class St_gcn(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, t_kernel_size=1, stride=1,
                 dropout=0.5, residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0]-1)//2, 0)

        self.gcn = SpatialConv(in_channels, out_channels, kernel_size[1], t_kernel_size)
        self.tcn = nn.Sequential(nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
                                 nn.BatchNorm2d(out_channels),
                                 nn.Dropout(dropout, inplace=True))
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                                          nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A_skl):
        res = self.residual(x)
        x = self.gcn(x, A_skl)
        x = self.tcn(x) + res
        return self.relu(x)


class SpatialConv(nn.Module):

    def __init__(self, in_channels, out_channels, k_num, 
                 t_kernel_size=1, t_stride=1, t_padding=0,
                 t_dilation=1, bias=True):
        super().__init__()

        self.k_num = k_num
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels*(k_num),
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)

    def forward(self, x, A_skl):
        x = self.conv(x)                                                   # [64, 128, 49, 21]
        n, kc, t, v = x.size()                                             # n = 64(batchsize), kc = 128, t = 49, v = 21
        x = x.view(n, self.k_num,  kc//(self.k_num), t, v)             # [64, 4, 32, 49, 21]
        A_all = A_skl
        x = torch.einsum('nkctv, kvw->nctw', (x, A_all))
        return x.contiguous()


class DecodeGcn(nn.Module):
    
    def __init__(self, in_channels, out_channels, k_num,
                 kernel_size=1, stride=1, padding=0,
                 dilation=1, dropout=0.5, bias=True):
        super().__init__()

        self.k_num = k_num
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels*(k_num), 
                              kernel_size=kernel_size,
                              stride=stride, 
                              padding=padding, 
                              dilation=dilation, 
                              bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, A_skl):      # x: [64, 256, 21] = N, d, V
        x = self.conv(x)
        x = self.dropout(x)
        n, kc, v = x.size()
        x = x.view(n, (self.k_num), kc//(self.k_num), v)          # [64, 4, 256, 21]
        x = torch.einsum('nkcv,kvw->ncw', (x, A_skl))           # [64, 256, 21]
        return x.contiguous()



class AveargeJoint(nn.Module):

    def __init__(self):
        super().__init__()
        self.torso = [8,9]
        self.left_leg_up = [0,1]
        self.left_leg_down = [2,3]
        self.right_leg_up = [4,5]
        self.right_leg_down = [6,7]
        self.head = [10,11]
        self.left_arm_up = [12,13]
        self.left_arm_down = [14,15]
        self.right_arm_up = [16,17]
        self.right_arm_down = [18,19]
        
    def forward(self, x):
        x_torso = F.avg_pool2d(x[:, :, :, self.torso], kernel_size=(1, 2))                                              # [N, C, T, V=1]
        x_leftlegup = F.avg_pool2d(x[:, :, :, self.left_leg_up], kernel_size=(1, 2))                                # [N, C, T, V=1]
        x_leftlegdown = F.avg_pool2d(x[:, :, :, self.left_leg_down], kernel_size=(1, 2))                     # [N, C, T, V=1]
        x_rightlegup = F.avg_pool2d(x[:, :, :, self.right_leg_up], kernel_size=(1, 2))                        # [N, C, T, V=1]
        x_rightlegdown = F.avg_pool2d(x[:, :, :, self.right_leg_down], kernel_size=(1, 2))                   # [N, C, T, V=1]
        x_head = F.avg_pool2d(x[:, :, :, self.head], kernel_size=(1, 2))                                              # [N, C, T, V=1]
        x_leftarmup = F.avg_pool2d(x[:, :, :, self.left_arm_up], kernel_size=(1, 2))                            # [N, C, T, V=1]
        x_leftarmdown = F.avg_pool2d(x[:, :, :, self.left_arm_down], kernel_size=(1, 2))                 # [N, C, T, V=1]
        x_rightarmup = F.avg_pool2d(x[:, :, :, self.right_arm_up], kernel_size=(1, 2))                        # [N, C, T, V=1]
        x_rightarmdown = F.avg_pool2d(x[:, :, :, self.right_arm_down], kernel_size=(1, 2))               # [N, C, T, V=1]
        x_part = torch.cat((x_leftlegup, x_leftlegdown, x_rightlegup, x_rightlegdown, x_torso, x_head,  x_leftarmup, x_leftarmdown, x_rightarmup, x_rightarmdown), dim=-1)               # [N, C, T, V=1]), dim=-1)        # [N, C, T, 10]
        return x_part



class AveargePart(nn.Module):

    def __init__(self):
        super().__init__()
        self.torso = [8,9,10,11]
        self.left_leg = [0,1,2,3]
        self.right_leg = [4,5,6,7]
        self.left_arm = [12,13,14,15]
        self.right_arm = [16,17,18,19]
        
    def forward(self, x):
        x_torso = F.avg_pool2d(x[:, :, :, self.torso], kernel_size=(1, 4))                                              # [N, C, T, V=1]
        x_leftleg = F.avg_pool2d(x[:, :, :, self.left_leg], kernel_size=(1, 4))                                # [N, C, T, V=1]
        x_rightleg = F.avg_pool2d(x[:, :, :, self.right_leg], kernel_size=(1, 4))                        # [N, C, T, V=1]
        x_leftarm = F.avg_pool2d(x[:, :, :, self.left_arm], kernel_size=(1, 4))                            # [N, C, T, V=1]
        x_rightarm = F.avg_pool2d(x[:, :, :, self.right_arm], kernel_size=(1, 4))                        # [N, C, T, V=1]
        x_body = torch.cat((x_leftleg, x_rightleg, x_torso, x_leftarm, x_rightarm), dim=-1)               # [N, C, T, V=1]), dim=-1)        # [N, C, T, 10]
        return x_body



class S1_to_S2(nn.Module):

    def __init__(self, n_j1, n_j2, n_p1, n_p2, t_kernel, t_stride, t_padding):
        super().__init__()

        self.embed_s1 = S1AttInform(n_j1, n_j2, t_stride[1], t_kernel, t_padding, drop=0.2, nmp=True)
        self.embed_s2 = S2AttInform(n_p1, n_p2, t_stride[1], t_kernel, t_padding, drop=0.2, nmp=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_s1, x_s2, relrec_s1, relsend_s1, relrec_s2, relsend_s2):                                                           # x: [64, 3, 49, 21]
        N, d, T, V = x_s1.size()
        N, d, T, W = x_s2.size()

        x_s1_att = self.embed_s1(x_s1, relrec_s1, relsend_s1)                                                                          # [64, 21, 784]
        x_s2_att = self.embed_s2(x_s2, relrec_s2, relsend_s2)
        Att = self.softmax(torch.matmul(x_s1_att, x_s2_att.permute(0,2,1)).permute(0,2,1))                                 # [64, 10, 21]

        x_s1 = x_s1.permute(0,3,2,1).contiguous().view(N,V,-1)                                                     # [64, 21, 49, 3] -> [64, 21, 147]
        x_s2_glb = torch.einsum('nwv, nvd->nwd', (Att, x_s1))                                                     # [64, 10, 147]
        x_s2_glb = x_s2_glb.contiguous().view(N, W, -1, d).permute(0,3,2,1)                                            # [64, 10, 784] -> [64, 10, 49, 16] -> [64, 3, 49, 10]

        return x_s2_glb


class S2_to_S3(nn.Module):

    def __init__(self, n_p1, n_p2, n_b1, n_b2, t_kernel, t_stride, t_padding):
        super().__init__()

        self.embed_s2 = S2AttInform(n_p1, n_p2, t_stride[1], t_kernel, t_padding, drop=0.2, nmp=True)
        self.embed_s3 = S3AttInform(n_b1, n_b2, t_stride[1], t_kernel, t_padding, drop=0.2, nmp=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_s2, x_s3, relrec_s2, relsend_s2, relrec_s3, relsend_s3):
        N, d, T, V = x_s2.size()
        N, d, T, W = x_s3.size()

        x_s2_att = self.embed_s2(x_s2, relrec_s2, relsend_s2)                                                        # [64, 21, 784]
        x_s3_att = self.embed_s3(x_s3, relrec_s3, relsend_s3)
        Att = self.softmax(torch.matmul(x_s2_att, x_s3_att.permute(0,2,1)).permute(0,2,1))                       # [64, 10, 21]

        x_s2 = x_s2.permute(0,3,2,1).contiguous().view(N,V,-1)                                                      # [64, 21, 49, 3] -> [64, 21, 147]
        x_s3_glb = torch.einsum('nwv, nvd->nwd', (Att, x_s2))                                                    # [64, 10, 147]
        x_s3_glb = x_s3_glb.contiguous().view(N, W, -1, d).permute(0,3,2,1)                                         # [64, 10, 784] -> [64, 10, 49, 16] -> [64, 3, 49, 10]
        
        return x_s3_glb
        

class S2_to_S1(nn.Module):

    def __init__(self, n_p1, n_p2, n_j1, n_j2, t_kernel, t_stride, t_padding):
        super().__init__()

        self.embed_s2 = S2AttInform(n_p1, n_p2, t_stride[1], t_kernel, t_padding, drop=0.2, nmp=True)
        self.embed_s1 = S1AttInform(n_j1, n_j2, t_stride[1], t_kernel, t_padding, drop=0.2, nmp=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_s2, x_s1, relrec_s2, relsend_s2, relrec_s1, relsend_s1):
        N, d, T, W = x_s2.size()
        N, d, T, V = x_s1.size()

        x_s2_att = self.embed_s2(x_s2, relrec_s2, relsend_s2)
        x_s1_att = self.embed_s1(x_s1, relrec_s1, relsend_s1)
        Att = self.softmax(torch.matmul(x_s2_att, x_s1_att.permute(0,2,1)).permute(0,2,1))

        x_s2 = x_s2.permute(0,3,2,1).contiguous().view(N,W,-1)                                                     # [64, 21, 49, 3] -> [64, 21, 147]
        x_s1_glb = torch.einsum('nvw, nwd->nvd', (Att, x_s2))                                                     # [64, 10, 147]
        x_s1_glb = x_s1_glb.contiguous().view(N, V, -1, d).permute(0,3,2,1)                                            # [64, 10, 784] -> [64, 10, 49, 16] -> [64, 3, 49, 10]
        
        return x_s1_glb


class S3_to_S2(nn.Module):

    def __init__(self, n_b1, n_b2, n_p1, n_p2, t_kernel, t_stride, t_padding):
        super().__init__()

        self.embed_s3 = S3AttInform(n_b1, n_b2, t_stride[1], t_kernel, t_padding, drop=0.2, nmp=True)
        self.embed_s2 = S2AttInform(n_p1, n_p2, t_stride[1], t_kernel, t_padding, drop=0.2, nmp=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_s3, x_s2, relrec_s3, relsend_s3, relrec_s2, relsend_s2):
        N, d, T, W = x_s3.size()
        N, d, T, V = x_s2.size()

        x_s3_att = self.embed_s3(x_s3, relrec_s3, relsend_s3)
        x_s2_att = self.embed_s2(x_s2, relrec_s2, relsend_s2)
        Att = self.softmax(torch.matmul(x_s3_att, x_s2_att.permute(0,2,1)).permute(0,2,1))

        x_s3 = x_s3.permute(0,3,2,1).contiguous().view(N,W,-1)                                                     # [64, 21, 49, 3] -> [64, 21, 147]
        x_s2_glb = torch.einsum('nvw, nwd->nvd', (Att, x_s3))                                                     # [64, 10, 147]
        x_s2_glb = x_s2_glb.contiguous().view(N, V, -1, d).permute(0,3,2,1)                                            # [64, 10, 784] -> [64, 10, 49, 16] -> [64, 3, 49, 10]
        
        return x_s2_glb
