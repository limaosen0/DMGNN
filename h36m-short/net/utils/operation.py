import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp_JpTrans(nn.Module):

    def __init__(self, n_in, n_hid, n_out, do_prob=0.5, out_act=True):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid+n_in, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout = nn.Dropout(p=do_prob)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.init_weights()
        self.out_act = out_act

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, x):
        x_skip = x
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(torch.cat((x,x_skip),-1))
        x = self.batch_norm(x)
        x = self.leaky_relu(x) if self.out_act==True else x
        return x


class PartLocalInform(nn.Module):

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

    def forward(self, part):
        N, d, T, w = part.size()  # [64, 256, 7, 10]
        x = part.new_zeros((N, d, T, 20))

        x[:,:,:,self.left_leg_up] = torch.cat((part[:,:,:,0].unsqueeze(-1), part[:,:,:,0].unsqueeze(-1)),-1)
        x[:,:,:,self.left_leg_down] = torch.cat((part[:,:,:,1].unsqueeze(-1), part[:,:,:,1].unsqueeze(-1)),-1)
        x[:,:,:,self.right_leg_up] = torch.cat((part[:,:,:,2].unsqueeze(-1), part[:,:,:,2].unsqueeze(-1)),-1)
        x[:,:,:,self.right_leg_down] = torch.cat((part[:,:,:,3].unsqueeze(-1), part[:,:,:,3].unsqueeze(-1)),-1)
        x[:,:,:,self.torso] = torch.cat((part[:,:,:,4].unsqueeze(-1), part[:,:,:,4].unsqueeze(-1)),-1)
        x[:,:,:,self.head] = torch.cat((part[:,:,:,5].unsqueeze(-1), part[:,:,:,5].unsqueeze(-1)),-1)
        x[:,:,:,self.left_arm_up] = torch.cat((part[:,:,:,6].unsqueeze(-1),part[:,:,:,6].unsqueeze(-1)),-1)
        x[:,:,:,self.left_arm_down] = torch.cat((part[:,:,:,7].unsqueeze(-1),part[:,:,:,7].unsqueeze(-1)),-1)
        x[:,:,:,self.right_arm_up] = torch.cat((part[:,:,:,8].unsqueeze(-1),part[:,:,:,8].unsqueeze(-1)),-1)
        x[:,:,:,self.right_arm_down] = torch.cat((part[:,:,:,9].unsqueeze(-1),part[:,:,:,9].unsqueeze(-1)),-1)

        return x


class BodyLocalInform(nn.Module):

    def __init__(self):
        super().__init__()

        self.torso = [8,9,10,11]
        self.left_leg = [0,1,2,3]
        self.right_leg = [4,5,6,7]
        self.left_arm = [12,13,14,15]
        self.right_arm = [16,17,18,19]

    def forward(self, body):
        N, d, T, w = body.size()  # [64, 256, 7, 10]
        x = body.new_zeros((N, d, T, 20))

        x[:,:,:,self.left_leg] = torch.cat((body[:,:,:,0:1], body[:,:,:,0:1], body[:,:,:,0:1], body[:,:,:,0:1]),-1)
        x[:,:,:,self.right_leg] = torch.cat((body[:,:,:,1:2], body[:,:,:,1:2], body[:,:,:,1:2], body[:,:,:,2:3]),-1)
        x[:,:,:,self.torso] = torch.cat((body[:,:,:,2:3], body[:,:,:,2:3], body[:,:,:,2:3], body[:,:,:,2:3]),-1)
        x[:,:,:,self.left_arm] = torch.cat((body[:,:,:,3:4], body[:,:,:,3:4], body[:,:,:,3:4], body[:,:,:,3:4]),-1)
        x[:,:,:,self.right_arm] = torch.cat((body[:,:,:,4:5], body[:,:,:,4:5], body[:,:,:,4:5], body[:,:,:,4:5]),-1)

        return x
        

def node2edge(x, rel_rec, rel_send):
    receivers = torch.matmul(rel_rec, x)
    senders = torch.matmul(rel_send, x)
    distance = receivers - senders
    edges = torch.cat([receivers, distance], dim=2)
    return edges

def edge2node_mean(x, rel_rec, rel_send):
    incoming = torch.matmul(rel_rec.t(), x)
    nodes = incoming/incoming.size(1)
    return nodes


class S1AttInform(nn.Module):

    def __init__(self, n_joint1, n_joint2, t_stride, t_kernel, t_padding, drop=0.2, layer1=False, nmp=False):
        super().__init__()
        self.time_conv = nn.Sequential(nn.Conv2d(n_joint1, n_joint1, kernel_size=(t_kernel, 1), stride=(t_stride, 1), padding=(t_padding, 0), bias=True),
                                                 nn.BatchNorm2d(n_joint1),
                                                 nn.Dropout(drop, inplace=True))
        if nmp==True:
            self.mlp1 = Mlp_JpTrans(n_joint2[0], n_joint2[1], n_joint2[1], drop)
            self.mlp2 = Mlp_JpTrans(n_joint2[1]*2, n_joint2[1], n_joint2[1], drop)
            self.mlp3 = Mlp_JpTrans(n_joint2[1]*2, n_joint2[1], n_joint2[1], drop, out_act=False)
        else:
            self.mlp1 = Mlp_JpTrans(n_joint2[0], n_joint2[1], n_joint2[1], drop, out_act=False)
        self.init_weights()
        self.layer1 = layer1
        self.nmp = nmp

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, x, rel_rec, rel_send):                                           # x: [64, 32, 49, 10]
        N, D, T, V = x.size()
        x_ = x if self.layer1==True else self.time_conv(x)
        x_ = x_.permute(0,2,3,1)
        x_ = x_.contiguous().view(N,V,-1)
        x_node = self.mlp1(x_)
        if self.nmp==True:
            x_node_skip = x_node
            x_edge = node2edge(x_node, rel_rec, rel_send)                               # [64, 420, 512]
            x_edge = self.mlp2(x_edge)                                                  # [64, 420, 256]
            x_node = edge2node_mean(x_edge, rel_rec, rel_send)                          # [64, 21, 256]
            x_node = torch.cat((x_node, x_node_skip), -1)                               # [64, 21, 512]
            x_node = self.mlp3(x_node)                                                  # [64, 21, 256]
        return x_node


class S3AttInform(nn.Module):

    def __init__(self, n_body1, n_body2, t_stride, t_kernel, t_padding, drop=0.2, layer1=False, nmp=True):
        super().__init__()
        self.time_conv = nn.Sequential(nn.Conv2d(n_body1, n_body1, kernel_size=(t_kernel, 1), stride=(t_stride, 1), padding=(t_padding, 0), bias=True),
                                                 nn.BatchNorm2d(n_body1),
                                                 nn.Dropout(drop, inplace=True))
        if nmp==True:
            self.mlp1 = Mlp_JpTrans(n_body2[0], n_body2[1], n_body2[1], drop)
            self.mlp2 = Mlp_JpTrans(n_body2[1]*2, n_body2[1], n_body2[1], drop)
            self.mlp3 = Mlp_JpTrans(n_body2[1]*2, n_body2[1], n_body2[1], drop, out_act=False)
        else:
            self.mlp1 = Mlp_JpTrans(n_body2[0], n_body2[1], n_body2[1], drop, out_act=False)
        self.init_weights()
        self.layer1 = layer1
        self.nmp = nmp

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, x, rel_rec, rel_send):                                           # x: [64, 32, 49, 10]
        N, D, T, V = x.size()
        x_ = x if self.layer1==True else self.time_conv(x)
        x_ = x_.permute(0,2,3,1)
        x_ = x_.contiguous().view(N,V,-1)

        x_node = self.mlp1(x_)                                                          # [64, 21, 256]
        if self.nmp==True:
            x_node_skip = x_node
            x_edge = node2edge(x_node, rel_rec, rel_send)                              # [64, 420, 512]
            x_edge = self.mlp2(x_edge)                                                  # [64, 420, 256]
            x_node = edge2node_mean(x_edge, rel_rec, rel_send)                   # [64, 21, 256]
            x_node = torch.cat((x_node, x_node_skip), -1)                  # [64, 21, 512]
            x_node = self.mlp3(x_node)                                                 # [64, 21, 256]
        return x_node
        

class S2AttInform(nn.Module):

    def __init__(self, n_part1, n_part2, t_stride, t_kernel, t_padding, drop=0.2, layer1=False, nmp=False):
        super().__init__()
        self.time_conv = nn.Sequential(nn.Conv2d(n_part1, n_part1, kernel_size=(t_kernel, 1), stride=(t_stride, 1), padding=(t_padding, 0), bias=True),
                                                 nn.BatchNorm2d(n_part1),
                                                 nn.Dropout(drop, inplace=True))
        if nmp==True:
            self.mlp1 = Mlp_JpTrans(n_part2[0], n_part2[1], n_part2[1], drop)
            self.mlp2 = Mlp_JpTrans(n_part2[1]*2, n_part2[1], n_part2[1], drop)
            self.mlp3 = Mlp_JpTrans(n_part2[1]*2, n_part2[1], n_part2[1], drop, out_act=False)
        else:
            self.mlp1 = Mlp_JpTrans(n_part2[0], n_part2[1], n_part2[1], drop, out_act=False)
        self.init_weights()
        self.layer1 = layer1
        self.nmp = nmp

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, x, rel_rec, rel_send):                                           # x: [64, 32, 49, 10]
        N, D, T, V = x.size()
        x_ = x if self.layer1==True else self.time_conv(x)
        x_ = x_.permute(0,2,3,1)
        x_ = x_.contiguous().view(N,V,-1)
        x_node = self.mlp1(x_)
        if self.nmp==True:
            x_node_skip = x_node
            x_edge = node2edge(x_node, rel_rec, rel_send)                              # [64, 420, 512]
            x_edge = self.mlp2(x_edge)                                                  # [64, 420, 256]
            x_node = edge2node_mean(x_edge, rel_rec, rel_send)                   # [64, 21, 256]
            x_node = torch.cat((x_node, x_node_skip), -1)                  # [64, 21, 512]
            x_node = self.mlp3(x_node)                                                 # [64, 21, 256]
        return x_node