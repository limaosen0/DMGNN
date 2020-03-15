import sys
import argparse
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .io import IO
from .data_tools import *
# import data_tools as tools


class Processor(IO):

    def __init__(self, argv=None):

        self.load_arg(argv)
        self.init_environment()
        self.load_model()
        self.load_weights()
        self.gpu()
        self.load_data()
        self.load_optimizer()

    def init_environment(self):
        super().init_environment()
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)


    def load_optimizer(self):
        pass


    def load_data(self):
        if self.arg.debug==True:
            self.actions = define_actions('walking')
        else:
            self.actions = define_actions(self.arg.actions)
        self.train_dict, self.complete_train = load_data(self.arg.train_dir, self.actions)
        self.test_dict, self.complete_test = load_data(self.arg.test_dir, self.actions)
        self.data_mean, self.data_std, self.dim_ignore, self.dim_use, self.dim_zero, self.dim_nonzero = normalization_stats(self.complete_train)


    def show_epoch_info(self):
        for k, v in self.epoch_info.items():
            self.io.print_log('\t{}: {}'.format(k, v))
        if self.arg.pavi_log:
            self.io.log('train', self.meta_info['iter'], self.epoch_info)


    def show_iter_info(self):
        if self.meta_info['iter'] % self.arg.log_interval == 0:
            info ='\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)
            self.io.print_log(info)

            if self.arg.pavi_log:
                self.io.log('train', self.meta_info['iter'], self.iter_info)


    def train(self):
        for _ in range(100):
            self.iter_info['loss'] = 0
            self.show_iter_info()
            self.meta_info['iter'] += 1
        self.epoch_info['mean loss'] = 0
        self.show_epoch_info()


    def test(self):
        for _ in range(100):
            self.iter_info['loss'] = 1
            self.show_iter_info()
        self.epoch_info['mean loss'] = 1
        self.show_epoch_info()


    def start(self):
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))

        if self.arg.phase == 'train':
            self.MAE_tensor = np.zeros((self.arg.iter_num//self.arg.eval_interval, 8, 13))
            self.mask = torch.ones(25).to(self.dev)
            self.mask[10:] = 2
            for itr in range(self.arg.iter_num):
                self.train()
                if ((itr+1) % self.arg.save_interval==0) or (itr+1==self.arg.iter_num):
                    filename = 'iter{}_model.pt'.format(itr+1)
                    self.io.save_model(self.model, filename)
                if ((itr+1) % self.arg.eval_interval==0) or (itr+1==self.arg.iter_num):
                    if (itr+1) % self.arg.savemotion_interval ==0:
                        save_motion = True
                    else:
                        save_motion = False
                    self.io.print_log('eval Iteration: {}'.format(itr+1))
                    self.test(iter_time=itr//self.arg.eval_interval, save_motion=save_motion)
            self.MAE = self.MAE_tensor.min(axis=0)
            self.MAE[:,-1] = self.MAE.mean(axis=-1)*13/10.

            print_str = "{0: <16} |".format("milliseconds")
            for ms in [40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 560, 1000]:
                print_str = print_str + " {0:5d} |".format(ms)
            self.io.print_log(print_str)
            for idx, action in enumerate(self.actions):
                print_str = "{0: <16} |".format(action)
                for ms_idx, ms in enumerate([0,1,2,3,4,5,6,7,8,9,10,11]):
                    if self.arg.target_seq_len >= ms+1:
                        print_str = print_str + " {0:.3f} |".format(self.MAE[idx, ms])
                    else:
                        print_str = print_str + "   n/a |"
                self.io.print_log(print_str)

            # for act_num in range(8):
            #     print_str = str(self.MAE[act_num])
            #     self.io.print_log(print_str)

        elif self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.io.print_log('Model:   {}.'.format(self.arg.model))
            self.io.print_log('Weights: {}.'.format(self.arg.weights))
            self.io.print_log('Evaluation Start:')
            self.test(phase=True)


    @staticmethod
    def get_parser(add_help=False):

        parser = argparse.ArgumentParser( add_help=add_help, description='Base Processor')

        parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
        parser.add_argument('-c', '--config', default=None, help='path to the configuration file')

        # processor
        parser.add_argument('--phase', default='train', help='must be train or test')
        parser.add_argument('--save_result', type=str2bool, default=False, help='if ture, the output of the model will be stored')
        parser.add_argument('--iter_num', type=int, default=10000, help='stop training in which epoch')
        parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPUs or not')
        parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')

        # visulize and debug
        parser.add_argument('--log_interval', type=int, default=100, help='the interval for printing messages (#iteration)')
        parser.add_argument('--save_interval', type=int, default=200, help='the interval for storing models (#iteration)')
        parser.add_argument('--eval_interval', type=int, default=200, help='the interval for evaluating models (#iteration)')
        parser.add_argument('--savemotion_interval', type=int, default=2000000000000, help='the interval for evaluating models (#iteration)')
        parser.add_argument('--save_log', type=str2bool, default=True, help='save logging or not')
        parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')
        parser.add_argument('--pavi_log', type=str2bool, default=False, help='logging on pavi or not')

        # data loading
        parser.add_argument('--actions', default='all', help='data loader will be used')
        parser.add_argument('--train_dir', default='../data/cmu_mocap/train', help='data loader will be used')
        parser.add_argument('--test_dir', default='../data/cmu_mocap/test', help='data loader will be used')
        parser.add_argument('--sample_dir', default='../samples', help='save generated samples')
        parser.add_argument('--batch_size', type=int, default=64, help='which Top K accuracy will be shown')
        parser.add_argument('--source_seq_len', type=int, default=50, help='which Top K accuracy will be shown')
        parser.add_argument('--target_seq_len', type=int, default=25, help='which Top K accuracy will be shown')

        # model
        parser.add_argument('--model', default=None, help='the model will be used')
        parser.add_argument('--model_args', action=DictAction, default=dict(), help='the arguments of model')
        parser.add_argument('--edge_weighting', type=bool, default=True, help='Add edge importance weighting')
        parser.add_argument('--weights', default=None, help='the weights for network initialization')
        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')

        return parser