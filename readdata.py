#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import platform as plat
import os

import numpy as np
from general_function.file_wav import *
from general_function.file_dict import *

import random


class DataSpeech():

    def __init__(self, path, type, LoadToMem=False, MemWavCount=10000):
        '''
        初始化
        参数：
            path：数据存放位置根目录
        '''

        system_type = plat.system()  # 由于不同的系统的文件路径表示不一样，需要进行判断

        self.datapath = path  # 数据存放位置根目录
        self.type = type  # 数据类型，分为三种：训练集(train)、验证集(dev)、测试集(test)

        self.dic_wavlist_thchs30 = {}
        self.dic_symbollist_thchs30 = {}
        self.list_symbolnames_thchs30 = []
        # self.dic_wavlist_stcmds = {}
        # self.dic_symbollist_stcmds = {}

        self.SymbolNum = 0  # 记录拼音符号数量
        self.list_symbol = self.GetSymbolList()  # 全部汉语拼音符号列表
        self.list_wavnum = []  # wav文件标记列表
        self.list_symbolnum = []  # symbol标记列表

        self.MAX_LABEL_LEN = 96

        self.DataNum = 0  # 记录数据量
        self.LoadDataList()

        self.wavs_data = []
        self.LoadToMem = LoadToMem
        self.MemWavCount = MemWavCount
        pass

    def LoadDataList(self):
        '''
        加载用于计算的数据列表
        参数：
            type：选取的数据集类型
                train 训练集
                dev 开发集
                test 测试集
        '''
        # 读取数据列表，wav文件列表和其对应的符号列表
        self.dic_wavlist_thchs30, self.dic_symbollist_thchs30, self.list_symbolnames_thchs30 = self.get_wavname_and_labels()
        # self.dic_wavlist_stcmds, self.list_wavnum_stcmds = get_wav_list(
        # self.datapath + filename_wavlist_stcmds)

        # self.dic_symbollist_thchs30, self.list_symbolname_thchs30 = self.get_wavname_and_symbols()
        # self.dic_symbollist_stcmds, self.list_symbolnum_stcmds = get_wav_symbol(
        # self.datapath + filename_symbollist_stcmds)
        self.DataNum = self.GetDataNum()

    def get_wavname_and_labels(self):
        '''
        从指定文件夹读取所有文件名和文件路径对应的wavepath和音素标签
        return dic_wavlist_thchs30: dict of wavename to filepath
        return dic_symbollist_thchs30: dict of wavename to symbol list
        return list_wavname_thchs30: list of all wavnames
        '''

        dirname = os.path.join(self.datapath, 'data_thchs30', self.type)
        all_files = os.listdir(dirname)
        dic_wavlist_thchs30 = {}
        dic_symbollist_thchs30 = {}
        list_wavname_thchs30 = []
        max_len = 0
        for filename in all_files:
            basename, ext = os.path.splitext(filename)
            if ext == '.wav':
                dic_wavlist_thchs30[basename] = os.path.join(dirname, filename)
                list_wavname_thchs30.append(basename)

                with open(os.path.join(dirname, filename+'.trn')) as f:
                    symbol_path = f.read().strip()
                with open(os.path.join(dirname, symbol_path)) as f:
                    symbols_chinese_charactor = f.readline().strip()
                    symbols_chinese_pinyin = f.readline().strip()
                    symbols_phone = f.readline().strip().split()
                    if len(symbols_phone) > max_len:
                        max_len = len(symbols_phone)
                dic_symbollist_thchs30[basename] = symbols_phone
        return dic_wavlist_thchs30, dic_symbollist_thchs30, list_wavname_thchs30

    def GetDataNum(self):
        '''
        获取数据的数量
        当wav数量和symbol数量一致的时候返回正确的值，否则返回-1，代表出错。
        '''
        num_wavlist_thchs30 = len(self.dic_wavlist_thchs30)
        num_symbollist_thchs30 = len(self.dic_symbollist_thchs30)
        # num_wavlist_stcmds = len(self.dic_wavlist_stcmds)
        # num_symbollist_stcmds = len(self.dic_symbollist_stcmds)
        if(num_wavlist_thchs30 == num_symbollist_thchs30):
            DataNum = num_wavlist_thchs30
        else:
            DataNum = -1
        return DataNum

    def GetData(self, n_start, n_amount=1):
        '''
        读取数据，返回神经网络输入值和输出值矩阵(可直接用于神经网络训练的那种)
        参数：
            n_start：从编号为n_start数据开始选取数据
            n_amount：选取的数据数量，默认为1，即一次一个wav文件
        返回：
            三个包含wav特征矩阵的神经网络输入值，和一个标定的类别矩阵神经网络输出值
        '''
        filename = self.dic_wavlist_thchs30[self.list_symbolnames_thchs30[n_start]]
        list_symbols_of_the_wave = self.dic_symbollist_thchs30[self.list_symbolnames_thchs30[n_start]]

        wavsignal, fs = read_wav_data(filename)

        # 获取音素标签
        labels_of_a_wave = []
        for i in list_symbols_of_the_wave:
            if('' != i):
                n = self.SymbolToNum(i)
                # v=self.NumToVector(n)
                # feat_out.append(v)
                labels_of_a_wave.append(n)

        # 获取输入特征
        # data_input = GetFrequencyFeature3(wavsignal,fs)
        data_input = getSTFTFeature(wavsignal, fs)
        data_input = np.array(data_input)
        data_label = np.array(labels_of_a_wave)
        return data_input, data_label

    def data_genetator(self, batch_size=32, audio_length=1600):
        '''
        数据生成器函数，用于Keras的generator_fit训练
        batch_size: 一次产生的数据量
        需要再修改。。。
        '''

        #labels = []
        # for i in range(0,batch_size):
        #    #input_length.append([1500])
        #    labels.append([0.0])

        #labels = np.array(labels, dtype = np.float)
        labels = np.zeros((batch_size, 1), dtype=np.int32)
        # print(input_length,len(input_length))

        while True:
            X = np.zeros((batch_size, audio_length, 200), dtype=np.float)
            #y = np.zeros((batch_size, 64, self.SymbolNum), dtype=np.int16)
            y = np.zeros((batch_size, self.MAX_LABEL_LEN), dtype=np.int32)

            #generator = ImageCaptcha(width=width, height=height)
            input_length = []
            label_length = []

            for i in range(batch_size):
                ran_num = random.randint(0, self.DataNum - 1)  # 获取一个随机数
                data_input, data_labels = self.GetData(ran_num)  # 通过随机数取一个数据
                # data_input, data_labels = self.GetData((ran_num + i) % self.DataNum)  # 从随机数开始连续向后取一定数量数据

                # 关于下面这一行取整除以8 并加8的余数，在实际中如果遇到报错，可尝试只在有余数时+1，没有余数时+0，或者干脆都不加，只留整除
                input_length.append(
                    data_input.shape[0] // 8 + data_input.shape[0] % 8)
                #print(data_input, data_labels)
                # print('data_input长度:',len(data_input))

                X[i, 0:len(data_input)] = data_input
                # print('data_labels长度:',len(data_labels))
                # print(data_labels)
                y[i, 0:len(data_labels)] = data_labels
                # print(i,y[i].shape)
                #y[i] = y[i].T
                # print(i,y[i].shape)
                label_length.append([len(data_labels)])

            label_length = np.matrix(label_length)
            input_length = np.array([input_length]).T
            #input_length = np.array(input_length)
            # print('input_length:\n',input_length)
            #X=X.reshape(batch_size, audio_length, 200, 1)
            # print(X)
            yield [X, y, input_length, label_length], labels
        pass

    def GetSymbolList(self):
        '''
        加载音素符号列表，用于标记符号
        返回一个列表list类型变量
        '''
        with open('lexicon.txt', 'r', encoding='utf8') as f:
            text = f.read()
        text_lines = text.split()
        list_symbol = text_lines
        list_symbol.append('_')
        self.SymbolNum = len(list_symbol)
        print(self.SymbolNum)
        return list_symbol

    def GetSymbolNum(self):
        '''
        获取音素符号数量
        '''
        return len(self.list_symbol)

    def SymbolToNum(self, symbol):
        '''
        符号转为数字
        '''
        if(symbol != ''):
            return self.list_symbol.index(symbol)
        return self.SymbolNum

    def NumToVector(self, num):
        '''
        数字转为对应的向量
        '''
        v = np.zeros((len(self.list_symbol)), dtype=np.int32)
        v[num] = 1
        return v


if(__name__ == '__main__'):
    # path='E:\\语音数据集'
    l = DataSpeech('dataset', 'test')
    a, b, c = l.get_wavname_and_labels()
    print(a[c[0]])
    # l.LoadDataList('train')
    # print(l.GetDataNum())
    # print(l.GetData(0))
    # aa=l.data_genetator()
    # for i in aa:
    # a,b=i
    # print(a,b)
    pass
