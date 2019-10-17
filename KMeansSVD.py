#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 17:22:28 2019

@author: haijiesong
"""


import numpy as np
import math
from numpy import linalg as la
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
from pandas.core.common import SettingWithCopyWarning

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)


class KMeansSVD(object):
    def __init__(self):
        self.base_data = pd.DataFrame()
        self.result_data = pd.DataFrame()
        self.normalized_data = pd.DataFrame()
        self.dr_data = pd.DataFrame()
        self.km_output_data = pd.DataFrame()
        self.u = None
        self.s = None
        self.v = None
        self.cum_var_fig = plt.figure(figsize=(7, 5))
        self.km_fig = plt.figure(figsize=(7, 5))
        self.result_fig = plt.figure(figsize=(14, 7))
        self.feature_num = 0
        self.class_number = 0
        self.mean_distortions = []
        self.cum_var = []
        pass

    def rebuild_data(self, input_data, drop_min=None, drop_max=None):
        """
        rebuild data method. Convert the number to a percentage
        :param data: a data frame. The first column must be id, the other columns record the performance of the study
                     unit in each axon.
        :param drop_min: default None. You can set this limit if you want to drop the event sum less than the number.
        :param drop_max: default None. You can set this limit if you want to drop the event sum greater than the number.
        """
        if type(input_data) != pd.core.frame.DataFrame:
            raise BaseException('input data must be pandas.core.frame.DataFrame')
        old_columns = input_data.columns[1:]
        self.base_data = input_data.copy(deep=True)
        data = input_data.copy(deep=True)
        data['all'] = data[old_columns].apply(lambda x: x.sum(), axis=1)
        if drop_min is not None:
            data = data[data['all'] > drop_min]
        if drop_max is not None:
            data = data[data['all'] < drop_max]
        for column in old_columns:
            data[column] = data[column] / data['all']
        self.normalized_data = data.iloc[:, 0: -1].reset_index(drop=True)
        self.normalized_data.columns = ['id'] + list(old_columns)
        self.base_data.columns = ['id'] + list(old_columns)

    def replace_id(self, data):
        self.base_data = data.copy(deep=True)
        self.base_data.columns = ['id'] + list(data.columns[1:])
        self.normalized_data = self.base_data.copy(deep=True)

    def cal_number_of_feature(self, min_explanatory_ratio: float = 0.99, set_feature_number: int = -1):
        """

        :param min_explanatory_ratio: float, The minimum interpretation ratio you can accept
        :param set_feature_number: you can set the feature number if you want
        """

        if min_explanatory_ratio >= 1:
            raise BaseException('min explanatory ratio must less then 1')
        svd_model_data = self.normalized_data.iloc[:, 1:].values
        self.u, self.s, self.v = la.svd(svd_model_data, full_matrices=False)
        if set_feature_number == -1:
            for i in range(len(self.s)):
                if i == 0:
                    self.cum_var.append(self.s[i] ** 2)
                else:
                    self.cum_var.append(self.cum_var[-1] + self.s[i] ** 2)
            cum_var_percentage = (self.cum_var / self.cum_var[-1])
            ax = self.cum_var_fig.add_subplot(111)
            ax.plot(range(len(cum_var_percentage)), cum_var_percentage, 'bx-')
            ax.set_ylim(math.floor(cum_var_percentage.min() * 10) / 10, 1.1)
            ax.set_yticks([i / 10 for i in list(range(math.floor(cum_var_percentage.min() * 10), 11, 1))])
            ax.set_yticklabels(
                ['{}%'.format(i * 10) for i in list(range(math.floor(cum_var_percentage.min() * 10), 11, 1))])
            ax.set_xlabel('feature number')
            ax.set_ylabel('explanatory ratio')
            for i in cum_var_percentage:
                if i < min_explanatory_ratio:
                    self.feature_num += 1
        else:
            self.feature_num = set_feature_number

    def get_dr_data(self, verbose=True):
        """
        Get dimensionality reduction data
        """
        km_model_data = self.normalized_data['id']
        corr = [[] for i in range(self.feature_num)]
        svd_model_data = self.normalized_data.iloc[:, 1:].values
        for i in range(len(svd_model_data)):
            if verbose:
                if i % int(len(svd_model_data) / 10) == 0:
                    if i == 0:
                        print('start build dimensionality reduction data')
                    else:
                        print("已经完成{}%".format(str(round(i / len(svd_model_data), 2) * 100)))
            for j in range(self.feature_num):
                corr[j].append(np.corrcoef(svd_model_data[i], self.v[j, :])[0][1])
        for i in range(self.feature_num):
            km_model_data = pd.concat([km_model_data, pd.Series(corr[i], index=km_model_data.index)], axis=1)
        km_model_data.columns = ['id'] + ['v{}'.format(i) for i in range(self.feature_num)]
        self.dr_data = km_model_data

    def train_km(self, min_mean_distortions=0.1, min_class: int = 1, max_class: int = 20, set_class: int = -1, vmax=1):
        x_train = self.dr_data.iloc[:, 1:].values
        if set_class == -1:
            if min_class <= 0:
                raise BaseException('min class must greater than 0')
            for i in range(min_class, max_class + 1):
                km = KMeans(n_clusters=i)
                km.fit(x_train)
                self.mean_distortions.append(
                    sum(np.min(cdist(x_train, km.cluster_centers_, "euclidean"), axis=1)) / x_train.shape[0])
            ax = self.km_fig.add_subplot(111)
            ax.plot(range(min_class, max_class + 1), self.mean_distortions, 'bx-')
            ax.set_xlabel('class number')
            ax.set_ylabel('mean distortions')
            ax.set_title('KMeans result')
            self.class_number = min_class
            for i in (range(0, max_class - min_class)):
                if self.mean_distortions[i] > min_mean_distortions:
                    self.class_number += 1
        else:
            self.class_number = set_class
        km = KMeans(n_clusters=self.class_number).fit(x_train)
        km_group_out = km.labels_
        self.km_output_data = pd.concat([self.dr_data['id'], pd.Series(km_group_out, index=self.dr_data.index)], axis=1)
        self.km_output_data.columns = ['id', 'group_id']
        self.km_output_data = pd.merge(self.km_output_data, self.normalized_data, on='id')
        self.base_data['all'] = self.base_data.iloc[:, 1:].apply(lambda x: x.sum(), axis=1)
        self.km_output_data = pd.merge(self.km_output_data, self.base_data[['id', 'all']], on='id')
        group_num = self.km_output_data[['id', 'group_id']].groupby('group_id').count().reset_index()
        group_num.columns = ['group_id', 'count_of_id_in_a_group']
        self.km_output_data = pd.merge(self.km_output_data, group_num, on='group_id', how='left')
        self.km_output_data.sort_values(by=['count_of_id_in_a_group', 'group_id', 'all'], inplace=True)
        self.km_output_data.reset_index(drop=True)
        temp = self.km_output_data[['id', 'group_id', 'count_of_id_in_a_group']]
        temp['sort_group'] = temp.apply(lambda x: x[2] * 1000 + x[1], axis=1)
        group_num = temp[['id', 'sort_group']].groupby(
            'sort_group').count().cumsum().reset_index(drop=True)
        ax1 = self.result_fig.add_subplot(121)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax1.imshow(self.km_output_data.iloc[:, 2:-2], aspect='auto', cmap=mpl.cm.viridis, vmax=vmax)
        for i in range(self.class_number - 1):
            ax1.hlines(y=int(group_num['id'][i]) - 1, xmin=-1, xmax=self.normalized_data.shape[1], color='white',
                       linewidth=2)
        ax1.set_xlim(-1, self.normalized_data.shape[1] - 2)
        ax1.set_yticks([0] + [group_num['id'][i] for i in range(self.class_number)])
        ax1.set_yticklabels([0] + [group_num['id'][i] for i in range(self.class_number)])
        plt.colorbar(im, cax=cax, orientation='vertical')

        ax2 = self.result_fig.add_subplot(122)
        data = pd.merge(self.base_data, self.km_output_data[['id', 'group_id']], on='id', how='left')
        self.result_data = data.copy(deep=True)
        distribution = data.groupby(by='group_id').mean()
        distribution[distribution.columns[1: -1]].T.plot(ax=ax2)
        for tick in ax2.get_xticklabels():
            tick.set_rotation(15)

    def fit(self, data, if_normalized=True, drop_min=None, drop_max=None, min_explanatory_ratio: float = 0.99,
            set_feature_number: int = -1, min_mean_distortions=0.1, min_class: int = 1, max_class: int = 20,
            set_class: int = -1, vmax=1, verbose=True):
        if if_normalized:
            self.rebuild_data(data, drop_min=drop_min, drop_max=drop_max)
        else:
            self.replace_id(data)
        self.cal_number_of_feature(min_explanatory_ratio=min_explanatory_ratio, set_feature_number=set_feature_number)
        self.get_dr_data(verbose=verbose)

        self.train_km(min_mean_distortions=min_mean_distortions, min_class=min_class, max_class=max_class,
                      set_class=set_class, vmax=vmax)
