#
# Copyright (c) 2023 Naoki Masuyama (masuyama@omu.ac.jp)
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
#

from collections.abc import Iterable, Iterator

import networkx as nx
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


class CAplus(BaseEstimator):
    """ CIM-based Adaptive Resonance Theory plus (CA+)"""

    def __init__(
            self,
            G_=nx.Graph(),
            dim_=None,
            num_signal_=0,
            V_thres_=1.0,
            sigma_=None,
            n_clusters_=0,
            active_node_idx_=None,
            flag_set_lambda_=False,
            n_init_data_=10,
            n_active_nodes_=np.inf,
            div_mat_=None,
            div_threshold_=1.0e-6,
            # div_threshold_=0.01,
            div_lambda_=np.inf,
    ):

        self.G_ = G_  # network
        self.dim_ = dim_  # Number of variables in an instance
        self.num_signal_ = num_signal_  # Counter for training instances
        self.sigma_ = sigma_  # An estimated sigma for CIM
        self.V_thres_ = V_thres_  # Similarity threshold
        self.n_clusters_ = n_clusters_  # Number of clusters
        self.active_node_idx_ = active_node_idx_  # Indexes of active nodes
        self.flag_set_lambda_ = flag_set_lambda_  # Flag for setting \lambda
        self.n_init_data_ = n_init_data_  # Number of signals for initialization of sigma
        self.n_active_nodes_ = n_active_nodes_  # Number of buffer nodes for calculating \sigma
        self.div_mat_ = div_mat_  # A matrix for diversity via determinants
        self.div_threshold_ = div_threshold_  # A threshold for diversity via determinants
        self.div_lambda_ = div_lambda_  # \lambda determined by diversity via determinants

    def fit(self, x: np.ndarray):
        """
        train data in batch manner
        :param x: array-like or ndarray
        """
        self.initialization(x)

        for signal in x:
            self.input_signal(signal, x)  # training a network

        return self

    def predict(self, x: np.ndarray):
        """
        predict cluster index for each sample.
        :param x: array-like or ndarray
        :rtype list:
            cluster index for each sample.
        """

        self.labels_ = self.__labeling_sample_for_clustering(x)

        return self.labels_

    def initialization(self, x: np.ndarray):
        """
        Initialize parameters
        :param x: array-like or ndarray
        """
        # set graph
        if len(list(self.G_.nodes)) == 0:
            self.G_ = nx.Graph()

        # set dimension of x
        if self.dim_ is None:
            self.dim_ = x.shape[1]

    def input_signal(self, signal: np.ndarray, x: np.ndarray):
        """
        Input a new signal one by one, which means training in online manner.
        fit() calls __init__() before training, which means resetting the state. So the function does batch training.
        :param signal: A new input signal
        :param x: array-like or ndarray
            data
        """

        if self.num_signal_ == x.shape[0]:
            self.num_signal_ = 1
        else:
            self.num_signal_ += 1

        if self.num_signal_ == 1 and self.G_.number_of_nodes() == 0:
            self.__calculate_sigma_by_active_nodes(x[0:self.n_init_data_, :], None)  # set init \sigma

        if self.flag_set_lambda_ is False or self.G_.number_of_nodes() < self.n_active_nodes_:
            new_node_idx = self.__add_node(signal)
            self.__update_active_node_index(signal, new_node_idx)

            # setup initial n_active_nodes_, div_lambda_, and V_thres_
            self.__setup_init_params()

        else:
            node_list, cim = self.__calculate_cim(signal)
            s1_idx, s1_cim, s2_idx, s2_cim = self.__find_nearest_node(node_list, cim)

            if self.V_thres_ < s1_cim or self.G_.number_of_nodes() < self.n_active_nodes_:
                new_node_idx = self.__add_node(signal)
                self.__update_active_node_index(signal, new_node_idx)
                self.__calculate_sigma_by_active_nodes(None, new_node_idx)
            else:
                self.__update_s1_node(s1_idx, signal)
                self.__update_active_node_index(signal, s1_idx)

                if self.V_thres_ >= s2_cim:
                    self.__update_s2_node(s2_idx, signal)

    def __setup_init_params(self):
        """
        Initialize n_active_nodes_, div_lambda_, and V_thres_.
        """

        if self.G_.number_of_nodes() >= 2 and self.flag_set_lambda_ is False:
            # calculate n_active_nodes_ and div_lambda_ based on diversity via determinants
            self.__setup_n_active_nodes_and_div_lambda()

        if self.G_.number_of_nodes() == self.n_active_nodes_:
            self.flag_set_lambda_ = True

            # estimate \sigma by using active nodes
            self.__calculate_sigma_by_active_nodes()

            # overwrite \sigma of all nodes
            [nx.set_node_attributes(self.G_, {k: {'sigma': self.sigma_}}) for k in list(self.G_.nodes)]

            # get similarity threshold
            self.__calculate_threshold_by_active_nodes()

    def __setup_n_active_nodes_and_div_lambda(self):
        """
        Set n_active_nodes_ and div_lambda_ by diversity of nodes.
        https://proceedings.neurips.cc/paper/2020/hash/d1dc3a8270a6f9394f88847d7f0050cf-Abstract.html
        """

        nodes_list = list(self.G_.nodes)
        _, correntropy = self.__calculate_correntropy(self.G_.nodes[nodes_list[-1]]['weight'])

        if self.G_.number_of_nodes() == 2:
            self.div_mat_ = np.array([[correntropy[1], correntropy[0]], [correntropy[0], correntropy[1]]])
        else:
            self.div_mat_ = np.insert(self.div_mat_, self.div_mat_.shape[1], correntropy[0:self.div_mat_.shape[1]],
                                      axis=0)
            self.div_mat_ = np.insert(self.div_mat_, self.div_mat_.shape[1], correntropy, axis=1)

        # div_cim = np.linalg.det(self.div_mat_)
        div_cim = np.linalg.det(np.exp(self.div_mat_))

        if div_cim < self.div_threshold_:
            self.n_active_nodes_ = self.G_.number_of_nodes()
            self.div_lambda_ = self.n_active_nodes_ * 2

    def __calculate_sigma_by_active_nodes(self, weight: np.ndarray = None, new_node_idx: int = None):
        """
        Calculate sigma for CIM based on active nodes.
        """

        if weight is None:
            active_node_idx_ = list(self.active_node_idx_)
            n_selected_weights = np.minimum(len(active_node_idx_), self.n_active_nodes_)
            selected_weights = list(
                self.__get_node_attributes_from('weight', active_node_idx_[0:int(n_selected_weights)]))
            std_weights = np.std(selected_weights, axis=0, ddof=1)
        else:
            selected_weights = weight
            std_weights = np.std(weight, axis=0, ddof=1)
        np.putmask(std_weights, std_weights == 0.0, 1.0e-6)  # If value=0, add a small value for avoiding an error.

        # Silverman's Rule
        a = np.power(4 / (2 + self.dim_), 1 / (4 + self.dim_))
        b = np.power(np.array(selected_weights).shape[0], -1 / (4 + self.dim_))
        s = a * std_weights * b
        self.sigma_ = np.median(s)

        if new_node_idx is not None:
            nx.set_node_attributes(self.G_, {new_node_idx: {'sigma': self.sigma_}})

    def __calculate_cim(self, signal: np.ndarray):
        """
        Calculate CIM between a signal and nodes.
        Return node indexes and CIM values.
        """
        node_list = list(self.G_.nodes)
        weights = list(self.__get_node_attributes_from('weight', node_list))
        sigma = list(self.__get_node_attributes_from('sigma', node_list))
        c = np.exp(-(signal - np.array(weights)) ** 2 / (2 * np.mean(np.array(sigma)) ** 2))
        return node_list, np.sqrt(1 - np.mean(c, axis=1))

    def __calculate_correntropy(self, signal: np.ndarray):
        """
        Calculate correntropy between a signal and nodes.
        """
        node_list = list(self.G_.nodes)
        weights = list(self.__get_node_attributes_from('weight', node_list))
        sigma = list(self.__get_node_attributes_from('sigma', node_list))
        c = np.exp(-(signal - np.array(weights)) ** 2 / (2 * np.mean(np.array(sigma)) ** 2))
        return node_list, np.mean(c, axis=1)

    def __add_node(self, signal: np.ndarray) -> int:
        """
        Add a new node to G with winning count, sigma, and label_counts.
        Return an index of the new node.
        """
        if len(self.G_.nodes) == 0:  # for the first node
            new_node_idx = 0
        else:
            new_node_idx = max(self.G_.nodes) + 1

        # Generate node
        self.G_.add_node(new_node_idx, weight=signal, winning_counts=1, sigma=self.sigma_)

        return new_node_idx

    def __update_active_node_index(self, signal, winner_idx):
        if self.active_node_idx_ is None:
            self.active_node_idx_ = np.array([winner_idx])
        else:
            delete_idx = np.where(self.active_node_idx_ == winner_idx)
            self.active_node_idx_ = np.delete(self.active_node_idx_, delete_idx)
            self.active_node_idx_ = np.append(winner_idx, self.active_node_idx_)

    def __delete_active_node_index(self, deleted_node_list: list):
        delete_idx = [np.where(self.active_node_idx_ == deleted_node_list[k]) for k in range(len(deleted_node_list))]
        self.active_node_idx_ = np.delete(self.active_node_idx_, delete_idx)

    def __calculate_threshold_by_active_nodes(self) -> float:
        """
        Calculate a similarity threshold by using active nodes.
        """

        active_node_idx_ = list(self.active_node_idx_)
        n_selected_weights = np.minimum(len(active_node_idx_), self.n_active_nodes_)
        selected_weights = list(self.__get_node_attributes_from('weight', active_node_idx_[0:int(n_selected_weights)]))
        cims = [self.__calculate_cim(w)[1] for w in selected_weights]  # Calculate a pairwise cim among nodes
        [np.putmask(cims[k], cims[k] == 0.0, 1.0) for k in range(len(cims))]  # Set cims[k][k] = 1.0
        self.V_thres_ = np.mean([np.min(cims[k]) for k in range(len(cims))])

    def __find_nearest_node(self, node_list: list, cim: np.ndarray):
        """
        Get 1st and 2nd nearest nodes from a signal.
        Return indexes and weights of the 1st and 2nd nearest nodes from a signal.
        """

        if len(node_list) == 1:
            node_list = node_list + node_list
            cim = np.array(list(cim) + [np.inf])

        idx = np.argsort(cim)
        return node_list[idx[0]], cim[idx[0]], node_list[idx[1]], cim[idx[1]]

    def __update_s1_node(self, idx, signal):
        """
        Update weight and winning count for the nearest node.
        """
        # update weight and winning_counts
        weight = self.G_.nodes[idx].get('weight')
        new_winning_count = self.G_.nodes[idx].get('winning_counts') + 1
        new_weight = weight + (signal - weight) / new_winning_count
        nx.set_node_attributes(self.G_, {idx: {'weight': new_weight, 'winning_counts': new_winning_count}})

    def __get_node_attributes_from(self, attr: str, node_list: Iterable[int]) -> Iterator:
        """
        Get a node attribute from selected nodes.
        """
        att_dict = nx.get_node_attributes(self.G_, attr)
        return map(att_dict.get, node_list)

    def __update_s2_node(self, idx, signal):
        """
        Update weight for the second-nearest node.
        """
        weight = self.G_.nodes[idx].get('weight')
        winning_counts = self.G_.nodes[idx].get('winning_counts')
        new_weight = weight + (signal - weight) / (100 * winning_counts)
        nx.set_node_attributes(self.G_, {idx: {'weight': new_weight}})

    def __labeling_sample_for_clustering(self, x: np.ndarray) -> np.ndarray:
        """
        Labeled samples should be evaluated by using clustering metrics.

        """
        # get cluster of nodes and order of nodes
        # compute cim between x and nodes
        weights = list(self.__get_node_attributes_from('weight', list(self.G_.nodes)))
        sigmas = list(self.__get_node_attributes_from('sigma', list(self.G_.nodes)))
        c = [np.exp(-(x[k, :] - np.array(weights)) ** 2 / (2 * np.mean(np.array(sigmas)) ** 2)) for k in range(len(x))]
        cim = [np.sqrt(1 - np.mean(c[k], axis=1)) for k in range(len(x))]

        # get indexes of the nearest neighbor
        nearest_node_idx = np.argmin(cim, axis=1)

        return nearest_node_idx

class ClusterCAplus(CAplus, ClusterMixin):
    pass
