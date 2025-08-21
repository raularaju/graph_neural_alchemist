# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
import torch.nn.functional as F
from config import *
from scipy.special import softmax
from .shapelet_utils import shapelet_distance, adjacent_matrix
from ..utils.base_utils import Debugger
from .time_aware_shapelets import learn_time_aware_shapelets
from ..utils.base_utils import ModelUtils, evaluate_performance
import numpy as np
from ..utils.gat_utils import GATDataloader, GATDataset
from ..utils.gat import GAT, accuracy_torch
class Time2Graph(ModelUtils):
    """
    Time2GraphGAT model
    Hyper-parameters:
        K: number of learned shapelets
        C: number of candidates
        A: number of shapelets assigned to each segment
        tflag: timing flag
    """

    def __init__(
        self,
        args
    ):
        super(Time2Graph, self).__init__(kernel="xgb")
        self.K = args.K
        self.C = args.C
        self.seg_length = args.seg_length
        self.num_segment = args.num_segment
        self.data_size = args.data_size
        self.warp = args.warp
        self.tflag = args.tflag
        self.gpu_enable = args.gpu_enable        
        self.shapelets = None
        self.append = args.append
        self.percentile = args.percentile
        self.threshold = None
        self.sort = args.sort
        self.aggregate = args.aggregate            
        self.softmax = args.softmax
        self.diff = args.diff
        self.standard_scale = args.standard_scale                
        self.feat_flag = args.feat_flag
        self.feat_norm = args.feat_norm
        self.lr = args.lr
        self.p = args.percentile
        self.alpha = args.alpha
        self.beta = args.beta
        self.debug = args.debug
        self.measurement = args.measurement
        self.batch_size = args.batch_size
        self.init = args.init       
        self.out_clf = True
        self.niter = args.epochs
        self.cuda = self.gpu_enable and torch.cuda.is_available()
        
        self.n_features = self.num_segment    
        self.n_class = args.num_classes
        
        self.xgb = self.clf__(use_label_encoder=False,  eval_metric="logloss", verbosity=0)
        
        self.gat = GAT(
            nfeat=self.n_features, nhid=8, nclass=self.n_class,
            dropout=0.2, nheads=8, alpha=0.2, nnodes=self.K, aggregate=self.aggregate
        )        
        
        Debugger.info_print(
            "initialize time2graph+ model with {}".format(self.__dict__)
        )

    def learn_shapelets(self, x, y, num_segment, data_size):
        assert x.shape[1] == num_segment * self.seg_length
        Debugger.info_print(
            "basic statistics before learn shapelets: max {:.4f}, min {:.4f}".format(
                np.max(x), np.min(x)
            )
        )
        self.shapelets = learn_time_aware_shapelets(
            time_series_set=x,
            label=y,
            K=self.K,
            C=self.C,
            p=self.p,
            num_segment=num_segment,
            seg_length=self.seg_length,
            data_size=data_size,
            lr=self.lr,
            alpha=self.alpha,
            beta=self.beta,
            num_batch=int(x.shape[0] / self.batch_size),
            measurement=self.measurement,
            gpu_enable=self.gpu_enable,
            **self.kwargs
        )
        
    def __gat_features__(self, X, train=False):
        __shapelet_distance = shapelet_distance(
            time_series_set=X, shapelets=self.shapelets, seg_length=self.seg_length,
            tflag=self.tflag, tanh=self.kwargs.get('tanh', False), debug=self.debug,
            init=self.init, warp=self.warp, measurement=self.measurement)
        threshold = None if train else self.threshold
        adj_matrix, self.threshold = adjacent_matrix(
            sdist=__shapelet_distance, num_time_series=X.shape[0], num_segment=int(X.shape[1] / self.seg_length),
            num_shapelet=self.K, percentile=self.percentile, threshold=threshold, debug=self.debug)
        __shapelet_distance = np.transpose(__shapelet_distance, axes=(0, 2, 1))
        if self.sort:
            __shapelet_distance = softmax(-1 * np.sort(__shapelet_distance, axis=1), axis=1)
        if self.softmax and not self.sort:
            __shapelet_distance = softmax(__shapelet_distance, axis=1)
        if self.append:
            origin = np.array([v[0].reshape(-1) for v in self.shapelets], dtype=np.float32).reshape(1, self.K, -1)
            return np.concatenate((__shapelet_distance, np.tile(origin, (__shapelet_distance.shape[0], 1, 1))),
                                axis=2).astype(np.float32), adj_matrix
        else:
            return __shapelet_distance.astype(np.float32), adj_matrix

    def __fit_gat(self, X, Y):
        feats, adj = self.__gat_features__(X=X, train=True)
        optimizer = optim.Adam(self.gat.parameters(), lr=self.lr, weight_decay=5e-4)
                       
        # weight = torch.FloatTensor([float(sum(Y) / len(Y)), 1 - float(sum(Y) / len(Y))])
        
        # Calculate weights for each class
        class_counts = np.array(np.bincount(Y), dtype=np.float32)
        class_weights = 1.0 / class_counts
        weight = torch.FloatTensor(class_weights / np.sum(class_weights))        
        
        # class_weights = 1.0 / class_counts
        # weight = torch.FloatTensor([class_weights / class_weights.sum()])
        
        if self.cuda:
            self.gat = self.gat.cuda()
            weight = weight.cuda()

        for epoch in range(self.niter):
            dataset = GATDataset(feat=feats, adj=adj, y=Y)
            dataloader = GATDataloader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=28)
            for i, (feat_batch, adj_batch, lb_batch) in enumerate(dataloader, 0):
                self.gat.train()
                optimizer.zero_grad()
                if self.cuda:
                    feat_batch = feat_batch.cuda()
                    adj_batch = adj_batch.cuda()
                    lb_batch = lb_batch.cuda()
                output_batch = self.gat(feat_batch, adj_batch)
                loss_train = F.nll_loss(output_batch, lb_batch, weight=weight)
                acc_train = accuracy_torch(output_batch, lb_batch)
                loss_train.backward()
                optimizer.step()
                self.gat.eval()
                output_batch = self.gat(feat_batch, adj_batch)
                loss_val = F.nll_loss(output_batch, lb_batch)
                acc_val = accuracy_torch(output_batch, lb_batch)
                Debugger.debug_print(
                    msg='Epoch: {:04d}-{:02d}, train loss: {:.4f} accu: {:.4f}, val loss: {:.4f}, accu: {:.4f}'.format(
                        epoch + 1, i + 1, loss_train.data.item(), acc_train.data.item(), loss_val.data.item(),
                        acc_val.data.item()), debug=self.debug)
        y_pred = self.___gat_predict(feat=feats, adj=adj)
        accu, prec, recall, f1 = evaluate_performance(y_pred=y_pred, y_true=Y)
        Debugger.info_print('fitting gat: accu {:.4f}, prec {:.4f}, recall {:.4f}, f1 {:.4f}'.format(
            accu, prec, recall, f1))

    def ___gat_predict(self, feat, adj):
        y_batch_list = []
        dataset = GATDataset(feat=feat, adj=adj)
        dataloader = GATDataloader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=28)
        for i, (feat_batch, adj_batch) in enumerate(dataloader, 0):
            self.gat.eval()
            if self.cuda:
                feat_batch = feat_batch.cuda()
                adj_batch = adj_batch.cuda()
            output_batch = self.gat(feat_batch, adj_batch)
            y_batch = output_batch.max(1)[1].type(torch.IntTensor)
            if self.cuda:
                y_batch_list.append(y_batch.data.cpu().numpy())
            else:
                y_batch_list.append(y_batch.data.numpy())
        return np.concatenate(y_batch_list, axis=0)

    def __gat_hidden_feature(self, feat, adj, X=None, train=False):
        feat_batch_list = []
        dataset = GATDataset(feat=feat, adj=adj)
        dataloader = GATDataloader(dataset, batch_size=self.batch_size, shuffle=False)
        for i, (feat_batch, adj_batch) in enumerate(dataloader, 0):
            self.gat.eval()
            if self.cuda:
                feat_batch = feat_batch.cuda()
                adj_batch = adj_batch.cuda()
            output_batch = self.gat(feat_batch, adj_batch, feat_flag=True)
            if self.cuda:
                feat_batch_list.append(output_batch.data.cpu().numpy())
            else:
                feat_batch_list.append(output_batch.data.numpy())
        feat_batch = np.concatenate(feat_batch_list, axis=0)
        if self.feat_flag:
            assert X is not None, 'time series data not provided when feat_flag is set as True'
            feat = self.fm.extract_features(samples=X)
            if train and self.feat_norm:
                feat = self.fm_scaler.fit_transform(feat)
            elif self.feat_norm:
                feat = self.fm_scaler.transform(feat)
            return np.concatenate((feat_batch, feat), axis=1)
        else:
            return feat_batch
    
    def preprocess_input_data(self, X):
        X_scale = X.copy()
        if self.diff:
            X_scale[:, :-1, :] = X[:, 1:, :] - X[:, :-1, :]
            X_scale[:, -1, :] = 0
            Debugger.debug_print("conduct time differing...")
        if self.standard_scale:
            for i in range(self.data_size):
                X_std = np.std(X_scale[:, :, i], axis=1).reshape(X.shape[0], -1)
                X_std[X_std == 0] = 1.0
                X_scale[:, :, i] = (
                    X_scale[:, :, i]
                    - np.mean(X_scale[:, :, i], axis=1).reshape(X.shape[0], -1)
                ) / X_std
                Debugger.debug_print(
                    f"Conducting z-normalization on data-{i}, with mean {np.mean(X_scale[0, :, i]):.2f} and std {np.std(X_scale[0, :, i]):.2f}"
                )
        return X_scale    
        
    def fit(self, X, Y):
        assert self.data_size == X.shape[-1]
        X_scale = self.preprocess_input_data(X)        
        
        self.__fit_gat(X=X_scale, Y=Y)
        X_feat, X_adj = self.__gat_features__(X_scale)

        Debugger.info_print('using default xgboost parameters')
        self.xgb.fit(self.__gat_hidden_feature(feat=X_feat, adj=X_adj, X=X, train=True), Y)            
        
        y_pred = self.___gat_predict(feat=X_feat, adj=X_adj)
        accu, prec, recall, f1 = evaluate_performance(y_pred=y_pred, y_true=Y)
        Debugger.info_print('fully-connected-layer res on training set: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(
            accu, prec, recall, f1))
        if self.out_clf:
            y_pred = self.xgb.predict(self.__gat_hidden_feature(feat=X_feat, adj=X_adj, X=X))
            accu, prec, recall, f1 = evaluate_performance(y_pred=y_pred, y_true=Y)
            Debugger.info_print('out-classifier on training set: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(
                accu, prec, recall, f1))

    def predict(self, X):
        assert self.shapelets is not None, 'shapelets has not been learnt yet...'
        X_scale = self.preprocess_input_data(X)
        x, adj = self.__gat_features__(X_scale)
        
        if self.out_clf:
            return self.xgb.predict(self.__gat_hidden_feature(feat=x, adj=adj, X=X))
        else:
            return self.___gat_predict(feat=x, adj=adj)
    
    def save_model(self, fpath):
        ret = {}
        for key, val in self.__dict__.items():
            if key != 'xgb':
                ret[key] = val
        self.xgb.save_model('{}.xgboost'.format(fpath))
        torch.save(ret, fpath)

    def load_model(self, fpath, map_location='cuda:0'):
        # @TODO: specify map_location
        cache = torch.load(fpath, map_location=map_location)
        for key, val in cache.items():
            self.__dict__[key] = val
        self.xgb.load_model('{}.xgboost'.format(fpath))
        
    def save_shapelets(self, fpath):
        torch.save(self.shapelets, fpath)

    def load_shapelets(self, fpath, map_location="cuda:0"):
        self.shapelets = torch.load(fpath, map_location=map_location)