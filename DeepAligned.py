import copy
from collections import Counter
import torch
from model import *
from init_parameter import *
from dataloader import *
from pretrain import *
from util import *
from time import time
import os
torch.set_num_threads(6)
import fitlog
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class ModelManager:
    def __init__(self, args, data, pretrained_model=None):
        set_seed(args.seed)

        if pretrained_model is None:
            pretrained_model = BertForModel.from_pretrained("./model", cache_dir = "", num_labels = data.n_known_cls)
            b = torch.load("./../NID_ACL/MTP2step_sf_12.1")
            for key in list(b.keys()):
                if "backbone" in key:
                    b[key[9:]] = b.pop(key)
            pretrained_model.load_state_dict(b, strict=False)
            #pretrained_model = BertForModel.from_pretrained(gei, cache_dir="",num_labels=data.n_known_cls)

        self.model = pretrained_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.freeze_bert_parameters_em:
            self.freeze_parameters(self.model)
        self.model.to(self.device)

        if args.cluster_num_factor > 1:
            self.num_labels = self.predict_k(args, data)
        else:
            self.num_labels = data.num_labels

        num_train_examples = len(data.train_labeled_examples) + len(data.train_unlabeled_examples)
        self.num_train_optimization_steps = int(num_train_examples / args.train_batch_size) * args.num_train_epochs
        self.optimizer = self.get_optimizer(args,self.model)
        self.best_eval_score = 0
        self.centroids = None
        self.test_results = None
        self.predictions = None
        self.true_labels = None

    def alignment(self, km, args):

        if self.centroids is not None:

            old_centroids = self.centroids.cpu().numpy()
            new_centroids = km.cluster_centers_

            DistanceMatrix = np.linalg.norm(old_centroids[:, np.newaxis, :] - new_centroids[np.newaxis, :, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(DistanceMatrix)

            new_centroids = torch.tensor(new_centroids).to(self.device)
            self.centroids = torch.empty(self.num_labels, args.feat_dim).to(self.device)

            alignment_labels = list(col_ind)
            for i in range(self.num_labels):
                label = alignment_labels[i]
                self.centroids[i] = new_centroids[label]

            pseudo2label = {label: i for i, label in enumerate(alignment_labels)}
            pseudo_labels = np.array([pseudo2label[label] for label in km.labels_])

        else:
            self.centroids = torch.tensor(km.cluster_centers_).to(self.device)
            pseudo_labels = km.labels_

        pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long).to(self.device)

        return pseudo_labels

    def get_features_labels(self, dataloader, model, args):

        model.eval()
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                feature = model(input_ids, segment_ids, input_mask, feature_ext=True)

            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))

        return total_features, total_labels

    def predict_k(self, args, data):

        feats, _ = self.get_features_labels(data.train_semi_dataloader, self.model, args)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters=data.num_labels).fit(feats)
        y_pred = km.labels_

        pred_label_list = np.unique(y_pred)
        drop_out = len(feats) / (data.num_labels)
        cnt = 0
        for label in pred_label_list:
            num = len(y_pred[y_pred == label])
            if num < drop_out:
                cnt += 1
        print(cnt)
        num_labels = len(pred_label_list) - cnt
        print('pred_num', num_labels)

        return num_labels

    def get_optimizer(self, args, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.lr,
                             warmup=args.warmup_proportion,
                             t_total=self.num_train_optimization_steps)
        return optimizer

    def evaluation(self, args, data):
        feats, labels = self.get_features_labels(data.test_dataloader, self.model, args)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters=self.num_labels).fit(feats)
        y_pred = km.labels_
        y_true = labels.cpu().numpy()
        results = clustering_score(y_true, y_pred)
        print('results', results)
        ind, _ = hungray_aligment(y_true, y_pred)
        map_ = {i[0]: i[1] for i in ind}
        y_pred = np.array([map_[idx] for idx in y_pred])
        cm = confusion_matrix(y_true, y_pred)
        print('confusion matrix', cm)
        self.test_results = results
        self.save_results(args)



    def update_pseudo_labels(self, pseudo_labels, args, input_ids,input_mask,segment_ids,label_ids):
        train_data = TensorDataset(input_ids, input_mask, segment_ids, pseudo_labels,
                                   label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        return train_dataloader

    def update_dataset(self,km,feats,k):
        def top_K_idx(data,k):
            data=np.array(data)
            idx=data.argsort()[-k:][::-1]
            return list(idx)
        updata_semi_label=copy.deepcopy(data.semi_label_ids)
        for a,example in enumerate(data.train_labeled_examples):
            plabel=km.labels_[a]
            #ndarray[9003]
            same_plabel_idx=[]
            top = []
            for idx,label in enumerate(km.labels_):
                if label==plabel and idx>len(data.train_labeled_examples):
                    same_plabel_idx.append(idx)
            for idx in same_plabel_idx:
                top.append(1 - cosine(feats[a], feats[idx]))
            idxlist = top_K_idx(top, k)
            semi_idxlist=[same_plabel_idx[i] for i in idxlist]
            for i in semi_idxlist:
                updata_semi_label[i]=updata_semi_label[a]
        return updata_semi_label





    def get_semi_loader(self, semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids, args):
        semi_data = TensorDataset(semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids)
        semi_sampler = SequentialSampler(semi_data)
        semi_dataloader = DataLoader(semi_data, sampler=semi_sampler, batch_size = args.train_batch_size)
        return semi_dataloader

    def train(self, args, data):
        bestresults = {'ACC': 0,
                       'ARI': 0,
                       'NMI': 0}
        jsonresults = {}
        for epoch in range(int(args.num_train_epochs)):
            s = time()
            feats, labels = self.get_features_labels(data.train_semi_dataloader, self.model, args)
            feats = feats.cpu().numpy()
            t0 = time()
            km = KMeans(n_clusters=self.num_labels).fit(feats)
            t1 = time()
            kmeans_time = t1 - t0
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            self.model.train()
            t0 = time()
            pseudo_labels = self.alignment(km, args)

            if args.augment_data_2:
                updata_semi_label=self.update_dataset(km,feats,args.k)
                train_semi_dataloader=self.update_pseudo_labels(pseudo_labels, args, data.semi_input_ids,data.semi_input_mask,data.semi_segment_ids,updata_semi_label)
            else:
                train_semi_dataloader = self.update_pseudo_labels(pseudo_labels, args, data.semi_input_ids,data.semi_input_mask,data.semi_segment_ids,data.semi_label_ids)

            '''
            pseudo_labels = self.alignment(km, args)
            train_semi_dataloader = self.update_pseudo_labels(pseudo_labels, args, data)
            '''

            for batch in train_semi_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, pseudo_label_ids, label_ids = batch

                # contrastive loss
                label_matrix = torch.zeros(input_ids.size(0), input_ids.size(0))
                labels = pseudo_label_ids
                for i in range(input_ids.size(0)):
                    label_matrix[i] = (labels == labels[i])

                feats = self.model(input_ids, segment_ids, input_mask, mode="sim", feature_ext=True)
                feats = F.normalize(feats, 2)
                sim_matrix = torch.exp(torch.matmul(feats, feats.t()) / args.t)
                sim_matrix = sim_matrix - sim_matrix.diag().diag()

                pos_matrix = torch.zeros_like(sim_matrix)
                pos_mask = np.where(label_matrix != 0)
                pos_matrix[pos_mask] = sim_matrix[pos_mask]

                cl_loss = pos_matrix / sim_matrix.sum(1).view(-1, 1)
                cl_loss = cl_loss[cl_loss != 0]
                cl_loss = -torch.log(cl_loss).mean()
                if torch.isnan(cl_loss):
                    cl_loss = 0

                # cross entropy loss
                ind = (label_ids != -1)
                if any(ind) is False:
                    ce_loss=0
                else:
                    input_ids = input_ids[ind]
                    input_mask = input_mask[ind]
                    segment_ids = segment_ids[ind]
                    label_ids = label_ids[ind]
                    ce_loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode="train")

                loss=(1 - args.beta) * cl_loss + args.beta * ce_loss
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                self.optimizer.step()
                self.optimizer.zero_grad()

            t1 = time()
            cont_time = t1 - t0
            tr_loss = tr_loss / nb_tr_steps
            print('train_loss = ', tr_loss)
            e = time()
            print("kmeans:{:.2f} cont:{:.2f} total:{:.2f}".format(kmeans_time, cont_time, e - s))

            feats, labels = self.get_features_labels(data.test_dataloader, self.model, args)
            feats = feats.cpu().numpy()
            km = KMeans(n_clusters=self.num_labels).fit(feats)

            y_pred = km.labels_
            y_true = labels.cpu().numpy()

            results = clustering_score(y_true, y_pred)
            jsonresults.update({epoch:results})
            print(results)
            '''
            feats, labels = self.get_features_labels(data.test_labeled_dataloader, self.model, args)
            feats = feats.cpu().numpy()
            km = KMeans(n_clusters=len(data.known_label_list)).fit(feats)
            y_pred = km.labels_
            y_true = labels.cpu().numpy()
            results = clustering_score(y_true, y_pred)
            jsonresults.update({(str(epoch)+"known"): results})
            print("known:"+str(results))

            feats, labels = self.get_features_labels(data.test_unlabeled_dataloader, self.model, args)
            feats = feats.cpu().numpy()
            km = KMeans(n_clusters=len(data.unknown_label_list)).fit(feats)
            y_pred = km.labels_
            y_true = labels.cpu().numpy()
            results = clustering_score(y_true, y_pred)
            jsonresults.update({(str(epoch) + "unknown"): results})
            print("unknown:" + str(results))
            '''
            if results["ACC"]+results["ARI"]+results["NMI"]>bestresults["ACC"]+bestresults["ARI"]+bestresults["NMI"]:
                bestresults=results

            jsonresults.update({"best": bestresults})
            '''
            if epoch+1==1 or (epoch+1)%5==0:
                a=self.eval_pretrain()
                jsonresults.update({"eval"+str(epoch): a})
            '''

            import json
            info_json=json.dumps(jsonresults,sort_keys=False,indent=4,separators=(",",": "))
            f=open('./outputs/info_{}.json'.format(args.name),'w')
            f.write(info_json)

    def eval_pretrain(self):
        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, data.n_known_cls)).to(self.device)

        for batch in tqdm(data.eval_dataloader, desc="Eval Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                _, logits = self.model(input_ids, segment_ids, input_mask, mode='eval')
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))

        total_probs, total_preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        return acc
    def load_pretrained_model(self, args):
        pretrained_dict = self.pretrained_model.state_dict()
        classifier_params = ['classifier.weight', 'classifier.bias']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)

    def restore_model(self, args, model):
        output_model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        model.load_state_dict(torch.load(output_model_file))
        return model

    def freeze_parameters(self, model):
        for name, param in model.bert.named_parameters():
            param.requires_grad = False
            param_grad_ = False
            for i in range(6, 12):
                name_str = "encoder.layer." + str(i)
                if name_str in name:
                    param_grad_ = True
            if param_grad_ or "pooler" in name:
                param.requires_grad = True

    def save_results(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        # var = [args.dataset, args.known_cls_ratio, args.labeled_ratio, args.cluster_num_factor, args.seed, self.num_labels]
        # names = ['dataset', 'known_cls_ratio', 'labeled_ratio', 'cluster_num_factor','seed', 'K']
        names = ['dataset', 'alpha', 'beta', 'batch_size', 'seed', 'K',"name"]
        var = [args.dataset, args.t, args.beta, args.train_batch_size, args.seed, self.num_labels, args.name]
        vars_dict = {k: v for k, v in zip(names, var)}
        results = dict(self.test_results, **vars_dict)
        keys = list(results.keys())
        values = list(results.values())

        file_name = '%s.csv' % args.dataset
        results_path = os.path.join(args.save_results_path, file_name)

        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori, columns=keys)
            df1.to_csv(results_path, index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results, index=[1])
            df1 = df1.append(new, ignore_index=True)
            df1.to_csv(results_path, index=False)
        data_diagram = pd.read_csv(results_path)

        print('test_results', data_diagram)


if __name__ == '__main__':

    parser = init_model()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    fitlog.set_rng_seed(args.seed)
    print('_________Prepare {} data__________'.format(args.dataset))
    data = Data(args)

    if args.pretrain:
        manager_p = PretrainModelManager(args, data)
        manager_p.train(args)
        torch.cuda.empty_cache()
        manager = ModelManager(args, data, manager_p.model)
    else:
        args.pretrain_dir = 'pretrained_' + args.dataset
        manager = ModelManager(args, data)

    print('Training begin...')
    manager.train(args, data)
    print('Training finished!')

    print('Evaluation begin...')
    manager.evaluation(args, data)
    print('Evaluation finished!')

    # manager.save_results(args)

