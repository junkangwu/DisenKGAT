from helper import *
from data_loader import *
from model import * 
def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True
# sys.path.append('./')

class Runner(object):

    def load_data(self):
        """
        Reading in raw triples and converts it into a standard format.

        Parameters
        ----------
        self.p.dataset:         Takes in the name of the dataset (FB15k-237)

        Returns
        -------
        self.ent2id:            Entity to unique identifier mapping
        self.id2rel:            Inverse mapping of self.ent2id
        self.rel2id:            Relation to unique identifier mapping
        self.num_ent:           Number of entities in the Knowledge graph
        self.num_rel:           Number of relations in the Knowledge graph
        self.embed_dim:         Embedding dimension used
        self.data['train']:     Stores the triples corresponding to training dataset
        self.data['valid']:     Stores the triples corresponding to validation dataset
        self.data['test']:      Stores the triples corresponding to test dataset
        self.data_iter:     The dataloader for different data splits

        """

        ent_set, rel_set = OrderedSet(), OrderedSet()
        for split in ['train', 'test', 'valid']:
            for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                ent_set.add(sub)
                rel_set.add(rel)
                ent_set.add(obj)

        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        self.rel2id.update({rel + '_reverse': idx + len(self.rel2id) for idx, rel in enumerate(rel_set)})

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.p.num_ent = len(self.ent2id)
        self.p.num_rel = len(self.rel2id) // 2
        self.p.embed_dim = self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim
        self.logger.info('num_ent {} num_rel {}'.format(self.p.num_ent, self.p.num_rel))
        self.data = ddict(list)
        sr2o = ddict(set)

        for split in ['train', 'test', 'valid']:
            for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                self.data[split].append((sub, rel, obj))

                if split == 'train':
                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel + self.p.num_rel)].add(sub)
        # self.data: all origin train + valid + test triplets
        self.data = dict(self.data)
        # self.sr2o: train origin edges and reverse edges
        self.sr2o = {k: list(v) for k, v in sr2o.items()}
        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel + self.p.num_rel)].add(sub)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.triples = ddict(list)

        # for (sub, rel), obj in self.sr2o.items():
        #     self.triples['train'].append({'triple': (sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})
        if self.p.strategy == 'one_to_n':
            for (sub, rel), obj in self.sr2o.items():
                self.triples['train'].append({'triple': (sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})
        else:
            for sub, rel, obj in self.data['train']:
                rel_inv = rel + self.p.num_rel
                sub_samp = len(self.sr2o[(sub, rel)]) + len(self.sr2o[(obj, rel_inv)])
                sub_samp = np.sqrt(1 / sub_samp)

                self.triples['train'].append(
                    {'triple': (sub, rel, obj), 'label': self.sr2o[(sub, rel)], 'sub_samp': sub_samp})
                self.triples['train'].append(
                    {'triple': (obj, rel_inv, sub), 'label': self.sr2o[(obj, rel_inv)], 'sub_samp': sub_samp})

        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                rel_inv = rel + self.p.num_rel
                self.triples['{}_{}'.format(split, 'tail')].append(
                    {'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)]})
                self.triples['{}_{}'.format(split, 'head')].append(
                    {'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})
            self.logger.info('{}_{} num is {}'.format(split, 'tail', len(self.triples['{}_{}'.format(split, 'tail')])))
            self.logger.info('{}_{} num is {}'.format(split, 'head', len(self.triples['{}_{}'.format(split, 'head')])))

        self.triples = dict(self.triples)

        def get_data_loader(dataset_class, split, batch_size, shuffle=True):
            return DataLoader(
                dataset_class(self.triples[split], self.p),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=max(0, self.p.num_workers),
                collate_fn=dataset_class.collate_fn
            )

        self.data_iter = {
            'train': get_data_loader(TrainDataset, 'train', self.p.batch_size),
            'valid_head': get_data_loader(TestDataset, 'valid_head', self.p.test_batch_size),
            'valid_tail': get_data_loader(TestDataset, 'valid_tail', self.p.test_batch_size),
            'test_head': get_data_loader(TestDataset, 'test_head', self.p.test_batch_size),
            'test_tail': get_data_loader(TestDataset, 'test_tail', self.p.test_batch_size),
        }
        self.logger.info('num_ent {} num_rel {}\n'.format(self.p.num_ent, self.p.num_rel))
        self.logger.info('train set num is {}\n'.format(len(self.triples['train'])))
        self.logger.info('{}_{} num is {}\n'.format('test', 'tail', len(self.triples['{}_{}'.format('test', 'tail')])))
        self.logger.info('{}_{} num is {}\n'.format('valid', 'tail', len(self.triples['{}_{}'.format('valid', 'tail')])))
        self.edge_index, self.edge_type = self.construct_adj()

    def construct_adj(self):
        """
        Constructor of the runner class

        Parameters
        ----------

        Returns
        -------
        Constructs the adjacency matrix for GCN

        """
        edge_index, edge_type = [], []

        for sub, rel, obj in self.data['train']:
            edge_index.append((sub, obj))
            edge_type.append(rel)

        # Adding inverse edges
        for sub, rel, obj in self.data['train']:
            edge_index.append((obj, sub))
            edge_type.append(rel + self.p.num_rel)
        # edge_index: 2 * 2E, edge_type: 2E * 1
        edge_index = torch.LongTensor(edge_index).to(self.device).t()
        edge_type = torch.LongTensor(edge_type).to(self.device)

        return edge_index, edge_type

    def __init__(self, params):
        """
        Constructor of the runner class

        Parameters
        ----------
        params:         List of hyper-parameters of the model

        Returns
        -------
        Creates computational graph and optimizer

        """
        self.p = params
        self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

        self.logger.info(vars(self.p))
        pprint(vars(self.p))

        if self.p.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        self.load_data()
        self.model = self.add_model(self.p.model, self.p.score_func)
        self.optimizer, self.optimizer_mi = self.add_optimizer(self.model)

    def add_model(self, model, score_func):
        """
        Creates the computational graph

        Parameters
        ----------
        model_name:     Contains the model name to be created

        Returns
        -------
        Creates the computational graph for model and initializes it

        """
        model_name = '{}_{}'.format(model, score_func)

        if model_name.lower() == 'disenkgat_transe':
            model = DisenKGAT_TransE(self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower() == 'disenkgat_distmult':
            model = DisenKGAT_DistMult(self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower() == 'disenkgat_conve':
            model = DisenKGAT_ConvE(self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower() == 'disenkgat_interacte':
            model = DisenKGAT_InteractE(self.edge_index, self.edge_type, params=self.p)
        else:
            raise NotImplementedError

        model.to(self.device)
        return model

    def add_optimizer(self, model):
        """
        Creates an optimizer for training the parameters

        Parameters
        ----------
        parameters:         The parameters of the model

        Returns
        -------
        Returns an optimizer for learning the parameters of the model

        """
        if self.p.mi_train and self.p.mi_method.startswith('club'):
            mi_disc_params = list(map(id,model.mi_Discs.parameters()))
            rest_params = filter(lambda x:id(x) not in mi_disc_params, model.parameters())
            for m in model.mi_Discs.modules():
                self.logger.info(m)
            for name, parameters in model.named_parameters():
                print(name,':',parameters.size())
            return torch.optim.Adam(rest_params, lr=self.p.lr, weight_decay=self.p.l2), torch.optim.Adam(model.mi_Discs.parameters(), lr=self.p.lr, weight_decay=self.p.l2)
        else:
            return torch.optim.Adam(model.parameters(), lr=self.p.lr, weight_decay=self.p.l2), None

    def read_batch(self, batch, split):
        """
        Function to read a batch of data and move the tensors in batch to CPU/GPU

        Parameters
        ----------
        batch:      the batch to process
        split: (string) If split == 'train', 'valid' or 'test' split


        Returns
        -------
        Head, Relation, Tails, labels
        """
        # if split == 'train':
        #     triple, label = [_.to(self.device) for _ in batch]
        #     return triple[:, 0], triple[:, 1], triple[:, 2], label
        # else:
        #     triple, label = [_.to(self.device) for _ in batch]
        #     return triple[:, 0], triple[:, 1], triple[:, 2], label
        if split == 'train':
            if self.p.strategy == 'one_to_x':
                triple, label, neg_ent, sub_samp = [_.to(self.device) for _ in batch]
                return triple[:, 0], triple[:, 1], triple[:, 2], label, neg_ent, sub_samp
            else:
                triple, label = [_.to(self.device) for _ in batch]
                return triple[:, 0], triple[:, 1], triple[:, 2], label, None, None
        else:
            triple, label = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label

    def save_model(self, save_path):
        """
        Function to save a model. It saves the model parameters, best validation scores,
        best epoch corresponding to best validation, state of the optimizer and all arguments for the run.

        Parameters
        ----------
        save_path: path where the model is saved

        Returns
        -------
        """
        state = {
            'state_dict': self.model.state_dict(),
            'best_val': self.best_val,
            'best_epoch': self.best_epoch,
            'optimizer': self.optimizer.state_dict(),
            'args': vars(self.p)
        }
        torch.save(state, save_path)

    def load_model(self, load_path):
        """
        Function to load a saved model

        Parameters
        ----------
        load_path: path to the saved model

        Returns
        -------
        """
        state = torch.load(load_path)
        state_dict = state['state_dict']
        self.best_val = state['best_val']
        self.best_val_mrr = self.best_val['mrr']

        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])

    def evaluate(self, split, epoch):
        """
        Function to evaluate the model on validation or test set

        Parameters
        ----------
        split: (string) If split == 'valid' then evaluate on the validation set, else the test set
        epoch: (int) Current epoch count

        Returns
        -------
        resutls:            The evaluation results containing the following:
            results['mr']:          Average of ranks_left and ranks_right
            results['mrr']:         Mean Reciprocal Rank
            results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """
        left_results = self.predict(split=split, mode='tail_batch')
        right_results = self.predict(split=split, mode='head_batch')
        results = get_combined_results(left_results, right_results)
        res_mrr = '\n\tMRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}\n'.format(results['left_mrr'],
                                                                              results['right_mrr'],
                                                                              results['mrr'])
        res_mr = '\tMR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}\n'.format(results['left_mr'],
                                                                          results['right_mr'],
                                                                          results['mr'])
        res_hit1 = '\tHit-1: Tail : {:.5}, Head : {:.5}, Avg : {:.5}\n'.format(results['left_hits@1'],
                                                                               results['right_hits@1'],
                                                                               results['hits@1'])
        res_hit3 = '\tHit-3: Tail : {:.5}, Head : {:.5}, Avg : {:.5}\n'.format(results['left_hits@3'],
                                                                               results['right_hits@3'],
                                                                               results['hits@3'])
        res_hit10 = '\tHit-10: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(results['left_hits@10'],
                                                                               results['right_hits@10'],
                                                                               results['hits@10'])
        log_res = res_mrr + res_mr + res_hit1 + res_hit3 + res_hit10
        if (epoch + 1) % 10 == 0 or split == 'test':
            self.logger.info(
                '[Evaluating Epoch {} {}]: {}'.format(epoch, split, log_res))
        else:
            self.logger.info(
                '[Evaluating Epoch {} {}]: {}'.format(epoch, split, res_mrr))

        return results

    def predict(self, split='valid', mode='tail_batch'):
        """
        Function to run model evaluation for a given mode

        Parameters
        ----------
        split: (string)     If split == 'valid' then evaluate on the validation set, else the test set
        mode: (string):     Can be 'head_batch' or 'tail_batch'

        Returns
        -------
        resutls:            The evaluation results containing the following:
            results['mr']:          Average of ranks_left and ranks_right
            results['mrr']:         Mean Reciprocal Rank
            results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """
        self.model.eval()

        with torch.no_grad():
            results = {}
            train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

            for step, batch in enumerate(train_iter):
                sub, rel, obj, label = self.read_batch(batch, split)
                pred, _ = self.model.forward(sub, rel, None, split)
                b_range = torch.arange(pred.size()[0], device=self.device)
                target_pred = pred[b_range, obj]
                # filter setting
                pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, obj] = target_pred
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj]

                ranks = ranks.float()
                results['count'] = torch.numel(ranks) + results.get('count', 0.0)
                results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
                results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
                for k in range(10):
                    results['hits@{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                        'hits@{}'.format(k + 1), 0.0)

                # if step % 100 == 0:
                #     self.logger.info('[{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.p.name))

        return results

    def run_epoch(self, epoch, val_mrr=0):
        """
        Function to run one epoch of training

        Parameters
        ----------
        epoch: current epoch count

        Returns
        -------
        loss: The loss value after the completion of one epoch
        """
        self.model.train()
        losses = []
        losses_train = []
        corr_losses = []
        lld_losses = []
        train_iter = iter(self.data_iter['train'])

        for step, batch in enumerate(train_iter):
            self.optimizer.zero_grad()
            if self.p.mi_train and self.p.mi_method.startswith('club'):
                self.model.mi_Discs.eval()
            sub, rel, obj, label, neg_ent, sub_samp = self.read_batch(batch, 'train')

            pred, corr = self.model.forward(sub, rel, neg_ent, 'train')

            loss = self.model.loss(pred, label)
            if self.p.mi_train:
                losses_train.append(loss.item())
                loss = loss + self.p.alpha * corr 
                corr_losses.append(corr.item())

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            # start to compute mi_loss
            if self.p.mi_train and self.p.mi_method.startswith('club'):
                for i in range(self.p.mi_epoch):
                    self.model.mi_Discs.train()
                    lld_loss = self.model.lld_best(sub, rel)
                    self.optimizer_mi.zero_grad()
                    lld_loss.backward()
                    self.optimizer_mi.step()
                    lld_losses.append(lld_loss.item())

            if step % 100 == 0:
                if self.p.mi_train:
                    self.logger.info('[E:{}| {}]: total Loss:{:.5}, Train Loss:{:.5}, Corr Loss:{:.5}, Val MRR:{:.5}\t{}'.format(epoch, step, np.mean(losses),
                                                                                           np.mean(losses_train), np.mean(corr_losses), self.best_val_mrr,
                                                                                           self.p.name))                     
                else:
                    self.logger.info('[E:{}| {}]: Train Loss:{:.5},  Val MRR:{:.5}\t{}'.format(epoch, step, np.mean(losses),
                                                                                           self.best_val_mrr,
                                                                                           self.p.name))

        loss = np.mean(losses_train) if self.p.mi_train else np.mean(losses)
        if self.p.mi_train:
            loss_corr = np.mean(corr_losses)
            if self.p.mi_method.startswith('club') and self.p.mi_epoch == 1:
                loss_lld = np.mean(lld_losses)
                return loss, loss_corr, loss_lld
            return loss, loss_corr, 0.
        # self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
        return loss, 0., 0.

    def fit(self):
        """
        Function to run training and evaluation of model

        Parameters
        ----------

        Returns
        -------
        """
        try:
            self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
            save_path = os.path.join('./checkpoints', self.p.name)

            if self.p.restore:
                self.load_model(save_path)
                self.logger.info('Successfully Loaded previous model')
            val_results = {}
            val_results['mrr'] = 0
            kill_cnt = 0
            for epoch in range(self.p.max_epochs):
                train_loss, corr_loss, lld_loss = self.run_epoch(epoch, val_mrr)
                # if ((epoch + 1) % 10 == 0):
                val_results = self.evaluate('valid', epoch)
        
                if val_results['mrr'] > self.best_val_mrr:
                    self.best_val = val_results
                    self.best_val_mrr = val_results['mrr']
                    self.best_epoch = epoch
                    self.save_model(save_path)
                    kill_cnt = 0
                else:
                    kill_cnt += 1
                    if kill_cnt % 10 == 0 and self.p.gamma > self.p.max_gamma:
                        self.p.gamma -= 5 
                        self.logger.info('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
                    if kill_cnt > self.p.early_stop: 
                        self.logger.info("Early Stopping!!")
                        break
                if self.p.mi_train:
                    if self.p.mi_method == 'club_s' or self.p.mi_method == 'club_b':
                        self.logger.info(
                            '[Epoch {}]: Training Loss: {:.5}, corr Loss: {:.5}, lld loss :{:.5}, Best valid MRR: {:.5}\n\n'.format(epoch, train_loss, corr_loss,
                                                                                             lld_loss, self.best_val_mrr))
                    else:
                        self.logger.info(
                            '[Epoch {}]: Training Loss: {:.5}, corr Loss: {:.5}, Best valid MRR: {:.5}\n\n'.format(epoch, train_loss, corr_loss,
                                                                                             self.best_val_mrr))
                else:
                    self.logger.info(
                        '[Epoch {}]: Training Loss: {:.5}, Best valid MRR: {:.5}\n\n'.format(epoch, train_loss,
                                                                                         self.best_val_mrr))

            self.logger.info('Loading best model, Evaluating on Test data')
            self.load_model(save_path)
            test_results = self.evaluate('test', self.best_epoch)
        except Exception as e:
            self.logger.debug("%s____%s\n"
                              "traceback.format_exc():____%s" % (Exception, e, traceback.format_exc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-name', default='testrun', help='Set run name for saving/restoring models')
    parser.add_argument('-data', dest='dataset', default='FB15k-237', help='Dataset to use, default: FB15k-237')
    parser.add_argument('-model', dest='model', default='disenkgat', help='Model Name')
    parser.add_argument('-score_func', dest='score_func', default='interacte', help='Score Function for Link prediction')
    parser.add_argument('-opn', dest='opn', default='cross', help='Composition Operation to be used in RAGAT')
    # opn is new hyperparameter
    parser.add_argument('-batch', dest='batch_size', default=2048, type=int, help='Batch size')
    parser.add_argument('-test_batch', dest='test_batch_size', default=2048, type=int,
                        help='Batch size of valid and test data')
    parser.add_argument('-gamma', type=float, default=9.0, help='Margin')
    parser.add_argument('-gpu', type=str, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-epoch', dest='max_epochs', type=int, default=1500, help='Number of epochs')
    parser.add_argument('-l2', type=float, default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('-lr', type=float, default=0.001, help='Starting Learning Rate')
    parser.add_argument('-lbl_smooth', dest='lbl_smooth', type=float, default=0.1, help='Label Smoothing')
    parser.add_argument('-num_workers', type=int, default=10, help='Number of processes to construct batches')
    parser.add_argument('-seed', dest='seed', default=41504, type=int, help='Seed for randomization')

    parser.add_argument('-restore', dest='restore', action='store_true', help='Restore from the previously saved model')
    parser.add_argument('-bias', dest='bias', action='store_true', help='Whether to use bias in the model')

    parser.add_argument('-init_dim', dest='init_dim', default=100, type=int,
                        help='Initial dimension size for entities and relations')
    parser.add_argument('-gcn_dim', dest='gcn_dim', default=200, type=int, help='Number of hidden units in GCN')
    parser.add_argument('-embed_dim', dest='embed_dim', default=200, type=int,
                        help='Embedding dimension to give as input to score function')
    parser.add_argument('-gcn_layer', dest='gcn_layer', default=1, type=int, help='Number of GCN Layers to use')
    parser.add_argument('-gcn_drop', dest='dropout', default=0.4, type=float, help='Dropout to use in GCN Layer')
    parser.add_argument('-hid_drop', dest='hid_drop', default=0.3, type=float, help='Dropout after GCN')

    # ConvE specific hyperparameters
    parser.add_argument('-hid_drop2', dest='hid_drop2', default=0.3, type=float, help='ConvE: Hidden dropout')
    parser.add_argument('-feat_drop', dest='feat_drop', default=0.3, type=float, help='ConvE: Feature Dropout')
    parser.add_argument('-k_w', dest='k_w', default=10, type=int, help='ConvE: k_w')
    parser.add_argument('-k_h', dest='k_h', default=20, type=int, help='ConvE: k_h')
    parser.add_argument('-num_filt', dest='num_filt', default=200, type=int,
                        help='ConvE: Number of filters in convolution')
    parser.add_argument('-ker_sz', dest='ker_sz', default=7, type=int, help='ConvE: Kernel size to use')

    parser.add_argument('-logdir', dest='log_dir', default='./log/', help='Log directory')
    parser.add_argument('-config', dest='config_dir', default='./config/', help='Config directory')
    parser.add_argument('-num_bases',   dest='num_bases',   default=-1,     type=int,   help='Number of basis relation vectors to use')
    # InteractE hyperparameters
    parser.add_argument('-neg_num', dest="neg_num", default=1000, type=int,
                        help='Number of negative samples to use for loss calculation')
    parser.add_argument("-strategy", type=str, default='one_to_n', help='Training strategy to use')
    parser.add_argument('-form', type=str, default='plain', help='The reshaping form to use')
    parser.add_argument('-ik_w', dest="ik_w", default=10, type=int, help='Width of the reshaped matrix')
    parser.add_argument('-ik_h', dest="ik_h", default=20, type=int, help='Height of the reshaped matrix')
    parser.add_argument('-inum_filt', dest="inum_filt", default=200, type=int, help='Number of filters in convolution')
    parser.add_argument('-iker_sz', dest="iker_sz", default=9, type=int, help='Kernel size to use')
    parser.add_argument('-iperm', dest="iperm", default=1, type=int, help='Number of Feature rearrangement to use')
    parser.add_argument('-iinp_drop', dest="iinp_drop", default=0.3, type=float, help='Dropout for Input layer')
    parser.add_argument('-ifeat_drop', dest="ifeat_drop", default=0.4, type=float, help='Dropout for Feature')
    parser.add_argument('-ihid_drop', dest="ihid_drop", default=0.3, type=float, help='Dropout for Hidden layer')
    parser.add_argument('-head_num', dest="head_num", default=1, type=int, help="Number of attention heads")
    parser.add_argument('-num_factors', dest="num_factors", default=3, type=int, help="Number of factors")
    parser.add_argument('-alpha', dest="alpha", default=1e-1, type=float, help='Dropout for Feature')
    parser.add_argument('-mi_train', dest='mi_train', action='store_true', help='whether to disentangle')
    parser.add_argument('-early_stop', dest="early_stop", default=200, type=int, help="number of early_stop")
    parser.add_argument('-no_act', dest='no_act', action='store_true', help='whether to use non_linear function')
    parser.add_argument('-no_enc', dest='no_enc', action='store_true', help='whether to use non_linear function')
    parser.add_argument('-mi_method', dest='mi_method', default='club_b', help='Composition Operation to be used in RAGAT')
    parser.add_argument('-att_mode', dest='att_mode', default='dot_weight', help='Composition Operation to be used in RAGAT')
    parser.add_argument('-mi_epoch', dest="mi_epoch", default=1, type=int, help="Number of MI_Disc training times")
    parser.add_argument('-score_method', dest='score_method', default='dot_rel', help='Composition Operation to be used in RAGAT')
    parser.add_argument('-score_order', dest='score_order', default='after', help='Composition Operation to be used in RAGAT')
    parser.add_argument('-mi_drop', dest='mi_drop', action='store_true', help='whether to use non_linear function')
    parser.add_argument('-max_gamma', type=float, default=5.0, help='Margin')
    parser.add_argument('-fix_gamma', dest='fix_gamma', action='store_true', help='whether to use non_linear function')
    parser.add_argument('-init_gamma', type=float, default=9.0, help='Margin')
    parser.add_argument('-gamma_method', dest='gamma_method', default='norm', help='Composition Operation to be used in RAGAT')
    args = parser.parse_args()

    if not args.restore:
        args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')

    set_gpu(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = Runner(args)
    model.fit()
