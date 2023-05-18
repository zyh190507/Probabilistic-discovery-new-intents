from util import *
from model import *
from dataloader import *

class PretrainModelManager:
    
    def __init__(self, args, data):
        set_seed(args.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.bert_model=="sentence-transformers/paraphrase-mpnet-base-v2":
            self.model = MPNetForModel(args.bert_model, num_labels=data.n_known_cls)
        else:
            self.model = BertForModel.from_pretrained(args.bert_model, cache_dir="", num_labels=data.n_known_cls)

        '''
        a=torch.load("./para_model_MTP2step")
        self.model.load_state_dict(a,strict=False)
        '''
        '''
        b = torch.load("./../NID_ACL/MTP2step")
        for key in list(b.keys()):
            if "backbone" in key:
                b[key[9:]] = b.pop(key)
        self.model.load_state_dict(b, strict=False)
        '''
        '''
        b=torch.load("./../NID_ACL/MTP2step_sf_12.1")
        for key in list(b.keys()):
            if "backbone" in key:
                b[key[9:]]=b.pop(key)
        self.model.load_state_dict(b,strict=False)
        '''
        '''
        b = torch.load("./stackoverflowMTP")
        for key in list(b.keys()):
            if "backbone" in key:
                b[key[9:]] = b.pop(key)
        self.model.load_state_dict(b, strict=False)
        '''
        if args.freeze_bert_parameters_pretrain:
            self.freeze_parameters(self.model)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.data=data
        self.num_train_optimization_steps = int(len(data.train_labeled_examples) / args.train_batch_size) * args.num_pretrain_epochs
        self.optimizer = self.get_optimizer(args)
        self.best_eval_score = 0

    def eval(self, args):
        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, self.data.n_known_cls)).to(self.device)
        
        for batch in tqdm(self.data.eval_dataloader, desc="Eval Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                _, logits = self.model(input_ids, segment_ids, input_mask, mode = 'eval')
                total_labels = torch.cat((total_labels,label_ids))
                total_logits = torch.cat((total_logits, logits))
        
        total_probs, total_preds = F.softmax(total_logits.detach(), dim=1).max(dim = 1)
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)

        return acc


    def train(self, args):
        self.model.to(self.device)
        print("Start finetune in labeled dataset")
        self.best_eval_score=0
        wait = 0
        best_model = None
        for epoch in trange(int(args.num_pretrain_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(self.data.train_labeled_dataloader, desc="Train Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):
                    loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode = "train")
                    loss.backward()
                    tr_loss += loss.item()
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
            
            loss = tr_loss / nb_tr_steps
            print('train_loss',loss)
            
            eval_score = self.eval(args)
            print('eval_score',eval_score)
            
            if eval_score > self.best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                self.best_eval_score = eval_score
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break
                
        self.model = best_model
        '''
        a = torch.load("./afterpretrain_pre")
        self.model.load_state_dict(a, strict=False)
        '''
        if args.save_model:
            self.save_model(args)

    def get_optimizer(self, args):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                         lr = args.lr_ft,
                         warmup = args.warmup_proportion,
                         t_total = self.num_train_optimization_steps)   
        return optimizer

    def save_model(self, args,pre=False):
        if not os.path.exists(args.pretrain_dir):
            os.makedirs(args.pretrain_dir)
            self.save_model = self.model.module if hasattr(self.model, 'module') else self.model
        model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        model_config_file = os.path.join(args.pretrain_dir, CONFIG_NAME)
        torch.save(self.save_model.state_dict(), model_file)
        with open(model_config_file, "w") as f:
            f.write(self.save_model.config.to_json_string())

    def freeze_parameters(self,model):
        for name, param in model.bert.named_parameters():
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
