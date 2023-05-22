# coding=utf-8
import tqdm

from util import *
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def difference(a, b):
    _b = set(b)
    return [item for item in a if item not in _b]

class Data:
    def __init__(self, args):
        set_seed(args.seed)
        max_seq_lengths = {'clinc': 30, 'stackoverflow': 45, 'banking': 55}
        args.max_seq_length = max_seq_lengths[args.dataset]

        processor = DatasetProcessor()
        self.data_dir = os.path.join(args.data_dir, args.dataset)
        self.all_label_list = processor.get_labels(self.data_dir)

        self.all_label_list = [str(item) for item in self.all_label_list]
        self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)
        print(self.n_known_cls)
        self.known_label_list = list(np.random.choice(np.array(self.all_label_list), self.n_known_cls, replace=False))
        self.unknown_label_list = difference(self.all_label_list, self.known_label_list)

        self.num_labels = int(len(self.all_label_list) * args.cluster_num_factor)

        self.train_labeled_examples, self.train_unlabeled_examples = self.get_examples(processor, args, 'train')
        print('num_labeled_samples', len(self.train_labeled_examples))
        print('num_unlabeled_samples', len(self.train_unlabeled_examples))
        self.eval_examples = self.get_examples(processor, args, 'eval')
        self.test_examples = self.get_examples(processor, args, 'test')
        self.test_labeled_examples, self.test_unlabeled_examples = self.get_examples(processor, args, 'test', True)
        self.test_labeled_dataloader = self.get_loader(self.test_labeled_examples, args, 'test', test_labeled=True)
        self.test_unlabeled_dataloader = self.get_loader(self.test_unlabeled_examples, args, 'test',
                                                         test_unlabeled=True)
        self.train_labeled_dataloader = self.get_loader(self.train_labeled_examples, args, 'train')
        if args.augment_data_simcse:
            self.train_labeled_examples_augment, jihe = self.data_augment(args.dataset, args.k, args.seed)
            self.train_unlabeled_examples_augment = []
            for i, item in enumerate(self.train_unlabeled_examples):
                if i not in jihe:
                    self.train_unlabeled_examples_augment.append(self.train_unlabeled_examples[i])
            self.semi_input_ids, self.semi_input_mask, self.semi_segment_ids, self.semi_label_ids = self.get_semi(
                self.train_labeled_examples_augment, self.train_unlabeled_examples_augment, args)
            self.train_labeled_dataloader = self.get_loader(self.train_labeled_examples_augment, args, 'train')
        else:
            # improve pretrain using augment_data_simcse?
            # self.train_labeled_examples_augment, jihe = self.data_augment(args.dataset, args.k, args.seed)
            # self.train_labeled_dataloader = self.get_loader(self.train_labeled_examples_augment, args, 'train')
            self.semi_input_ids, self.semi_input_mask, self.semi_segment_ids, self.semi_label_ids = self.get_semi(
                self.train_labeled_examples, self.train_unlabeled_examples, args)
        self.train_semi_dataloader = self.get_semi_loader(self.semi_input_ids, self.semi_input_mask,
                                                          self.semi_segment_ids, self.semi_label_ids, args)
        self.eval_dataloader = self.get_loader(self.eval_examples, args, 'eval')
        self.test_dataloader = self.get_loader(self.test_examples, args, 'test')

    def get_examples(self, processor, args, mode='train', separate=False):
        ori_examples = processor.get_examples(self.data_dir, mode)

        if mode == 'train':
            train_labels = np.array([example.label for example in ori_examples])
            train_labeled_ids = []
            for label in self.known_label_list:
                num = round(len(train_labels[train_labels == label]) * args.labeled_ratio)
                pos = list(np.where(train_labels == label)[0])
                train_labeled_ids.extend(random.sample(pos, num))

            train_labeled_examples, train_unlabeled_examples = [], []
            for idx, example in enumerate(ori_examples):
                if idx in train_labeled_ids:
                    train_labeled_examples.append(example)
                else:
                    train_unlabeled_examples.append(example)

            return train_labeled_examples, train_unlabeled_examples

        elif mode == 'eval':
            eval_examples = []
            for example in ori_examples:
                if example.label in self.known_label_list:
                    eval_examples.append(example)
            return eval_examples

        elif mode == 'test' and separate == False:
            return ori_examples
        else:
            test_labels = np.array([example.label for example in ori_examples])
            test_labeled_ids = []
            for label in self.known_label_list:
                num = len(test_labels[test_labels == label])
                pos = list(np.where(test_labels == label)[0])
                test_labeled_ids.extend(random.sample(pos, num))

            test_labeled_examples, test_unlabeled_examples = [], []
            for idx, example in enumerate(ori_examples):
                if idx in test_labeled_ids:
                    test_labeled_examples.append(example)
                else:
                    test_unlabeled_examples.append(example)

            return test_labeled_examples, test_unlabeled_examples

    def get_semi(self, labeled_examples, unlabeled_examples, args):
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
        labeled_features = convert_examples_to_features(labeled_examples, self.known_label_list, args.max_seq_length,
                                                        tokenizer)
        unlabeled_features = convert_examples_to_features(unlabeled_examples, self.all_label_list, args.max_seq_length,
                                                          tokenizer)

        labeled_input_ids = torch.tensor([f.input_ids for f in labeled_features], dtype=torch.long)
        labeled_input_mask = torch.tensor([f.input_mask for f in labeled_features], dtype=torch.long)
        labeled_segment_ids = torch.tensor([f.segment_ids for f in labeled_features], dtype=torch.long)
        labeled_label_ids = torch.tensor([f.label_id for f in labeled_features], dtype=torch.long)

        unlabeled_input_ids = torch.tensor([f.input_ids for f in unlabeled_features], dtype=torch.long)
        unlabeled_input_mask = torch.tensor([f.input_mask for f in unlabeled_features], dtype=torch.long)
        unlabeled_segment_ids = torch.tensor([f.segment_ids for f in unlabeled_features], dtype=torch.long)
        unlabeled_label_ids = torch.tensor([-1 for f in unlabeled_features], dtype=torch.long)

        semi_input_ids = torch.cat([labeled_input_ids, unlabeled_input_ids])
        semi_input_mask = torch.cat([labeled_input_mask, unlabeled_input_mask])
        semi_segment_ids = torch.cat([labeled_segment_ids, unlabeled_segment_ids])
        semi_label_ids = torch.cat([labeled_label_ids, unlabeled_label_ids])
        return semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids

    def get_semi_loader(self, semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids, args):
        semi_data = TensorDataset(semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids)
        semi_sampler = SequentialSampler(semi_data)
        semi_dataloader = DataLoader(semi_data, sampler=semi_sampler, batch_size=args.train_batch_size)

        return semi_dataloader

    def get_loader(self, examples, args, mode='train', test_labeled=False, test_unlabeled=False):
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
        if mode == 'train' or mode == 'eval':
            features = convert_examples_to_features(examples, self.known_label_list, args.max_seq_length, tokenizer)
        elif mode == 'test':
            if test_labeled == True:
                features = convert_examples_to_features(examples, self.known_label_list, args.max_seq_length, tokenizer)
            elif test_unlabeled == True:
                features = convert_examples_to_features(examples, self.unknown_label_list, args.max_seq_length,
                                                        tokenizer)
            else:
                features = convert_examples_to_features(examples, self.all_label_list, args.max_seq_length, tokenizer)

        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

        if mode == 'train':
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=args.train_batch_size)
        elif mode == 'eval' or mode == 'test':
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=args.eval_batch_size)
        return dataloader

    def data_augment(self, dataset, k, seed):
        def top_K_ids(data, k):
            data = np.array(data)
            idx = data.argsort()[-k:][::-1]
            return list(idx)

        if os.path.exists("./Neighbour/cosine_matrix_{}_topk{}_seed{}.npy".format(dataset, k, seed)):
            cosine_matrix = np.load('./Neighbour/cosine_matrix_{}_topk{}_seed{}.npy'.format(dataset, k, seed))
            cosine_matrix = cosine_matrix.tolist()
        else:
            tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
            model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            labeled_textlist = []
            labeled_labellist = []
            unlabeled_textlist = []
            unlabeled_labellist = []
            labeled_embeddinglist = []
            unlabeled_embeddinglist = []
            for example in self.train_labeled_examples:
                labeled_textlist.append(example.text_a)
                labeled_labellist.append(example.label)
            for example in self.train_unlabeled_examples:
                unlabeled_textlist.append(example.text_a)
                unlabeled_labellist.append(example.label)
            for example in tqdm(labeled_textlist, desc="labeled_embeddinglist"):
                inputs_labeled = tokenizer(example, padding=True, truncation=True, return_tensors="pt")
                inputs_labeled.to(device)
                with torch.no_grad():
                    embeddings_labeled = model(**inputs_labeled, output_hidden_states=True,
                                               return_dict=True).pooler_output
                    labeled_embeddinglist.append(embeddings_labeled[0])
            for example in tqdm(unlabeled_textlist, desc="unlabeled_embeddinglist"):
                inputs_unlabeled = tokenizer(example, padding=True, truncation=True, return_tensors="pt")
                inputs_unlabeled.to(device)
                with torch.no_grad():
                    embeddings_unlabeled = model(**inputs_unlabeled, output_hidden_states=True,
                                                 return_dict=True).pooler_output
                    unlabeled_embeddinglist.append(embeddings_unlabeled[0])

            cosine_matrix = []
            for i, tensor in tqdm(enumerate(labeled_embeddinglist), desc="findtopk"):
                top = []
                for j, tensor_un in enumerate(unlabeled_embeddinglist):
                    top.append(1 - cosine(tensor.cpu(), tensor_un.cpu()))
                idxlist = top_K_ids(top, k)
                cosine_matrix.append(idxlist)
            a = np.array(cosine_matrix)
            np.save('./Neighbour/cosine_matrix_{}_topk{}_seed{}.npy'.format(dataset, k, seed), a)  # 保存为.npy格式
        output = copy.deepcopy(self.train_labeled_examples)
        jihe = set()
        for i, idx_list in enumerate(cosine_matrix):
            for idx in idx_list:
                if idx not in jihe:
                    jihe.add(idx)
                    temp = copy.deepcopy(self.train_unlabeled_examples[idx])
                    temp.label = self.train_labeled_examples[i].label
                    output.append(temp)
        return output, jihe

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class DatasetProcessor(DataProcessor):

    def get_examples(self, data_dir, mode):
        if mode == 'train':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        elif mode == 'eval':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "train")
        elif mode == 'test':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self, data_dir):
        """See base class."""
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
        labels = np.unique(np.array(test['label']))

        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if len(line) != 2:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)  # For dialogue context
        else:
            tokens_b.pop()

