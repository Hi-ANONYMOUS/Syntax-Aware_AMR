import logging
import random
import torch
from cached_property import cached_property
from torch.utils.data import Dataset
from spring_amr.IO import read_raw_amr_data
import json

def reverse_direction(x, y, pad_token_id=1):
    input_ids = torch.cat([y['decoder_input_ids'], y['lm_labels'][:, -1:]], 1)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[input_ids == pad_token_id] = 0
    decoder_input_ids = x['input_ids'][:,:-1]
    lm_labels = x['input_ids'][:,1:]
    x = {'input_ids': input_ids, 'attention_mask': attention_mask}
    y = {'decoder_input_ids': decoder_input_ids, 'lm_labels': lm_labels}
    return x, y

class AMRDataset(Dataset):
    
    def __init__(
        self,
        paths,
        tokenizer,
        device=torch.device('cpu'),
        use_recategorization=False,
        remove_longer_than=None,
        remove_wiki=False,
        dereify=True,
        dep_file=None,
            dep_vocab_file=None,
    ):
        self.paths = paths
        self.tokenizer = tokenizer
        self.device = device
        graphs = read_raw_amr_data(paths, use_recategorization, remove_wiki=remove_wiki, dereify=dereify)
        self.all_dep_data = read_depparse_features(dep_file)
        self.all_subtoken_map, self.all_token_list = get_subtoken_map(dep_file, tokenizer)
        self.dep_tag2id = get_dep_tag_vocab(dep_vocab_file)
        self.graphs = []
        self.sentences = []
        self.linearized = []
        self.linearized_extra = []
        self.remove_longer_than = remove_longer_than
        for index, g in enumerate(graphs):
            l, e = self.tokenizer.linearize(g)
            # if e['graphs'].metadata['id'] == "bolt-eng-DF-170-181103-8887658_0014.12":
            #     print("daole")
            # l是该图的线性化表示（经过bpe的，ids表示）。e里面包含"'linearized_graphs"(该图的线性化表示，单词表示)和"graphs"（原图的三元组等）.
            try:
                self.tokenizer.batch_encode_sentences([g.metadata['snt']])
            except:
                logging.warning('Invalid sentence!')
                continue

            if remove_longer_than and len(l) > remove_longer_than:
                continue
            if len(l) > 1024:
                logging.warning('Sequence longer than 1024 included. BART does not support it!')

            self.sentences.append(g.metadata['snt'])
            self.graphs.append(g)
            self.linearized.append(l)
            self.linearized_extra.append(e)

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sample = {}
        sample['id'] = idx
        sample['sentences'] = self.sentences[idx]
        sample['dep_tags'] = self.all_dep_data[idx]
        sample['subtoken_map'] = self.all_subtoken_map[idx]
        sample['token_list'] = self.all_token_list[idx]
        if self.linearized is not None:
            sample['linearized_graphs_ids'] = self.linearized[idx]
            sample.update(self.linearized_extra[idx])            
        return sample
    
    def size(self, sample):
        return len(sample['linearized_graphs_ids'])
    
    def collate_fn(self, samples, device=torch.device('cpu')):
        x = [s['sentences'] for s in samples]
        dep_tags = [s['dep_tags'] for s in samples]
        input_tokens = [s['token_list'] for s in samples]
        subtoken_map = [s['subtoken_map'] for s in samples]
        # x, extra = self.tokenizer.batch_encode_sentences(x, device=device)
        x, extra = self.tokenizer.batch_encode_sentences_self(x, self.dep_tag2id, dep_tags, input_tokens, subtoken_map, device=device)
        if 'linearized_graphs_ids' in samples[0]:
            y = [s['linearized_graphs_ids'] for s in samples]
            y, extra_y = self.tokenizer.batch_encode_graphs_from_linearized(y, samples, device=device)
            extra.update(extra_y)
        else:
            y = None
        extra['ids'] = [s['id'] for s in samples]
        return x, y, extra
    
class AMRDatasetTokenBatcherAndLoader:
    
    def __init__(self, dataset, batch_size=800 ,device=torch.device('cpu'), shuffle=False, sort=False):
        assert not (shuffle and sort)
        self.batch_size = batch_size
        self.tokenizer = dataset.tokenizer
        self.dataset = dataset
        self.device = device
        self.shuffle = shuffle
        self.sort = sort

    def __iter__(self):
        it = self.sampler()
        it = ([[self.dataset[s] for s in b] for b in it])
        it = (self.dataset.collate_fn(b, device=self.device) for b in it)
        return it

    @cached_property
    def sort_ids(self):
        lengths = [len(s.split()) for s in self.dataset.sentences]
        ids, _ = zip(*sorted(enumerate(lengths), reverse=True))
        ids = list(ids)
        return ids

    def sampler(self):
        ids = list(range(len(self.dataset)))[::-1]
        
        if self.shuffle:
            random.shuffle(ids)
        if self.sort:
            ids = self.sort_ids.copy()

        batch_longest = 0
        batch_nexamps = 0
        batch_ntokens = 0
        batch_ids = []

        def discharge():
            nonlocal batch_longest
            nonlocal batch_nexamps
            nonlocal batch_ntokens
            ret = batch_ids.copy()
            batch_longest *= 0
            batch_nexamps *= 0
            batch_ntokens *= 0
            batch_ids[:] = []
            return ret

        while ids:
            idx = ids.pop()
            size = self.dataset.size(self.dataset[idx])
            cand_batch_ntokens = max(size, batch_longest) * (batch_nexamps + 1)
            if cand_batch_ntokens > self.batch_size and batch_ids:
                yield discharge()
            batch_longest = max(batch_longest, size)
            batch_nexamps += 1
            batch_ntokens = batch_longest * batch_nexamps
            batch_ids.append(idx)

            if len(batch_ids) == 1 and batch_ntokens > self.batch_size:
                yield discharge()

        if batch_ids:
            yield discharge()


# ------------------------自己加-------------------------
def read_json(input_tag_file):
    input_tag_data = []
    with open(input_tag_file, "r", encoding='utf-8') as reader:
        for line in reader:
            input_tag_data.append(json.loads(line))
    return input_tag_data


def read_depparse_features(data_path):
    with open(data_path, "r") as f:
        examples = [json.loads(jsonline) for jsonline in f.readlines()]

    dependencies = []

    for example in examples:
        depparse_features = example['dep_parse']
        dependency_features = []
        for sent_feature in depparse_features:
            temp_dependency = []
            for feature in sent_feature:
                word_id = feature['id']
                head_id = feature['head_id']
                deprel = feature['deprel']
                temp_dependency.append([deprel, head_id, word_id])

            dependency_features.append(temp_dependency)
        dependencies.append(dependency_features)

    return dependencies


def get_subtoken_map(data_path, tokenizer):
    with open(data_path, "r") as f:
        examples = [json.loads(jsonline) for jsonline in f.readlines()]

    all_subtoken_map = []
    all_token_list = []

    for example in examples:
        token_list = example['all_token_list']
        subtoken_map = []
        word_idx = -1
        # 一个段落可能有多个句子
        for sent_tokens in token_list:
            # 每个句子
            for token in sent_tokens:
                word_idx += 1
                subtokens = tokenizer.tokenize(token, add_special_tokens=False)
                for sidx, subtoken in enumerate(subtokens):
                    subtoken_map.append(word_idx)

        all_subtoken_map.append(subtoken_map)
        all_token_list.append(token_list)

    return all_subtoken_map, all_token_list


def get_dep_tag_vocab(data_path):
    tag2id = {}
    with open(data_path) as f:
        for line in f.readlines():
            tag, idx = line.strip().split(" ")
            tag2id[tag] = int(idx)
    return tag2id

