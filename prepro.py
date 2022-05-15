import numpy
import ujson as json
import os
import pickle
import random
import numpy as np
from tqdm import tqdm

docred_rel2id = json.load(open('./meta/rel2id.json', 'r'))

class ReadDataset:
    def __init__(self, dataset: str, tokenizer, max_seq_Length: int = 1024,
             transformers: str = 'bert') -> None:
        self.transformers = transformers
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_Length = max_seq_Length

    def read(self, file_in: str):
        save_file = file_in.split('.json')[0] + '_' + self.transformers + '_' \
                        + self.dataset + '.pkl'
        if self.dataset == 'docred':
            return read_docred(self.transformers, file_in, save_file, self.tokenizer, self.max_seq_Length)
        else:
            raise RuntimeError("No read func for this dataset.")



def read_docred(transfermers, file_in, save_file, tokenizer, max_seq_length=1024):
    if os.path.exists(save_file):
        with open(file=save_file, mode='rb') as fr:
            features = pickle.load(fr)
            fr.close()
        print('load preprocessed data from {}.'.format(save_file))
        return features
    else:
        max_len = 0
        up512_num = 0
        i_line = 0
        pos_samples = 0
        neg_samples = 0
        features = []
        if file_in == "":
            return None
        with open(file_in, "r") as fh:
            data = json.load(fh)
        if transfermers == 'bert':
            # entity_type = ["ORG", "-",  "LOC", "-",  "TIME", "-",  "PER", "-", "MISC", "-", "NUM"]
            entity_type = ["-", "ORG", "-",  "LOC", "-",  "TIME", "-",  "PER", "-", "MISC", "-", "NUM"]


        for sample in tqdm(data, desc="Example"):
            sents = []
            sent_map = []

            entities = sample['vertexSet']
            entity_start, entity_end = [], []
            mention_types = []
            for entity in entities:
                for mention in entity:
                    sent_id = mention["sent_id"]
                    pos = mention["pos"]
                    entity_start.append((sent_id, pos[0]))
                    entity_end.append((sent_id, pos[1] - 1))
                    mention_types.append(mention['type'])

            for i_s, sent in enumerate(sample['sents']):
                new_map = {}
                for i_t, token in enumerate(sent):
                    tokens_wordpiece = tokenizer.tokenize(token)
                    if (i_s, i_t) in entity_start:
                        t = entity_start.index((i_s, i_t))
                        if transfermers == 'bert':
                            mention_type = mention_types[t]
                            special_token_i = entity_type.index(mention_type)
                            special_token = ['[unused' + str(special_token_i) + ']']
                        else:
                            special_token = ['*']
                        tokens_wordpiece = special_token + tokens_wordpiece
                        # tokens_wordpiece = ["[unused0]"]+ tokens_wordpiece

                    if (i_s, i_t) in entity_end:
                        t = entity_end.index((i_s, i_t))
                        if transfermers == 'bert':
                            mention_type = mention_types[t]
                            special_token_i = entity_type.index(mention_type) + 50
                            special_token = ['[unused' + str(special_token_i) + ']']
                        else:
                            special_token = ['*']
                        tokens_wordpiece = tokens_wordpiece + special_token

                        # tokens_wordpiece = tokens_wordpiece + ["[unused1]"]
                        # print(tokens_wordpiece,tokenizer.convert_tokens_to_ids(tokens_wordpiece))

                    new_map[i_t] = len(sents)
                    sents.extend(tokens_wordpiece)
                new_map[i_t + 1] = len(sents)
                sent_map.append(new_map)

            if len(sents)>max_len:
                max_len=len(sents)
            if len(sents)>512:
                up512_num += 1

            train_triple = {}
            if "labels" in sample:
                for label in sample['labels']:
                    evidence = label['evidence']
                    r = int(docred_rel2id[label['r']])
                    if (label['h'], label['t']) not in train_triple:
                        train_triple[(label['h'], label['t'])] = [
                            {'relation': r, 'evidence': evidence}]
                    else:
                        train_triple[(label['h'], label['t'])].append(
                            {'relation': r, 'evidence': evidence})

            entity_pos = []
            for e in entities:
                entity_pos.append([])
                mention_num = len(e)
                for m in e:
                    start = sent_map[m["sent_id"]][m["pos"][0]]
                    end = sent_map[m["sent_id"]][m["pos"][1]]
                    entity_pos[-1].append((start, end,))


            relations, hts = [], []
            relation_matrix = np.zeros((len(entity_pos), len(entity_pos), len(docred_rel2id)))
            # Get positive samples from dataset
            for h, t in train_triple.keys():
                relation = [0] * len(docred_rel2id)
                for mention in train_triple[h, t]:
                    relation[mention["relation"]] = 1
                    relation_matrix[h, t, mention["relation"]] = 1
                    evidence = mention["evidence"]
                relations.append(relation)
                hts.append([h, t])
                pos_samples += 1

            # Get negative samples from dataset
            for h in range(len(entities)):
                for t in range(len(entities)):
                    if h != t and [h, t] not in hts:
                        relation = [1] + [0] * (len(docred_rel2id) - 1)
                        relations.append(relation)
                        relation_matrix[h, t, docred_rel2id['Na']] = 1
                        hts.append([h, t])
                        neg_samples += 1

            assert len(relations) == len(entities) * (len(entities) - 1)

            if len(hts)==0:
                print(len(sent))
            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            i_line += 1
            feature = {'input_ids': input_ids,
                       'entity_pos': entity_pos,
                       'labels': relations,
                       'hts': hts,
                       'title': sample['title'],
                       'label_matrix': relation_matrix,
                       }
            features.append(feature)



        print("# of documents {}.".format(i_line))
        print("# of positive examples {}.".format(pos_samples))
        print("# of negative examples {}.".format(neg_samples))
        print("# {} examples len>512 and max len is {}.".format(up512_num, max_len))


        with open(file=save_file, mode='wb') as fw:
            pickle.dump(features, fw)
        print('finish reading {} and save preprocessed data to {}.'.format(file_in, save_file))

        return features



if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    read_docred('roberta', './dataset/docred/train_distant.json', 'train.pkl', tokenizer)