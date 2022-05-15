import torch
import torch.nn as nn
from transformers import AutoModel

from model.attention import AxisAttention
from model.bilinear import Bilinear
from long_seq import process_long_input
from loss import multilabel_categorical_crossentropy


class DocRE(nn.Module):
    def __init__(self, cfg, args):
        super(DocRE, self).__init__()
        self.bert_model = AutoModel.from_pretrained(cfg.bert_path)
        self.embedding_size = self.bert_model.config.hidden_size
        self.W_dim = cfg.W_dim
        if (self.W_dim / cfg.split_k) == (self.W_dim // cfg.split_k):
            self.split_k = cfg.split_k
        else:
            raise ValueError(f"{self.W_dim} can't split to {cfg.split_k} parts")
        self.W_s = nn.Parameter(torch.ones(self.embedding_size, cfg.W_dim))
        self.W_o = nn.Parameter(torch.ones(self.embedding_size, cfg.W_dim))
        self.W_c = nn.Parameter(torch.ones(self.embedding_size, cfg.W_dim))

        nn.init.xavier_uniform_(self.W_s)
        nn.init.xavier_uniform_(self.W_o)
        nn.init.xavier_uniform_(self.W_c)

        self.bilinear = Bilinear(self.W_dim // self.split_k, self.W_dim // self.split_k, cfg.W_dim)
        self.cls_token_id = args.cls_token_id
        self.sep_token_id = args.sep_token_id
        self.atte_layer = AxisAttention(cfg.W_dim, cfg.attn_dim)
        self.classifier = nn.Linear(cfg.W_dim, cfg.num_class)
        self.loss_fn = multilabel_categorical_crossentropy
        self.num_class = cfg.num_class

    def _get_embed_and_att(self, input_ids, attention_mask):
        sequence_output, attention = process_long_input(self.bert_model, input_ids, attention_mask, [self.cls_token_id],
                                                        [self.sep_token_id])
        return sequence_output, attention

    def better_output(self, outputs, features, id2rel, threshold=0.0, drop_na=True):
        """
        :param outputs: list of output , n , n , num_class
        :return:
        """
        fn = lambda x: x[-1] != 0
        res = []
        assert len(outputs) == len(features)
        for i in range(len(outputs)):
            output, title = outputs[i], features[i]['title']
            if drop_na:
                pred = list(filter(fn, zip(*[i.tolist() for i in torch.where(output > threshold)])))
            else:
                pred = list(zip(*[i.tolist() for i in torch.where(output > threshold)]))

            res.extend(list(map(lambda x:{'title':title, 'h_idx':x[0], 't_idx':x[1], 'r': id2rel[x[2]]}, pred)))

        return res

    def forward(self, input_ids=None, attention_mask=None, label_matrix=None, entity_pos=None, return_device='cpu'):
        """

        :param input_ids:
        :param attention_mask:
        :param entity_pos: 开始标记的位置 batch, entity, mention,  list list list
        :return:
        """
        batch_size = input_ids.size(0)
        sequence_output, attention = self._get_embed_and_att(input_ids, attention_mask)
        outputs = []
        if label_matrix is not None:
            loss = 0
        for i in range(batch_size):
            cur_global_rep = []
            cur_global_att = []
            for e in entity_pos[i]:
                cur_global_rep.append(torch.logsumexp(sequence_output[i, e], dim=0))
                cur_global_att.append(torch.sum(attention[i, :, e], dim=1))
            cur_global_rep = torch.stack(cur_global_rep)  # N, hidden_size
            cur_global_att = torch.stack(cur_global_att)  # N, H, l
            N = cur_global_rep.size(0)

            cur_global_att = torch.sum(cur_global_att.unsqueeze(1) * cur_global_att.unsqueeze(0), dim=-2,
                                       keepdim=True)  # N, N, 1, l
            cur_global_att = cur_global_att / cur_global_att.sum(-1, keepdim=True)
            cur_global_att = torch.matmul(cur_global_att, sequence_output[i]).squeeze(-2)  # 得到c(s,o), 对称的

            cur_global_att = torch.matmul(cur_global_att, self.W_c)
            Z_sub = torch.tanh((torch.matmul(cur_global_rep, self.W_s).unsqueeze(1) + cur_global_att))
            Z_obj = torch.tanh((torch.matmul(cur_global_rep, self.W_o).unsqueeze(1) + cur_global_att))

            Z_sub = torch.stack(Z_sub.split(self.split_k, -1), -1).unsqueeze(-2)
            Z_obj = torch.stack(Z_obj.split(self.split_k, -1), -1).unsqueeze(-2)

            g_s_o = self.bilinear(Z_sub, Z_obj).squeeze().sum(dim=-2)

            r_s_o = self.atte_layer(g_s_o)
            output = self.classifier(r_s_o)
            if return_device == 'cpu':
                outputs.append(output.cpu())
            elif return_device == 'cuda':
                outputs.append(output.cuda())
            output = output.permute(2, 0, 1)
            if label_matrix is not None:
                label = torch.from_numpy(label_matrix[i]).permute(2, 0, 1).to(input_ids)
                loss += self.loss_fn(output.unsqueeze(0), label.unsqueeze(0)).mean()

        if label_matrix is not None:
            loss /= batch_size
            return loss, outputs
        return outputs


if __name__ == '__main__':
    from transformers import AutoTokenizer
    import argparse
    from torch.utils.data import DataLoader
    from utlis import collate_fn
    # from prepro import read_docred
    import pickle
    from model_config import docred_config, Config

    tokenizer = AutoTokenizer.from_pretrained(docred_config.bert_path)
    args = Config(cls_token_id=tokenizer.cls_token_id, sep_token_id=tokenizer.sep_token_id)
    features = pickle.load(open('../dataset/docred/train_annotated_roberta_docred.pkl', 'rb'))
    dataloader = DataLoader(features, batch_size=2, collate_fn=collate_fn, shuffle=True)

    it = iter(dataloader)
    a = next(it)

    model = DocRE(docred_config, args)
    model.cuda()
    rs = model(a[0].cuda(), a[1].cuda(), a[2], a[3])
