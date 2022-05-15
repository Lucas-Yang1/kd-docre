import argparse
import os
import time
from datetime import datetime
import numpy as np
import torch

import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model.docre import DocRE
from utlis import set_seed, collate_fn
from evaluation import official_evaluate
from prepro import ReadDataset
from model_config import docred_config
from generate_pseudo import DistantDataset

rel2id = json.load(open('./meta/rel2id.json', 'r'))
id2rel = {value: key for key, value in rel2id.items()}


def train(args, model, train_features, dev_features, test_features):
    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(args.log_dir, 'a+') as f_log:
                f_log.write(s + '\n')
    def finetune(features, optimizer, num_epoch, num_steps, model):
        cur_model = model.module if hasattr(model, 'module') else model
        if args.train_from_saved_model != '':
            best_score = torch.load(args.train_from_saved_model)["best_f1"]
            epoch_delta = torch.load(args.train_from_saved_model)["epoch"] + 1
        else:
            epoch_delta = 0
            best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        train_iterator = [epoch + epoch_delta for epoch in range(num_epoch)]
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        global_step = 0
        log_step = 100
        total_loss = 0
        


        #scaler = GradScaler()
        for epoch in train_iterator:
            start_time = time.time()
            optimizer.zero_grad()

            for step, batch in enumerate(train_dataloader):
                model.train()

                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'label_matrix': batch[2],
                          'entity_pos': batch[3]
                          }
                #with autocast():
                outputs = model(**inputs)
                loss = outputs[0] / args.gradient_accumulation_steps
                total_loss += loss.item()
                #    scaler.scale(loss).backward()
               

                loss.backward()

                if step % args.gradient_accumulation_steps == 0:
                    #scaler.unscale_(optimizer)
                    if args.max_grad_norm > 0:
                        # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(cur_model.parameters(), args.max_grad_norm)
                    #scaler.step(optimizer)
                    #scaler.update()
                    #scheduler.step()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    num_steps += 1
                    if global_step % log_step == 0:
                        cur_loss = total_loss / log_step
                        elapsed = time.time() - start_time
                        logging(
                            '| epoch {:2d} | step {:4d} | min/b {:5.2f} | lr {} | train loss {:5.3f}'.format(
                                epoch, global_step, elapsed / 60, scheduler.get_last_lr(), cur_loss * 1000))
                        total_loss = 0
                        start_time = time.time()

                if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                # if step ==0:
                    logging('-' * 89)
                    eval_start_time = time.time()
                    dev_score, dev_output = evaluate(args, model, dev_features, id2rel=id2rel, threshold=0.0, tag="dev")

                    logging(
                        '| epoch {:3d} | time: {:5.2f}s | dev_result:{}'.format(epoch, time.time() - eval_start_time,
                                                                                dev_output))
                    logging('-' * 89)
                    if dev_score > best_score:
                        best_score = dev_score
                        logging(
                            '| epoch {:3d} | best_f1:{}'.format(epoch, best_score))
                        pred = report(args, model, test_features)
                        with open("./submit_result/best_result.json", "w") as fh:
                            json.dump(pred, fh)
                        if args.save_path != "":
                            torch.save({
                                'epoch': epoch,
                                'checkpoint': cur_model.state_dict(),
                                'best_f1': best_score,
                                'optimizer': optimizer.state_dict()
                            }, args.save_path
                            , _use_new_zipfile_serialization=False)
        return num_steps

    cur_model = model.module if hasattr(model, 'module') else model
    extract_layer = [ "bilinear"]
    bert_layer = ['bert_model']
    optimizer_grouped_parameters = [
        {"params": [p for n, p in cur_model.named_parameters() if any(nd in n for nd in bert_layer)], "lr": args.bert_lr},
        {"params": [p for n, p in cur_model.named_parameters() if any(nd in n for nd in extract_layer)], "lr": 1e-3},
        {"params": [p for n, p in cur_model.named_parameters() if not any(nd in n for nd in extract_layer + bert_layer)]},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.train_from_saved_model != '':
        optimizer.load_state_dict(torch.load(args.train_from_saved_model)["optimizer"])
        print("load saved optimizer from {}.".format(args.train_from_saved_model))
    

    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps, model)


def evaluate(args, model, features, id2rel, threshold, tag="dev"):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    total_loss = 0
    for i, batch in enumerate(dataloader):
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'label_matrix': batch[2],
                  'entity_pos': batch[3]
                  }
        with torch.no_grad():
            output = model(**inputs)
            loss = output[0]
            pred = output[1]
            preds.extend(pred)
            total_loss += loss.item()

    average_loss = total_loss / (i + 1)
    ans = model.better_output(preds, features, id2rel, threshold)

    if len(ans) > 0:
        best_f1, _, best_f1_ign, _, re_p, re_r = official_evaluate(ans, args.data_dir)
    output = {
        tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
        tag + "_re_p": re_p * 100,
        tag + "_re_r": re_r * 100,
        tag + "_average_loss": average_loss
    }
    return best_f1, output


def report(args, model, features, threshold=0.0):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  # 'label_matrix': batch[2],
                  'entity_pos': batch[3]
                  }

        with torch.no_grad():
            pred = model(**inputs)
            preds.extend(pred)

    preds = model.better_output(preds, features, id2rel, threshold)
    return preds


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--dataset", default='docred', type=str)
    parser.add_argument("--log_dir", default='log.log', type=str)
    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--distant_file", default='train_distant.json', type=str)
    parser.add_argument("--train_distant", default=True, type=bool)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_path", default="./checkpoint/docred/best2.pth", type=str)
    parser.add_argument("--load_path", default="", type=str)
    parser.add_argument("--train_from_saved_model", default='./checkpoint/docred/best.pth', type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=1, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=4, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=4e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--bert_lr", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                         help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=5000, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")



    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    print('args:',args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else docred_config.bert_path,
    )

    Dataset = ReadDataset(args.dataset, tokenizer, args.max_seq_length, transformers=docred_config.transformer_type)

    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)
    train_features = Dataset.read(train_file)
    dev_features = Dataset.read(dev_file)
    test_features = Dataset.read(test_file)

    args.cls_token_id = tokenizer.cls_token_id
    args.sep_token_id = tokenizer.sep_token_id

    set_seed(args)
    model = DocRE(docred_config, args)
    if args.train_from_saved_model != '':
        model.load_state_dict(torch.load(args.train_from_saved_model)["checkpoint"])
        print("load saved model from {}.".format(args.train_from_saved_model))
    
    # if torch.cuda.device_count() > 1:
    #         print("Let's use", torch.cuda.device_count(), "GPUs!")
    #         model = torch.nn.DataParallel(model, device_ids = list(range(torch.cuda.device_count())))
    model.to(device)

    if args.train_distant:
        model_ = DocRE(docred_config, args)
        if args.train_from_saved_model == '':
            print("WARNING, distant_pseudo_label will generated by random model params!!!")
        model_.load_state_dict(torch.load(args.train_from_saved_model)["checkpoint"])
        train_features = DistantDataset.from_json_to_pseduo(model_, tokenizer, os.path.join(args.data_dir, args.distant_file), docred_config.transformer_type,
                                                            remake=False, max_seq_length=args.max_seq_length, device=args.device)

    if args.load_path == "":  # Training
        train(args, model, train_features, dev_features, test_features)
    else:  # Testing
        model.load_state_dict(torch.load(args.load_path)['checkpoint'])
        T_features = test_features  # Testing on the test set
        T_score, T_output = evaluate(args, model, T_features, tag="test")
        pred = report(args, model, T_features)
        with open("./submit_result/result.json", "w") as fh:
            json.dump(pred, fh)


if __name__ == "__main__":
    main()
