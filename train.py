import sys
import os
import torch
import random
import numpy as np
import time

from sklearn import metrics
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from transformers import BertTokenizer
from snippet import *
from model import ALSA_DCT
import test


def set_random_seed(args):
    # set random seed
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # random.seed(args.seed)
    # torch.cuda.manual_seed(args.seed)

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def evaluate(model, dataloader, args, vocab):
    token_vocab = vocab['token']
    polarity_vocab = vocab['polarity']
    predictions, labels = [], []

    val_loss, val_acc = 0.0, 0.0

    for step, batch in enumerate(dataloader):
        model.eval()
        with torch.no_grad():
            batch = [b.to(args.device) for b in batch]
            inputs = batch[:-1]
            label = batch[-1]

            logits = model(inputs)
            loss = F.cross_entropy(logits, label, reduction='mean')
            val_loss += loss.data

            predictions += np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
            labels += label.data.cpu().numpy().tolist()

    val_acc = metrics.accuracy_score(labels, predictions) * 100.0
    f1_score = metrics.f1_score(labels, predictions, average='macro')

    return val_loss / len(dataloader), val_acc, f1_score


def train(args, vocab, tokenizer, train_dataloader, valid_dataloader, test_dataloader, model, optimizer):
    ############################################################
    # train
    writer = SummaryWriter(log_dir=args.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    with open(args.log_path + 'log.txt', 'a') as f:
        print("Training Set: {}".format(len(train_dataloader)))
        print("Valid Set: {}".format(len(valid_dataloader)))
        print("Test Set: {}".format(len(test_dataloader)))
        f.write("Training Set: {}".format(len(train_dataloader)))
        f.write("Valid Set: {}".format(len(valid_dataloader)))
        f.write("Test Set: {}".format(len(test_dataloader)))

    train_acc_history, train_loss_history = [0.0], [0.0]
    val_acc_history, val_history, val_f1_score_history = [0.0], [0.0], [0.0]
    model_path = args.model_path + 'ALSA_DCT' + '.pth'

    in_test_epoch, in_test_acc, in_test_f1 = 0, 0.0, 0.0
    patience = 0
    for epoch in range(1, args.num_epoch + 1):
        begin_time = time.time()
        with open(args.log_path + 'log.txt', 'a') as f:
            print("Epoch {}".format(epoch) + "-" * 60)
            f.write("Epoch {}".format(epoch) + "-" * 60 + '\n')

        train_loss, train_acc, train_step = 0.0, 0.0, 0

        train_all_predict = 0
        train_all_correct = 0

        for i, batch in enumerate(train_dataloader):
            model.train()
            optimizer.zero_grad()

            batch = [b.to(args.device) for b in batch]
            inputs = batch[:-1]
            label = batch[-1]

            logits = model(inputs)
            loss = F.cross_entropy(logits, label, reduction='mean')
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()

            train_all_predict += label.size()[0]
            train_all_correct += corrects.item()

            train_step += 1

            if train_step % args.log_step == 0:
                with open(args.log_path + 'log.txt', 'a') as f:
                    print('{}/{} train_loss:{:.6f}, train_acc:{:.4f}'.format(
                        i, len(train_dataloader), train_loss / train_step, 100.0 * train_all_correct / train_all_predict
                    ))
                    f.write('{}/{} train_loss:{:.6f}, train_acc:{:.4f}\n'.format(
                        i, len(train_dataloader), train_loss / train_step, 100.0 * train_all_correct / train_all_predict
                    ))
            train_acc = 100.0 * train_all_correct / train_all_predict
            val_loss, val_acc, val_f1 = evaluate(model, valid_dataloader, args, vocab)
            writer.add_scalar("loss/train", loss.item(), (epoch - 1) * len(train_dataloader) + i)
            writer.add_scalar("loss/val", val_loss, (epoch - 1) * len(train_dataloader) + i)
            writer.add_scalar("acc/train", train_acc, (epoch - 1) * len(train_dataloader) + i)
            writer.add_scalar("acc/val", val_acc, (epoch - 1) * len(train_dataloader) + i)

        # train_acc = 100.0 * train_all_correct / train_all_predict
        val_loss, val_acc, val_f1 = evaluate(model, valid_dataloader, args, vocab)
        with open(args.log_path + 'log.txt', 'a') as f:
            print(
                "[{:.2f}s] Pass!\nEnd of {} train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, f1_score: {:.4f}\n".format(
                    time.time() - begin_time, epoch, train_loss / train_step, train_acc, val_loss, val_acc, val_f1
                )
            )
            f.write(
                "[{:.2f}s] Pass!\nEnd of {} train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, f1_score: {:.4f}\n".format(
                    time.time() - begin_time, epoch, train_loss / train_step, train_acc, val_loss, val_acc, val_f1
                )
            )

        train_acc_history.append(train_acc)
        train_loss_history.append(train_loss / train_step)

        if epoch == 1 or float(val_acc) > max(val_acc_history) or float(val_f1) > max(val_f1_score_history):
            # save
            torch.save(model.state_dict(), model_path)

            patience = 0
            test_loss, test_acc, test_f1 = evaluate(model, test_dataloader, args, vocab)

            in_test_epoch = epoch
            in_test_acc = test_acc
            in_test_f1 = test_f1
            with open(args.log_path + 'log.txt', 'a') as f:
                print('-->In test: patience:{}, test_acc:{}, test_f1:{}'.format(patience, test_acc, test_f1))
                f.write('-->In test: patience:{}, test_acc:{}, test_f1:{}\n'.format(patience, test_acc, test_f1))
        else:
            patience += 1

        val_acc_history.append(float(val_acc))
        val_f1_score_history.append(val_f1)

        if patience >= args.max_patience:
            with open(args.log_path + 'log.txt', 'a') as f:
                print('Exceeding max patience', patience)
                f.write('Exceeding max patience {}\n'.format(patience))

    writer.close()

    with open(args.log_path + 'log.txt', 'a') as f:
        print('Best epoch of val: {}\t f1: {:.5f}\n\n'.format(val_f1_score_history.index(max(val_f1_score_history)),
                                                              max(val_f1_score_history)))
        f.write('Best epoch of val: {}\t f1: {:.5f}\n\n'.format(val_f1_score_history.index(max(val_f1_score_history)),
                                                                max(val_f1_score_history)))

    with open(args.log_path + 'log.txt', 'a') as f:
        print('Training ended with {} epoches.'.format(epoch))
        f.write('Training ended with {} epoches.\n'.format(epoch))

    # test
    model.load_state_dict(torch.load(model_path))
    _, last_test_acc, last_test_f1 = evaluate(model, test_dataloader, args, vocab)

    with open(args.log_path + 'log.txt', 'a') as f:
        print('----------------------------------------------------------')
        print('In Results: test_epoch:{}, test_acc:{}, test_f1:{}'.format(in_test_epoch, in_test_acc, in_test_f1))
        print('Last In Results: test_epoch:{}, test_acc:{}, test_f1:{}'.format(epoch, last_test_acc, last_test_f1))
        f.write('----------------------------------------------------------\n')
        f.write('In Results: test_epoch:{}, test_acc:{}, test_f1:{}\n'.format(in_test_epoch, in_test_acc, in_test_f1))
        f.write('Last In Results: test_epoch:{}, test_acc:{}, test_f1:{}\n'.format(epoch, last_test_acc, last_test_f1))


def run(args, vocab, tokenizer):
    print_arguments(args)

    # data
    train_dataloader, valid_dataloader, test_dataloader = load_data(args, vocab, tokenizer=tokenizer)

    # model
    model = ALSA_DCT(args).to(device=args.device)
    print(model)
    print('# parameters:', totally_parameters(model))

    # optimizer
    bert_model = model.intra_context_module.context_encoder
    bert_params_dict = list(map(id, bert_model.parameters()))

    if not args.borrow_encoder and args.plus_AA:
        bert_model_2 = model.inter_context_module.con_aspect_graph_encoder
        bert_params_dict_2 = list(map(id, bert_model_2.parameters()))
        bert_params_dict += bert_params_dict_2

    base_params = filter(lambda p: id(p) not in bert_params_dict, model.parameters())
    optimizer_grouped_parameters = [
        {"params": [p for p in base_params if p.requires_grad]},
        {"params": [p for p in bert_model.parameters() if p.requires_grad], "lr": args.bert_lr}
    ]

    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.l2)

    train(args, vocab, tokenizer, train_dataloader, valid_dataloader, test_dataloader, model, optimizer)
    print('End!')


def replace_labels(lst):
    return ['negative' if x == 0 else 'neutral' if x == 1 else 'positive' for x in lst]


def Test(tokenizer):
    model_path = args.model_path + 'ALSA_DCT' + '.pth'
    model = ALSA_DCT(args).to(device=args.device)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    test_dataloader = load_one_data(args, file_name=args.data_dir + '/test1_new.json',
                                    vocab=vocab, tokenizer=tokenizer, is_shuffle=False)

    token_vocab = vocab['token']
    polarity_vocab = vocab['polarity']
    predictions, labels = [], []

    val_loss, val_acc = 0.0, 0.0

    for step, batch in enumerate(test_dataloader):
        model.eval()
        with torch.no_grad():
            batch = [b.to(args.device) for b in batch]
            inputs = batch[:-1]
            label = batch[-1]

            logits = model(inputs)
            # loss = F.cross_entropy(logits, label, reduction='mean')
            # val_loss += loss.data

            predictions += np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
            labels += label.data.cpu().numpy().tolist()

    # val_acc = metrics.accuracy_score(labels, predictions) * 100.0
    # f1_score = metrics.f1_score(labels, predictions, average='macro')
    return replace_labels(predictions)


# parameter
args = get_parameter()
set_random_seed(args)
vocab = load_vocab(args)
bert_tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer_path)

if __name__ == '__main__':

    if args.task == 'train':
        run(args, vocab, tokenizer=bert_tokenizer)
    else:
        Test(bert_tokenizer)
