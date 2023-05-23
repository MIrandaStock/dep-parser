import os
import torch
import logging
from torch.utils.data import DataLoader
import datetime
# import numpy as np

from data.Dataset import UDEnglishDataset, batch_to_variable
from data.Vocab import creatVocab
from model.Args import get_args
from model.Model import ParserModel
from model.Train_helper import compute_loss_and_acc, ensure_path, get_data_path, parse
from utils.Optimizer import Optimizer
from utils.best_result import BestResult
from data.Dependency import evalDepTree, batch_variable_depTree


def train(model, train_dataloader, device, epoch, optimizer, vocab):
    model.train()
    for batch_id, batch_samples in enumerate(train_dataloader):
        words, extwords, tags, heads, rels, lengths, masks = batch_to_variable(
            batch_samples, vocab)
        words, extwords, tags, masks, heads, rels = words.to(
            device), extwords.to(device), tags.to(device), masks.to(
                device), heads.to(device), rels.to(device)

        arc_logits, rel_logits = model(words, extwords, tags, masks)
        loss, uas, las = compute_loss_and_acc(heads, rels, lengths, arc_logits,
                                              rel_logits, device)

        # Update parameter
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log
        log_str = 'Epoch {} [train] progress {}/{}, loss: {:.2f}, uas: {:.2f}, las: {:.2f}'.format(
            epoch, batch_id, len(train_dataloader), loss.item(), uas, las)
        if batch_id % 10 == 0:
            print(log_str)
            logging.info(log_str)


# def evaluate(model, dev_dataloader, device, epoch, vocab):
#     model.eval()
#     with torch.no_grad():
#         total_loss, total_uas, total_las = [], [], []
#         for batch_samples in dev_dataloader:
#             words, extwords, tags, heads, rels, lengths, masks = batch_to_variable(batch_samples, vocab)
#             words, extwords, tags, masks, heads, rels = words.to(device), extwords.to(
#                 device), tags.to(device), masks.to(device), heads.to(device), rels.to(device)

#             arc_logits, rel_logits = model(words, extwords, tags, masks)
#             loss, uas, las = compute_loss_and_acc(heads, rels, lengths, arc_logits, rel_logits, device)
#             total_loss.append(loss.item())
#             total_uas.append(uas)
#             total_las.append(las)
#         loss, uas, las = np.mean(total_loss), np.mean(total_uas), np.mean(total_las)

#         # log
#         log_str = 'Epoch {} [dev], loss: {:.2f}, uas: {:.2f}, las: {:.2f}\n'.format(epoch, loss, uas, las)
#         logging.info(log_str)
#         print(log_str)

#         return las


def test(model, test_dataloader, vocab, device):
    with torch.no_grad():
        # 保存测试生成的依存树
        # output = open(ensure_path(os.path.join(args.test_result_dir,
        #                                        args.target_domain + '.txt')), 'w', encoding='utf-8')
        arc_total_test, arc_correct_test, rel_total_test, rel_correct_test = 0, 0, 0, 0
        for batch_samples in test_dataloader:
            words, extwords, tags, heads, rels, lengths, masks, batch_samples = batch_to_variable(
                batch_samples, vocab, need_raw_sentence=True)
            words, extwords, tags, masks, heads, rels = words.to(
                device), extwords.to(device), tags.to(device), masks.to(
                    device), heads.to(device), rels.to(device)
            arc_logits, rel_logits = model(words, extwords, tags, masks)
            arcs_batch, rels_batch = parse(arc_logits, rel_logits, lengths,
                                           vocab.ROOT)
            for tree, gold_tree in zip(
                    batch_variable_depTree(batch_samples, arcs_batch, rels_batch, lengths, vocab),
                    batch_samples):
                # printDepTree(output, tree)
                arc_total, arc_correct, rel_total, rel_correct = evalDepTree(
                    predict=tree, gold=gold_tree)
                arc_total_test += arc_total
                arc_correct_test += arc_correct
                rel_total_test += rel_total
                rel_correct_test += rel_correct
        # output.close()
        test_uas = arc_correct_test * 100.0 / arc_total_test
        test_las = rel_correct_test * 100.0 / rel_total_test

    return test_uas, test_las


if __name__ == "__main__":
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Get data path
    train_file = get_data_path(args.data_path, args.source_domain,
                               'train.conllu')  # 多个源领域文件列表
    dev_file = get_data_path(args.data_path, args.target_domain,
                             'dev.conllu')  # 只有1个目标领域文件
    test_file = get_data_path(args.data_path, args.target_domain,
                              'test.conllu')  # 只有1个目标领域文件

    # vocab_path = ensure_path(os.path.join(args.best_model_dir, args.target_domain, 'vocab.pt'))
    vocab = creatVocab(train_file, min_occur_count=2)
    pretrained_emb = vocab.load_pretrained_embs(args.emb)
    # print(vocab.print_rels())
    # exit()
    # torch.save(vocab, vocab_path)
    # torch.save(pretrained_emb, os.path.join(args.best_model_dir, "glove_100.pt"))

    train_data = UDEnglishDataset(train_file, vocab)
    train_dataloader = DataLoader(train_data,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  collate_fn=lambda x: x)
    devDataset = UDEnglishDataset(dev_file, vocab)
    devDataloader = DataLoader(devDataset,
                               batch_size=args.test_batch_size,
                               shuffle=True,
                               collate_fn=lambda x: x)
    testDataset = UDEnglishDataset(test_file, vocab)
    testDataloader = DataLoader(testDataset,
                                batch_size=args.test_batch_size,
                                shuffle=False,
                                collate_fn=lambda x: x)

    # log
    now = datetime.datetime.now()
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt='%Y-%m-%d  %H:%M:%S %a',
        filename=ensure_path(
            os.path.join(args.log_dir, args.target_domain,
                         now.strftime("%Y-%m-%d %H:%M:%S") + '_log.txt')))

    # Init model
    model = ParserModel(vocab, args, pretrained_emb)
    print(model)
    model = model.to(device)

    # Optimizer
    optimizer = Optimizer(model.parameters(), lr=args.lr, decay_step=100)

    # Record
    best_result = BestResult()

    # Train loop
    for epoch in range(args.num_epochs):
        print('[{}] Epoch {}/{} ...'.format(args.target_domain, epoch,
                                            args.num_epochs))
        train(model,
              train_dataloader,
              device,
              epoch=epoch,
              optimizer=optimizer,
              vocab=vocab)

        # test
        test_uas, test_las = test(model, testDataloader, vocab, device)
        logging.info('Epoch {} [Test] UAS = {:.2f}, LAS = {:.2f}'.format(
            epoch, test_uas, test_las))
        if best_result.is_new_record(UAS=test_uas, LAS=test_las, epoch=epoch):
            logging.info('Epoch {} [Test] '.format(epoch) + str(best_result))
            torch.save(
                model.state_dict(),
                ensure_path(
                    os.path.join(args.best_model_dir, args.target_domain,
                                 'best_meta_{}.pt'.format(epoch))))
        if epoch - best_result.best_LAS_epoch >= 10:
            info = "[Epoch {}] Stop training because there is no promotion in 10 epochs.".format(
                epoch)
            print(info)
            print('[BEST RESULT in Epoch {}] '.format(epoch) +
                  str(best_result))
            logging.info('[BEST RESULT in Epoch {}] '.format(epoch) +
                         str(best_result))
            break

    print('[BEST RESULT in Epoch {}] '.format(epoch) + str(best_result))
    logging.info('[BEST RESULT in Epoch {}] '.format(epoch) + str(best_result))
