#author:zhaochao time:2021/6/4

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F

from torch.autograd import Variable
import os
import math
import data_loader_1d
import resnet18_1d as models
import torch.nn as nn
import  time
import numpy as np
import  random
import pickle

from utils import *

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # 并行gpu
    torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    torch.backends.cudnn.benchmark = True  # 训练集变化不大时使训练加速

def log(name,OS_Accuracy_list,OS_star_Accuracy_list,Unknow_list, Train_Time):
    f = open('./store/'+name+'.txt', 'w')


    f.write('OS_acc:')
    f.write(str(OS_Accuracy_list))
    f.write('\r\n')
    f.write('OS_star_loss:')
    f.write(str(OS_star_Accuracy_list))
    f.write('\r\n')
    f.write('Unknow_acc:')
    f.write(str(Unknow_list))
    f.write('\r\n')
    f.write('train_time:')
    f.write(str(Train_Time))
    f.close()

def normalization(input):
    kethe=0.0000000001
    output=(input-min(input))/(max(input)-min(input)+kethe)

    return output


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Training settings



momentum = 0.9
no_cuda = False
seed = 8
log_interval = 10
l2_decay = 5e-4


criterion_bce = nn.BCELoss()

def train(model):
    src_iter = iter(src_loader)
    tgt_iter = iter(tgt_train_loader)
    src_dlabel = Variable(torch.ones(batch_size).long().cuda())
    tgt_dlabel = Variable(torch.zeros(batch_size).long().cuda())

    OS_Accuracy_list = []
    OS_star_Accuracy_list = []
    Unknow_list = []



    for i in range(1, iteration + 1):
        model.train()
        G_LEARNING_RATE = G_lr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        C_LEARNING_RATE = C_lr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)

        D_LEARNING_RATE = Dlr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)


        p = i / iteration
        constant = 2. / (1. + np.exp(-10 * p)) - 1
        if (i - 1) % 100 == 0:
            print('learning rate{: .4f}'.format(G_LEARNING_RATE))

        optimizer = torch.optim.Adam([
            {'params': model.sharedNet.parameters(),'lr': G_LEARNING_RATE},
            {'params': model.cls_fc.parameters(), 'lr': C_LEARNING_RATE},
        ], weight_decay=l2_decay)

        optimizer_critic = torch.optim.Adam([
            {'params': model.domain_fc.parameters()},
             {'params': model.au_domain_fc.parameters()}


        ], lr=D_LEARNING_RATE, weight_decay=l2_decay)



        try:
            src_data, src_label = src_iter.next()
            tgt_data, tgt_label = tgt_iter.next()
        except Exception as err:
            src_iter = iter(src_loader)
            src_data, src_label = src_iter.next()
            tgt_iter = iter(tgt_train_loader)
            tgt_data, tgt_label = tgt_iter.next()

        if cuda:
            src_data, src_label = src_data.cuda(), src_label.cuda()
            tgt_data, tgt_label = tgt_data.cuda(), tgt_label.cuda()

        optimizer.zero_grad()

        src_pred,src_domain_pred,src_au_domain_pred,src_feature= model(src_data)
        tgt_pred,tgt_domain_pred,tgt_au_domain_pred,tgt_feature,tgt_un_pred= model(tgt_data, constant = constant, adaption = True)



        cls_loss = F.nll_loss(F.log_softmax(src_pred, dim=1), src_label)

        output_t_prob_unk = F.softmax(tgt_pred, dim=1)[:, -1]
        loss_adv = criterion_bce(output_t_prob_unk, torch.tensor([threshold] * batch_size).cuda())



        new_label_pred = torch.cat((src_domain_pred, tgt_domain_pred), 0)
        new_label_pred_au = torch.cat((src_au_domain_pred,tgt_au_domain_pred), 0)

        confusion_loss = nn.CrossEntropyLoss(reduction='none')


        confusion_loss_total= confusion_loss(new_label_pred, torch.cat((src_dlabel, tgt_dlabel), 0))
        confusion_loss_total_au = confusion_loss(new_label_pred_au, torch.cat((src_dlabel, tgt_dlabel), 0))



        tar_weight = normalization(confusion_loss_total_au[batch_size:])



        weight = confusion_loss_total_au * 1
        weight[:batch_size] = 1
        weight[batch_size:] = tar_weight

        confusion_loss_total = torch.sum((confusion_loss_total.reshape(-1, 1)).mul(weight.reshape(-1, 1))) / (2 * batch_size)

        confusion_loss_total_au=confusion_loss_total_au.sum() / (2 * batch_size)

        confu=confusion_loss_total+confusion_loss_total_au




        lambd = 2 / (1 + math.exp(-10 * (i) / iteration)) - 1




        m = nn.Softmax(dim=1)



        ce_ep = CrossEntropyLoss(m(tgt_un_pred), Variable(torch.from_numpy(
            np.concatenate((np.zeros((batch_size, len(Source_class[TT]))), np.ones((batch_size, 1))), axis=-1).astype(
                'float32'))).cuda(), instance_level_weight=(1-tar_weight))



        loss = cls_loss - confusion_loss_total + loss_adv * A + ce_ep * B * lambd






        loss.backward(retain_graph=True)
        optimizer.step()


        optimizer_critic.zero_grad()
        confu.backward(retain_graph=True)
        optimizer_critic.step()




        if i % log_interval == 0:
            print('Train iter: {} [({:.0f}%)]\tcls_Loss: {:.6f}\tEX_adv_Loss: {:.6f}\tadv_Loss: {:.6f}\tAU_adv_Loss: {:.6f}\tce_Loss: {:.6f}'.format(
                i, 100. * i / iteration,  cls_loss.item(), loss_adv.item(),confusion_loss_total.item(),confusion_loss_total_au.item(),ce_ep,))

        if i % (log_interval * 10) == 0:



            zeros = torch.zeros(class_num, 1).cuda()
            weight_list = zeros.scatter_add(0, tgt_label.reshape(-1, 1), tar_weight.reshape(-1, 1))

            ones = torch.ones_like(tgt_label, dtype=torch.float)
            zeros = torch.zeros(class_num).cuda()
            t_n_classes = zeros.scatter_add(0, tgt_label, ones)

            weight_l = torch.div(weight_list, t_n_classes.reshape(-1, 1))

            llable = torch.linspace(1, class_num, steps=class_num, out=None).cuda()

            zeros = torch.zeros(class_num, 2).cuda()

            zeros[:, 0] = llable
            zeros[:, 1] = weight_l.squeeze(dim=1)
            print(zeros)

            train_correct,train_loss = test_source(model,src_loader)
            tar_OS,tar_OS_,tar_UNK= test_target(model,tgt_test_known_loader,tgt_test_unknown_loader)

            OS_Accuracy_list.append(tar_OS)

            OS_star_Accuracy_list.append(tar_OS_)

            Unknow_list.append(tar_UNK)



def test_target(model,test_know_loader,test_unknow_loader):
    model.eval()
    test_loss = 0
    correct_know = 0
    correct_unknow = 0



    with torch.no_grad():
        for tgt_test_data, tgt_test_label in test_know_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)

            tgt_pred,_,_,_= model(tgt_test_data)
            test_loss += F.nll_loss(F.log_softmax(tgt_pred, dim=1), tgt_test_label,
                                    reduction='sum').item()  # sum up batch loss
            pred = tgt_pred.data.max(1)[1]  # get the index of the max log-probability

            correct_know += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()

        for tgt_test_data, tgt_test_label in test_unknow_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)

            tgt_pred,_,_,_ = model(tgt_test_data)
            test_loss += F.nll_loss(F.log_softmax(tgt_pred, dim=1), tgt_test_label,
                                    reduction='sum').item()  # sum up batch loss
            pred = tgt_pred.data.max(1)[1]  # get the index of the max log-probability


            correct_unknow += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()




    OS_=np.array(correct_know)/len(test_know_loader.dataset)
    UNK=np.array(correct_unknow)/len(test_unknow_loader.dataset)
    OS=(len(Source_class[TT])*OS_+UNK)/(len(Source_class[TT])+1)

    print('\n{} set - {} set, global Accuracy:({:.2f}%), know Accuracy: {}/{} ({:.2f}%),unknow Accuracy:{}/{}  ({:.2f}%)\n'.format
          (src_name,tgt_name,OS*100,correct_know,len(test_know_loader.dataset),OS_*100,correct_unknow,len(test_unknow_loader.dataset),UNK*100))




    return OS,OS_,UNK


def test_source(model,test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for tgt_test_data, tgt_test_label in test_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)

            tgt_pred,_,_,_= model(tgt_test_data)
            test_loss += F.nll_loss(F.log_softmax(tgt_pred, dim=1), tgt_test_label,
                                    reduction='sum').item()  # sum up batch loss
            pred = tgt_pred.data.max(1)[1]  # get the index of the max log-probability

            correct += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()


    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(tgt_name,test_loss, correct, len(test_loader.dataset),10000. * correct / len(test_loader.dataset)))
    return correct,test_loss


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print('Total:{} Trainable:{}'.format( total_num, trainable_num))


if __name__ == '__main__':
    # setup_seed(seed)
    iteration = 10000
    batch_size = 64
    G_lr   =0.0001
    C_lr   =0.0001
    Dlr    =0.0001
    Dlr_au =0.0001



    FFT = True

    threshold=0.5

    dataset = 'CWRU'
    class_num = 10
    Task_name = np.array(['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10'])
    src_tar = np.array([[0, 3], [0, 3], [0, 3], [0, 3], [0, 3], [3, 0], [3, 0], [3, 0], [3, 0], [3, 0]])

    Source_class = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5, 7, 8, 9, 10],
        [1, 2, 3, 5, 6, 8, 9],
        [1, 2, 5, 8, 10],
        [1, 2, 10],
        [1, 2, 4, 5, 6, 8, 9, 10],
        [1, 2, 5, 6, 8, 9],
        [1, 8, 9, 10],
        [1, 5, 9],
        [1, 7]
    ])


    A=0.1
    B=0.1


    for TT in range(1):

        all_category = np.linspace(1, class_num, class_num, endpoint=True)
        partial = Source_class[TT]
        notargetclasses = list(set(all_category) ^ set(partial))# 要删去的类别

        print(notargetclasses)

        source = src_tar[TT][0]
        target = src_tar[TT][1]


        for repeat in range(1):

            root_path = '/home/dlzhaochao/deeplearning/DTL/data/' + dataset + 'data' + str(class_num) + '.mat'

            src_name = 'load' + str(source) + '_train'
            tgt_name = 'load' + str(target) + '_train'
            test_name = 'load' + str(target) + '_test'

            cuda = not no_cuda and torch.cuda.is_available()
            torch.manual_seed(seed)
            if cuda:
                torch.cuda.manual_seed(seed)

            kwargs = {'num_workers': 1, 'pin_memory': False} if cuda else {}

            src_loader = data_loader_1d.load_training(notargetclasses, root_path, src_name, FFT,
                                                      len(Source_class[TT]), batch_size, kwargs)

            tgt_train_loader = data_loader_1d.load_training([], root_path, tgt_name, FFT, class_num,
                                                            batch_size, kwargs)

            tgt_test_known_loader = data_loader_1d.load_testing_known(notargetclasses, root_path, test_name,
                                                                      FFT,
                                                                      len(Source_class[TT]),
                                                                      batch_size, kwargs)
            tgt_test_unknown_loader = data_loader_1d.load_testing_unknown(partial, root_path, test_name, FFT,
                                                                          len(notargetclasses), batch_size,
                                                                          kwargs)
            src_dataset_len = len(src_loader.dataset)

            src_loader_len = len(src_loader)
            model = models.M18(num_classes=len(Source_class[TT]) + 1)
            # get_parameter_number(model) 计算模型训练参数个数
            print(model)
            if cuda:
                model.cuda()
            train(model)



























