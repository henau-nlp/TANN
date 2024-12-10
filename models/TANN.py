import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer
from torch.autograd import Function
import time

def get_time_dif(start_time):
    """计算使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return time.strftime("%H:%M:%S", time.gmtime(time_dif))

class Config(object):
    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'TANN'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt').readlines()]  # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 100000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 5  # epoch数
        self.batch_size = 32  # mini-batch大小
        self.pad_size = 64  # 每句话处理成的长度(短填长切)
        self.learning_rate = 2e-5  # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.num_filters = 256  # 卷积核数量
        self.filter_sizes = (2, 3, 4)
        self.dropout = 0.1
        self.event_num = 11
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambd = ctx.lambd
        return (grad_output * -lambd), None

def grad_reverse(x, lambd):
    return ReverseLayerF.apply(x, lambd)

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.event_num = config.event_num
        self.dropout = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_sizes])
        self.lstm = nn.LSTM(config.hidden_size, 256, 2, bidirectional=True, batch_first=True)
        self.domain_classifier = nn.Sequential(
            nn.Linear(2048, config.hidden_size),
            nn.LeakyReLU(True),
            nn.Linear(config.hidden_size, self.event_num)
        )
        self.fc = nn.Sequential(
            nn.Linear(2048, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.num_classes)
        )
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    def forward(self, x):
        context, seq_len, mask, event = x[0], x[1], x[2], x[3]
        # bert
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        # # cnn
        cnn_out = encoder_out.unsqueeze(1)
        cnn_out = torch.cat([self.conv_and_pool(cnn_out, conv) for conv in self.convs], 1)
        # lstm
        lstm_out, _ = self.lstm(encoder_out)
        out = torch.cat((text_cls, cnn_out, lstm_out[:, -1, :]), dim=-1)
        out = self.dropout(out)
        # classification
        lambd = 0.1
        reverse_feature = grad_reverse(out, lambd)  #out [b,embedding_dim]
        domain_outputs = self.domain_classifier(reverse_feature) #event output [b,embedding_dim]
        outputs = self.fc(out) #label output
        return outputs, domain_outputs