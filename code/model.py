import torch
import torch.nn as nn
from transformers import RobertaModel,RobertaConfig

class Self_Attention(nn.Module):
    def __init__(self,input_size,num_attention_heads):
        super(Self_Attention, self).__init__()
        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

class Bi_LSTMmodel(nn.Module):

    #vocab_size为一共有多少单词
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers,max_sent_len):
        super(Bi_LSTMmodel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True,bidirectional=True)
        self.linear = nn.Linear(hidden_size * 4 * max_sent_len, 4)
    def forward(self, x,y):

        x = self.embed(x)
        y = self.embed(y)
        out_x, _ = self.lstm(x)
        out_y, _ = self.lstm(y)
        out = torch.cat([out_x,out_y],1)
        out = out.reshape(out.shape[0],out.shape[1] * out.shape[2])
        out = self.linear(out)
        return out

class Transformer_model(nn.Module):

    #vocab_size为一共有多少单词
    def __init__(self):
        super(Transformer_model, self).__init__()
        self.bert = RobertaModel.from_pretrained("model\\roberta-base\\")
        self.linear = nn.Linear(self.bert.config.hidden_size, 4)
    def forward(self, x,z):

        out = self.bert(input_ids=x,attention_mask=z)
        out = out[1]
        # out = out[0]
        # out = out[:,0,:]
        out = self.linear(out)

        return out