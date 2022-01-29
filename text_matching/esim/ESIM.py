import torch
import torch.nn as nn
from transformers import AutoTokenizer,AutoModel
from model.SoftAttention import SoftmaxAttention


### BERT_ESIM
### ESIM部分来自  https://github.com/coetaur0/ESIM 项目
class ESIM(nn.Module):

    def __init__(self,bert_path='/data1/zsp/PreTrainModelStorage/self_pretrained_bert_11G/',
                    in_size = 768,
                    hidden_size = 300,
                    out_size = 2,
                    dropout = 0.5
                    ):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.dropout = dropout
        
        ## embedding
        self.embedding = AutoModel.from_pretrained(bert_path)
        for i,(name,para) in enumerate(self.embedding.named_parameters()):
            para.requires_grad = False    ## 冻结作为词向量
            # print(i,name,para.requires_grad)
        
        ## ESIM
        self.encoding = nn.LSTM(self.in_size,self.hidden_size,bidirectional=True)
        self.attention = SoftmaxAttention()
        self.projection = nn.Sequential(nn.Linear(4*2*self.hidden_size,self.hidden_size),  ## 4表示拼接的4个向量/2表示双向
                                                    nn.ReLU())
        self.composition = nn.LSTM(self.hidden_size,self.hidden_size,bidirectional=True)
        self.classifer = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(2*4*self.hidden_size,
                                                       self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size,
                                                       self.out_size))
    
    def forward(self,
                premises,
                premise_mask,
                premise_seg_ids,
                hypotheses,
                hypotheses_mask,
                hypo_seg_ids,
                ):
        embedded_premises = self.embedding(premises,premise_mask,premise_seg_ids)
        embedded_premises = embedded_premises[0]
        # print(embedded_premises)
        # print(type(embedded_premises))
        # print(embedded_premises)
        embedded_hypotheses = self.embedding(hypotheses,hypotheses_mask,hypo_seg_ids)
        embedded_hypotheses = embedded_hypotheses[0]
        # print(type(embedded_premises))
        encoded_premises,_ = self.encoding(embedded_premises)
        encoded_hypotheses,_ = self.encoding(embedded_hypotheses)
        # print(type(encoded_premises))
        # print(len(encoded_premises[0]))


        attended_premises, attended_hypotheses =\
            self.attention(encoded_premises, premise_mask,
                            encoded_hypotheses, hypotheses_mask)
        
        enhanced_premises = torch.cat([encoded_premises,
                                       attended_premises,
                                       encoded_premises - attended_premises,
                                       encoded_premises * attended_premises],
                                       dim=-1)
        enhanced_hypotheses = torch.cat([encoded_hypotheses,
                                         attended_hypotheses,
                                         encoded_hypotheses - attended_hypotheses,
                                         encoded_hypotheses * attended_hypotheses],
                                         dim=-1)
        projected_premises = self.projection(enhanced_premises)
        projected_hypotheses = self.projection(enhanced_hypotheses)

        v_ai,_ = self.composition(projected_premises)
        v_bj,_ = self.composition(projected_hypotheses)
        
        # print(v_ai)
        # print('--------')
        # print(premise_mask)
        # print('--------')
        # print(v_ai.size())
        # print(premise_mask.size())
        # print(torch.sum(v_ai * premise_mask.unsqueeze(1).transpose(2, 1), dim=1))
        v_a_avg = torch.sum(v_ai * premise_mask.unsqueeze(1)
                                                .transpose(2, 1), dim=1)\
            / torch.sum(premise_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * hypotheses_mask.unsqueeze(1)
                                                  .transpose(2, 1), dim=1)\
            / torch.sum(hypotheses_mask, dim=1, keepdim=True)
        
        # v_a_max, _ = replace_masked(v_ai, premise_mask, -1e7).max(dim=1)
        # v_b_max, _ = replace_masked(v_bj, hypotheses_mask, -1e7).max(dim=1)
        v_a_max,_ = v_ai.max(dim=1)
        v_b_max,_ = v_bj.max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        logits = self.classifer(v)
        probabilities = nn.functional.softmax(logits, dim=-1)

        return logits, probabilities


if __name__=='__main__':
    tokenizer = AutoTokenizer.from_pretrained('/data1/zsp/PreTrainModelStorage/self_pretrained_bert_11G/')
    a = tokenizer(['我喜欢北京'])
    input_ids = a['input_ids']
    seg_ids = a['token_type_ids']
    atten_mask = a['attention_mask']

    b = tokenizer(['我爱北京'])
    binput_ids = a['input_ids']
    bseg_ids = a['token_type_ids']
    batten_mask = a['attention_mask']
    # seg_ids
    # print(type(a))
    # print(a)
    emodel = ESIM()
    logits,p = emodel(torch.tensor(input_ids),torch.tensor(seg_ids),torch.tensor(atten_mask),
                            torch.tensor(binput_ids),torch.tensor(bseg_ids),torch.tensor(batten_mask))
    
    print(logits)
    print(p)
