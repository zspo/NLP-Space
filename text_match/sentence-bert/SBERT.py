import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModel

class SBERT(nn.Module):

    def __init__(self,
                 bert_path='/data1/zsp/PreTrainModelStorage/self_pretrained_bert_11G/'
                ):
        super().__init__()
        
        ## embedding
        self.embedding = AutoModel.from_pretrained(bert_path)
#         for i,(name,para) in enumerate(self.embedding.named_parameters()):
#             para.requires_grad = False    ## 冻结作为词向量
            # print(i,name,para.requires_grad)
        self.metric = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.hidden_size = 768
        
        self.fc = nn.Linear(self.hidden_size * 3, 2)
        
    def forward(self,
                premises,
                premise_mask,
                premise_seg_ids,
                hypotheses,
                hypotheses_mask,
                hypo_seg_ids,
                inference=False
                ):
        embedded_premises = self.embedding(premises,premise_mask,premise_seg_ids)
        embedded_premises = embedded_premises[0]

        embedded_hypotheses = self.embedding(hypotheses,hypotheses_mask,hypo_seg_ids)
        embedded_hypotheses = embedded_hypotheses[0]
        
        sen_a_len, sen_b_len = (premise_mask != 0).sum(dim=1, keepdim=True), (hypotheses_mask != 0).sum(dim=1, keepdim=True)
        sen_a_pooling, sen_b_pooling = embedded_premises.sum(dim=1) / sen_a_len, embedded_hypotheses.sum(dim=1) / sen_b_len
        
        if inference:
            # sen_a_norm = torch.norm(sen_a_pooling, dim=1)
            # sen_b_norm = torch.norm(sen_b_pooling, dim=1)
            # similarity = (sen_a_pooling * sen_b_pooling).sum(dim=1) / (sen_a_norm * sen_b_norm)
            similarity = F.cosine_similarity(sen_a_pooling, sen_b_pooling, dim=1)
            return similarity
        
        hidden = torch.cat([sen_a_pooling, sen_b_pooling, torch.abs(sen_a_pooling - sen_b_pooling)], dim=1)

        return self.fc(hidden)


if __name__=='__main__':
    tokenizer = AutoTokenizer.from_pretrained('/data1/zsp/PreTrainModelStorage/self_pretrained_bert_11G/')
    a = tokenizer(['我喜欢北京'])
    input_ids = a['input_ids']
    seg_ids = a['token_type_ids']
    atten_mask = a['attention_mask']

    b = tokenizer(['另外一个不相关的句子'])
    binput_ids = a['input_ids']
    bseg_ids = a['token_type_ids']
    batten_mask = a['attention_mask']
    # seg_ids
    # print(type(a))
    # print(a)
    emodel = SBERT()
    logits = emodel(torch.tensor(input_ids),torch.tensor(atten_mask),torch.tensor(seg_ids),
                    torch.tensor(binput_ids),torch.tensor(batten_mask),torch.tensor(bseg_ids)
                   )
    
    print(logits)
