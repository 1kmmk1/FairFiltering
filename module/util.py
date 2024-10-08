
from module.mlp import MLP
from module.bert_clf import BertWrapper, ViTWrapper
from module.resnet import resnet50

import pdb
import torch
import os, sys
from transformers import BertForSequenceClassification, BertModel
from transformers import ViTFeatureExtractor, ViTModel
        
def cnt_params(model):
    cnt = 0
    for m in model.parameters():
        if m.requires_grad == True:
            cnt += m.numel()
    return cnt

def get_model(model_tag, num_classes, train_clf):
    if model_tag == "MLP":
        model = MLP(num_classes=num_classes)
    elif model_tag == "resnet50":
        model = resnet50(pretrained=True, train_clf = train_clf, num_classes = num_classes)
    elif model_tag == "BERT":
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        model = BertWrapper(bert_model, num_classes = num_classes, train_clf = train_clf)
    else:
        raise NotImplementedError
    print("*"*10,"Parameters: ", cnt_params(model), "*"*10)
    return model 

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
