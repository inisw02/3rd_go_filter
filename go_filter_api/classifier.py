import json

from torch import nn
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from keras_preprocessing.sequence import pad_sequences

with open('config.json') as json_file:
    config = json.load(json_file)

class Classifier(nn.Module):
    def __init__(self, n_classes):
        super(Classifier, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(config["BERT_MODEL"], num_labels=n_classes, problem_type="multi_label_classification")
        self.model = self.model.cuda()
        self.model = self.model.load_state_dict(torch.load(config["PRE_TRAINED_MODEL"]))
        self.tokenizer = BertTokenizer(vocab_file=config["TOKENIZER_VOCAB"])
        self.device = torch.device("cuda")
        self.classes = ['욕설', '비난', '범죄', '차별', '혐오', '일반', '선정', '폭력']

    def tokenize(self, sentences):
        tokenized_texts = [self.tokenizer.tokenize(sent) for sent in sentences]
        input_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
        input_ids = pad_sequences(input_ids, maxlen=config["MAX_SEQUENCE_LEN"], dtype='long', truncating='post', padding='post')
        attention_mask = []
        for seq in input_ids:
            seq_mask = [float(i>0) for i in seq]
            attention_mask.append(seq_mask)
        tensor_input = torch.tensor(input_ids)
        tensor_mask = torch.tensor(attention_mask)
        return tensor_input, tensor_mask

    def inference(self, sentences):
        input_ids, attention_mask = Classifier.tokenize(sentences)
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_ids.to(self.device), attention_mask = attention_mask.to(self.device))
            result = self.classes[output[0].argmax()]
        return result


