import json
import torch
from transformers import BertTokenizer
from keras_preprocessing.sequence import pad_sequences

from classifier import Classifier

with open("config.json") as json_file:
    config = json.load(json_file)

class Model:
    def __init__(self):
        self.device = torch.device("cuda")
        self.tokenizer = BertTokenizer(vocab_file=config["TOKENIZER_VOCAB"])
        classifier = Classifier(len(config["CLASS_NAMES"]))
        classifier.load_state_dict(torch.load(config["PRE_TRAINED_MODEL"], map_location=self.device))
        classifier = classifier.eval()
        self.classifier = classifier.to(self.device)
    
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
        input_ids, attention_mask = Model.tokenize(sentences)
        with torch.no_grad():
            output = self.model(input_ids.to(self.device), attention_mask = attention_mask.to(self.device))
            result = self.classes[output[0].argmax()]
        return result

model = Model()

def get_model():
    return model
        
