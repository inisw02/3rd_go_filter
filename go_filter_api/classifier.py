import json

from transformers import BertTokenizer, BertForSequenceClassification


with open('config.json') as json_file:
    config = json.load(json_file)

class Classifier():
    def __init__(self, n_classes):
        self.model = BertForSequenceClassification.from_pretrained(
            config["BERT_MODEL"], 
            num_labels=n_classes, 
            problem_type="multi_label_classification"
            )
       




