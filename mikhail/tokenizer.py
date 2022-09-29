import torch

from transformers import DebertaTokenizer, DebertaForTokenClassification

if __name__ == '__main__':
    tokenizer = DebertaTokenizer.from_pretrained("dbsamu/deberta-base-finetuned-ner")
    inputs = tokenizer("HuggingFace is a company based in Paris and New York", add_special_tokens=False)

    #model = DebertaForTokenClassification.from_pretrained("dbsamu/deberta-base-finetuned-ner")