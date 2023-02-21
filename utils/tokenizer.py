
from torchtext.data.utils import get_tokenizer

class Tokenizer:
    
    def __init__(self):
        self.en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        self.de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')