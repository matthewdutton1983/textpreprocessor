import spacy
from text_preprocessor import TextPreprocessor
from typing import List, Tuple

class NLPProcessor(TextPreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nlp = spacy.load('en_core_web_sm')


    def pos_tagging(self, text: str) -> List[Tuple[str, str]]:
        """Tags parts of speech using spaCy"""
        doc = self.nlp(text)
        return [(token.text, token.pos) for token in doc]
  

    def entity_recognition(self, text: str) -> List[Tuple[str, str]]:
        """Performs named entity recognition using spaCy"""
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
        
