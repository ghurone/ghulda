import sys
import os
import string
from multiprocessing import Pool

import spacy

if 'ipykernel' in sys.modules and 'spyder' not in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def clean_space(text: str) -> str:
    return ' '.join(text.split())


class TextTokenizer(object):
    def __init__(self, nlp: spacy.language.Language, classes: list[str] = None, alfabeto: set = None,
                 stop_words: list = None, min_length: int = 1, lemmatize: bool = True):

        if alfabeto is None:
            self.alfabeto = set(string.ascii_letters + 'áàãâéêíóôõúüçÁÀÃÂÉÊÍÓÔÕÚÜÇ-')
        else:
            self.alfabeto = alfabeto

        if stop_words is None:
            self.stop_words = nlp.Defaults.stop_words
        else:
            self.stop_words = stop_words

        if classes is None:
            self.classes = ['PNONU', 'NOUN', 'ADJ', 'VERB']
        else:
            self.classes = classes

        self.min_length = min_length
        self.lemmatize = lemmatize
        self.nlp = nlp

    def text_tokenizer(self, text: str) -> list[str]:
        doc = self.nlp(text)

        temp = []
        for token in doc:
            palavra = token.orth_
            if len(token) >= self.min_length and token.pos_ in self.classes and self._is_alpha(
                    palavra) and (palavra.lower() not in self.stop_words or token.lemma_.lower() not in self.stop_words):

                word = token.lemma_ if self.lemmatize else palavra
                temp.append(word.lower())

        return temp

    def document_tokenizer(self, list_text: list[str], workers: int = -1, verbose: bool = True) -> list[list[str]]:
        cpu_count = os.cpu_count() or 1
        workers = max(1, min(cpu_count, cpu_count + workers + 1))

        if verbose:
            with Pool(processes=workers) as pool:
                docs = list(tqdm(pool.imap(self.text_tokenizer, list_text), total=len(list_text)))
        else:
            with Pool(processes=workers) as pool:
                docs = pool.map(self.text_tokenizer, list_text)

        return docs

    def _is_alpha(self, palavra) -> bool:
        return set(palavra).issubset(self.alfabeto)


