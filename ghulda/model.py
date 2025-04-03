import logging

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import Phrases, LdaModel
from gensim.models.callbacks import Callback
from gensim.models.coherencemodel import CoherenceModel
from tqdm import tqdm


logging.getLogger("gensim").setLevel(logging.WARNING)


def add_bigram(documentos: list[list[str]],
               min_count: int = 5,
               delimiter: str = "_",
               threshold: int = 5
               ) -> None:
    
    bigram = Phrases(documentos, min_count=min_count, delimiter=delimiter, threshold=threshold)
    for i in range(len(documentos)):
        bi = [token for token in bigram[documentos[i]] if "_" in token]
        documentos[i].extend(bi)


def create_dictionary(documentos: list[list[str]],
                      n_abaixo: int = None,
                      n_acima: float = None,
                      keep_n: int = 100000,
                      keep_tokens: list = None
                      ) -> Dictionary:
    
    dicio = Dictionary(documentos)

    if n_abaixo is not None or n_acima is not None:
        if n_abaixo is None:
            n_abaixo = 1
        if n_acima is None:
            n_acima = 0.5

        dicio.filter_extremes(no_below=n_abaixo, no_above=n_acima, keep_n=keep_n, keep_tokens=keep_tokens)

    return dicio


def create_corpus(dicionario: Dictionary,
                  documentos: list[list[str]],
                  verbose: bool = False
                  ) -> list[list[tuple[int, int]]]:
    
    if verbose:
        return [dicionario.doc2bow(doc) for doc in tqdm(documentos, desc="Corpus")]
    else:
        return [dicionario.doc2bow(doc) for doc in documentos]


def calc_coherence(model, documents, dictionary, corpus, method="c_v") -> CoherenceModel:
    return CoherenceModel(
        model=model,
        texts=documents,
        dictionary=dictionary,
        corpus=corpus,
        coherence=method,
    )


class TQDMbar(Callback):
    def __init__(self, epochs, metrics=None):
        super().__init__(metrics)
        self.pbar = tqdm(total=epochs, desc="Passes", leave=False)
        self.logger = logging.getLogger()

    def get_value(self, **args):
        self.pbar.update(1)
        return 1


class ModelLDA:
    def __init__(
        self,
        corpus,
        id2word,
        chunksize: int = 2000,
        iterations: int = 100,
        passes: int = 25,
        verbose: bool = True,
        random_seed: int = 99,
        dtype=np.float64,
    ):
        self.corpus = corpus
        self.id2word = id2word
        self.chunksize = chunksize
        self.iterations = iterations
        self.passes = passes
        self.seed = random_seed
        self.verb = verbose
        self.dtype = dtype
        self.model = None

    def run(self, n_topic, alpha="auto", eta="auto") -> LdaModel:
        self.model = LdaModel(
            corpus=self.corpus,
            id2word=self.id2word,
            chunksize=self.chunksize,
            alpha=alpha,
            eta=eta,
            iterations=self.iterations,
            num_topics=n_topic,
            passes=self.passes,
            random_state=self.seed,
            eval_every=None,
            dtype=self.dtype,
            callbacks=[TQDMbar(self.passes)] if self.verb else None,
        )
        return self.model
