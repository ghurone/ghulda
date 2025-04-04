import spacy
from tqdm.autonotebook import tqdm

_classes_padrao = ('PROPN', 'NOUN', 'ADJ', 'VERB')


class Tokenizer:
    def __init__(
        self,
        lang: str = 'pt_core_news_lg',
        disables: list[str] = None,
        classes: tuple[str] = None,
        lemma: bool = True,
        n_min_len: int = 1,
        batch_size: int = 32,
        n_process: int = -1
    ) -> None:
        
        if disables is None:
            self.nlp = spacy.load(lang, disable=["parser", "ner", "attribute_ruler"])
        else:
            self.nlp = spacy.load(lang, disable=disables)


        if not lemma and "lemmatizer" in self.nlp.pipe_names:
            self.nlp.remove_pipe("lemmatizer")

        self.classes = classes or _classes_padrao
        self.lemma = lemma
        self.n_min_len = n_min_len
        self.batch_size = batch_size
        self.n_process = n_process

    def _tokenize_text(self, doc: spacy.tokens.Doc) -> list[str]:
        tokens = []

        for token in doc:
            if (
                token.pos_ in self.classes
                and len(token.orth_) > self.n_min_len
                and token.orth_.replace('-', '').isalpha()
            ):
                word = token.lemma_ if self.lemma else token.orth_
                tokens.append(word.lower())

        return tokens

    def tokenize_texts(self, texts: list[str]) -> list[list[str]]:
        tokenized_texts = []

        for doc in tqdm(
            self.nlp.pipe(
                texts,
                batch_size=self.batch_size,
                n_process=self.n_process
            ),
            total=len(texts)
        ):
            doc_tokens = self._tokenize_text(doc)
            tokenized_texts.append(doc_tokens)

        return tokenized_texts
