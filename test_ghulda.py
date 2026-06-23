"""
Teste rapido das funcoes do ghulda com gensim 4.4.0 / spacy 3.8.x.

Como rodar (da raiz do repo):
    conda activate ghulda-test      # ou o env onde estao as versoes novas
    python test_ghulda.py

OBS: o guard `if __name__ == "__main__"` e OBRIGATORIO porque o CoherenceModel
(c_v) usa multiprocessing. No Python 3.12 / macOS o start method e 'spawn', que
re-importa o modulo no processo filho; sem o guard isso quebra.
"""
import warnings

from ghulda.preprocessing import Tokenizer
from ghulda.model import (
    add_bigram, create_dictionary, create_corpus, calc_coherence, ModelLDA,
)

textos = [
    "O gato preto subiu no telhado da casa velha durante a noite fria.",
    "A casa velha tem um telhado vermelho muito bonito e antigo.",
    "Cientistas estudam inteligencia artificial e aprendizado de maquina.",
    "O aprendizado de maquina usa redes neurais artificiais profundas.",
    "Inteligencia artificial transforma a industria e a economia mundial.",
    "O cachorro corria feliz pelo parque verde durante a tarde ensolarada.",
    "Redes neurais profundas processam grandes volumes de dados rapidamente.",
    "A economia mundial enfrenta desafios com a nova tecnologia digital.",
] * 4  # 32 docs pra dar corpo


def main():
    print("== 1. Tokenizer (spacy + modelo) ==")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # warning de incompat. de modelo vira erro
        try:
            tok = Tokenizer(n_process=1, batch_size=8)
            print("   classes type:", type(tok.classes).__name__, "->", tok.classes)
            docs = tok.tokenize_texts(textos)
        except Warning as w:
            print("   WARNING capturado:", w)
            warnings.simplefilter("default")
            tok = Tokenizer(n_process=1, batch_size=8)
            docs = tok.tokenize_texts(textos)
    print("   exemplo tokens[0]:", docs[0])

    print("\n== 1b. n_process=-1 (todos os cores?) ==")
    try:
        tok2 = Tokenizer(n_process=-1, batch_size=8)
        d2 = tok2.tokenize_texts(textos[:8])
        print("   n_process=-1 OK, tokens[0]:", d2[0])
    except Exception as e:
        print("   n_process=-1 FALHOU:", type(e).__name__, e)

    print("\n== 2. add_bigram (.freeze() + delimiter fix) ==")
    before = sum(len(d) for d in docs)
    add_bigram(docs, min_count=2, threshold=1)
    after = sum(len(d) for d in docs)
    bigrams = sorted({t for d in docs for t in d if "_" in t})
    print(f"   tokens antes={before} depois={after}; bigramas: {bigrams[:10]}")

    print("\n== 2b. add_bigram com delimiter custom '++' ==")
    docs_custom = [[t for t in d if "_" not in t] for d in docs]
    add_bigram(docs_custom, min_count=2, threshold=1, delimiter="++")
    custom_bi = sorted({t for d in docs_custom for t in d if "++" in t})
    print("   bigramas com '++':", custom_bi[:10] or "(nenhum)")

    print("\n== 3. create_dictionary / create_corpus ==")
    dic = create_dictionary(docs, n_abaixo=1, n_acima=0.9)
    print("   tamanho do dicionario:", len(dic))
    corpus = create_corpus(dic, docs, verbose=True)
    print("   docs no corpus:", len(corpus), "| bow[0][:5]:", corpus[0][:5])

    print("\n== 4. ModelLDA.run (alpha/eta auto) ==")
    m = ModelLDA(corpus, dic, passes=3, iterations=20, verbose=False)
    lda = m.run(n_topic=3)
    print("   topicos:")
    for i, t in lda.print_topics(num_words=4):
        print(f"     {i}: {t}")

    print("\n== 5. calc_coherence ==")
    cm = calc_coherence(lda, docs, dic, corpus, method="c_v")
    print("   coherence c_v:", round(cm.get_coherence(), 4))

    print("\nTODOS OS TESTES PASSARAM")


if __name__ == "__main__":
    main()
