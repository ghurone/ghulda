# GhuLDA

Funções simples para pré-processar textos e treinar modelos de tópicos **LDA** (Latent Dirichlet Allocation), construído sobre [spaCy](https://spacy.io/) e [gensim](https://radimrehurek.com/gensim/).

## Instalação

```bash
pip install GhuLDA
```

Baixe também o modelo de português do spaCy:

```bash
python -m spacy download pt_core_news_lg
```

## O que cada parte faz

| Função / Classe | Para que serve |
| --- | --- |
| `Tokenizer` | Tokeniza, lematiza e filtra tokens por classe gramatical (substantivo, verbo, adjetivo, nome próprio). |
| `add_bigram` | Junta pares de palavras que aparecem juntas com frequência (ex.: `aprendizado_maquina`). |
| `create_dictionary` | Cria o dicionário de termos, com filtro opcional de palavras muito raras/frequentes. |
| `create_corpus` | Converte os documentos em bag-of-words. |
| `ModelLDA` | Treina o modelo LDA (`alpha`/`eta` automáticos por padrão). |
| `calc_coherence` | Calcula a coerência dos tópicos (`c_v`, `u_mass`, etc.). |

## Requisitos

- Python 3.10+
- `gensim`, `spacy`, `tqdm` (instalados automaticamente)

## Licença

MIT — Erick Ghuron
