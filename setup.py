from setuptools import setup


with open('README.md', 'rb') as f:
    readme = f.read().decode('utf-8')

setup(
    name='GhuLDA',
    packages=['ghulda'],
    version='1.1.1',
    description='Pacote com funções para processamento de modelos LDA',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Erick Ghuron',
    author_email='ghuron@usp.br',
    url='https://github.com/ghurone/ghulab',
    install_requires=['gensim==4.3.3', 'spacy==3.7.5', 'tqdm>=4.66.1'],
    license='MIT',
    keywords=['ghu', 'lda'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)
