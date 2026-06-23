from setuptools import setup


with open('README.md', 'rb') as f:
    readme = f.read().decode('utf-8')

setup(
    name='GhuLDA',
    packages=['ghulda'],
    version='2.0.1',
    description='Pacote com funções para processamento de modelos LDA',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Erick Ghuron',
    author_email='ghuron@usp.br',
    url='https://github.com/ghurone/ghulda',
    install_requires=['gensim==4.4.0', 'spacy==3.8.14', 'tqdm>=4.66.1'],
    license='MIT',
    keywords=['ghu', 'lda'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)
