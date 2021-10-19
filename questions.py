from typing import Dict
import nltk
import sys
import cv2
import os
import numpy as np
from io import open
from os.path import isfile, join, isdir
from nltk import word_tokenize
from nltk.corpus import stopwords
from operator import itemgetter

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.

    """

    oraciones = dict()

    contenido = os.listdir(directory)

    for archivo in contenido:
        with open(os.path.join(directory, archivo), encoding=('utf-8')) as f:
            oraciones[archivo] = f.read()

    return oraciones

    raise NotImplementedError


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    palabras = []

    palabras_vacias = set(stopwords.words('english')) 

    frase = word_tokenize(document)

    for palabra in frase:
        minuscula = palabra.lower()
        if minuscula.isalpha():
            if minuscula not in palabras_vacias:
                palabras.append(minuscula)
    
   
    return palabras


    raise NotImplementedError


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words_idfs = dict()
    

    primer_document0 = list(documents.keys())[0]
  
    num_doc = len(documents)
    num_word = 0

    for articulo in documents:
        lista = []

        for palabra in documents[articulo]:
            if palabra not in lista:
                num = 1
                lista.append(palabra)
                if palabra not in words_idfs:
                    words_idfs.update({palabra:num})
                    num_word = num_word + 1
                else:
                    numero = words_idfs[palabra]
                    numero = numero + num   
                    words_idfs.update({palabra:numero})
  
    for word in words_idfs:
        numero = words_idfs[word]
        idf = np.log(num_doc / numero)
        words_idfs.update({word:idf})

    return words_idfs

    raise NotImplementedError

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """

    dic_words = {}

    for termino in query:
        lista = []
        for articulo in files:
            num = 0
            archivo = articulo
            dic_words.update({termino:[]})

            for palabra in files[articulo]:
                if termino == palabra:
                    num = num + 1
                    otro = {archivo:num}
            if num > 0:
                lista.append(otro)

        dic_words.update({termino:lista})

    dic_tfidf = {}

    for word in dic_words:
        if word in idfs:
            idf = idfs[word]
            lista = dic_words[word]
            for doc in lista:
                for k in doc.keys():
                    tf = doc[k]
                    tfidf = tf * idf
                    if k in dic_tfidf:
                        tfid = dic_tfidf[k]
                        tfidf = tfidf + tfid
                    dic_tfidf.update({k:tfidf})

    dic_tfidf_desc = dict(sorted(dic_tfidf.items(), key=itemgetter(1), reverse=True))

    lista_file = []
    file_select = 0
    for archivo in dic_tfidf_desc.keys():
        if file_select < n :
            lista_file.append(archivo)
            file_select = file_select + 1

    return lista_file

    raise NotImplementedError


def top_sentences(query, sentences, idfs, n):

    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    dic_query = {}
    for word in query:
        if word in idfs:
            valor_idf = idfs[word]
            dic_query.update({word:valor_idf})

    lista_oraciones = []
    dic_oraciones = {}

    iteracion = 1
    for oracion in sentences:
        palabras = sentences[oracion]
        largo = len(palabras)
        iteracion = iteracion + 1

        val_idf = 0
        palabra_usada = []
        frec_palabra = 0
        for palabra in palabras:
            if palabra in query:
                frec_palabra = frec_palabra + 1
                if oracion in dic_oraciones:
                    if palabra in palabra_usada:
                        continue
                    else:
                        val_idf = dic_query[palabra]  + dic_oraciones[oracion]
                        palabra_usada.append(palabra)
                else:
                    val_idf = dic_query[palabra]
                    palabra_usada.append(palabra)
                dic_oraciones.update({oracion:val_idf})

        densidad_termino = frec_palabra / largo
        val_idf = val_idf + densidad_termino
        dic_oraciones.update({oracion:val_idf})

    dic_oraciones_desc = dict(sorted(dic_oraciones.items(), key=itemgetter(1), reverse=True))

    file_select = 0
    for oracion in dic_oraciones_desc.keys():
        if file_select < n :
            lista_oraciones.append(oracion)
            file_select = file_select + 1

    return lista_oraciones

    raise NotImplementedError


if __name__ == "__main__":
    main()
