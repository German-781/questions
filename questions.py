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
    print("load files")
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.

    """

    oraciones = dict()

    print("directory ", directory)

    contenido = os.listdir(directory)

    for archivo in contenido:
        print("archivo ", archivo)
        with open(os.path.join(directory, archivo), encoding=('utf-8')) as f:
            oraciones[archivo] = f.read()

    return oraciones

    raise NotImplementedError


def tokenize(document):
    #print("tokenize")
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
    
    #print("palabras ", palabras)
    
    return palabras




    raise NotImplementedError


def compute_idfs(documents):
    print("compute idfs")
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words_idfs = dict()
    

    primer_document0 = list(documents.keys())[0]
    print("primer documento ", primer_document0)

    print("largo primer documento ", len(documents[primer_document0]))
  
    num_doc = len(documents)
    print("numero documentos ", num_doc)
    num_word = 0


    for articulo in documents:
        #print("articulo ", articulo)
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

        #print("largo lista ", articulo, len(lista))
    print("largo dicc ", len(words_idfs))
    print("num word ", num_word)

    idf1 = np.log(num_doc / 1)
    idf2 = np.log(num_doc / 2)
    idf3 = np.log(num_doc / 3)
    idf4 = np.log(num_doc / 4)
    idf5 = np.log(num_doc / 5)
    idf6 = np.log(num_doc / 6)

    print("idfs ", idf1, idf2, idf3, idf4, idf5, idf6)

    
    for word in words_idfs:
        numero = words_idfs[word]
        if numero == 1:
            words_idfs.update({word:idf1})
        elif numero == 2:
            words_idfs.update({word:idf2})
        elif numero == 3:
            words_idfs.update({word:idf3})
        elif numero == 4:
            words_idfs.update({word:idf4})
        elif numero == 5:
            words_idfs.update({word:idf5})
        elif numero == 6:
            words_idfs.update({word:idf6})

    #print("IA ", words_idfs["IA"])
    #print("probability ", words_idfs["probability"])
    #print("Python ", words_idfs["python"])
    #print("system ", words_idfs["system"])
    #print("neuronal ", words_idfs["neuronal"])
    #print("learning ", words_idfs["learning"])


    print("words_idfs largo ", len(words_idfs))

    return words_idfs


        #    print(documents[articulo])

    #raise NotImplementedError


def top_files(query, files, idfs, n):
    print("top files")
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """

    print("query ", query)
    print("n ", n)
    #print("files ", files)
    #print("idfs ", idfs)

    dic_words = {}

    for termino in query:
        lista = []
        for articulo in files:
            #print("articulo ", articulo)
            num = 0
            archivo = articulo
            dic_words.update({termino:[]})

            for palabra in files[articulo]:
                encuentra = False
                if termino == palabra:
                    encuentra = True
                    num = num + 1
                    otro = {archivo:num}
                    #print("termino ", termino)
                    #print("otro ", otro)
            if num > 0:
                lista.append(otro)
        
            #print("lista articulos ", termino, lista)

        dic_words.update({termino:lista})
        #print("termino en files ", dic_words)    


    print("frecuencia words ", dic_words)

    dic_tfidf = {}

    for word in dic_words:
        print("word ", word)
        if word in idfs:
            idf = idfs[word]
            lista = dic_words[word]
            for doc in lista:
                print("doc ", doc)
                for k in doc.keys():
                    tf = doc[k]
                    tfidf = tf * idf
                    #print("k ", k, "tf", tf, "idf", idf, "itf ", tfidf)
                    if k in dic_tfidf:
                        tfid = dic_tfidf[k]
                        print("tfid ", tfid)
                        tfidf = tfidf + tfid
                        print("tfidf ", tfidf)
                    dic_tfidf.update({k:tfidf})
                    #print("dic tfidf en k ", dic_tfidf)                
                    #print("k ", k, "tf", tf, "idf ", idf, "tfidf ", tfidf)
    #print("dic tfidf ", dic_tfidf)                

    dic_tfidf_desc = dict(sorted(dic_tfidf.items(), key=itemgetter(1), reverse=True))
    #print("dic_tfidf desc ", dic_tfidf_desc)
    #print(" dic keys ", dic_tfidf_desc.keys())

    lista_file = []
    file_select = 0
    for archivo in dic_tfidf_desc.keys():
        if file_select < n :
            lista_file.append(archivo)
            file_select = file_select + 1

    print("lista file ", lista_file)

    return lista_file

    raise NotImplementedError


def top_sentences(query, sentences, idfs, n):

    print("top sentences ")

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
    print("diccionario query ", dic_query)

    
    
    lista_oraciones = []
    dic_oraciones = {}

    print("n ", n)
    #print("oraciones ", sentences)
    print("numero de oraciones ", len(sentences))
    #print("idfs ", idfs)
    print("query ", query)

    iteracion = 1
    for oracion in sentences:
        #print("oracion ", oracion)
        palabras = sentences[oracion]
        #print("palabras ", palabras)
        iteracion = iteracion + 1
        #if iteracion > 20:
        #    break

        val_idf = 0
        for palabra in palabras:
            if palabra in query:
                #print("palabra ", palabra)
                #print("idf ", dic_query[palabra])
                if oracion in dic_oraciones:
                    val_idf = val_idf + dic_query[palabra]
                    continue
                else:
                    val_idf = dic_query[palabra]

                dic_oraciones.update({oracion:val_idf})

    print("dic oraciones ", dic_oraciones)
    print("numero de oraciones ", len(dic_oraciones))
    print("iteraciones ", iteracion)



    return lista_oraciones
    raise NotImplementedError


if __name__ == "__main__":
    main()
