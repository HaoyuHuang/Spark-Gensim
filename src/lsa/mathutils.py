__author__ = 'haoyu'
import math
from gensim.corpora import TextCorpus, MmCorpus, Dictionary
from gensim.models import LsiModel

def max(score=[]):
    """calculate the maximum score"""
    max = -9999.0;
    index = -1;
    for i in range(0, len(score)):
        if score[i] > max:
            max = score[i];
            index = i;
    return max, index;

def cossim(vec1=[], vec2=[]):
    """calculate the cosine similarity between two word vectors"""
    dotp = 0.0;
    for i in range(0, len(vec1)):
        dotp = dotp + vec1[i] * vec2[i];
    return dotp / (magnitude(vec1) * magnitude(vec2));


def magnitude(vec=[]):
    """calculate the magnitude of a given word vector"""
    sum = 0.0;
    for em in vec:
        sum = sum + em * em;
    return math.sqrt(sum);

def combine_and_normalize(vecs=[], size=0):
    """combine and normalize word vectors"""
    retvec = [0.0 for i in range(0, size)];
    for vec in vecs:
        for i in range(0, len(vec)):
            retvec[i] = retvec[i] + vec[i];

    for i in range(0, len(retvec)):
        retvec[i] = retvec[i] / len(vecs);
    return retvec;
