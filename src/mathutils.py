__author__ = 'haoyu'
import math

def max(score=[]):
    """calculate the maximum score"""
    max = -9999.0;
    index = -1;
    for i in range(0, len(score)):
        if score[i] > max:
            max = score[i];
            index = i;
    return max, index;

def calculate_score(context=[], blanks=[[]]):
    """calcualte the score given context words and blanks"""
    pass;

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

def combine_and_normalize(vecs=[[]], size=0):
    """combine and normalize word vectors"""
    retvec = [0 for i in range(0, size)];
    for vec in vecs:
        for i in range(0, len(vec)):
            retvec[i] = retvec[i] + vec[i];

    for i in range(0, len(retvec)):
        retvec[i] = retvec[i] / len(vecs);
    return retvec;

vecs = [[1.0,2.0,3.0],[2.0,3.0,4.0]]
print cossim([1,1,1], [1,2,3])