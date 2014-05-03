__author__ = 'haoyu'

from src import mathutils
import nltk
from src import lsamodel
from src import eval

answer = [];
cal_ans = [];
unseen_word = set();

def index2answer(index):
    return chr(index+65)

def load_gre_answer(answer_path):
    """load all gre sentences task answers"""
    file = open(answer_path);

    for line in file:
        if "answer:" in line:
            line = line.strip();
            index = line.index("answer:");
            answer.append(line[index+7:len(line)]);
    print len(answer);
    file.close();

def load_gre_sentence(sentence_path, lsi, word2id):
    """load all gre sentence tasks and select answers based on LSA"""
    file = open(sentence_path);

    taskScore = [];
    taskCount = 0;

    for line in file:
        if "" == line.strip():
            if len(taskScore) == 5:
                maxScore, index = mathutils.max(taskScore);
                canswer = index2answer(index)
                print 'score : ' + str(maxScore) + ' answer: ' + canswer;
                cal_ans.append(canswer);
                taskScore = [];
                taskCount = taskCount + 1;
            pass;
        else:
            sentence = "";
            li = -1;
            options = [];

            isInParenthese = False;
            for i in range(0, len(line)):
                if line[i] == '(':
                    li = i;
                    isInParenthese = True;
                elif line[i] == ')':
                    options.append(line[li+1:i]);
                    isInParenthese = False;
                elif isInParenthese == False:
                    sentence = sentence + line[i];
            context = nltk.word_tokenize(sentence);

            blanks = [];

            for blank in options:
                temp = nltk.word_tokenize(blank);
                # print temp;
                blanks.append(temp);
            taskScore.append(calculate_score(lsi, word2id=word2id, blanks=blanks, context=context));
    file.close();
    print taskCount;

def calculate_score(lsi, word2id = {}, context=[], blanks=[], alpha=0.5):
    """calcualte the score given context words and blanks"""
    # print len(blanks)
    # print blanks
    # print len(context)
    # print context

    score = 0.0
    blank_vecs = []
    score_per_blank = []
    for blank in blanks:
        vector = [];
        for em in blank:
            if word2id.has_key(em):
                vector.append(lsi.projection.u[int(word2id[em])])
            else:
                unseen_word.add(em)
        if len(vector)  > 0:
            # print vector
            blank_vecs.append(mathutils.combine_and_normalize(vector, len(vector[0])))

    if len(blank_vecs) == 0:
        return score

    for vec in blank_vecs:
        cnt = 0;
        temp_score = 0.0;
        for word in context:
            if word2id.has_key(word):
                cnt = cnt + 1
                word_vec = lsi.projection.u[int(word2id[word])]
                temp_score = temp_score + mathutils.cossim(vec, word_vec)
            else:
                unseen_word.add(word)
        if cnt != 0:
            score_per_blank.append(temp_score/cnt)

    if len(score_per_blank) == 2:
        score = alpha * score_per_blank[0] + (1 - alpha) * score_per_blank[1]
    elif len(score_per_blank) == 1:
        score = score_per_blank[0]
    return score;

load_gre_answer(answer_path='/Users/apple/Dropbox/NLP/GREVerbal.txt')
print 'finish loading answers'
#lsi = lsamodel.load_model(wordid_txt_file='/Users/apple/graduate/Courses/544NLP/data/wiki_article/wiki_wordids.txt', tfidf_txt_file='/Users/apple/graduate/Courses/544NLP/data/wiki_article/wiki_tfidf.mm',model_file='/Users/apple/graduate/Courses/544NLP/data/wiki_article/wiki_part_model.model')
lsi = lsamodel.load('/Users/apple/graduate/Courses/544NLP/data/wiki_article/wiki_part_model.model')
print 'finish loading lsa model'
word2id = lsamodel.load_word2id(dic_txt_file='/Users/apple/graduate/Courses/544NLP/data/wiki_article/wiki_wordids.txt')
print 'finish loading word2id dictionary'
load_gre_sentence(sentence_path="/Users/apple/Dropbox/NLP/bi_plaintext.txt", lsi=lsi, word2id=word2id);
print 'finish loading and selecting gre sentence completion task answers'
print 'accuracy: ' + str(eval.eval_accuracy(answer, cal_ans))
print len(unseen_word)
print unseen_word