import lsamodel, mathutils, eval

__author__ = 'haoyu'

import nltk,rake
import heapq
import gensim
my_dic = dict()

K_MAX_TOTAL_SIMILARITY = 1;
TOTAL_SIMILARITY = 2;
TOTAL_SIMILARITY_WITH_COMBINATION = 3;
TOTAL_SIMILARITY_WITH_RAKE = 4;

rake_obj = rake.Rake("SmartStoplist.txt")

prep = dict()
prep['to'] = True
prep['from'] = True
prep['of'] = True
prep['about'] = True
prep['against'] = True
prep['with'] = True
prep['by'] = True
prep['in'] = True
prep['on'] = True
prep['for'] = True
prep['up'] = True
prep['o'] = True
prep['among'] = True
prep['into'] = True
prep['a'] = True
prep['an'] = True
prep['between'] = True
prep['at'] = True

answer = [];
cal_ans = [];
unseen_word = set();
def load_dic(filepath):
    for l in open(filepath, 'r'):
        line = l.split()
        my_dic[line[0]] = True
        
load_dic('/Users/junchen/Documents/CSCI544/project/wiki_data/gutenberg_200.dic')
model = gensim.models.Word2Vec.load('/Users/junchen/Documents/CSCI544/project/wiki_data/gutenberg_640.model')
print len(my_dic)

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
    '''
    cnt = 0
    for line in file:
        answer.append(line[0])
        cnt += 1
    print 'cnt:',cnt
    '''
    print len(answer);
    file.close();

def clear_answer():
    cal_ans = []

def load_gre_sentence(sentence_path, lsi, word2id, algorithm=TOTAL_SIMILARITY,k=11):
    """load all gre sentence tasks and select answers based on LSA"""
    file = open(sentence_path);
    
    taskScore = [];
    taskCount = 0;
    f = open('/Users/junchen/Documents/CSCI544/project/wiki_data/w2v_score.txt','w')
    for line in file:
        if "" == line.strip():
            if len(taskScore) == 5:
                s = 0
                for t in taskScore:
                    s += t + 1.0
                for t in range(len(taskScore)):
                    f.write(str((taskScore[t] + 1.0)/s) + '\r\n')
                f.write('\r\n')
                maxScore, index = mathutils.max(taskScore);
                canswer = index2answer(index)
                #print 'score : ' + str(maxScore) + ' answer: ' + canswer;
                if len(cal_ans) == len(answer):
                    cal_ans[taskCount] = canswer
                else:
                    cal_ans.append(canswer);
                taskScore = [];
                taskCount = taskCount + 1;
            pass;
        else:
            text = "";
            sentence = "";
            li = -1;
            options = [];
            
            isInParenthese = False;
            for i in range(0, len(line)):
                if line[i] == '(':
                    li = i;
                    text = text + line[i]
                    isInParenthese = True;
                elif line[i] == ')':
                    options.append(line[li+1:i]);
                    isInParenthese = False;
                elif isInParenthese == False:
                    sentence = sentence + line[i];
                    text = text + line[i]
            context = nltk.word_tokenize(sentence);
            
            blanks = [];
            
            for blank in options:
                temp = nltk.word_tokenize(blank);
                # print temp;
                blanks.append(temp);
            if algorithm == TOTAL_SIMILARITY:
                taskScore.append(calculate_total_similarity(lsi, word2id=word2id, blanks=blanks, context=context));
            elif algorithm == TOTAL_SIMILARITY_WITH_COMBINATION:
                taskScore.append(calculate_total_similarity_with_combination(lsi, word2id=word2id, blanks=blanks, context=context));
            elif algorithm == K_MAX_TOTAL_SIMILARITY:
                taskScore.append(calculate_total_similarity_by_k_max(lsi, word2id=word2id, blanks=blanks, context=context, k=k));
            elif algorithm == TOTAL_SIMILARITY_WITH_RAKE:
                taskScore.append(calculate_total_similarity_with_rake(lsi, text, word2id, blanks,k))
    file.close();
    f.close()
    print taskCount;

def load_gre_sentence_definition(sentence_path, lsi, word2id, algorithm=TOTAL_SIMILARITY):
    """load all gre sentence tasks and select answers based on LSA"""
    file = open(sentence_path);
    
    taskScore = [];
    taskCount = 0;
    taskId = [];
    
    for line in file:
        if "" == line.strip():
            maxScore, index = mathutils.max(taskScore);
            print 'score : ' + str(maxScore) + ' answer: ' + taskId[index];
            cal_ans.append(taskId[index]);
            taskScore = [];
            taskCount = taskCount + 1;
        else:
            sentence = "";
            li = -1;
            options = [];
            taskId.append(line[0:1]);
            isInParenthese = False;
            for i in range(2, len(line)):
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
            if algorithm == TOTAL_SIMILARITY:
                taskScore.append(calculate_total_similarity(lsi, word2id=word2id, blanks=blanks, context=context));
            elif algorithm == TOTAL_SIMILARITY_WITH_COMBINATION:
                taskScore.append(calculate_total_similarity_with_combination(lsi, word2id=word2id, blanks=blanks, context=context));
            else:
                taskScore.append(calculate_total_similarity_by_k_max(lsi, word2id=word2id, blanks=blanks, context=context));
    file.close();
    print taskCount;

def calculate_total_similarity(lsi, word2id = {}, context=[], blanks=[]):
    """calcualte the score given context words and blanks"""
    score = 0.0
    cnt = 0.0
    for i in range(0, len(blanks)):
        for em in blanks[i] :
            for word in context:
                if my_dic.has_key(em.lower())and my_dic.has_key(word.lower()) and my_dic.has_key(em.lower()):
                    cnt = cnt + 1
                    score = score + model.similarity(em.lower(), word.lower())
    if cnt != 0:
        score = score / cnt
    return score;

def calculate_total_similarity_by_k_max(lsi, word2id = {}, context=[], blanks=[], k=4):
    """calcualte the score given context words and blanks"""
    score = 0.0
    queue = []
    for i in range(0, len(blanks)):
        for em in blanks[i]:
            if my_dic.has_key(em.lower()) and prep.has_key(em) == False:
                for j in range(0, len(blanks)):
                    if j != i:
                        for em2 in blanks[j] :
                            if my_dic.has_key(em2.lower()) and prep.has_key(em2) == False:
                                heapq.heappush(queue, model.similarity(em.lower(), em2.lower()))
                            else:
                                unseen_word.add(em2)
                
                for word in context:
                    if my_dic.has_key(word.lower()):
                        heapq.heappush(queue, model.similarity(em.lower(), word.lower()))
                    else:
                        unseen_word.add(word)
            else:
                unseen_word.add(em)
    kmax = heapq.nlargest(k, queue)
    for i in range(0, len(kmax)):
        if i < k:
            score = score + kmax[i]
        else:
            break
    if len(kmax) > k:
        score = score / k
    elif len(kmax) != 0:
        score = score / len(kmax)
    return score;


def calculate_total_similarity_with_combination(lsi, word2id = {}, context=[], blanks=[], alpha=0.5):
    """calcualte the score given context words and blanks"""
    score = 0.0
    blank_vecs = []
    score_per_blank = []
    for blank in blanks:
        vector = [];
        for em in blank:
            if my_dic.has_key(em.lower()):
                vector.append(model[em.lower()])
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
            if my_dic.has_key(word.lower()):
                cnt = cnt + 1
                word_vec = model[word.lower()]
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


def calculate_total_similarity_with_rake(lsi,text, word2id = {}, blanks=[], k=13):
    """calculate total similarity between the blanks and keywords identified by rake"""
    score = 0.0
    cnt = 0.0
    keywords = rake_obj.run(text)
    keyword_dict = {}
 
    # build the key word dictionary with the key is the token and value is the corresponding value
    for key in keywords:
        words = nltk.word_tokenize(key[0])
        for word in words:
            keyword_dict[word] = key[1]
    queue = []
    #print keyword_dict
    weight_sum = 0.0
    for i in range(0, len(blanks)):
        for em in blanks[i]:
            if my_dic.has_key(em.lower()) and prep.has_key(em) == False:
                for j in range(0, len(blanks)):
                    if j != i:
                        for em2 in blanks[j] :
                            if my_dic.has_key(em2.lower()) and prep.has_key(em2) == False:
                                heapq.heappush(queue, model.similarity(em.lower(), em2.lower()))
                            else:
                                unseen_word.add(em2)
                
                for key in keyword_dict.keys():
                    if my_dic.has_key(key.lower()):
                        cnt = cnt + 1
                        heapq.heappush(queue,model.similarity(em.lower(), key.lower()))
                    else:
                        unseen_word.add(word)
    kmax = heapq.nlargest(k, queue)
    for i in range(0, len(kmax)):
        if i < k:
            score = score + kmax[i]
        else:
            break
    if len(kmax) > k:
        score = score / k
    elif len(kmax) != 0:
        score = score / len(kmax)
    return score;

def wiki_50M_model_test():
    """train lsa model with 50M wikipedia dumps, normal accuracy 24.2% with 2100 unseen words, definition accuracy 17% with 4000+ unseen words"""
    load_gre_answer(answer_path='/Users/apple/Dropbox/NLP/GREVerbal.txt')
    print 'finish loading answers'
    lsi = lsamodel.load('/Users/apple/graduate/Courses/544NLP/data/wiki_article/wiki_part_model.model')
    print 'finish loading lsa model'
    word2id = lsamodel.load_word2id(dic_txt_file='/Users/apple/graduate/Courses/544NLP/data/wiki_article/wiki_wordids.txt')
    print 'finish loading word2id dictionary'
    load_gre_sentence_definition(sentence_path="/Users/apple/Dropbox/NLP/bi_d_plaintext.txt", lsi=lsi, word2id=word2id);
    #load_gre_sentence(sentence_path="/Users/apple/Dropbox/NLP/bi_plaintext.txt", lsi=lsi, word2id=word2id);
    print 'finish loading and selecting gre sentence completion task answers'
    print 'accuracy: ' + str(eval.eval_accuracy(answer, cal_ans))
    print len(unseen_word)
    print unseen_word


def wiki_10G_model_test():
    """train lsa model with wiki 10G data, 753 unseen words, 23.4% accuracy, definition accuracy 20.6% with 1810 unseen words"""
    load_gre_answer(answer_path='/Users/junchen/Documents/CSCI544/project/GREVerbal.txt')
    print 'finish loading answers'
    lsi = lsamodel.load('/Users/junchen/Documents/CSCI544/project/lsi model/lsi.model')
    print 'finish loading lsa model'
    word2id = lsamodel.load_word2id(dic_txt_file='/Users/junchen/Documents/CSCI544/project/wiki_data/wiki_en_wordids.txt')
    print 'finish loading word2id dictionary'
    for i in range(2, 3):
        #load_gre_sentence_definition(sentence_path="/Users/junchen/Documents/CSCI544/project/bi_d_plaintext.txt", lsi=lsi, word2id=word2id);
        load_gre_sentence(sentence_path="/Users/junchen/Documents/CSCI544/project/bi_plaintext.txt", lsi=lsi, word2id=word2id, algorithm=TOTAL_SIMILARITY_WITH_RAKE , k=i)
        print 'finish loading and selecting gre sentence completion task answers'
        print str(i) + ': accuracy: ' + str(eval.eval_accuracy(answer, cal_ans))
        clear_answer()
 #   load_gre_sentence_definition(sentence_path="/Users/junchen/Documents/CSCI544/project/bi_d_plaintext.txt", lsi=lsi, word2id=word2id);
    # load_gre_sentence(sentence_path="/Users/apple/Dropbox/NLP/bi_plaintext.txt", lsi=lsi, word2id=word2id, algorithm=TOTAL_SIMILARITY_WITH_RAKE);
    
    print len(unseen_word)
    print unseen_word

if __name__ == '__main__':
    wiki_10G_model_test()

