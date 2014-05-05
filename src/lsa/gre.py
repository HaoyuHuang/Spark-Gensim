__author__ = 'haoyu'
import lsamodel, mathutils, eval, rake
import nltk
import heapq

TOTAL_SIMILARITY_K_MAX = 1;
TOTAL_SIMILARITY = 2;
TOTAL_SIMILARITY_WITH_COMBINATION = 3;
TOTAL_SIMILARITY_WITH_RAKE = 4;

rake_obj = rake.Rake("SmartStoplist.txt")

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

def load_gre_sentence(sentence_path, lsi, word2id, algorithm=TOTAL_SIMILARITY):
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
            text = ""
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
            elif algorithm == TOTAL_SIMILARITY_K_MAX:
                taskScore.append(calculate_total_similarity_by_k_max(lsi, word2id=word2id, blanks=blanks, context=context));
            elif algorithm == TOTAL_SIMILARITY_WITH_RAKE:
                taskScore.append(calculate_total_similarity_with_rake(lsi, text, word2id, blanks))
    file.close();
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
            text = "";
            sentence = "";
            li = -1;
            options = [];
            taskId.append(line[0:1]);
            isInParenthese = False;
            for i in range(2, len(line)):
                if line[i] == '(':
                    li = i;
                    text = text + line[i]
                    isInParenthese = True;
                elif line[i] == ')':
                    options.append(line[li+1:i]);
                    isInParenthese = False;
                elif isInParenthese == False:
                    text = text + line[i]
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
            elif algorithm == TOTAL_SIMILARITY_K_MAX:
                taskScore.append(calculate_total_similarity_by_k_max(lsi, word2id=word2id, blanks=blanks, context=context));
            elif algorithm == TOTAL_SIMILARITY_WITH_RAKE:
                taskScore.append(calculate_total_similarity_with_rake(lsi=lsi,text=text, word2id=word2id, blanks=blanks));
    file.close();
    print taskCount;

def calculate_total_similarity(lsi, word2id = {}, context=[], blanks=[]):
    """calcualte the score given context words and blanks"""
    score = 0.0
    cnt = 0.0
    for i in range(0, len(blanks)):
        for em in blanks[i]:
            if word2id.has_key(em):
                em_vec = lsi.projection.u[int(word2id[em])]

                for j in range(0, len(blanks)):
                    if j != i:
                        for em2 in blanks[j]:
                            if word2id.has_key(em2):
                                cnt = cnt + 1
                                word_vec = lsi.projection.u[int(word2id[em2])];
                                score = score + mathutils.cossim(em_vec, word_vec)

                for word in context:
                    if word2id.has_key(word):
                        cnt = cnt + 1
                        word_vec = lsi.projection.u[int(word2id[word])]
                        score = score + mathutils.cossim(em_vec, word_vec)
    if cnt != 0:
        score = score / cnt
    return score;

def calculate_total_similarity_by_k_max(lsi, word2id = {}, context=[], blanks=[], k=2):
    """calcualte the score given context words and blanks"""
    score = 0.0
    queue = []
    for i in range(0, len(blanks)):
        for em in blanks[i]:
            if word2id.has_key(em):
                em_vec = lsi.projection.u[int(word2id[em])]
                # loop all other blanks
                for j in range(0, len(blanks)):
                    if j != i:
                        for em2 in blanks[j]:
                            if word2id.has_key(em2):
                                word_vec = lsi.projection.u[int(word2id[em2])];
                                heapq.heappush(queue, mathutils.cossim(em_vec, word_vec))
                            else:
                                unseen_word.add(em2)

                for word in context:
                    if word2id.has_key(word):
                        word_vec = lsi.projection.u[int(word2id[word])]
                        heapq.heappush(queue, mathutils.cossim(em_vec, word_vec))
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

def calculate_total_similarity_with_rake(lsi,text, word2id = {}, blanks=[]):
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

    # print keyword_dict
    weight_sum = 0.0
    for blank in blanks:
        for em in blank:
            if word2id.has_key(em):
                em_vec = lsi.projection.u[int(word2id[em])]
                for key in keyword_dict.keys():
                    if word2id.has_key(key):
                        cnt = cnt + 1
                        score = score + mathutils.cossim(em_vec, lsi.projection.u[int(word2id[key])]) * keyword_dict[key]
                        # weight_sum = weight_sum + keyword_dict[key]

    if cnt != 0:
        score = score / (cnt)
    return score;


def calculate_total_similarity_with_combination(lsi, word2id = {}, context=[], blanks=[], alpha=0.5):
    """calcualte the score given context words and blanks"""
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

def wiki_50M_model_test():
    """train lsa model with 50M wikipedia dumps, normal accuracy 24.2% with 2100 unseen words, definition accuracy 17% with 4000+ unseen words"""
    load_gre_answer(answer_path='/Users/apple/Dropbox/NLP/GREVerbal.txt')
    print 'finish loading answers'
    lsi = lsamodel.load('/Users/apple/graduate/Courses/544NLP/data/wiki_article/wiki_part_model.model')
    print 'finish loading lsa model'
    word2id = lsamodel.load_word2id(dic_txt_file='/Users/apple/graduate/Courses/544NLP/data/wiki_article/wiki_wordids.txt')
    print 'finish loading word2id dictionary'
    # load_gre_sentence_definition(sentence_path="/Users/apple/Dropbox/NLP/bi_d_plaintext.txt", lsi=lsi, word2id=word2id);
    load_gre_sentence(sentence_path="/Users/apple/Dropbox/NLP/bi_plaintext.txt", lsi=lsi, word2id=word2id);
    print 'finish loading and selecting gre sentence completion task answers'
    print 'accuracy: ' + str(eval.eval_accuracy(answer, cal_ans))
    print len(unseen_word)
    print unseen_word

def wiki_10G_model_test():
    """train lsa model with wiki 10G data, 753 unseen words, 23.4% accuracy, definition accuracy 20.6% with 1810 unseen words"""
    load_gre_answer(answer_path='/Users/apple/Dropbox/NLP/GREVerbal.txt')
    print 'finish loading answers'
    lsi = lsamodel.load('/Users/apple/graduate/Courses/544NLP/data/wiki_article/wiki_latest_model/lsi.model')
    print 'finish loading lsa model'
    word2id = lsamodel.load_word2id(dic_txt_file='/Users/apple/graduate/Courses/544NLP/data/wiki_article/wiki_latest_model/wiki_en_wordids.txt')
    print 'finish loading word2id dictionary'
    load_gre_sentence_definition(sentence_path="/Users/apple/Dropbox/NLP/bi_d_plaintext.txt", lsi=lsi, word2id=word2id, algorithm=TOTAL_SIMILARITY_WITH_RAKE);
    # load_gre_sentence(sentence_path="/Users/apple/Dropbox/NLP/bi_plaintext.txt", lsi=lsi, word2id=word2id, algorithm=TOTAL_SIMILARITY_WITH_RAKE);
    print 'finish loading and selecting gre sentence completion task answers'
    print 'accuracy: ' + str(eval.eval_accuracy(answer, cal_ans))
    print len(unseen_word)
    print unseen_word

if __name__ == '__main__':
    wiki_10G_model_test()
