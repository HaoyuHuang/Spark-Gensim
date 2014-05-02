__author__ = 'haoyu'

from src import mathutils
import nltk

answer = [];
cal_ans = [];

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

def load_gre_sentence(sentence_path):
    """load all gre sentence tasks and select answers based on LSA"""
    file = open(sentence_path);

    taskScore = [];
    taskCount = 0;

    for line in file:
        if "" == line.strip():
            if len(taskScore) == 5:
                maxScore, index = mathutils.max(taskScore);
                cal_ans.append(index);
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

            blanks = [[]];

            for blank in options:
                temp = nltk.word_tokenize(blank);
                print temp;
                blanks.append(temp);

            taskScore.append(mathutils.calculate_score(context=context, blanks=blanks));

    file.close();
    print taskCount;


#load_gre_sentence(answer_path="/Users/apple/Dropbox/NLP/GREVerbal.txt");
load_gre_sentence("/Users/apple/Dropbox/NLP/bi_d_plaintext.txt");