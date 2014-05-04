__author__ = 'haoyu'

def eval_accuracy(answers=[], pred=[]):
    """evaluate the accuracy"""
    correct = 0.0;
    for i in range(0, len(answers)):
        if answers[i] == pred[i]:
            correct = correct + 1
    return correct / len(answers);
