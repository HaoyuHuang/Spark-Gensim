f = open('/Users/junchen/Documents/CSCI544/project/wiki_data/w2v_score.txt','r')
a = []
b = []
for l in f:
    if (len(l.split())) > 0:    
        a.append(float(l))
f = open('/Users/junchen/Documents/CSCI544/project/wiki_data/gre_lm_score_gb(1).txt','r')
s = 0
tp = []
cnt = 0
last = ""
ll = ""
for l in f:
    last = l
    if (len(l.split())) > 0:
        x = float(l)
        s += x
        tp.append(x)
    else:
        ll = l
        for i in range(len(tp)):
            b.append(tp[i]/s)
        tp = []
        s = 0
        cnt += 1
for i in range(len(tp)):
    b.append(tp[i]/s)
o = open('/Users/junchen/Documents/CSCI544/project/wiki_data/ensemble_ans.txt','w')
ans = []
f = open('/Users/junchen/Documents/CSCI544/project/answer.txt','r')
for l in f:
    line = l.split()
    ans.append(line[0])

mp = dict()
mp[0] = 'A'
mp[1] = 'B'
mp[2] = 'C'
mp[3] = 'D'
mp[4] = 'E'
for i in range(10):
    print b[i]
print mp[1] == ans[0]
ma_correct = 0
rate = 0
tot = 0
for time in range(100):
    bootstrap = []
    import random
    random.seed(time)
    for i in range(10000):
        r = random.randint(0, 382)
        bootstrap.append(r)
    for x in range(1, 101):
    
        alpha = x / 100.0
        my_ans = []
        correct = 0
        for k in range(len(bootstrap)):
            i = bootstrap[k]
            ma = -9999
            index = 0
            for j in range(5):
                if a[i * 5 + j] * alpha  - b[i * 5 + j] * (1 - alpha) > ma:
                    ma = a[i * 5 + j] * alpha - b[i * 5 + j] * (1 - alpha)
                    index = j
            #print ans[i], mp[index]
            if ans[i] == mp[index]:
                correct += 1
        #print correct
        if correct * 1.0 / len(bootstrap) > ma_correct:
            ma_correct = correct * 1.0 / len(bootstrap)
            rate = alpha
    print ma_correct, rate
    tot += ma_correct
print tot / 100.0
