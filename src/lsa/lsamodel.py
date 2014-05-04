from gensim.corpora import TextCorpus, MmCorpus, Dictionary
from gensim.models import LsiModel

text_corpus_file = '/Users/apple/graduate/Courses/544NLP/Workspace/Spark/gutenberg/gutenberg_7g_lowcase.txt'
dict_file = '/Users/apple/graduate/Courses/544NLP/Workspace/Spark/gutenberg/gutenberg_7g_lowcase_dic.dic'
dic_txt_file = '/Users/apple/graduate/Courses/544NLP/Workspace/Spark/gutenberg/gutenberg_7g_lowcase_dic.txt'
mm_corpus_file = '/Users/apple/graduate/Courses/544NLP/Workspace/Spark/gutenberg/gutenberg_7g_lowcase_corpus.mm'
model_file = '/Users/apple/graduate/Courses/544NLP/Workspace/Spark/gutenberg/gutenberg_7g_lowcase.model'

def pretrain():
    """pre train the text corpus and build the dictionary"""
    gutenberg_corpus = TextCorpus(text_corpus_file)
    gutenberg_corpus.dictionary.save(dict_file)
    gutenberg_corpus.dictionary.save_as_text(dic_txt_file)
    mm = MmCorpus.serialize(mm_corpus_file, gutenberg_corpus)
    print mm;

def train(text_corpus_file, dict_file):
    """train lsi model from text corpus"""
    gutenberg_corpus = TextCorpus(text_corpus_file)
    dict = Dictionary.load(dict_file)
    lsi = LsiModel(corpus=gutenberg_corpus, id2word=dict, num_topics=400)
    lsi.save(model_file)
    print lsi.projection.u
    print lsi.projection.u.size
    print lsi.projection.u[0].size

def load(model_file):
    """load the lsi model into memory"""
    lsi = LsiModel.load(model_file)
    return lsi;

def load_word2id(dic_txt_file):
    file = open(dic_txt_file)
    word2id = {};
    for line in file:
        ems = line.split('\t');
        word2id[ems[1]] = ems[0];
    return word2id;

def load_model(wordid_txt_file, tfidf_txt_file, model_file):
    id2word = Dictionary.load_from_text(wordid_txt_file)
    mm = MmCorpus(tfidf_txt_file)
    lsi = LsiModel(corpus=mm, id2word=id2word, num_topics=400)
    lsi.save(model_file)
    return lsi

if __name__ == '__main__':
    pretrain()