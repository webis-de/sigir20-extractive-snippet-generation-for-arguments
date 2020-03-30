import json
import re
import spacy
from typing import List, Any
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import pickle

# import utils

class FeatureGeneration:
    nlp = spacy.load('en_core_web_sm')

    def depth_dep_tree(self, corpus):
        """
        Create matrix containing depth of dependeny tree as features for each sentence.

        :param corpus: list of documents containing lists of sentences (or list of arguments)
        :return: numpy array; shape number sentences x 1
        """

        depth_vec = []
        for doc in corpus:
            for sentence in doc:
                s = self.nlp(sentence)
                root = [token for token in s if token.head == token][0]
                depth_vec.append(self.get_depth(root))
        return np.array(depth_vec).reshape(len(depth_vec), 1)

    def get_depth(self, root, depth=0):

        if not list(root.children):
            return 0
        else:
            return 1 + max(self.get_depth(x) for x in root.children)

    def number_tokens(self, corpus):
        """
        Create matrix with number of tokens per sentence

        :param corpus: list of documents containing lists of sentences (or list of arguments)
        :return: numpy array; shape number of sentences x 1
        """
        token_vec = []
        for doc in corpus:
            for sentence in doc:
                token_len = []
                s = self.nlp(sentence)
                token_len.append(len(s))
                token_vec.append(token_len)
        return np.array(token_vec)

    def relative_position(self, corpus):
        """
        Create matrix containing the position of the sentence divided by the number of sentences in the document

        :param corpus: corpus: list of documents containing lists of sentences (or list of arguments)
        :return: numpy array; shape number sentences x 1
        """
        position_vec = []
        for doc in corpus:
            i = 0
            for sentence in doc:
                # print(sentence)
                i += 1
                position_vec.append(i / len(doc))
        return np.array(position_vec).reshape(len(position_vec), 1)

    def num_3pos_tags(self, corpus):
        """
        Create matrix containing the number of trigram pos tags  of  sentences


        :param corpus: list of documents containing lists of sentences (or list of arguments)
        :return:  numpy array; shape number sentences x number trigram pos tags
        """
        pos_vec: List[List[int]] = []
        dict = {}
        key = ""
        for tag in self.nlp.tokenizer.vocab.morphology.tag_map:
            key1 = tag + " "
            for tag in self.nlp.tokenizer.vocab.morphology.tag_map:
                key2 = tag + " "
                for tag in self.nlp.tokenizer.vocab.morphology.tag_map:
                    dict[key1 + key2 + tag] = 0

        for doc in corpus:
            for sentence in doc:
                s = self.nlp(sentence)
                for i in range((len(s) - 2)):
                    key = ""
                    key = s[i].tag_ + " " + s[i + 1].tag_ + " " + s[i + 2].tag_
                    # print(key)
                    dict[key] += 1
                    # print(dict[key])
                pos_vec.append(list(dict.values())[:])
                # print(pos_vec)
                for key in dict:
                    dict[key] = 0
        return np.array(pos_vec)

    def num_pos_tags(self, corpus):
        """
        Create matrix containing feature with number of pos tags occurring in each sentence.

        :param corpus: corpus: list of documents containing lists of sentences (or list of arguments)
        :return: numpy array; shape number of sentences x number of pos tags
        """
        pos_vec = []
        dict = {}
        for tag in self.nlp.tokenizer.vocab.morphology.tag_map:
            # print(tag)
            dict[tag] = 0

        for doc in corpus:
            for sentence in doc:
                s = self.nlp(sentence)
                for token in s:
                    dict[token.tag_] = dict[token.tag_] + 1
                pos_vec.append(list(dict.values())[:])
                dict = {}
                for tag in self.nlp.tokenizer.vocab.morphology.tag_map:
                    dict[tag] = 0
        # print(pos_vec)
        return np.array(pos_vec)

    def num_named_entity(self, corpus):
        """
        Create matrix containing feature with number of entity tags occurring in each sentence.

        :param corpus: corpus: list of documents containing lists of sentences (or list of arguments)
        :return: numpy array; shape number of sentences x number of entity tags
        """

        entity_vec = []
        dict = {}
        for e in self.nlp.entity.labels:
            dict[e] = 0

        for doc in corpus:
            for sentence in doc:
                s = self.nlp(sentence)
                for ent in s.ents:
                    dict[ent.label_] = dict[ent.label_] + 1
                entity_vec.append(list(dict.values())[:])
                dict = {}
                for e in self.nlp.entity.labels:
                    dict[e] = 0
        # print(entity_vec)
        return np.array(entity_vec)

    def num_of_words(self, corpus):
        """
        Create matrix containing number of words per sentence.

        :param corpus:  corpus: list of documents containing lists of sentences (or list of arguments)
        :return: numpy array; shape sentences x 1
        """
        count_words = []
        for doc in corpus:
            for sentence in doc:
                count = len(re.findall(r'\w+', sentence))
                count_words.append([count])
        return np.array(count_words)

    def unigrams(self, vocab, corpus, count_vectorizer=None):
        """
        Create matrix of unigram features

        :param vocab: vocabulary (list of words that correspond to feature dimensions)
        :param corpus: corpus:  corpus: list of documents containing lists of sentences (or list of arguments)
        :param count_vectorizer: scikit CountVectorizer
        :return: numpy array; shape number sentences x length vocab
        """
        if count_vectorizer is None:
            vectorizer = CountVectorizer(max_features=5000)
            vectorizer.fit(vocab)
        else:
            vectorizer = count_vectorizer
        bag_of_words: List[Any] = []
        # print(corpus)
        for doc in corpus:
            for sentence in doc:
                # print(sentence)
                vec = vectorizer.transform([sentence]).toarray()
                bag_of_words.append(vec[0])
        # print(bag_of_words)
        return np.array(bag_of_words)

    def avg_max_tf_idf(self, vocab, corpus, tfidf_vectorizer=None):
        """

        :param vocab: vocabulary (list of words that correspond to feature dimensions)
        :param corpus:  corpus: list of documents containing lists of sentences (or list of arguments)
        :param tfidf_vectorizer: scikit TfIdfVectorizer
        :return: numpy array; shape number of sentences x 2
        """
        if tfidf_vectorizer is None:
            vectorizer = TfidfVectorizer()
            vectorizer.fit(vocab)
        else:
            vectorizer = tfidf_vectorizer
        max_avg: List[Any] = []
        for doc in corpus:
            for sentence in doc:
                vec = vectorizer.transform([sentence]).toarray()
                max_avg.append(np.hstack((np.amax(vec), np.average(vec))))
        # print(max_avg)
        return np.array(max_avg)

    def num_pos_neg_neu(self, corpus):
        """
        Create marix containing the number of positive, negative and neutral words for each sentences according to the lexicon by Theresa Wilson, Janyce Wiebe and Paul Hoffmann (2005). Recognizing Contextual
        Polarity in Phrase-Level Sentiment Analysis.
                
        :param corpus:  corpus: list of documents containing lists of sentences (or list of arguments)
        :return:  numpy array; shape number of sentences x 3
        """
        # type, len, word1, pos1, stemmed1, polarity, priorpolarity

        header = ["type", "len", "word1", "pos1", "stemmed1", "polarity", "priorpolarity"]
        table = []
        dicts_from_file = defaultdict(list)
        with open('subjectivity_clues_hltemnlp05/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff',
                  'r') as mpqa:
            for line in mpqa:
                line = line.strip('\n')
                l = line.split(" ")
                list_element = dict(s.split('=') for s in l)
                # print(list_element)
                dicts_from_file[list_element["word1"]].append(list_element)
        # pos,neutral,neg
        pos_neg_neu = []
        for doc in corpus:
            dict_sent = {"positive": 0,
                         "neutral": 0,
                         "negative": 0,
                         "both": 0
                         }
            for sentence in doc:
                d = self.nlp(sentence)
                for token in d:
                    if token.text.lower() in dicts_from_file:
                        dict_sent[dicts_from_file[token.text.lower()][0]["priorpolarity"]] += 1
                pos_neg_neu.append(list(dict_sent.values()))
        return np.array(pos_neg_neu)
        # print(pos_neg_neu)

        # for list in dicts_from_file:
        #     for head in header
        # for doc in corpus:
        #     for sentence in doc:
        #

    def centroidnes(self, vocab, corpus, threshold=1.7, tfidf_vectorizer=None):
        """
        Create matrix containing score for sentences according to the centroid-based summarization algorihm by Radev et. al
        Radev, D. R., Jing, H., Styś, M., & Tam, D. (2004). Centroid-based summarization of multiple documents. Information Processing & Management, 40(6), 919-938.

        :param vocab:  vocabulary (list of words that correspond to feature dimensions)
        :param corpus:  corpus: list of documents containing lists of sentences (or list of arguments)
        :param threshold: tfidf value threshold for token to be included in the centroid document
        :param tfidf_vectorizer:  scikit TfIdfVectorizer
        :return: np.array containing centroid score per sentence
        """

        if tfidf_vectorizer is None:
            vectorizer = TfidfVectorizer(max_features=5000)
            vectorizer.fit(vocab)
            tfidf = vectorizer.vocabulary_
        else:
            vectorizer = tfidf_vectorizer
            tfidf = vectorizer.vocabulary_
        # print(tfidf)
        # print(vectorizer.idf_)
        centroid = {}
        for word in tfidf:
            if vectorizer.idf_[tfidf[word]] > threshold:
                centroid[word] = vectorizer.idf_[tfidf[word]]
            else:
                centroid[word] = 0
        centroid_feature = []
        counter = 0
        for doc in corpus:
            for sentence in doc:
                centroid_feature.append([0])
                d = self.nlp(sentence)
                for token in d:
                    # print(centroid_feature)
                    if token.text.lower() in tfidf:
                        centroid_feature[counter][0] += centroid[token.text.lower()]
                counter += 1
        return np.array(centroid_feature)

    def punctuation(self, corpus):
        """
        Create matrix of feature represenation of each sentence from a corpus.
        Each dimension coresponds to a punctuation mark.


        :param corpus: corpus:  corpus: list of documents containing lists of sentences (or list of arguments)
        :return: numpy array; shape number of sentences x 8
        """
        punc_vec = []
        punct_dict = {
            "!": 0,
            "?": 0,
            "'": 0,
            ":": 0,
            ".": 0,
            ",": 0,
            "-": 0,
            ";": 0,
            ",": 0,
        }
        for doc in corpus:
            for sentence in doc:
                s = self.nlp(sentence)
                for token in s:
                    if token.text in punct_dict:
                        punct_dict[token.text] += 1
                punc_vec.append(list(punct_dict.values())[:])
                for key in punct_dict:
                    punct_dict[key] = 0

        return np.array(punc_vec)

    def save_vectorizer(self, vocab, name_tfidf, name_count):
        """
        Train TfidfVectorizer with vocabulary and pickle for later usage.

        :param vocab: vocabulary (list of words that correspond to feature dimensions)
        :param name_tfidf: name for pickle file for tfidf
        :param name_count: name for pickle file for count
        :return:
        """
        tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        tfidf_vectorizer.fit(vocab)
        pickle.dump(tfidf_vectorizer, open(name_tfidf, "wb"))
        vectorizer = CountVectorizer(max_features=5000)
        vectorizer.fit(vocab)
        pickle.dump(vectorizer, open(name_count, "wb"))

    def create_stacked_features(self, tfidf_vectorizer, count_vectorizer, corpus, vocab=None):
        """
        Stack features for RidgeRegression training.

        :param tfidf_vectorizer:
        :param count_vectorizer:
        :param corpus: corpus: list of documents containing lists of sentences (or list of arguments)
        :param vocab: vocabulary (list of words that correspond to feature dimensions)
        :return:
        """
        feature_1 = FeatureGeneration().avg_max_tf_idf(vocab, corpus, tfidf_vectorizer=tfidf_vectorizer)
        feature_2 = FeatureGeneration().num_pos_tags(corpus)
        feature_3 = FeatureGeneration().num_pos_neg_neu(corpus)
        feature_4 = FeatureGeneration().centroidnes(vocab, corpus, tfidf_vectorizer=tfidf_vectorizer)
        feature_5 = FeatureGeneration().unigrams(vocab, corpus, count_vectorizer=count_vectorizer)
        feature_6 = FeatureGeneration().num_named_entity(corpus)
        feature_7 = FeatureGeneration().num_of_words(corpus)
        return np.hstack((feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7))

    def stack_all(self, tfidf_vectorizer, count_vectorizer, corpus):
        """
        Stack all features defined in this class for argument mining tasks.
        :param tfidf_vectorizer:
        :param count_vectorizer:
        :param corpus: corpus: list of documents containing lists of sentences (or list of arguments)
        :return: numpy arrays of stacked features
        """


        vocab = []
        features = [(self.depth_dep_tree, [corpus]), (self.number_tokens, [corpus]), (self.relative_position, [corpus]),
                    (self.num_3pos_tags, [corpus]), (self.num_pos_neg_neu, [corpus]),
                    (self.num_named_entity, [corpus]), (self.num_of_words, [corpus]),
                    (self.unigrams, [vocab, corpus, count_vectorizer]), (
                        self.avg_max_tf_idf, [vocab, corpus, tfidf_vectorizer]),
                    (self.centroidnes, [vocab, corpus, 1.7, tfidf_vectorizer]), (self.punctuation, [corpus])]
        x = []
        for f in features:
            X_i = f[0](*f[1])
            x.append(X_i)
        return np.hstack(tuple(x))


if __name__ == '__main__':
    # print("test")
    with open("norm_scored_debate.json", "r") as debate_file:
        data = json.load(debate_file)
    label_train = []
    # list of documents
    corpus_train = []
    # list of sentences
    doc = []
    vocab = []
    train_length = round(len(data) * 0.9)
    print(train_length)
    t = 0
    for i in range(train_length):
        doc = []
        text = ''
        for j in range(len(data[i])):
            label_train.append(data[i][j][1])
            doc.append(data[i][j][0])
            text = text + data[i][j][0] + ' '
        t += len(data[i])
        print(len(data[i]))
        vocab.append(text)
        # print(vocab)
        corpus_train.append(doc)
    print(t)
    label_train = np.array(label_train).reshape((len(label_train), 1))

    print("train ready")
    t = 0
    label_dev = []
    corpus_dev = []
    k = train_length
    for k in range(len(data)):
        doc = []
        text = ''
        for j in range(len(data[i])):
            label_dev.append(data[i][j][1])
            doc.append(data[i][j][0])
            text = text + data[i][j][0] + ' '
        t += len(data[i])
        vocab.append(text)
        # print(vocab)
        corpus_dev.append(doc)
    label_dev = np.array(label_dev).reshape((len(label_dev), 1))
    print(t)
    # feature_generation().save_vectorizer(vocab)
    print("saved")
    # #
    # # print("dev ready")
    # #
    # count_vec = pickle.load(open("CountVectorizer.p", "rb"))
    # tfidf_vec = pickle.load( open( "TfidfVectorizer.p", "rb" ) )
    # #
    # # # for filename in os.listdir("args_me"):
    # # #     with open("args_me/" + filename, "r", encoding="utf8") as debate_file:
    # # #         data = json.load(debate_file)
    # # #         argumentList = list(map(Argument.from_json, data["arguments"]))
    # # # print(len(feature_generation().sample_indices(100, len(argumentList))))
    # # # for arg in argumentList:
    # # #     text = ""
    # # #     for s in arg:
    # # #         text = text + s
    # # #     vocab.append(text)
    # # #
    # # # feature_generation().save_vectorizer(vocab)
    # # # print("done")
    # # # feature = feature_generation().avg_max_tf_idf(vocab,corpus)
    # # # feature_generation().num_pos_tags(corpus)
    # # # feature_generation().num_pos_neg_neu(corpus)
    # # # feature_generation().centroidnes(vocab,corpus)
    # # # pos = feature_generation().depth_dep_tree(corpus)
    # #
    # #
    feature = FeatureGeneration().avg_max_tf_idf(vocab, corpus_train, tfidf_vectorizer=None)
    feature1 = FeatureGeneration().num_pos_tags(corpus_train)
    feature2 = FeatureGeneration().num_pos_neg_neu(corpus_train)
    feature3 = FeatureGeneration().centroidnes(vocab, corpus_train, tfidf_vectorizer=None)
    feature4 = FeatureGeneration().unigrams(vocab, corpus_train, count_vectorizer=None)
    feature5 = FeatureGeneration().num_named_entity(corpus_train)
    feature6 = FeatureGeneration().num_of_words(corpus_train)

    r = np.hstack((feature, feature1, feature2, feature3, feature4, feature5, feature6))
    print("r stacked")
    feature_dev = FeatureGeneration().avg_max_tf_idf(vocab, corpus_dev, tfidf_vectorizer=None)
    feature_dev1 = FeatureGeneration().num_pos_tags(corpus_dev)
    feature_dev2 = FeatureGeneration().num_pos_neg_neu(corpus_dev)
    feature_dev3 = FeatureGeneration().centroidnes(vocab, corpus_dev, tfidf_vectorizer=None)
    feature_dev4 = FeatureGeneration().unigrams(vocab, corpus_dev, count_vectorizer=None)
    feature_dev5 = FeatureGeneration().num_named_entity(corpus_dev)
    feature_dev6 = FeatureGeneration().num_of_words(corpus_dev)

    r_dev = np.hstack((feature_dev, feature_dev1, feature_dev2, feature_dev3, feature_dev4, feature_dev5, feature_dev6))
    print("r_dev stacked")
    np.save("r_train", r)
    np.save("r_dev", r_dev)
    np.save("label_train", label_train)
    np.save("label_dev", label_dev)
    print("r" + str(r.shape))

    # r = np.load("r_train.npy", mmap_mode="r+")
    # label_train = np.load("label_train.npy", mmap_mode="r+")

    r_pairwise_differences = []
    difference = []
    length = 0
    print(train_length)
    for i in range(train_length):
        for j in range(len(data[i])):
            for k in range(j + 1, len(data[i])):
                print(length, k, j, len(data[i]))
                r_pairwise_differences.append(r[length + k] - r[length + j])
        length += len(data[i])
    print("finish")
    r_pairwise_differences = np.array(r_pairwise_differences)
    L = np.ones((np.array(r_pairwise_differences).shape[0], 1))
    reg = RidgeRegression.RidgeRegressor()
    print("r" + str(r.shape))
    print("label_train" + str(label_train.shape))
    print("r_pairwise_differences" + str(r_pairwise_differences.shape))
    print("L" + str(L.shape))
    R = r_pairwise_differences
    # r_dev = np.load("r_dev.npy")
    # print("r_dev" + str(r_dev.shape))
    # label_dev = np.load("label_dev.npy")
    # print("label_dev" + str(label_dev.shape))
    reg.fit_best_params(r, label_train, R, L, r_dev, label_dev)
    pickle.dump(reg, open("Regression.p", "wb"))
    # args = utils.return_indexed_args("indices.json")

    # feature_pred = feature_generation().avg_max_tf_idf(vocab=None, args, tfidf_vectorizer=tfidf_vec)
    # feature_pred1 = feature_generation().num_pos_tags(args)
    # feature_pred2 = feature_generation().num_pos_neg_neu(args)
    # feature_pred3 = feature_generation().centroidnes(vocab, args, tfidf_vectorizer=tfidf_vec)
    # feature_pred4 = feature_generation().unigrams(vocab, args, count_vectorizer=count_vec)
    # feature_pred5 = feature_generation().num_named_entity(args)
    # feature_pred6 = feature_generation().num_of_words(args)

    # r_pred = np.hstack((feature_pred, feature_pred1, feature_pred2, feature_pred3, feature_pred4, feature_pred5, feature_pred6))
    # pred = reg.predict(args)
    # print(pred)

#     print(np.dot(np.dot(R.T, 0), R).shape)
#     reg.fit(r, label_train, r_pairwise_differences, L )
#     X = r
#     beta_ = 3 * np.eye(X.shape[1])
#     lamda_ = 4 * np.eye(R.shape[0])
#     y = label_train
#    # print((np.dot(X.T,X) + np.dot(np.dot(R.T,lamda_),R) + beta_))
#    # print((np.dot(X.T,y) + np.dot(np.dot(R.T,lamda_),L)))
#    # print(np.linalg.inv(np.dot(X.T,X) + np.dot(np.dot(R.T,lamda_),R) + beta_))
#    # print(np.dot(np.linalg.inv(np.dot(X.T,X) + np.dot(np.dot(R.T,lamda_),R) + beta_),(np.dot(X.T,y) + np.dot(np.dot(R.T,lamda_),L))))
#     #print(feature_generation().punctuation(corpus))
#
#
#
