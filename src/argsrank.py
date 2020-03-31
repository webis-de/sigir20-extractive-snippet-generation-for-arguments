import json
import re
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
from textwrap import wrap
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from src.feature_generation import FeatureGeneration
import time
from src.Argument import Argument
import spacy


class ArgsRank:

    def __init__(self):
        script_dir = os.path.dirname(__file__)
        f = open(os.path.join(script_dir,"../data/ClaimLexicon.txt"))
        self.claim_markers = f.read().split(", ")
        # for m in CLAIM_MARKERS:

        #     print(m)

        self.discourse_markers = ["for example", "such as", "for instance", "in the case of", "as revealed by",
                                  "illustrated by",
                                  "because", "so", "therefore", "thus", "consequently", "hence", "similariy",
                                  "likewise",
                                  "as with",
                                  "like", "equally", "in the same way", "first", "second ",
                                  "third,", "finally", "next", "meanwhile", "after", "then", "subsequently",
                                  "above all",
                                  "in particular", "especially", "significantly", "indeed", "notably", "but", "however",
                                  "although",
                                  "unless", "except", "apart from", "as long as", "if", "whereas", "instead of",
                                  "alternatively", "otherwise", "unlike", "on the other hand", "conversely"]

        # WEIGHT_SYN = 0.3
        # c_simw = 1
        #
        # d = 0.15
        # dev = False
        # class_mapping = None
        # classes = {}
        self.WEIGHT_SYN = 0.3
        self.c_simw = 0.5
        self.classes = {}
        self.class_mapping = None
        self.d = 0.15

        self.scaler = MinMaxScaler()
        # self.arg_model = pickle.load(open("objects/arg_model2.p", "rb"))
        # self.tfidf_vec = pickle.load(open("objects/tfidf_arg.p", "rb"))
        # self.count_vec = pickle.load(open("objects/count_arg.p", "rb"))

        module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
        self.embed = hub.Module(module_url)

    def plot_similarity(self, labels, features, rotation):
        corr = np.inner(features, features)
        print(corr.shape)
        labels = ["\n".join(wrap(l, 25)) for l in labels]
        sns.set(font_scale=0.6)
        g = sns.heatmap(
            corr,
            xticklabels=labels,
            yticklabels=labels,
            vmin=0,
            vmax=1,
            cmap="YlOrRd")
        g.set_xticklabels(labels, rotation=rotation)
        g.set_yticklabels(labels, rotation=0)
        g.set_title("Semantic Textual Similarity")

        plt.autoscale()
        plt.tight_layout()
        plt.show()

    def power_method(self, M, epsilon):
        """
        Apply power methode to calculate stationary distribution for Matrix M

        :param M: numpy matrix
        :param epsilon:
        :return:
        """
        t = 0
        p_t = (np.ones((M.shape[0], 1)) * (1 / M.shape[0]))
        while True:

            t = t + 1
            p_prev = p_t
            p_t = np.dot(M.T, p_t)
            residual = np.linalg.norm(p_t - p_prev)

            if residual < epsilon:
                break
        return p_t

    def create_test_cluster(self, path):
        cluster = []
        for filename in os.listdir(path):
            with open(path + "/" + filename, "r", encoding="utf8") as text_file:
                data = text_file.read()
                arg = Argument()
                arg.set_sentences(data)
                cluster.append(arg)
        return cluster

    def normalize_by_rowsum(self, M):
        for i in range(M.shape[0]):
            sum = np.sum(M[i])
            for j in range(M.shape[1]):
                M[i][j] = M[i][j] / sum
        return M

    def run_and_plot(self, session_, input_tensor_, messages_, encoding_tensor):
        message_embeddings_ = session_.run(
            encoding_tensor, feed_dict={input_tensor_: messages_})
        # plot_similarity(messages_, message_embeddings_, 0)
        return message_embeddings_

    def add_syn_similarity(self, matrix, cluster, weight):
        """
        Add score for syntactic similarity of resultng from PageRank
        :param matrix: semantic similarity matrix
        :param cluster: list of arguments
        :param weight:
        :return:
        """

        feature = []

        start = time.time()
        f_position = FeatureGeneration().relative_position(cluster)
        end = time.time()
        feature.append(f_position)
        start = time.time()
        f_number_tokens = FeatureGeneration().number_tokens(cluster)
        end = time.time()
        start = time.time()
        feature.append(f_number_tokens)
        f_depth_dep_tree = FeatureGeneration().depth_dep_tree(cluster)
        end = time.time()
        start = time.time()
        feature.append(f_depth_dep_tree)
        f_punctuation = FeatureGeneration().punctuation(cluster)
        end = time.time()
        start = time.time()
        feature.append(f_punctuation)
        f_num_pos = FeatureGeneration().num_3pos_tags(cluster)
        feature.append(f_num_pos)
        end = time.time()

        matrix_N = []
        for f in feature:
            if f.size > 0:
                self.scaler.fit_transform(f)
                N = euclidean_distances(f, f)
                for i in range(N.shape[0]):
                    for j in range(N.shape[1]):
                        N[i][j] = 1 / (1 + N[i][j])
                if len(matrix_N) > 0:
                    matrix_N = matrix_N + (1 / len(feature) * N)
                else:
                    matrix_N = ((1 / len(feature)) * N)

        matrix_N = self.normalize_by_rowsum(np.array(matrix_N))
        # matrix_N = add_tp_ratio(matrix_N, cluster, 0.15)
        # matrix = add_tp_ratio(matrix_N, cluster, 0.15)
        matrix_N = np.array(matrix_N) * (1 - self.d) + matrix * self.d
        p = self.power_method(matrix_N, 0.00000001)
        x = 0
        for i in range(len(cluster)):
            if not cluster[i].score:
                score_exists = False
            else:
                score_exists = True
            for j in range(len(cluster[i].sentences)):
                if score_exists:
                    cluster[i].score[j] += (p[x][0]) * weight
                else:
                    cluster[i].score.append(p[x][0] * weight)
                x += 1

    def create_class_file(self, arg, name):
        """
        Precompute a file containing lists of predicted classes for all arguments belonging to the clusters of arg.

        :param arg: list of arguments
        :param name: file name
        :return:
        """
        classes = {}
        clusters = []
        for a in arg:
            clusters.append((a.cluster_map[a.context["sourceId"]]))
        for cluster in clusters:
            for arg in cluster:
                feature = FeatureGeneration().stack_all(self.tfidf_vec, self.count_vec, [arg])
                if arg.id not in classes:
                    classes[arg.id] = []
                    for idx, sentence in enumerate(arg):
                        pred = self.arg_model.predict(feature[idx].reshape(1, -1))
                        classes[arg.id].append(int(pred[0]))
        with open(name, 'w') as f:
            json.dump(classes, f)

    def add_tp_ratio(self, cluster):
        """
        Create numpy array with aij = argumentativeness of sentence j


        :param cluster: cluster of arguments
        :return: (numpy array) teleportation marix
        """
        start = time.time()
        message_embedding = []
        row = []

        for argument_j in cluster:
            # feature = feature_generation().stack_all(tfidf_vec, count_vec, [argument_j], )
            for idx, sentence_j in enumerate(argument_j):
                value = 1.0
                for marker in self.discourse_markers:
                    if marker in sentence_j.lower():
                        value += 1
                if any(claim_ind in sentence_j.lower() for claim_ind in self.claim_markers):
                    value += 1
                # pred = self.class_mapping[argument_j.id][idx]
                # if pred[0] == 0:
                #    print(sentence_j)'
                # value += self.classes[pred]
                row.append(value)

        message_embedding = []
        for argument in cluster:
            for sentence in argument:
                message_embedding.append(row)

        message_embedding = np.array(message_embedding)
        message_embedding = self.normalize_by_rowsum(message_embedding)
        # matrix = np.array(matrix) * (1 - d) + np.array(message_embedding) * d
        end = time.time()
        return np.array(message_embedding)

    def sem_similarty_scoring(self, clusters):
        """
        Run biased PageRank using Universal Sentence Encoder to receive embedding.
        Calls add add_tp_ratio() and add_syn_similarity().
        Computes similarity to conclusion.

        :param clusters:
        :return:
        """
        tf.compat.v1.reset_default_graph()
        start = time.time()
        messages = []

       
        similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
        similarity_message_encodings = self.embed(similarity_input_placeholder)
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())
            for idx, cluster in enumerate(clusters):
                messages = []
                for argument in cluster:
                    messages = messages + argument.sentences

                message_embedding = self.run_and_plot(session, similarity_input_placeholder, messages,
                                                      similarity_message_encodings)

                sim = np.inner(message_embedding, message_embedding)
                sim_message = self.normalize_by_rowsum(sim)
                matrix = self.add_tp_ratio(cluster)
                if (self.WEIGHT_SYN != 0):
                    self.add_syn_similarity(matrix, cluster, self.WEIGHT_SYN)
                M = np.array(sim_message) * (1 - self.d) + np.array(matrix) * self.d
                p = self.power_method(M, 0.0000001)
                end = time.time()
                x = 0
                for i in range(len(cluster)):
                    if cluster[i].conclusion is not None:

                        conclusion = self.run_and_plot(session, similarity_input_placeholder, [cluster[i].conclusion],
                                                       similarity_message_encodings)
                    if not cluster[i].score:
                        score_exists = False
                    else:
                        score_exists = True
                    for j in range(len(cluster[i].sentences)):
                        if score_exists:
                            cluster[i].score[j] += (p[x][0] * (1 - self.WEIGHT_SYN))
                            cluster[i].score[j] = cluster[i].score[j] / 2

                        else:
                            cluster[i].score.append(p[x][0] * (1 - self.WEIGHT_SYN))
                        x += 1
                    if (len(cluster[i].score) > 1):
                        cluster[i].score = list(
                            (cluster[i].score - min(cluster[i].score)) / (
                                    max(cluster[i].score) - min(cluster[i].score)))
                    else:
                        cluster[i].score = [1]
                    for j in range(len(cluster[i].sentences)):
                        if  cluster[i].conclusion is not None:
                            c_sim_score = self.c_simw * np.inner(conclusion,
                                                             self.run_and_plot(session, similarity_input_placeholder,
                                                                               [cluster[i].sentences[j]],
                                                                               similarity_message_encodings))
                            cluster[i].score[j] += c_sim_score[0][0]
                # self.add_syn_similarity(matrix, cluster, self.WEIGHT_SYN)

    def slice_it(self, li, cols):
        start = 0
        for i in range(cols):
            stop = start + len(li[i::cols])
            yield li[start:stop]
            start = stop

    # start = time.time()
    # cluster_map = {}
    # for filename in os.listdir("args_me"):
    #     with open("args_me/" + filename, "r", encoding="utf8") as debate_file:
    #         data = json.load(debate_file)
    #         argumentList = list(map(Argument.from_json, data["arguments"], repeat(cluster_map) ))
    # end = time.time()

    # messages = ["Well, \"gun control\" is such a broad topic, but I'm going to talk about universal background checks mainly.","Although gun bans are unconstitutional, the amendment does allow for regulation within reason.", "The United States leads in per-capita gun ownership worldwide" ]
    # cluster = create_test_cluster("test_doc")
    # messages = []
    # args_sentences = []
    # for argument in cluster:
    #   for sentence in argument:
    #       print(sentence)

    # cluster_map = {}
    # for filename in os.listdir("args_me"):
    #     with open("args_me/" + filename, "r", encoding="utf-8") as args_me:
    #         data = json.load(args_me)
    #         argumentList = list(map(Argument.from_json, data["arguments"], repeat(cluster_map)))
    #         pickle.dump(argumentList, open("argumentList.p", "wb")

    def run_argsRank(self, arg=None, cluster=True, dev_mapping=False, pWEIGHT_SYN=0, pc_simw=0, pc_0=2, pc_1=2, pc_2=1,
                     pc_3=1, pd=0.3):
        """
        Generate snippets for all arguments belonging to the cluster of arg.

        :param arg: list of arguments
        :param list:
        :param dev_mapping:
        :param pWEIGHT_SYN: weight given to syntactic similarity
        :param pc_simw: weight given to similarity of conclustion
        :param pc_0: value given to class 0 in teleportation ratio
        :param pc_1: value given to class 1 in teleportation ratio
        :param pc_2: value given to class 2 in teleportation ratio
        :param pc_3: value given to class 3 in teleportation ratio
        :param pd: damping factor used for PageRank
        :return: returns list of snippets (2 sentence length)
        """
        # global WEIGHT_SYN
        # global c_simw
        # global classes
        # global class_mapping
        # global d

        self.WEIGHT_SYN = pWEIGHT_SYN
        self.c_simw = pc_simw

        self.classes[0] = pc_0
        self.classes[1] = pc_1
        self.classes[2] = pc_2
        self.classes[3] = pc_3
        self.d = pd

        # if dev_mapping is False:
        #     with open("data/class_mapping_test", 'r') as f:
        #         self.class_mapping = json.load(f)
        # else:
        #     with open("data/class_mapping_dev", 'r') as f:
        #         self.class_mapping = json.load(f)

        #print("Args Rank")
        # if arg == None:
        #     argumentList = pickle.load(open("argumentList.p", "rb"))
        #     cluster_map = argumentList[0].cluster_map
        #     print("Length:")
        #     print(len(cluster_map))
        #
        #     with open("indices.json", "r") as indices_file:
        #         data = json.load(indices_file)
        #         data = sorted(data, key=str)
        #
        #     with open("args_rank_result.txt", "a+", encoding="utf-8", ) as summaries:
        #         # add_syn_similarity(argumentList, WEIGHT_SYN)
        #         n = 0
        #         clusters = []
        #         for index in data:
        #             clusters.append((cluster_map[argumentList[index].context["sourceId"]]))
        #
        #         self.sem_similarty_scoring(clusters)
        #         for index in data:
        #             self.add_syn_similarity(cluster_map[argumentList[index].context["sourceId"]], WEIGHT_SYN)
        #             summaries.write(" ".join(argumentList[index].get_topK(3)))
        #             summaries.write("\n")
        #             n += 1
        #             print("Round %d", n)

        if cluster == True:
            snippet = []
            clusters = []
            print(len(arg))
            for a in arg:
                clusters.append((a.cluster_map[a.context["sourceId"]]))

            start = time.time()
            self.sem_similarty_scoring(clusters)
            end = time.time()
            print("Scoring")
            print(end - start)
            # create_class_file(clusters)
            #  '   print(a.id)
            # pool_size = 5
            # clusters_list = []
            #
            # for c in slice_it(clusters, 5):
            #     clusters_list.append(c)
            # # size = 0
            # # if len(clusters) < pool_size:
            # #     size = len(clusters)
            # #     for i in range(len(clusters)):
            # #         clusters_list.append([])
            # #     for idx, cluster in enumerate(clusters):
            # #         clusters_list[idx % size].append(cluster)
            # # else:
            # #     size = pool_size
            # #     for i in range(pool_size):
            # #         clusters_list.append([])
            # #     for idx, cluster in enumerate(clusters):
            # #         clusters_list[idx % size].append(cluster)
            #
            # pool = Pool(pool_size)
            #
            #
            # #     clusters_list.append(c)
            # for clusters in clusters_list:
            #     for cluster in clusters:
            #         for arg in cluster:
            #             print(arg.score)
            #             print(arg.id)
            # pool.map(sem_similarty_scoring, clusters_list)
            #
            # for clusters in clusters_list:
            #     for cluster in clusters:
            #         for arg in cluster:
            #             print(arg.score)
            #             print(arg.id)
            # # #sem_similarty_scoring(clusters)
            # # for a in arg:
            # #     add_syn_similarity(a.cluster_map[a.context["sourceId"]], WEIGHT_SYN)'
            for a in arg:
                snippet.append(a.get_topK(2))

            return snippet
        else:
            snippets = []

            self.sem_similarty_scoring([arg])
            for a in arg:
                snippets.append(a.get_topK(2).tolist())
            indices = []
            for idx, snippet in enumerate(snippets):
                index = []
                for sentence in snippet:
                    p = re.compile(sentence)
                    m = p.search(arg[idx].premises[0]["text"])
                    index.append(list(m.span()))
                indices.append(index)

            return indices

    def dynamic_snippet_scoring(self, arg, query, weight):
        spacy_nlp = spacy.load('en_core_web_sm')
        query_doc = spacy_nlp(query)
        tokens_query = [token.text for token in query_doc]
        for idx, s in enumerate(arg.sentences):
            doc = spacy_nlp(s)
            tokens_arg = [token.text for token in doc]
            for t in query_doc:
                if t in tokens_arg:
                    arg.score[idx] += weight


    def generate_snippet(self, data, arg):
        self.WEIGHT_SYN = 0
        self.c_simw = 0

        self.classes[0] = 0
        self.classes[1] = 0
        self.classes[2] = 0
        self.classes[3] = 0
        self.d = 0.15

        indices = []

        if len(sys.argv) > 1:
            if sys.argv[1] == "regenerate":
                snippets = []
                self.sem_similarty_scoring([arg])
                for a in arg:
                    snippets.append(a.get_topK(2).tolist())

                for idx, snippet in enumerate(snippets):
                    index = []
                    for sentence in snippet:
                        p = re.compile(sentence)
                        m = p.search(arg[idx].premises[0]["text"])
                        index.append(list(m.span()))
                    data[arg[idx].id] = index
                    indices.append(index)
        else:
            for a in arg:
                if a.id in data:
                    indices.append(data[a.id])
                else:
                    indices.append(None)

            if None in indices:
                self.sem_similarty_scoring([arg])
                for idx in range(len(indices)):
                    if indices[idx] is None:
                        print(idx)
                        snippet = arg[idx].get_topK(2).tolist()
                        new_index = []
                        for sentence in snippet:
                            p = re.compile(sentence)
                            m = p.search(arg[idx].premises[0]["text"])
                            new_index.append(list(m.span()))
                        data[arg[idx].id] = new_index
                        indices[idx] = new_index
        return indices

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)


    argsRank = ArgsRank()
    with open(os.path.join(script_dir, "../src/snippets.txt")) as json_file:
        data = json.load(json_file)
        while True:
            input_data = sys.stdin.readline()
            if input_data == "store":
                with open(os.path.join(script_dir, "../src/snippets.txt")) as json_file:
                    json.dump(data, json_file)

            #input_data = "[{\"id\":\"0191\",\"text\":\"Abortion is wrong! Abortion Is gross! Abortion is MURDER!!!!\"},{\"id\":\"02391\",\"text\":\"This is a test. Let's look if this works.\"}]"
            json_input = json.loads(input_data)
            #print(json_input)
            cluster = []
            for argument in json_input:
                arg = Argument()
                arg.premises= [{"text": argument["text"]}]
                arg.id = argument["id"]
                arg.set_sentences(argument["text"])
                cluster.append(arg)

            snippets = argsRank.generate_snippet(data, cluster)
            # return the snippets
            #print(snippets)
            sys.stdout.write(json.dumps(snippets))
            sys.stdout.write('\n')
            sys.stdout.flush()



