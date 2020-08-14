import json
import re
import sys
import os

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from discreteMarkovChain import markovChain


from flask import current_app, g


class ArgsRank:

    def __init__(self, flask_app):
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



        self.d = 0.15
        self.flask_app = flask_app
        self.scaler = MinMaxScaler()


        # Create graph and finalize (optional but recommended).

        # g = tf.Graph()
        # with g.as_default():
        #   self.text_input = tf.placeholder(dtype=tf.string, shape=[None])
        #   embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
        #   self.embed_result = embed(self.text_input)
        #   init_op   = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
        # g.finalize()

        # self.tf_session = tf.Session(graph=g)
        # self.tf_session.run(init_op)
        self.embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


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
            
            if np.isnan(p_t).any():
                break

            residual = np.linalg.norm(p_t - p_prev)

            if residual < epsilon:
                break
        return p_t



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



    def add_tp_ratio(self, cluster):
        """
        Create numpy array with aij = argumentativeness of sentence j


        :param cluster: cluster of arguments
        :return: (numpy array) teleportation marix
        """

        row = []

        for argument_j in cluster:
            for idx, sentence_j in enumerate(argument_j):
                value = 1.0
                for marker in self.discourse_markers:
                    if marker in sentence_j.lower():
                        value += 1
                if any(claim_ind in sentence_j.lower() for claim_ind in self.claim_markers):
                    value += 1

                row.append(value)

        message_embedding = []
        for argument in cluster:
            for sentence in argument:
                message_embedding.append(row)

        message_embedding = np.array(message_embedding)
        message_embedding = self.normalize_by_rowsum(message_embedding)
        return np.array(message_embedding)

    def sem_similarty_scoring(self, clusters):
        """
        Run biased PageRank using Universal Sentence Encoder to receive embedding.
        Calls add add_tp_ratio() and add_syn_similarity().
        Computes similarity to conclusion.

        :param clusters:
        :return:
        """
        messages = []

        for idx, cluster in enumerate(clusters):
            messages = []
            for argument in cluster:
                messages = messages + argument.sentences

            message_embedding = [ message.numpy() for message in self.embed(messages)] #self.tf_session.run(self.embed_result, feed_dict={self.text_input: messages})

            sim = np.inner(message_embedding, message_embedding)
            sim_message = self.normalize_by_rowsum(sim)
            matrix = self.add_tp_ratio(cluster)
            M = np.array(sim_message) * (1 - self.d) + np.array(matrix) * self.d
            
            #p = self.power_method(M, 0.0000001)
            mc = markovChain(M)
            mc.computePi('power')
            p  = mc.pi

            x = 0
            for i in range(len(cluster)):
                if not cluster[i].score:
                    score_exists = False
                else:
                    score_exists = True
                for j in range(len(cluster[i].sentences)):
                    if score_exists:
                        cluster[i].score[j] += p[x]
                        cluster[i].score[j] = cluster[i].score[j]

                    else:
                        cluster[i].score.append(p[x])
                    x += 1
                if (len(cluster[i].score) > 1):
                    cluster[i].score = list(
                        (cluster[i].score - min(cluster[i].score)) / (
                                max(cluster[i].score) - min(cluster[i].score)))
                else:
                    cluster[i].score = [1]



    def find_span(self, arg_txt, sent_txt):
        try:
            p = re.compile(re.escape(sent_txt))
            m = p.search(arg_txt)
            return list(m.span())
        except Exception as e:
            self.flask_app.logger.info('Failed in matching sentence in argument..')
            self.flask_app.logger.error(str(e))
            return None


    def generate_snippet(self, args):

        self.d = 0.15

        output = []

        self.sem_similarty_scoring([args])
        for arg in args:
            arg_text = arg.premises[0]["text"]
            arg_snippet = {}

            #processing snippet title
            snippet_title = arg.get_topK(1).tolist()[0]
            snippet_title_span = self.find_span(arg_text, snippet_title)
            arg_snippet['title'] = {'span': snippet_title_span, 'text': snippet_title}


            #processing snippet body
            snippet_body_sentences  = arg.get_topK(2).tolist()

            snippet_body = []
            for sentence in snippet_body_sentences:
                try:
                    sentence_span = self.find_span(arg_text, sentence)
                    snippet_body.append({'span': sentence_span, 'text': sentence})
                except Exception as e:
                    self.flask_app.logger.error(str(e))

            arg_snippet['body'] = snippet_body


            output.append(arg_snippet)


        return output

def init_snippet_gen_app():
    g.snippet_gen_app = ArgsRank()


def get_stored_snippets():
    script_dir = os.path.dirname(__file__)
    stored_snippets = json.load(open(os.path.join(script_dir, "../data/snippets.txt")))
    return stored_snippets

def get_snippet_gen_app():
    if 'snippet_gen_app' not in g:
        g.snippet_gen_app = ArgsRank()

    return g.snippet_gen_app