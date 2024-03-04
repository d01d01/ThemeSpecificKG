import hashlib
import json
import math
import os
import pickle
import sqlite3

import gensim
import numpy as np
import regex
import tensorflow as tf

from bilm import dump_bilm_embeddings, dump_bilm_embeddings_inner, initialize_sess
from flask import g
from scipy.spatial.distance import cosine


class ElmoProcessor:

    RANKED_RETURN_NUM = 20

    def __init__(self, allow_tensorflow):
        self.datadir = os.path.join('bilm-tf', 'model')
        self.vocab_file = os.path.join(self.datadir, 'vocab_test.txt')
        self.options_file = os.path.join(self.datadir, 'elmo_2x4096_512_2048cnn_2xhighway_options.json')
        self.weight_file = os.path.join(self.datadir, 'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5')
        with open('data/sent_example.pickle', 'rb') as handle:
            self.sent_example_map = pickle.load(handle)
        self.target_embedding_map = {}
        self.wikilinks_embedding_map = {}
        self.target_output_embedding_map = {}
        self.wikilinks_output_embedding_map = {}
        self.allow_tensorflow = allow_tensorflow
        self.stop_sign = "STOP_SIGN_SIGNAL"
        self.batcher, self.ids_placeholder, self.ops, self.sess = initialize_sess(self.vocab_file, self.options_file, self.weight_file)
        self.db_loaded = False
        self.server_mode = False
        self.word2vec = None

    def load_sqlite_db(self, path, server_mode=False):
        if not os.path.isfile(path):
            return False
        self.db_conn = sqlite3.connect(path)
        self.db_path = path
        self.server_mode = server_mode
        self.db_loaded = True
        return True

    def query_sqlite_db(self, candidates):
        if not self.db_loaded:
            return {}
        if self.server_mode:
            db = getattr(g, '_database', None)
            if db is None:
                db = g._database = sqlite3.connect(self.db_path)
            cursor = db.cursor()
        else:
            cursor = self.db_conn.cursor()
        ret_map = {}
        for candidate in candidates:
            cursor.execute("SELECT value FROM data WHERE title=?", [candidate])
            result = cursor.fetchone()
            if result is not None:
                result_str = result[0]
                assert(result_str[0] == '[')
                assert(result_str[-1] == ']')
                result_str = result_str[1:-1]
                result_arr = [float(x) for x in result_str.split(",")]
                ret_map[candidate] = result_arr
        return ret_map

    def process_batch_continuous(self, sentences):
        tokenized_context = [sentence.strip().split() for sentence in sentences]
        freq_map = {}

        for i in range(0, len(tokenized_context)):
            for j in range(0, len(tokenized_context[i])):
                key = tokenized_context[i][j]
                if key in freq_map:
                    freq_map[key] = freq_map[key] + 1.0
                else:
                    freq_map[key] = 1.0

        ret_map = {}
        for sent_id in range(0, len(sentences)):
            sentence = ' '.join(tokenized_context[sent_id])
            tokens = sentence.strip().split()
            char_ids = self.batcher.batch_sentences([tokens])
            sent_embedding = self.sess.run(
                self.ops['lm_embeddings'], feed_dict={self.ids_placeholder: char_ids}
            )[0]
            for i in range(0, len(tokenized_context[sent_id])):
                key = tokenized_context[sent_id][i]
                concat = np.concatenate([
                    sent_embedding[0][i],
                    sent_embedding[1][i],
                    sent_embedding[2][i]
                ])
                if key in ret_map:
                    ret_map[key] = ret_map[key] + concat
                else:
                    ret_map[key] = concat
                assert(len(ret_map[key]) == 3 * 1024)
        ret_map_avg = {}
        for key in ret_map:
            dividend = freq_map[key]
            ret_map_avg[key] = list(ret_map[key] / dividend)
        tf.reset_default_graph()
        return ret_map_avg

    """
    @sentences: A list of string
    """
    def process_batch(self, sentences):
        tokenized_context = [sentence.strip().split() for sentence in sentences]
        freq_map = {}

        for i in range(0, len(tokenized_context)):
            for j in range(0, len(tokenized_context[i])):
                key = tokenized_context[i][j]
                if key in freq_map:
                    freq_map[key] = freq_map[key] + 1.0
                else:
                    freq_map[key] = 1.0

        embedding_map = dump_bilm_embeddings(
            self.vocab_file, sentences, self.options_file, self.weight_file
        )

        ret_map = {}
        for sent_id in range(0, len(sentences)):
            sent_embedding = embedding_map[sent_id]
            for i in range(0, len(tokenized_context[sent_id])):
                key = tokenized_context[sent_id][i]
                concat = np.concatenate([
                    sent_embedding[0][i],
                    sent_embedding[1][i],
                    sent_embedding[2][i]
                ])
                if key in ret_map:
                    ret_map[key] = ret_map[key] + concat
                else:
                    ret_map[key] = concat
                assert(len(ret_map[key]) == 3 * 1024)
        ret_map_avg = {}
        for key in ret_map:
            dividend = freq_map[key]
            ret_map_avg[key] = list(ret_map[key] / dividend)
        tf.reset_default_graph()
        return ret_map_avg

    def process_single_continuous(self, sentence):
        tokens = sentence.strip().split()
        char_ids = self.batcher.batch_sentences([tokens])
        embedding = self.sess.run(
            self.ops['lm_embeddings'], feed_dict={self.ids_placeholder: char_ids}
        )[0]
        ret_map = {}
        for i in range(0, len(tokens)):
            ret_map[tokens[i]] = list(embedding[0][i]) + list(embedding[1][i]) + list(embedding[2][i])
            assert(len(ret_map[tokens[i]]) == 3 * 1024)
        return ret_map

    """
    @sentences: A string
    """
    def process_single(self, sentence):
        tokens = sentence.split()
        embedding = dump_bilm_embeddings_inner(
            self.vocab_file, sentence, self.options_file, self.weight_file
        )
        assert(len(embedding[0]) == len(tokens))
        ret_map = {}
        for i in range(0, len(tokens)):
            ret_map[tokens[i]] = list(embedding[0][i]) + list(embedding[1][i]) + list(embedding[2][i])
            assert(len(ret_map[tokens[i]]) == 3 * 1024)
        tf.reset_default_graph()
        return ret_map

    """
    @vec_a, vec_b: A list of numbers
    """
    @staticmethod
    def cosine_helper(vec_a, vec_b):
        vec_a_np = np.array(vec_a)
        vec_b_np = np.array(vec_b)
        return 1.0 - cosine(vec_a_np, vec_b_np)

    """
    Helper function that loads pre-computed ELMo representations to save time.
    @target_file_name: A pickle file that caches test-corpus
    @wikilinks_file_name: A pickle file that caches Wikilinks sentences generated from the test-corpus
    """
    def load_cached_embeddings(self, target_file_name, wikilinks_file_name):
        with open(target_file_name, "rb") as handle:
            self.target_embedding_map = pickle.load(handle)
        with open(wikilinks_file_name, "rb") as handle:
            self.wikilinks_embedding_map = pickle.load(handle)

    """
    @sentence: A zoe_utils.Sentence
    @candidates: A list of (string, float) pair
    @return: A list of (string, float) pair of title to ELMo scores
    """
    def rank_candidates(self, sentence, candidates):
        candidates = [x[0] for x in candidates]
        target_vec = []
        if sentence.get_mention_surface() not in self.target_embedding_map and self.allow_tensorflow:
            target_additional_map = self.process_single_continuous(sentence.get_sent_str())
            if sentence.get_mention_surface() in target_additional_map:
                target_vec = target_additional_map[sentence.get_mention_surface()]
        if sentence.get_mention_surface() in self.target_embedding_map:
            target_vec = self.target_embedding_map[sentence.get_mention_surface()]
        sentences_to_process = []
        if self.db_loaded:
            wikilinks_embedding_map = self.query_sqlite_db(candidates)
        else:
            wikilinks_embedding_map = self.wikilinks_embedding_map
        for candidate in candidates:
            if candidate not in self.sent_example_map:
                continue
            if candidate in wikilinks_embedding_map:
                continue
            example_sentences_str = self.sent_example_map[candidate]
            example_sentences = example_sentences_str.split("|||")
            for i in range(0, min(len(example_sentences), 10)):
                sentences_to_process.append(example_sentences[i])
        wikilinks_additional_map = {}
        if len(sentences_to_process) > 0 and self.allow_tensorflow:
            wikilinks_additional_map = self.process_batch_continuous(sentences_to_process)
        if len(target_vec) == 0:
            return [(self.stop_sign, 0.0)]
        self.target_output_embedding_map[sentence.get_mention_surface()] = target_vec
        results = {}
        for candidate in candidates:
            wikilinks_vec = []
            if candidate in wikilinks_embedding_map:
                wikilinks_vec = wikilinks_embedding_map[candidate]
            if candidate in wikilinks_additional_map:
                wikilinks_vec = wikilinks_additional_map[candidate]
            self.wikilinks_output_embedding_map[candidate] = wikilinks_vec
            if len(wikilinks_vec) > 0:
                results[candidate] = ElmoProcessor.cosine_helper(target_vec, wikilinks_vec)
            else:
                results[candidate] = 0.0
        sorted_results = sorted(results.items(), key=lambda kv: kv[1], reverse=True)
        return [(x[0], x[1]) for x in sorted_results][:self.RANKED_RETURN_NUM]

    def word2vec_helper(self, input):
        vec = np.zeros(300)
        if self.word2vec is None:
            return None
        if input in self.word2vec:
            return self.word2vec[input]
        if input.lower() in self.word2vec:
            return self.word2vec[input.lower()]
        count = 0.0
        for token in input.split("_"):
            if token in self.word2vec:
                vec += self.word2vec[token]
                count += 1.0
            elif token.lower() in self.word2vec:
                vec += self.word2vec[token.lower()]
                count += 1.0
        if count == 0.0:
            return None
        return vec / count

    def rank_candidates_vec(self, sentence=None, candidates=None):
        data_path = "data/word2vec/GoogleNews-vectors-negative300.bin"
        if not os.path.isfile(data_path):
            return candidates
        if self.word2vec is None:
            print("self.word2vec is None")
            self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(data_path, binary=True)
        if sentence is None:
            return None
        candidates = [x[0] for x in candidates]
        target_vec = self.word2vec_helper(sentence.get_mention_surface())
        if target_vec is None:
            print(sentence.get_mention_surface() + " not found in word2vec")
            return [(x, 0.0) for x in candidates]
        assert(len(target_vec) == 300)
        results = {}
        for candidate in candidates:
            candidate_vec = self.word2vec_helper(candidate)
            if candidate_vec is None:
                similarity = 0
            else:
                similarity = cosine(target_vec, candidate_vec)
                print(similarity)
            results[candidate] = similarity
        sorted_results = sorted(results.items(), key=lambda kv: kv[1], reverse=True)
        return [(x[0], x[1]) for x in sorted_results][:self.RANKED_RETURN_NUM]

    """
    To save the cache maps generated by the processor instance 
    """
    def save_cached_maps(self, target_file_name, wikilinks_file_name):
        max_bytes = 2 ** 31 - 1
        with open(target_file_name, 'wb') as handle:
            pickle.dump(self.target_output_embedding_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
        bytes_out = pickle.dumps(self.wikilinks_output_embedding_map, protocol=pickle.HIGHEST_PROTOCOL)
        with open(wikilinks_file_name, 'wb') as handle:
            for idx in range(0, len(bytes_out), max_bytes):
                handle.write(bytes_out[idx:idx + max_bytes])


class EsaProcessor:

    N_DOCUMENTS = 24504233.0
    RETURN_NUM = 300

    def __init__(self):
        with open('data/esa/esa.pickle', 'rb') as handle:
            self.esa_map = pickle.load(handle)
        with open('data/esa/freq.pickle', 'rb') as handle:
            self.freq_map = pickle.load(handle)
        with open('data/esa/invcount.pickle', 'rb') as handle:
            self.invcount_map = pickle.load(handle)

    """
    @map_val: A map representation in string.
              [key]::[val]|[key]::[val]|...
    """
    @staticmethod
    def str2map(map_val):
        ret_map = {}
        entries = map_val.split("|")
        for entry in entries:
            key = entry.split("::")[0]
            val = entry.split("::")[1]
            ret_map[key] = float(val)
        return ret_map

    """
    @sentence: A zoe_utils.Sentence
    @return: A list of (string, float) pair of title to ESA scores
    """
    def get_candidates(self, sentence):
        tokens = sentence.tokens
        overall_map = {}
        doc_freq_map = {}
        max_acc = 0
        for token in tokens:
            if token in doc_freq_map:
                acc = doc_freq_map[token] + 1
            else:
                acc = 1
            if acc > max_acc:
                max_acc = acc
            doc_freq_map[token] = acc
        for token in tokens:
            if token in self.esa_map:
                idf_score = math.log(self.N_DOCUMENTS / float(self.freq_map[token]))
                tf_score = 0.5 + 0.5 * (float(doc_freq_map[token]) / float(max_acc))
                inv_freq = float(self.invcount_map[token])
                sub_map = EsaProcessor.str2map(self.esa_map[token])
                for key in sub_map:
                    weight = idf_score * tf_score * sub_map[key] / inv_freq
                    if key in overall_map:
                        overall_map[key] = overall_map[key] + weight
                    else:
                        overall_map[key] = weight
        sorted_overall_map = sorted(overall_map.items(), key=lambda kv: kv[1], reverse=True)
        return [(x[0], x[1]) for x in sorted_overall_map][:self.RETURN_NUM]


class InferenceProcessor:

    # P(title|surface) min threshold
    PROB_TRUST_THRESHOLD = 0.5
    # The multiplier of the size of ESA candidates to ELMo candidates
    ELMO_TO_ESA_MULTIPLIER = 15.0
    # Top N candidates we trust to vote for fine types
    TRUST_CANDIDATE_SIZE = 20
    # A elmo score threshold above which a fine type will be added without voting
    MIN_ELMO_SCORE_THRESHOLD = 0.65
    # Voting threshold when title is selected via ESA
    VOTING_THRESHOLD_NORMAL = 0.8
    # Voting threshold when title is selected via P(title|surface)
    VOTING_THRESHOLD_PRIOR = 0.3

    """
    It's important to define a @mode as it defines type mappings etc.
    """
    def __init__(self, mode, do_inference=True, use_prior=True, use_context=True, resource_loader=None, custom_mapping=None):
        self.mode = mode
        self.mapping = {}
        self.do_inference = do_inference
        self.use_prior = use_prior
        self.use_context = use_context
        if custom_mapping is None:
            mapping_file_name = "mapping/" + self.mode + ".mapping"
            with open(mapping_file_name) as f:
                for line in f:
                    line = line.strip()
                    self.mapping[line.split("\t")[0]] = line.split("\t")[1]
        else:
            self.mapping = custom_mapping
        if resource_loader is None:
            with open("data/prior_prob.pickle", "rb") as handle:
                self.prior_prob_map = pickle.load(handle)
            with open("data/title2freebase.pickle", "rb") as handle:
                self.freebase_map = pickle.load(handle)
        else:
            self.prior_prob_map = resource_loader.prior_prob_map
            self.freebase_map = resource_loader.freebase_map
        self.logic_mappings = []
        if custom_mapping is None:
            logic_mapping_file_name = "mapping/" + self.mode + ".logic.mapping"
            with open(logic_mapping_file_name) as f:
                for line in f:
                    line = line.strip()
                    self.logic_mappings.append(line)

    """
    Compute a unique signature of the current inference mode
    """
    def signature(self):
        return hashlib.sha224(str(self.mapping).encode('utf-8')).hexdigest()

    """
    Process logic mappings (i.e. additional target_taxonomy to target_taxonomy mappings)
    and then returns a list of adjusted types
    """
    def get_final_types(self, current_set):
        for line in self.logic_mappings:
            line_group = line.split("\t")
            if line_group[0] == "+":
                if line_group[1] in current_set:
                    current_set.add(line_group[2])
            if line_group[0] == "-":
                if line_group[1] in current_set and line_group[2] in current_set:
                    current_set.remove(line_group[2])
                if line_group[1] in current_set and line_group[2] == "ALL_OTHER":
                    to_remove = set()
                    for t in current_set:
                        if not t.startswith(line_group[1]):
                            to_remove.add(t)
                    ret_current_set = set()
                    for t in current_set:
                        if t not in to_remove:
                            ret_current_set.add(t)
                    current_set = ret_current_set
        return current_set

    """
    @surface: A string tokenized by spaces
    """
    def get_prob_title(self, surface):
        surface = surface.lower()
        if surface in self.prior_prob_map:
            prior_prob = self.prior_prob_map[surface]
            if prior_prob[1] > self.PROB_TRUST_THRESHOLD:
                return prior_prob[0]
        return ""

    """
    Get direct mapped types from FreeBase->Target mappings
    @title: A string of title
    """
    def get_mapped_types_of_title(self, title):
        if " " in title:
            title = title.replace(" ", "_")
        if regex.match(r'\d{4}', title):
            self.freebase_map[title] = ""
        if title.lower() == title:
            concat = ""
            for token in title.split("_"):
                if len(token) == 0:
                    continue
                concat += token[0:1].upper()
                if len(token) > 1:
                    concat += token[1:]
                concat += "_"
            if len(concat) > 0:
                concat = concat[:-1]
            title = concat
        freebase_types = []
        if title in self.freebase_map:
            freebase_types = self.freebase_map[title].split(",")
        mapped_set = set()
        for t in freebase_types:
            converted_type = "/" + t.replace(".", "/")
            if converted_type in self.mapping:
                mapped_set.add(self.mapping[converted_type])
        if ("organization.non_profit_organization" in freebase_types) or ("education.academic_institution" in freebase_types):
            if "/organization/company" in mapped_set:
                mapped_set.remove("/organization/company")
        if ("/organization/educational_institution" in mapped_set) and ("/organization/company" in mapped_set):
            mapped_set.remove("/organization/company")
        if len(mapped_set) == 0 and "EMPTY" in self.mapping and title in self.freebase_map:
            mapped_set.add(self.mapping["EMPTY"])
        return mapped_set

    """
    @title: A string 
    """
    def get_coarse_types_of_title(self, title):
        fine_types = self.get_types_of_title(title)
        ret = set()
        for t in fine_types:
            ret.add("/" + t.split("/")[1])
        return ret

    """
    @title: A string
    """
    def get_types_of_title(self, title):
        mapped_set = self.get_mapped_types_of_title(title)
        mapped_set_list = list(mapped_set)
        for t in mapped_set:
            if len(t.split("/")) >= 2:
                mapped_set_list.append("/" + t.split("/")[1])
        return self.get_final_types(set(mapped_set_list))

    """
    Vote for a best coarse type via candidates' ELMo scores
    @title: A string
    @candidates: A list of string
    @type_score: A map of (string: float)
    """
    def get_voted_coarse_type_of_title(self, title, candidates, type_score):
        mapped_set = self.get_mapped_types_of_title(title)
        coarse_freq = {}
        for t in mapped_set:
            key = "/" + t.split("/")[1]
            if key not in self.get_coarse_types_of_title(title):
                continue
            if key in coarse_freq:
                coarse_freq[key] = coarse_freq[key] + 1
            else:
                coarse_freq[key] = 1
        pairs = list(coarse_freq.items())
        pairs.sort(key=lambda kv: kv[1], reverse=True)
        highest_score = pairs[0][1]
        coarse_type = pairs[0][0]
        duel_titles = set()
        for pair in pairs:
            if pair[1] == highest_score:
                duel_titles.add(pair[0])
        duel_freq_map = {}
        for candidate in candidates:
            for coarse_type_candidate in duel_titles:
                if coarse_type_candidate in self.get_coarse_types_of_title(candidate):
                    if coarse_type_candidate in duel_freq_map:
                        duel_freq_map[coarse_type_candidate] += type_score[coarse_type_candidate]
                    else:
                        duel_freq_map[coarse_type_candidate] = type_score[coarse_type_candidate]
        pairs = list(duel_freq_map.items())
        pairs.sort(key=lambda kv: kv[1], reverse=True)
        if len(pairs) > 0:
            coarse_type = pairs[0][0]
        for line in self.logic_mappings:
            line_group = line.split("\t")
            if line_group[0] == "=" and coarse_type == line_group[1] and line_group[2] in mapped_set:
                coarse_type = line_group[2]
        return coarse_type

    """
    @titles: A list of string
    """
    def compute_set_freq(self, titles):
        freq_map = {}
        for title in titles:
            title_types = self.get_types_of_title(title)
            for t in title_types:
                if t in freq_map:
                    freq_map[t] = freq_map[t] + 1
                else:
                    freq_map[t] = 1
        return freq_map

    """
    @candidates: A list of string
    @type_scores: A map of (string: float)
    """
    def select_in_order(self, candidates, type_scores):
        for candidate in candidates:
            if len(self.get_mapped_types_of_title(candidate)) == 0:
                continue
            coarse_types = self.get_coarse_types_of_title(candidate)
            for ct in coarse_types:
                if ct in type_scores and type_scores[ct] > 1.0:
                    return candidate
        return candidates[0]

    """
    Get a type's average ELMo score
    @candidates: A map of (string: float)
    """
    def get_elmo_type_scores(self, candidates):
        ret_map = {}
        ret_map_freq = {}
        for title in candidates:
            score = candidates[title]
            for t in self.get_types_of_title(title):
                if t in ret_map:
                    ret_map[t] = ret_map[t] + score
                    ret_map_freq[t] = ret_map_freq[t] + 1.0
                else:
                    ret_map[t] = score
                    ret_map_freq[t] = 1.0
        for key in ret_map:
            ret_map[key] = ret_map[key] / ret_map_freq[key]
        return ret_map

    """
    Helper function that infer types
    @selected: A string of title that is selected as best one
    @candidates: A list of (string: float) pairs
    @elmo_type_score: results from self.get_elmo_type_scores()
    @from_prior: A bool indicating whether @selected comes from P(title|surface)
    """
    def get_inferred_types(self, selected, candidates, elmo_type_score, from_prior):
        if len(self.get_mapped_types_of_title(selected)) == 0:
            return []
        candidates = [x[0] for x in candidates]
        coarse_type = self.get_voted_coarse_type_of_title(selected, candidates, elmo_type_score)
        filtered_types = set()
        filtered_types.add(coarse_type)
        for t in self.get_mapped_types_of_title(selected):
            if t.startswith(coarse_type):
                filtered_types.add(t)
        freq_map = {}
        total = 0
        trusted_candidates = set()
        trusted_candidates.add(selected)
        for candidate in candidates[:self.TRUST_CANDIDATE_SIZE]:
            trusted_candidates.add(candidate)
        for candidate in trusted_candidates:
            if coarse_type in self.get_coarse_types_of_title(candidate):
                total += 1
                for t in self.get_mapped_types_of_title(candidate):
                    if t.startswith(coarse_type):
                        if t in freq_map:
                            freq_map[t] = freq_map[t] + 1
                        else:
                            freq_map[t] = 1
        selected_types = set()
        for key in freq_map:
            if key in elmo_type_score and elmo_type_score[key] > self.MIN_ELMO_SCORE_THRESHOLD:
                selected_types.add(key)
        consider_types = [x[0] for x in freq_map.items()]
        voting_threshold = self.VOTING_THRESHOLD_NORMAL
        if from_prior:
            consider_types = filtered_types
            voting_threshold = self.VOTING_THRESHOLD_PRIOR

        selected_types.add(coarse_type)
        for t in consider_types:
            if t in freq_map:
                if float(freq_map[t]) > float(total) * voting_threshold and freq_map[t] > 1:
                    selected_types.add(t)

        to_be_removed_types = set()
        for t in selected_types:
            if len(t.split("/")) <= 2:
                continue
            for compare_type in freq_map:
                if compare_type in elmo_type_score and t in elmo_type_score:
                    if compare_type.startswith(coarse_type) and (compare_type not in selected_types) and elmo_type_score[compare_type] > elmo_type_score[t]:
                        to_be_removed_types.add(t)
        final_ret_types = set()
        for t in selected_types:
            if t not in to_be_removed_types:
                final_ret_types.add(t)
        return final_ret_types

    def get_all_possible_coarse_types(self, candidates):
        candidates = [x[0] for x in candidates]
        freq_map = {}
        for candidate in candidates:
            for ct in self.get_coarse_types_of_title(candidate):
                if ct in freq_map:
                    freq_map[ct] += 1
                else:
                    freq_map[ct] = 1
        sorted_types = sorted(freq_map.items(), key=lambda kv: kv[1], reverse=True)
        return [x[0] for x in sorted_types[:3]]

    """
    Inference utility function which make predictions and set results to the input @sentence
    @sentence: A zoe_utils.Sentence
    @elmo_candidates: A list of (title, score) pairs
    @esa_candidates: A list of (title, score) pairs
    @return: None
    """
    def inference(self, sentence, elmo_candidates, esa_candidates):
        elmo_titles = [x[0] for x in elmo_candidates]
        esa_titles = [x[0] for x in esa_candidates]
        elmo_freq = self.compute_set_freq(elmo_titles)
        esa_freq = self.compute_set_freq(esa_titles)
        type_promotion_score_map = {}
        for t in elmo_freq:
            esa_freq_t = 0.0
            if t in esa_freq:
                esa_freq_t = float(esa_freq.get(t))
            type_promotion_score_map[t] = float(elmo_freq.get(t)) * self.ELMO_TO_ESA_MULTIPLIER / esa_freq_t

        selected_title = self.select_in_order(elmo_titles, type_promotion_score_map)

        prob_title = self.get_prob_title(sentence.get_mention_surface_raw())
        if not self.use_prior:
            prob_title = ""
        from_prior = False
        if prob_title != "" and len(self.get_mapped_types_of_title(prob_title)) > 0:
            prob_title_coarse_types = self.get_coarse_types_of_title(prob_title)
            for t in prob_title_coarse_types:
                if t in type_promotion_score_map and type_promotion_score_map[t] > 1.0:
                    selected_title = prob_title
                    from_prior = True
        # Now we have the most trust-worthy title
        elmo_score_map = {}
        for (title, score) in elmo_candidates:
            elmo_score_map[title] = score
        if from_prior:
            elmo_score_map[selected_title] = 1.0
        elmo_type_score = self.get_elmo_type_scores(elmo_score_map)
        inferred_types = self.get_inferred_types(selected_title, elmo_candidates, elmo_type_score, from_prior)
        could_also_be_types = self.get_all_possible_coarse_types(elmo_candidates)
        final_types = self.get_final_types(set(inferred_types))
        if len(final_types) == 0 and "EMPTY" in self.mapping:
            final_types.add(self.mapping["EMPTY"])
        if not self.do_inference:
            final_types = self.get_types_of_title(selected_title)
        if not self.use_context:
            final_types = self.get_types_of_title(prob_title)
        # set predictions
        sentence.set_predictions(final_types)
        sentence.set_could_also_be_types(could_also_be_types)
        sentence.set_esa_candidates(esa_titles)
        sentence.set_elmo_candidates(elmo_titles)
        sentence.set_selected_candidate(selected_title)
        if from_prior:
            sentence.selected_title = "SURF-" + selected_title
        else:
            sentence.selected_title = "ELMO-" + selected_title
        sentence.set_signature(self.signature())


class Sentence:

    def __init__(self, tokens, mention_start, mention_end, gold_types=None):
        self.tokens = tokens
        self.mention_start = int(mention_start)
        self.mention_end = int(mention_end)
        self.gold_types = gold_types
        if self.gold_types is None:
            self.gold_types = []
        self.predicted_types = []
        self.could_also_be_types = []
        self.esa_candidate_titles = []
        self.elmo_candidate_titles = []
        self.selected_title = ""
        self.selected_candidate = ""
        self.inference_signature = ""

    """
    @returns: A string tokenized by "_"
    """
    def get_mention_surface(self):
        concat = ""
        for i in range(self.mention_start, self.mention_end):
            concat += self.tokens[i] + "_"
        if len(concat) > 0:
            concat = concat[:-1]
        return concat

    """
    @returns: A string tokenized by " "
    """
    def get_mention_surface_raw(self):
        return self.get_mention_surface().replace("_", " ")

    def get_sent_str(self):
        concat = ""
        i = 0
        while i < len(self.tokens):
            if i == self.mention_start:
                concat += self.get_mention_surface()
                i = self.mention_end - 1
            else:
                concat += self.tokens[i]
            i += 1
            concat += " "
        if len(concat) > 0:
            concat = concat[:-1]
        return concat

    def set_predictions(self, predicted_types):
        self.predicted_types = predicted_types

    def set_could_also_be_types(self, could_also_be_types):
        self.could_also_be_types = list(set(could_also_be_types) - set(self.predicted_types))

    def set_esa_candidates(self, esa_candidate_titles):
        self.esa_candidate_titles = esa_candidate_titles

    def set_elmo_candidates(self, elmo_candidate_titles):
        self.elmo_candidate_titles = elmo_candidate_titles

    def set_selected_candidate(self, selected):
        self.selected_candidate = selected

    def set_signature(self, signature):
        self.inference_signature = signature

    def print_self(self):
        print(self.get_sent_str())
        print(self.get_mention_surface())
        print("Gold\t: " + str(self.gold_types))
        print("Predicted\t" + str(self.predicted_types))
        print("ESA Candidate Titles: " + str(self.esa_candidate_titles))
        print("ELMo Candidate Titles: " + str(self.elmo_candidate_titles))
        print("Selected Candidate: " + str(self.selected_candidate))


class Evaluator:

    def __init__(self):
        self.sentences = []
        self.total_gold_types = 0
        self.total_predicted_types = 0
        self.total_matches = 0
        self.total_macro_precision = 0.0
        self.total_macro_recall = 0.0
        self.perfect_match = 0

    @staticmethod
    def compute_matches(set_a, set_b):
        count = 0
        for item in set_a:
            if item in set_b:
                count += 1
        return count

    @staticmethod
    def get_if_perfect_match(set_a, set_b):
        if len(set_a) == len(set_b):
            for item in set_a:
                if item not in set_b:
                    return False
            return True
        return False

    @staticmethod
    def compute_f1(precision, recall):
        if precision + recall == 0.0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def print_performance(self, sentences):
        self.sentences = sentences
        for sentence in self.sentences:
            if len(sentence.gold_types) == 0:
                print("[ERROR]: encountered examples without correct answer.")
                return
            matches = self.compute_matches(sentence.gold_types, sentence.predicted_types)
            self.total_matches += matches
            self.total_gold_types += len(sentence.gold_types)
            self.total_predicted_types += len(sentence.predicted_types)
            if len(sentence.predicted_types) > 0:
                self.total_macro_precision += float(matches) / float(len(sentence.predicted_types))
            if len(sentence.gold_types) > 0:
                self.total_macro_recall += float(matches) / float(len(sentence.gold_types))
            if self.get_if_perfect_match(sentence.gold_types, sentence.predicted_types):
                self.perfect_match += 1
        strict_accuracy = 0.0
        if len(self.sentences) > 0:
            strict_accuracy = float(self.perfect_match) / float(len(self.sentences))

        micro_precision = 0.0
        if self.total_predicted_types > 0.0:
            micro_precision = float(self.total_matches) / float(self.total_predicted_types)
        micro_recall = 0.0
        if self.total_gold_types > 0.0:
            micro_recall = float(self.total_matches) / float(self.total_gold_types)
        micro_f1 = self.compute_f1(micro_precision, micro_recall)

        macro_precision = 0.0
        macro_recall = 0.0
        if len(self.sentences) > 0:
            macro_precision = float(self.total_macro_precision) / float(len(self.sentences))
            macro_recall = float(self.total_macro_recall) / float(len(self.sentences))
        macro_f1 = self.compute_f1(macro_precision, macro_recall)

        print("=========Performance==========")
        print("Strict Accuracy:\t" + str(strict_accuracy))
        print("---------------")
        print("Micro Precision:\t" + str(micro_precision))
        print("Micro Recall:\t" + str(micro_recall))
        print("Micro F1:\t" + str(micro_f1))
        print("---------------")
        print("Macro Precision:\t" + str(macro_precision))
        print("Macro Recall:\t" + str(macro_recall))
        print("Macro F1:\t" + str(macro_f1))
        print("==============================")


class DataReader:

    def __init__(self, data_file_name, size=-1, unique=False):
        self.sentences = []
        self.unique = unique
        if not os.path.isfile(data_file_name):
            print("[ERROR] No sentences read.")
            return
        with open(data_file_name) as f:
            for line in f:
                line = line.strip()
                data = json.loads(line)
                tokens = data['tokens']
                mentions = data['mentions']
                for mention in mentions:
                    self.sentences.append(Sentence(tokens, mention['start'], mention['end'], mention['labels']))
                    if self.unique:
                        break
        if size > 0:
            self.sentences = self.sentences[:size]
