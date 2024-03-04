import requests
import json
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
import Levenshtein
import re

import spacy  # version 3.5
import os
import pickle
import sys

from zoe_utils import DataReader
from zoe_utils import ElmoProcessor
from zoe_utils import EsaProcessor
from zoe_utils import Evaluator
from zoe_utils import InferenceProcessor

datapath = '../dataset/EVB/'
datainput = datapath + 'evbattery_phrase.json'
output = datapath + 'evbattery_typed.json'
output_category = datapath + 'category_dict.json'
root = "Battery_(electricity)"
theme = "electrical vehicle battery"


# initialize language model
nlp = spacy.load("en_core_web_md")

# add pipeline (declared through entry_points in setup.py)
nlp.add_pipe("entityLinker", last=True)

# pre-transformer loading
from sentence_transformers import SentenceTransformer
sent_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#ZOE loading
class ZoeRunner:

    """
    @allow_tensorflow sets whether the system will do run-time ELMo processing.
                      It's set to False in experiments as ELMo results are cached,
                      but please set it to default True when running on new sentences.
    """
    def __init__(self, allow_tensorflow=True):
        self.elmo_processor = ElmoProcessor(allow_tensorflow)
        self.esa_processor = EsaProcessor()
        self.inference_processor = InferenceProcessor("figer")
        self.evaluator = Evaluator()
        self.evaluated = []

    """
    Process a single sentence
    @sentence: a sentence in zoe_utils.Sentence structure
    @return: a sentence in zoe_utils that has predicted types set
    """
    def process_sentence(self, sentence, inference_processor=None):
        esa_candidates = self.esa_processor.get_candidates(sentence)
        elmo_candidates = self.elmo_processor.rank_candidates(sentence, esa_candidates)
        if len(elmo_candidates) > 0 and elmo_candidates[0][0] == self.elmo_processor.stop_sign:
            return -1
        if inference_processor is None:
            inference_processor = self.inference_processor
        inference_processor.inference(sentence, elmo_candidates, esa_candidates)
        return sentence

    def process_sentence_vec(self, sentence, inference_processor=None):
        esa_candidates = self.esa_processor.get_candidates(sentence)
        elmo_candidates = self.elmo_processor.rank_candidates_vec(sentence, esa_candidates)
        if len(elmo_candidates) > 0 and elmo_candidates[0][0] == self.elmo_processor.stop_sign:
            return -1
        if inference_processor is None:
            inference_processor = self.inference_processor
        inference_processor.inference(sentence, elmo_candidates, esa_candidates)
        return sentence

    """
    Helper function to evaluate on a dataset that has multiple sentences
    @file_name: A string indicating the data file. 
                Note the format needs to be the common json format, see examples
    @mode: A string indicating the mode. This adjusts the inference mode, and set caches etc.
    @return: None
    """
    def evaluate_dataset(self, file_name, mode, do_inference=True, use_prior=True, use_context=True, size=-1):
        if not os.path.isfile(file_name):
            print("[ERROR] Invalid input data file.")
            return
        self.inference_processor = InferenceProcessor(mode, do_inference, use_prior, use_context)
        dataset = DataReader(file_name, size)
        for sentence in dataset.sentences:
            processed = self.process_sentence(sentence)
            if processed == -1:
                continue
            self.evaluated.append(processed)
            processed.print_self()
            evaluator = Evaluator()
            evaluator.print_performance(self.evaluated)

    """
    Helper function that saves the predicted sentences list to a file.
    @file_name: A string indicating the target file path. 
                Note it will override the content
    @return: None
    """
    def save(self, file_name):
        with open(file_name, "wb") as handle:
            pickle.dump(self.evaluated, handle, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def evaluate_saved_runlog(log_name):
        with open(log_name, "rb") as handle:
            sentences = pickle.load(handle)
        evaluator = Evaluator()
        evaluator.print_performance(sentences)
        
class Sentence:

    def __init__(self, tokens,tokens_lemma, mention_start, mention_end,senid,fileid):
        self.tokens = tokens
        self.tokens_lemma = tokens_lemma
        self.mention_start = int(mention_start)
        self.mention_end = int(mention_end)
        self.gold_types = ''
        self.predicted_types =''
        self.could_also_be_types = []
        self.esa_candidate_titles = []
        self.elmo_candidate_titles = []
        self.selected_title = ""
        self.selected_candidate = ""
        self.inference_signature = ""
        self.senid = senid
        self.fileid = fileid

    """
    @returns: A string tokenized by "_"
    """
    def get_mention_surface(self):
        concat = ""
        for i in range(self.mention_start, self.mention_end):
            concat += self.tokens_lemma[i] + "_"
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


class TypedSentence:

    def __init__(self, tokens,tokens_lemma, senid,fileid):
        self.tokens = tokens
        self.tokens_lemma = tokens_lemma
        self.mentions = []
        self.senid = senid
        self.fileid = fileid

    """
    @returns: A string tokenized by "_"
    """

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
                tokens_lemma = data['tokens_lemma']
                mentions = data['mentions']
                sentid = data['senid']
                fileid = data['fileid']
                for mention in mentions:
                    self.sentences.append(Sentence(tokens,tokens_lemma , mention['start'], mention['end'],sentid,fileid))
                    if self.unique:
                        break
        if size > 0:
            self.sentences = self.sentences[:size]

def obtain_ontology_words(file_name, root):
    words = []
    words_level =[]
    words_dic = {}

    with open(file_name, "r") as f:
        for line in f.readlines():
            # extract the words of each line
            ws = line.split(",")
            if root not in ws: 
                continue
            else:
                ws = ws[ws.index(root):]
                print(ws)
                
            # add unique words
            w_tmp = []
            for i,w in enumerate(ws):
                if w =='' or w == '\n':
                    continue
                lemma = w.replace("\n", "").lower().replace("_"," ")
                if (w.replace("\n", "") not in words) and (len(w) > 0) and (w != "\n"):
                    words.append(w.replace("\n", ""))
 
                    words_level.append((w.replace("\n", ""),i))
                    if i>0 and w.replace("\n", "") not in words_dic.keys():
                        words_dic[lemma] = w_tmp[0:]
                w_tmp.append(lemma)
                    
    return words_level,words_dic

def choose_relation(context,relations):
    sentences = [context] + relations
    embeddings = sent_model.encode(sentences)
    max_cos = 0
    max_i = -1
    max_theme = 0
    max_self = 0
    i=-1
    for i in range(len(relations)):
        cosine = np.dot(embeddings[0],embeddings[i+1])
        if  cosine>max_cos:
            max_i = i
            max_cos = cosine
        print(relations[i],cosine)

    if max_cos>0.15 or i==-1:
        result = relations[max_i]
    else:
        result = ''

    return result


def trans_similarity(sentences):
    embeddings = sent_model.encode(sentences)
    cosine = np.dot(embeddings[0],embeddings[1])
    return cosine

def choose_cate(mention,categories):
    if len(categories)==0:
        return ''
    sentences = [mention] + categories + [theme]
    embeddings = sent_model.encode(sentences)
    max_cos = 0
    max_i = -1
    max_theme = 0
    max_self = 0
    i=-1
    for i in range(len(categories)):
        cosine_self = np.dot(embeddings[0],embeddings[i+1])
        cosine_theme = np.dot(embeddings[i+1],embeddings[-1])
        if cosine_theme>0.05:
            cosine = cosine_self 
            if  cosine>max_cos:
                max_i = i
                max_cos = cosine
                max_self = cosine_self 
                max_theme = cosine_theme
            
    if max_cos>0.15 or i==-1:
        result = categories[max_i]
    else:
        result = mention
        
    print(mention,categories[max_i],cosine_self, cosine_theme)
    return result

#case1: match
#case2: search 

def choose_prior_category(context,relations):
    if len(relations)==0:
        return []
    sentences = [context] + relations
    embeddings = sent_model.encode(sentences)
    max_i = -1 
    max_cos = -1
    res = []
    for i in range(len(relations)):
        cosine = np.dot(embeddings[0],embeddings[i+1])
        if  cosine>max_cos:
            max_i = i
            max_cos = cosine
        if cosine > 0.5:
            res.append(relations[i])
    return res
        

def get_prior_category(category):
    if  category in words_dic.keys():
        return 
    url = "https://en.wikipedia.org/wiki/Category:"+category
    response = requests.get(url)
    res_parse = response.text.split('Hidden categories:')[0].split('title=\"Help:Category\"')
    m = []
    if len(res_parse)>1:
        m = re.findall(r'title=\"Category:.*?\"\>', res_parse[1])
    res = []
    for result in m:
        label = result[16:-2]
        if  'Wikidata' not in label and 'Category' not in label:
            if label not in res:
                res.append(label)
    print(res)
    res = choose_prior_category(category,res)
    print(res)
    words_dic[category] = res
    
def get_category(mention,mentionid):
    print(mention,mentionid)
    results = []
    match_results = []
    if mentionid!= 0:
        for result in sent_tool.get_parents(mentionid,10):
            if result[1]:
                label = result[1].lower()
                if label in init_categories.keys():
                    results.append(label)
                else:
                    if label != 'disambiguation pages' and trans_similarity([theme,label])>0:
                        categories[label] = 6         
                        results.append(label)
                        get_prior_category(label)

    if len(results)==0:
        url = "https://en.wikipedia.org/wiki/"+mention
        # 发送请求并检索响应
        response = requests.get(url)
        m = re.findall(r'title=\"Category:.*?\"\>', response.text.split('Hidden categories:')[0])
        if mention in categories.keys():
            match_results.append(mention)

        for result in m:
            label = result[16:-2].lower()
            if label in init_categories.keys():
                match_results.append(label)
            else:
                if label != 'disambiguation pages' and trans_similarity([theme,label])>0.1:
                    categories[label] = 6         
                    results.append(label)
                    get_prior_category(label)


    return results if len(match_results) == 0 else match_results

def get_category_by_submention(mention):
    results = []
    match_results = []
    url = "https://en.wikipedia.org/wiki/"+mention
    # 发送请求并检索响应
    response = requests.get(url)
    m = re.findall(r'title=\"Category:.*?\"\>', response.text.split('Hidden categories:')[0])
    new_title = re.findall(r'\"wgPageName\":\".*?\"', response.text.split('Hidden categories:')[0])
    
    if mention in categories.keys():
        match_results.append(mention)
    
    for result in m:
        label = result[16:-2].lower()
        if label in categories.keys():
            match_results.append(label)
        else:
            if label != 'disambiguation pages' :
                categories[label] = 6         
                results.append(label)
                get_prior_category(label)

    if len(new_title)>0 and len(m)>0:
        label = new_title[0][14:-1].lower().replace('_',' ')
        if label in categories.keys():
            match_results.append(label)
        else:
            if label != 'disambiguation pages' :
                categories[label] = 6         
                results.append(label)
                get_prior_category(label)

    return results if len(match_results) == 0 else match_results

def get_cate_subpart(sentence):
    ment_list = sentence.tokens_lemma[sentence.mention_start: sentence.mention_end]
    results = []
    num = len(ment_list)
    i = 0
    while len(results)==0 and i <= num-1:
        mention = ' '.join(ment_list[i:])
        results = get_category_by_submention(mention)
        if len(results)>0 and i>0:
             print(mention)
        i=i+1
    return results 


# initialize the models and ontology
runner = ZoeRunner(allow_tensorflow=True)
data = DataReader(datainput)
file_name = datapath + root + ".csv"
words ,words_dic= obtain_ontology_words(file_name, root)
init_categories = {}
for word in words:
    init_categories[word[0].lower().replace("_"," ")] = word[1]    
categories = init_categories.copy()
mention_cate_dict = {}

# get category candidates
for i,sent in tqdm(enumerate(data.sentences)):
    mention = sent.get_mention_surface().replace('_'," ")
    cates = set()
    # print(mention)
    if mention in mention_cate_dict.keys():
        continue
    url = f'https://www.wikidata.org/w/api.php?action=wbsearchentities&search={mention}&language=en&format=json'
    req = requests.get(url)
    info = json.loads(req.text)
    match = dict()
    if len(info['search']) !=0:
        for result in info['search']:
            if len(cates)>0:
                break
            dis = Levenshtein.distance(mention,result['label'])/len(mention) 
            if dis<0.2:
                cate = get_category(result['label'],result['url'][25:])
                match[(result['label'],result['url'])]=cate
                cates.update(cate)
    if len(match.keys())==0:
        cate =  get_cate_subpart(sent)
        cates.update(cate)
    theme_score = trans_similarity([theme, ", ".join(list(cates)+[mention])])
    print(theme_score)
    if theme_score>0.12:
        print(cates)
    mention_cate_dict[mention] = cates
    
#choose categories
mention_cate_dict_final = {}
for i,sent in tqdm(enumerate(data.sentences)):
    mention = sent.get_mention_surface().replace('_'," ")
    if mention not in mention_cate_dict_final.keys():
        cates = mention_cate_dict[mention]
        cates_new = []
        for cate in cates:
            cates_new.append(cate)
            if cate in words_dic.keys() and len(words_dic[cate])>0:
                cates_new.append(words_dic[cate][-1])
            
        theme_score = trans_similarity([theme,", ".join(list(cates)+[mention])])
        if  theme_score>0.12 and len(cates)>0:
            mention_cate_dict_final[mention] = choose_cate(mention,list(cates))
        else:
            mention_cate_dict_final[mention] = 'None'
            print(mention,theme_score,len(cates))
    data.sentences[i].predicted_types = mention_cate_dict_final.get(mention,'')
    data.sentences[i].could_also_be_types = cates

#generate output
typedsentences = []
sentences_id = []
for i,sent in enumerate(data.sentences):
    senid = sent.senid
    if sent.predicted_types!='None':
        if  senid in sentences_id:
            typedsentences[-1].mentions.append({'start':sent.mention_start,'end':sent.mention_end,'catagory':sent.predicted_types})
        else:
            sentences_id.append(senid)
            tmp = TypedSentence(sent.tokens,sent.tokens_lemma,senid,sent.fileid)
            tmp.mentions.append({'start':sent.mention_start,'end':sent.mention_end,'catagory':sent.predicted_types})
            typedsentences.append(tmp)
            
with open (output,"w+") as f:
    for sentence in typedsentences:
        sj = json.dumps(sentence.__dict__)
        f.write(sj+'\n')
        
with open (output_category, "w+") as f:
    sj = json.dumps(words_dic)
    f.write(sj)
