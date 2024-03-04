import openai
import json
from itertools import combinations
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import os
import re

#settings
datapath = '../dataset/EVB/'
datainput = datapath + 'evbattery_typed.json'
datainput_category = datapath + 'category_dict.json'
output = datapath + 'triples.json'
use_root = True # if use parent category to retrieve relation candidates 


# add your open ai token for gpt-4
def get_api_key():
    return ''

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

    def __init__(self, tokens,tokens_lemma,mentions, senid,fileid):
        self.tokens = tokens
        self.tokens_lemma = tokens_lemma
        self.mentions = mentions
        self.senid = senid
        self.fileid = fileid

    """
    @returns: A string tokenized by "_"
    """

class Triples:

    def __init__(self, tokens,tokens_lemma,mention1,mention2, senid,fileid,relation_candidates):
        self.tokens = tokens
        self.tokens_lemma = tokens_lemma
        self.mention1 = mention1
        self.mention2 = mention2
        self.senid = senid
        self.fileid = fileid
        self.relation_candidates = relation_candidates
        self.relations = ""

    """
    @returns: A string tokenized by "_"
    """
class Triple:

    def __init__(self, tokens,tokens_lemma,mention1,mention2, senid,fileid,relation_candidates):
        self.tokens = tokens
        self.tokens_lemma = tokens_lemma
        self.mention1 = mention1
        self.mention2 = mention2
        self.senid = senid
        self.fileid = fileid
        self.relation_candidates = relation_candidates
        self.relations = ""
    def get_context(self):
        return " ".join(self.tokens)
    def get_mention_surface(self):
        concat1 = ""
        for i in range(self.mention1['start'], self.mention1['end']):
            concat1 += self.tokens_lemma[i] + "_"
        if len(concat1) > 0:
            concat1 = concat1[:-1]
        concat2 = ""
        for i in range(self.mention2['start'], self.mention2['end']):
            concat2 += self.tokens_lemma[i] + "_"
        if len(concat1) > 0:
            concat2 = concat2[:-1]
            
        return (concat1,concat2)

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
                self.sentences.append(TypedSentence(tokens,tokens_lemma , mentions,sentid,fileid))
        if size > 0:
            self.sentences = self.sentences[:size]



TEMPLATE = '''You are a very warm and helpful assistant for instance/entity relation identification. The user will give you two category words from the Wikidata. You will use your world knowledge to generate all possible relations between two instances respectively from these two categories. The input format is:

category1,category2

And the here is the output format:

relation1,relation2,relation3...

Please follow the template and do not output anything else!
Please follow the template and do not output anything else!!
Please follow the template and do not output anything else!!!



Input:
'''


TEMPLATE_2 = '''You are a very warm and helpful assistant for instance/entity relation identification. The user will give you two category words from the Wikidata. You will use your world knowledge to generate all possible relations between two instances respectively from these two categories. The input format is:

category1,category2

And the here is the output format:

category1, relation1,category2
category1, relation2,category2
category1, relation3,category2
......

Please follow the template and do not output anything else!
Please follow the template and do not output anything else!!
Please follow the template and do not output anything else!!!

'''

'''
Here are the examples:
Input: batteries, electrical vehicles
Output:
batteries, be power source of, electrical vehicles
batteries, be recycled from, electrical vehicles
batteries, be managed by , electrical vehicles

Input:
'''

TEMPLATE_3 = '''You are a very warm and helpful assistant for relation extraction. The user will give you the context and the pair of entity mentions and the set of candidate relations. You should choose the relation in set for the entities according to the context. if there is no relations of two entities in the context,please choose none. The input format is:

Context: context
Entity: entity1, entity2
Relation candidate: relation1, relation2, relation3

And the output format should be:

entity1, relation, entity2

Please follow the template and do not output anything else!
Please follow the template and do not output anything else!!
Please follow the template and do not output anything else!!!
'''

'''
Here the examples:
Input: 
Context:There are two main types of lead acid batteries : automobile engine starter batteries , and deep cycle batteries. 
Entity: deep cycle batteries, automobile engine starter batteries 
Relation candidate: be a type of, be compatible with, be used in, none

Output:
deep cycle batteries, none, automobile engine starter batteries

Input:
Context: Lithium ion batteries are currently used in most portable consumer electronics such as cell phones and laptops because of their high energy per unit mass and volume relative to other electrical energy storage systems.
Entity: portable consumer electronics, cell phones 
Relation candidate: be used in, include, manufacture, supply parts for,market,none

Output:
portable consumer electronics, include, cell phones 

Input:
'''


def obtain_relations(TEMPLATE, category1, category2):
    q = TEMPLATE + f"{category1},{category2}"
    
    rsp = openai.ChatCompletion.create(
      model="gpt-4",
      messages=[
            {"role": "user", "content": q}
        ]
    )
    return rsp.choices[0].message.content

def generate_relations(cate1,cate2):
    q = "list all potential relations in "+cate1+" and "+cate2+", in the format of " + "1."+cate1+" () "+cate2+"\n2."+cate1+" () "+cate2+"\n3."+cate1+" () "+cate2+"\n...\nPlease fill the blank and follow the format without other words."
    print(q)
    rsp = openai.ChatCompletion.create(
      model="gpt-4-1106-preview",
      messages=[
            {"role": "user", "content": q}
        ]
    )

    print(rsp.choices[0].message.content.lower())
    q2 = "Group the sentences with same meaning : " + rsp.choices[0].message.content.lower() +'''The result should Following the template:
    ** Meaning
    Sentence1:
    Sentence2:
    ** Meaning
    Sentence3:
    Sentence4:
    '''

    rsp2 = openai.ChatCompletion.create(
      model="gpt-4-1106-preview",
      messages=[
            {"role": "user", "content": q2}
        ]
    )   
    
    return rsp2.choices[0].message.content.lower() 

def generate_relations_2 (cate1,cate2):
    #q = "list all potential relations in "+cate1+" and "+cate2+", in the format of " + "1."+cate1+" () "+cate2+"\n2."+cate1+" () "+cate2+"\n3."+cate1+" () "+cate2+"\n...\nPlease use simple words fill the () and follow the format without other words."
    q = TEMPLATE_2 + cate1+', '+cate2
    rsp = openai.ChatCompletion.create(
      model="gpt-4-1106-preview",
      messages=[
            {"role": "user", "content": q}
        ]
    )
    print(cate1+', '+cate2)
    print(rsp.choices[0].message.content.lower())
    return rsp.choices[0].message.content.lower()    

def parse_relations_2(relation_str,cate1,cate2):
    relations = relation_str.split('\n')
    results  = []
    for instance in relations:
        triple = instance.split(',')
        if len(triple)==3:
            results.append(triple[1].strip())
    return results

def relation_classify(triple):

    context = ' '.join(triple.tokens)
    mention1 = get_mention(triple.tokens,triple.mention1['start'], triple.mention1['end'])
    mention2 = get_mention(triple.tokens,triple.mention2['start'], triple.mention2['end'])
    
    q = TEMPLATE_3+ '\n'+ "Context: "+context+ '\n'+'Entity: '+mention1+', '+mention2+'\n'+ 'Relation candidate: '+', '.join(triple.relation_candidates+['None'])
    rsp = openai.ChatCompletion.create(
      model="gpt-4-1106-preview",
      messages=[
            {"role": "user", "content": q}
        ]
    )
    triple.relations = rsp.choices[0].message.content.lower()   
    # print(rsp.choices[0].message.content.lower())
    return rsp.choices[0].message.content.lower()  

def relation_classify_2(triple,i):

    context = ' '.join(triples_v2[i])
    mention1 = get_mention(triple.tokens,triple.mention1['start'], triple.mention1['end'])
    mention2 = get_mention(triple.tokens,triple.mention2['start'], triple.mention2['end'])
    
    q = TEMPLATE_3+ '\n'+ "Context: "+context+ '\n'+'Entity: '+mention1+', '+mention2+'\n'+ 'Relation candidate: '+', '.join(triple.relation_candidates+['None'])
    rsp = openai.ChatCompletion.create(
      model="gpt-4-1106-preview",
      messages=[
            {"role": "user", "content": q}
        ]
    )
    triple.relations = rsp.choices[0].message.content.lower()   
    print(rsp.choices[0].message.content.lower())
    return rsp.choices[0].message.content.lower()

def get_mention(tokens,mention_start, mention_end):
    concat = ""
    for i in range(mention_start, mention_end):
        concat += tokens[i] + "_"
    if len(concat) > 0:
        concat = concat[:-1]

openai.api_key = get_api_key()
sent_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

data = DataReader(datainput)
with open(datainput_category) as f:
    words_dic_text = f.readline()
    words_dic = json.loads(words_dic_text)
    
relation_dict = {}
relation_dict_2 = {}
triples = []

for sent in tqdm(data.sentences):
    num = len(sent.mentions)
    for i in tqdm(range(num)):
        for j in tqdm(range(i+1,num)):
            mention_cate1 = sent.mentions[i]['catagory']
            mention_cate2 = sent.mentions[j]['catagory']
            relation_candidates = []
            if (mention_cate1,mention_cate2) in relation_dict_2.keys():
                relation_candidates = relation_dict_2[(mention_cate1,mention_cate2)]
            elif (mention_cate2,mention_cate1) in relation_dict_2.keys():
                relation_cadidates = relation_dict_2[(mention_cate2,mention_cate1)]
            else:
                relation_str = generate_relations_2(mention_cate1 ,mention_cate2)
                relation_candidates = parse_relations_2(relation_str,mention_cate1 ,mention_cate2)
                relation_dict_2[(mention_cate1,mention_cate2)] = relation_candidates

            root_relation_candidates = []
            if use_root:
                root1 = words_dic.get(mention_cate1,[])
                root2 = words_dic.get(mention_cate2,[])
                if len(root1)==0 and len(root2)==0:
                    pass
                else:
                    if len(root1)>0:
                        mention_cate1 = root1[-1]
                    if len(root2)>0:
                        mention_cate2 = root2[-1]
                    if (mention_cate1,mention_cate2) in relation_dict_2.keys():
                        relation_candidates = relation_dict_2[(mention_cate1,mention_cate2)]
                    elif (mention_cate2,mention_cate1) in relation_dict_2.keys():
                        relation_cadidates = relation_dict_2[(mention_cate2,mention_cate1)]
                    else:
                        relation_str = generate_relations_2(mention_cate1 ,mention_cate2)
                        root_relation_candidates = parse_relations_2(relation_str,mention_cate1 ,mention_cate2)
                        relation_dict_2[(mention_cate1,mention_cate2)] = root_relation_candidates

            triple = Triple(sent.tokens,sent.tokens_lemma,sent.mentions[i],sent.mentions[j], sent.senid,sent.fileid,relation_candidates+root_relation_candidates)        
            triples.append(triple)
            
# relation classification
for triple in tqdm(triples):
    relation_classify(triple)
    
# result1
for triple in tqdm(triples):
    t = triple.relations.split(',')
    if len(t)==3:
        e1 =t[0].stripe()
        e2 = t[1].stripe()
        e3 = t[2].stripe()
        if e2!='none':
            print('{\"source\":\"'+e1+'\",\"relation\":\"'+e2+'\",\"target\":\"'+e3+'\"},')
            
#results final 
with open (output,"w+") as f:
    for triple in triples:
        sj = json.dumps(triple.__dict__)
        f.write(sj+'\n')

