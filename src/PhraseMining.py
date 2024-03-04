import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
import copy
import re
import collections
import operator
import numpy as np
from numpy.linalg import norm
import Levenshtein
from tqdm import tqdm
import requests
import json
from wikidata.client import Client
import queue

#Settings
#dataset
datapath = '../dataset/EVB/evbattery.txt'
outpath = '../dataset/EVB/evbattery_phrase.json'
#spacy tools
nlp = spacy.load("en_core_web_sm")

##wikidata client
client = Client()
## for test the client
'''
subclass_of = client.get('P279')  
instance_of = client.get('P31')
'''
buffer = {}

## idf stats
idf_dic={}
with open("enwiki-2023-04-13.txt",'r') as f:
    idf_lines = f.readlines()
    for idf_line in idf_lines:
        idf = idf_line.split()
        if len(idf)==2:
            idf_dic[idf[0]] =  float(idf[1])
            
## for phrase frequency count            
min_phrase = 1
max_phrase = 6
            
class Sentence:

    def __init__(self, tokens,tokens_lemma, mentions,senid,fileid=""):
        self.tokens = tokens
        self.tokens_lemma = tokens_lemma
        self.mentions = mentions
        self.senid = 0
        self.fileid = ""
        self.mentions_label = []

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
    def print_self(self):
        print(self.get_sent_str())
        print(self.get_mention_surface())

class ChunkToken:
    def __init__(self, text, pos, is_stop, frequency,lemma,start_token,end_token):
        self.text = text
        self.pos_ = pos
        self.is_stop = is_stop
        self.frequency = frequency
        self.lemma_ = lemma
        self.start = start
        self.end = end
    def __call__(self):
        return self.text

def print_chunk_list(chunk_list):
    for chunk in chunk_list:
        print([t.text for t in chunk], [t.pos_ for t in chunk],[t.is_stop for t in chunk],[t.frequency for t in chunk],[t.lemma_ for t in chunk])

def print_chunk_list_short(chunk_list):
    for chunk in chunk_list:
        print([t.text for t in chunk])
        
def if_match_wiki(mention):
    results = []
    match_results = []
    url = "https://en.wikipedia.org/wiki/"+mention
    # request wiki
    response = requests.get(url)
    m = re.findall(r'title=\"Category:.*?\"\>', response.text.split('Hidden categories:')[0])
    new_title = re.findall(r'\"wgPageName\":\".*?\"', response.text.split('Hidden categories:')[0])
    if len(m)==0:
        return False
    for result in m:
        label = result[16:-2].lower()
        if label == 'disambiguation pages':
            return False
    return True
def get_chunk_freq(chunk):
    terms = ' '.join([t.lemma_ for t in chunk])
    terms = re.findall(r"[\w']+", terms)
    terms = ' '.join(terms)
    chunk_freq = phrases_freq.get(terms,0)
    print(terms,chunk_freq)
    return chunk_freq
    
def parse(chunks):
    new_chunks = []
    for chunk in chunks:
        new_chunk = chunk[0:]
        print_chunk_list([new_chunk])
        pron_flag = False
        stop_flag = False
        adj_flag = False
        noun_flag = False
        general_flag = False
        tf_flag = False
        word_idx = 0
        stop_idx = -1000
        if len(chunk) <= 0:
            continue
        # without nouns
        if chunk[-1].pos_ != "NOUN" and chunk[-1].pos_!= "PROPN" :
            print("without noun")
            continue
        
        # pron/num, # cut the stopwords
        for t in list(chunk)[::-1]:
            #with pron/num
            if t.pos_ in ['PRON','NUM','PUNCT'] :
                pron_flag = True
                break
            # cut the stopwords
            if t.pos_ in ['DET'] :
                stop_flag = True
                stop_idx = max(stop_idx,word_idx) 
                print(t.text)
            word_idx = word_idx - 1
        if pron_flag:
            print('with pron')
            continue
        if stop_flag  and stop_idx != 0:
            print('delete stop',stop_idx)
            new_chunk = new_chunk[stop_idx:]

        
         # TF FREQ
        if len(new_chunk)>1:
            chunk_freq = get_chunk_freq(new_chunk)
            if (chunk_freq>2 and len(new_chunk)>2) or (chunk_freq>4):
                new_chunks.append(new_chunk)
                print('chunk freq>2')
                print([t.text for t in new_chunk])
                continue
                
        word_idx = 0
        stop_idx = -1000
        curr_word = ''
        for t in new_chunk[::-1]:
            curr_word = t.text + ' ' + curr_word
            if t.pos_ in ['ADJ','ADV'] and idf_dic.get(t.text,0)>100000:
                if not if_match_wiki(curr_word):
                    adj_flag = True
                    stop_idx = max(stop_idx,word_idx)
                else:
                    print('match_wiki:'+curr_word)
            word_idx = word_idx - 1 

        if adj_flag  and stop_idx != 0:
            print('delete adj')
            new_chunk = new_chunk[stop_idx:]
            print([t.text for t in new_chunk])
            
        if len(new_chunk)>2:
            general_flag = True
        else:
            for t in new_chunk:
                if t.frequency <130000*len(new_chunk) and t.pos_!= "PROPN":
                    general_flag = True
        if general_flag:
            new_chunks.append(new_chunk)
            print([t.text for t in new_chunk])
        else:
            print('general single word')
    return new_chunks

## dataloader
with open(datapath,'r') as f:
    text = '. '.join(f.readlines())

text = text.lower()
doc = nlp(text)
sentences = []
tokens = []
senid = 0
for token in doc:
    if token.text  == ".":
        tokens.append(token.text)
        sentences.append(Sentence(tokens[0:],[], [],senid))
        senid = senid +1
        tokens = []
        continue
    if token.text != "-" and token.text !="–":
        tokens.append(token.text)
        
lem = [token.lemma_ for token in doc]
lem_text = ' '.join(lem)
 
## in-corpus phrases freqency
#get list of all sentences in document
sentences = []
#with open ("./PhraseCount/input.txt") as document:
    #for line in document:
        #Add contained sentences to list (Currently assumes no abbreviations, acronyms, websites, decimals etc.)
lem_text.replace('\r\n', ' ')
sentences.extend(re.split(r"\.|,|\?|\!", lem_text))

#Build dictionary of phrases
phrases_freq = {}
for sentence in sentences:
    words = re.findall(r"[\w']+", sentence)

    #Build phrases starting from each word
    for index,word in enumerate(words):

        #Start with smallest phrase length and increase in size until end of sentence
        phrase_length = min_phrase;
        while (phrase_length <= max_phrase and  index + phrase_length <= len(words)):
            phrase =' '.join(words[index : (index + phrase_length)])
            
            #Add phrase to dictionary or increment dulplicate count if it already exists
            phrases_freq[phrase] = phrases_freq.get(phrase, 0) + 1
            phrase_length += 1
        
# process 
new_chunks = []
for chunk in doc.noun_chunks:
    if len(chunk)==0:
        continue
    new_chunk = []
    start = 0
    end = 0
    tmp = ChunkToken('','',True,0,'',start,end)
    merge_flag = False
    for i,t in enumerate(chunk):
        if i ==0 or tmp.text == '':
            tmp.text = t.text
            tmp.pos_ = t.pos_
            tmp.is_stop = t.is_stop
            tmp.frequency = idf_dic.get(t.lemma_,0)
            tmp.lemma_ = t.lemma_
            continue
        if merge_flag:
            merge_flag = False
            continue
        if (t.text == "-"  or t.text == "–") and i<len(chunk)-1:
            tmp.text = chunk[i-1].text+t.text+ chunk[i+1].text
            tmp.pos_ = chunk[i+1].pos_
            tmp.is_stop = chunk[i+1].is_stop
            tmp.frequency = idf_dic.get(tmp.text,0)
            tmp.lemma_ = tmp.text
            merge_flag = True
        elif  t.pos_ == 'CCONJ':
            new_chunk.append(copy.deepcopy(tmp))
            tmp = ChunkToken('','',True,0,'',start,end)
            new_chunks.append(new_chunk)
            new_chunk = []
        else:
            new_chunk.append(copy.deepcopy(tmp))
            tmp.text = t.text
            tmp.pos_ = t.pos_
            tmp.is_stop =t.is_stop
            tmp.frequency = idf_dic.get(t.lemma_,0)
            tmp.lemma_ = t.lemma_
    new_chunk.append(copy.deepcopy(tmp))
    new_chunks.append(new_chunk)
    
phrases = parse(new_chunks)
phrases_clean = []
for phrase in phrases:
    phrase_clean = []
    for p in phrase:
        if '-' in p.text:
            ps  = p.text.split('-')
            phrase_clean.extend(ps)
        elif '–' in p.text:
            ps  = p.text.split('–')
            phrase_clean.extend(ps)
        else:
            phrase_clean.append(p.text)
    phrases_clean.append(phrase_clean[0:])

##generate output
sentences = []
tokens = []
senid = 0
tokens_lemma = []
for token in doc:
    if token.text  == ".":
        tokens.append(token.text)
        tokens_lemma.append(token.lemma_ if token.pos_ == "NOUN" else token.text)
        sentences.append(Sentence(tokens[0:],tokens_lemma, [],senid))
        senid = senid +1
        tokens = []
        tokens_lemma = []
        continue
    if token.text != "-" and token.text != "–":
        tokens.append(token.text)
        tokens_lemma.append(token.lemma_ if token.pos_ == "NOUN" else token.text)
        
mentionid = 0
mention_num = len(phrases_clean)

for senid, sentence in enumerate(sentences):
    soffset = 0
    slen = len(sentence.tokens)
    if mentionid >= len(phrases_clean):
        break
    mention_len  = len(phrases_clean[mentionid])
    sentence.senid = senid
    while soffset < slen - mention_len and mentionid<mention_num:
        curv = soffset
        start = soffset
        for mention_token in phrases_clean[mentionid]:                
            if sentence.tokens[curv] != mention_token:
                break
            curv+=1
        if (curv-start) == mention_len:
            sentence.mentions.append({'start':start,'end':curv,'labels':['']})
            sentence.mentions_label.append({'start':start,'end':curv})
            soffset = curv
            mentionid = mentionid +1
            if mentionid<mention_num:
                mention_len  = len(phrases_clean[mentionid])
        else:
            soffset = soffset+1

# output
with open (outpath,"w+") as f:
    for sentence in sentences:
        sj = json.dumps(sentence.__dict__)
        f.write(sj+'\n')


            

