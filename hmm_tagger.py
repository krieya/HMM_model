### Import modules

from collections import defaultdict, Counter
import numpy as np
from nltk.corpus import brown,words
from nltk.corpus import treebank
from random import shuffle,seed,choice,random
from sklearn.metrics import adjusted_rand_score


### main sequential HMM class here

class SequentialHMM(object):
    
    def __init__(self,emissions,states,allow_unknown = False):
        '''emissions are a list of possible emissions (words)
        states are a list of possible hidden states (pos tags)
        allow_unknown is used to indicate that it will be possible to emit an unknown word
        (Optional)'''
        self.allow_unknown = allow_unknown

        self.state_lookup = {}
        self.rev_state_lookup = {}
        for state in states:
            self.state_lookup[state] = len(self.state_lookup)
            self.rev_state_lookup[self.state_lookup[state]] = state
        
        self.emissions_lookup = {}
        self.rev_emissions_lookup = {}
        for emission in emissions:
            self.emissions_lookup[emission] = len(self.emissions_lookup)
            self.rev_emissions_lookup[self.emissions_lookup[emission]] = emission
            
        init_probs = np.log(np.ones(len(states)) / len(states))
        tran_probs = np.log(np.ones((len(states), len(states))) / len(states))
        emis_probs = np.log(np.ones((len(states), len(emissions))) / (np.array([len(emissions)])))
        
        self.init_probs = init_probs
        self.tran_probs = tran_probs
        self.emis_probs = emis_probs
        
        
    def train(self,training_set):
        '''updates the start, transition, and emission matrices based on the provided training corpus,
        which should be a list of sentences where each sentence is a list of (word,pos) tuples
        '''
        word_counter = Counter()
        state_counts = defaultdict(int)
        first_state_counts = defaultdict(int)
        state_state_counts = defaultdict(int)
        state_emission_counts = defaultdict(int)
        for sent in training_set:
            word_counter.update([word[0] for word in sent])
            first_state_counts[sent[0][1]] += 1
            state_counts[sent[0][1]] += 1
            state_emission_counts[(sent[0][1], sent[0][0])] += 1
            for i in range(len(sent) -1):
                state_state_counts[(sent[i][1], sent[i+1][1])] += 1
                state_counts[sent[i+1][1]] += 1
                state_emission_counts[(sent[i+1][1], sent[i+1][0])] += 1
        
        if self.allow_unknown:
            single_occur_words = list()
            state_emission_counts_for_unk = defaultdict(int)
            for word, count in word_counter.items():
                if count == 1:
                    single_occur_words.append(word)
            for pair in state_emission_counts:
                if pair[0] in single_occur_words:
                    state_emission_counts_for_unk[(pair[0], '*UNK*')] = state_emission_counts[pair]
                else:
                    state_emission_counts_for_unk[pair] = state_emission_counts[pair]
                    
            emissions_with_unk = [item for item in list(word_counter.keys()) if item not in single_occur_words] + ['*UNK*']
            self.emissions_with_unk = emissions_with_unk
            
            # re-initialize the emission matrix and emission lookup:
            self.emis_probs = np.log(np.ones((len(state_counts), len(emissions_with_unk))) / (np.array([len(emissions_with_unk)])))
            self.emissions_lookup = {}
            self.rev_emissions_lookup = {}
            for emission in emissions_with_unk:
                self.emissions_lookup[emission] = len(self.emissions_lookup)
                self.rev_emissions_lookup[self.emissions_lookup[emission]] = emission
                    

        for i in range(len(self.init_probs)):
            state = self.rev_state_lookup[i]
            self.init_probs[i] = np.log((first_state_counts.get(state,0) + 1) / (len(training_set) + len(self.init_probs)))
            
            for tran_i in range(len(self.init_probs)):
                state_2 = self.rev_state_lookup[tran_i]
                pair = (state, state_2)
                self.tran_probs[i, tran_i] = np.log((state_state_counts.get(pair, 0) + 1) / (state_counts.get(state) + self.tran_probs.shape[1]))
            
            if self.allow_unknown:
                for emis_i in range(self.emis_probs.shape[1]):
                    w = self.rev_emissions_lookup[emis_i]
                    pair = (state, w)
                
                    self.emis_probs[i, emis_i] = np.log((state_emission_counts_for_unk.get(pair, 0) + 1) / (state_counts.get(state) + self.emis_probs.shape[1]))
                              
            else:
                for emis_i in range(self.emis_probs.shape[1]):
                    w = self.rev_emissions_lookup[emis_i]
                    pair = (state, w)
                
                    self.emis_probs[i, emis_i] = np.log((state_emission_counts.get(pair, 0) + 1) / (state_counts.get(state) + self.emis_probs.shape[1]))
            
    
    def decode(self, sentence):
        '''Carries out Viterbi decoding of the sentence, which is a list of words
        Returns a sentence where each word has been replace with a (word,pos) tuple with the best
        POS sequence assigned based on the provided transition probabilities
        '''
        
        back_pointer = dict()
        vite_probs = list()
        
        new_sentence = []
        for word in sentence:
            if word in self.emissions_lookup:
                new_sentence.append(word)
            else:
                new_sentence.append('*UNK*')
            
        
        vite_probs.append((self.init_probs + self.emis_probs[:, self.emissions_lookup[new_sentence[0]]]).reshape(-1, 1))
        
        for word_i in range(1, len(new_sentence)):
            
            cur_word_ind = self.emissions_lookup[new_sentence[word_i]]
            
            vite_prob = vite_probs[word_i - 1] + self.tran_probs + self.emis_probs[:, cur_word_ind] 
            
            vite_probs.append(np.max(vite_prob, axis = 0).reshape(-1, 1))
            back_pointer[word_i] = np.argmax(vite_prob, axis = 0) 
        
        pointer_list = []
        pointer_list.append(np.argmax(vite_probs[-1]))
        for i in range(len(new_sentence)-1, 0, -1):
            pointer_list.append(back_pointer[i][pointer_list[len(new_sentence) - i - 1]])
        
        pos_list = [self.rev_state_lookup[i] for i in reversed(pointer_list)]
        return [(sentence[i], pos_list[i]) for i in range(len(sentence))], np.max(vite_probs[-1])


### EM Algorithm for semi-supervised tagging 

def EM(hmm,corpus):
    '''
    hmm --- an already initialized (but not necessarily trained) SequentialHMM
    corpus --- a lists of sentences, where each sentence is a list of (word,pos) pairs
    
    returns a corpus with new pos tags after carrying out Viterbi-based expectation maximization'''
    last_perplexity = -1
    curr_perplexity = -1
    while last_perplexity == -1 or last_perplexity -curr_perplexity > 0.5:
        last_perplexity = curr_perplexity
        total_logp = 0
        total_words = 0
        corpus_with_new_tags = []
        
        hmm.train(corpus)
        for sent in corpus:
            new_sent = [word[0].lower() for word in sent if word[0].lower() in hmm.emissions_lookup]
            if len(new_sent) == len(sent):
                new_decode, logp = hmm.decode(new_sent)
                corpus_with_new_tags.append(new_decode)
                total_logp += np.log2(np.exp(logp))
                total_words += len(new_sent)
                
        corpus = corpus_with_new_tags
        curr_perplexity = 2**(-total_logp/total_words)
        
    return corpus_with_new_tags
    

### test functions for EM algorithm

def get_randomized_tags(corpus,percent_accurate,POS_tags):
    '''this function takes a gold standard tagged corpus and randomly keeps precent_accurate of them 
    correct while corrupting the rest by randomly selecting a tag from POS_tags'''
    random_tag_sents =[]
    for sent in corpus:
        new_sent = []
        for word,pos in sent:
            if random() < percent_accurate:
                new_sent.append((word,pos))
            else:
                new_sent.append((word, choice(POS_tags)))
        random_tag_sents.append(new_sent)
    return random_tag_sents

def accuracy(ref_tagged_sents,pred_tagged_sents):
    ''' this function calculates tagging accuracy'''
    correct = 0
    total = 0
    for i in range(len(ref_tagged_sents)):
        for j in range(len(ref_tagged_sents[i])):
            if ref_tagged_sents[i][j][1] == pred_tagged_sents[i][j][1]:
                correct +=1
            total +=1
    return correct/total



### Use brown corpus to train and test on the treebank corpus

if __name__ == '__main__':

    brown_vocab = set()
    brown_POS_tags = set()

    for word,pos in brown.tagged_words(tagset='universal'):
        brown_vocab.add(word.lower())
        brown_POS_tags.add(pos)
    
    hmm =  SequentialHMM(brown_vocab, brown_POS_tags)

    brown_corpus = [[(word.lower(),pos) for word,pos in sent] for sent in brown.tagged_sents(tagset='universal')]
    
    print("Not allowing unknown words:")
    hmm.train(brown_corpus) 
    
    print("1 example in treebank:")
    for sent in treebank.sents():
        new_sent = [word.lower() for word in sent if word.lower() in brown_vocab]
        if len(new_sent) == len(sent):
            print(hmm.decode(new_sent))
            break
            
    print("--------")
    print("Allowing unknown words:")

    hmm =  SequentialHMM(brown_vocab, brown_POS_tags,allow_unknown=True)
    hmm.train(brown_corpus)

    print("5 examples in treebank:")
    for sent in treebank.sents()[:5]:
        new_sent = [word.lower() for word in sent]
        print(hmm.decode(new_sent))
        print("---------")


    ## test the EM algorithm
    print("Now test the EM algorithm works")
    for accurate_data in [0.001, 0.01, 0.1]:
        randomized_corpus = get_randomized_tags(brown_corpus,accurate_data,list(brown_POS_tags))
        print("percent correct data")
        print(accurate_data)
        print("starting accuracy")
        start_accuracy = accuracy(brown_corpus,randomized_corpus)  
        print(start_accuracy)
        hmm =  SequentialHMM(brown_vocab,brown_POS_tags)
        output = EM(hmm,randomized_corpus)
        print("ending accuracy")
        end_accuracy = accuracy(brown_corpus,output)
        print(end_accuracy)


    









