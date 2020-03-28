# HMM_model

A script that does unsupervised part-of-speech tagging using HMMs and the EM algorithm.

### Description

I used HMMs and Viterbi-based expectation maximization (EM) for semi-supervised POS tagging. Specifically, I train my sequential HMM tagger on the brown corpus and test on the treebank corpus. To deal with unknown words, I used the simple dropout in my function. In my sample output, I printed out the tagging results along with the probability of the sentence. 

In the EM method, I tested a variety of correct data percentages and the accuracy after running EM. 

### Dependancies

- Python(>=3.6)
- nltk
- collections
- numpy
- random
- scikit-learn


### Usage

`python hmm_tagger.py`

### Sample output

```
Not allowing unknown words:
1 example in treebank:
([('there', 'PRT'), ('is', 'VERB'), ('no', 'DET'), ('asbestos', 'NOUN'), ('in', 'ADP'), ('our', 'DET'), ('products', 'NOUN'), ('now', 'ADV'), ('.', '.'), ("''", '.')], -63.86265242750288)
--------
Allowing unknown words:
5 examples in treebank:
([('pierre', 'NOUN'), ('vinken', '.'), (',', '.'), ('61', 'NUM'), ('years', 'NOUN'), ('old', 'ADJ'), (',', '.'), ('will', 'VERB'), ('join', 'VERB'), ('the', 'DET'), ('board', 'NOUN'), ('as', 'ADP'), ('a', 'DET'), ('nonexecutive', 'ADJ'), ('director', 'NOUN'), ('nov.', 'NOUN'), ('29', 'NUM'), ('.', '.')], -143.98383331045272)
---------
([('mr.', 'NOUN'), ('vinken', 'PRT'), ('is', 'VERB'), ('chairman', 'NOUN'), ('of', 'ADP'), ('elsevier', 'DET'), ('n.v.', 'NOUN'), (',', '.'), ('the', 'DET'), ('dutch', 'ADJ'), ('publishing', 'NOUN'), ('group', 'NOUN'), ('.', '.')], -105.44471263378199)
---------
([('rudolph', 'DET'), ('agnew', 'NOUN'), (',', '.'), ('55', 'NUM'), ('years', 'NOUN'), ('old', 'ADJ'), ('and', 'CONJ'), ('former', 'ADJ'), ('chairman', 'NOUN'), ('of', 'ADP'), ('consolidated', 'DET'), ('gold', 'ADJ'), ('fields', 'NOUN'), ('plc', '.'), (',', '.'), ('was', 'VERB'), ('named', 'VERB'), ('*-1', 'ADP'), ('a', 'DET'), ('nonexecutive', 'ADJ'), ('director', 'NOUN'), ('of', 'ADP'), ('this', 'DET'), ('british', 'ADJ'), ('industrial', 'ADJ'), ('conglomerate', 'NOUN'), ('.', '.')], -225.2869819708468)
---------
([('a', 'DET'), ('form', 'NOUN'), ('of', 'ADP'), ('asbestos', 'PRON'), ('once', 'ADV'), ('used', 'VERB'), ('*', 'PRT'), ('*', 'VERB'), ('to', 'PRT'), ('make', 'VERB'), ('kent', 'NOUN'), ('cigarette', 'NOUN'), ('filters', 'NOUN'), ('has', 'VERB'), ('caused', 'VERB'), ('a', 'DET'), ('high', 'ADJ'), ('percentage', 'NOUN'), ('of', 'ADP'), ('cancer', 'NOUN'), ('deaths', 'NOUN'), ('among', 'ADP'), ('a', 'DET'), ('group', 'NOUN'), ('of', 'ADP'), ('workers', 'NOUN'), ('exposed', 'VERB'), ('*', 'ADV'), ('to', 'ADP'), ('it', 'PRON'), ('more', 'ADV'), ('than', 'ADP'), ('30', 'NUM'), ('years', 'NOUN'), ('ago', 'ADV'), (',', '.'), ('researchers', 'PRON'), ('reported', 'VERB'), ('0', 'NUM'), ('*t*-1', 'NOUN'), ('.', '.')], -334.1062807985621)
---------
([('the', 'DET'), ('asbestos', 'ADJ'), ('fiber', 'NOUN'), (',', '.'), ('crocidolite', 'ADV'), (',', '.'), ('is', 'VERB'), ('unusually', 'ADV'), ('resilient', 'VERB'), ('once', 'ADV'), ('it', 'PRON'), ('enters', 'VERB'), ('the', 'DET'), ('lungs', 'NOUN'), (',', '.'), ('with', 'ADP'), ('even', 'ADV'), ('brief', 'ADJ'), ('exposures', 'NOUN'), ('to', 'ADP'), ('it', 'PRON'), ('causing', 'VERB'), ('symptoms', 'NOUN'), ('that', 'ADP'), ('*t*-1', 'PRON'), ('show', 'VERB'), ('up', 'PRT'), ('decades', 'VERB'), ('later', 'ADV'), (',', '.'), ('researchers', 'PRON'), ('said', 'VERB'), ('0', 'NUM'), ('*t*-2', 'NOUN'), ('.', '.')], -283.04594784458953)
---------
Now test the EM algorithm works
percent correct data
0.001
starting accuracy
0.08438053310735864
hmm_tagger.py:169: RuntimeWarning: divide by zero encountered in log2
  total_logp += np.log2(np.exp(logp))
hmm_tagger.py:157: RuntimeWarning: invalid value encountered in double_scalars
  while last_perplexity == -1 or last_perplexity -curr_perplexity > 0.5:
ending accuracy
0.1828448697545281
percent correct data
0.01
starting accuracy
0.09247135701933874
ending accuracy
0.5236860054151251
percent correct data
0.1
starting accuracy
0.1748393030609925
ending accuracy
0.8548620727666053

```
### Credits

This task is designed and mentored by my instructor [Julian Brooke](https://linguistics.ubc.ca/person/julian-brooke/) in my class COLX 563 (Unsupervised Learning) in UBC.