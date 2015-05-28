from nltk.parse import stanford

import os

os.environ['JAVA_HOME'] = 'C://Program Files//Java//jdk1.8.0_45//bin'
os.environ['STANFORD_PARSER'] = 'D://stanford-parser-full-2015-04-20'
os.environ['STANFORD_MODELS'] = 'D://stanford-parser-full-2015-04-20'

parser = stanford.StanfordParser(model_path='D://stanford-parser-full-2015-04-20//englishPCFG.ser.gz')

sentences_iter = parser.parse_sents([['Hello', ',', 'world', '!']])

sentences = []

for line in sentences_iter:
    for sentence in line:
        sentences.append(str(sentence))

