import sys
import gc

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

from datasets import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier

# Load training and testing set
test = pd.read_csv('./llm-detect-ai-generated-text/test_essays.csv')
sub = pd.read_csv('./llm-detect-ai-generated-text/sample_submission.csv')
org_train = pd.read_csv('./llm-detect-ai-generated-text/train_essays.csv')
train = pd.read_csv("./llm-detect-ai-generated-text/train_v2_drcat_02.csv", sep=',')

train = train.drop_duplicates(subset=['text'])
train.reset_index(drop=True, inplace=True)

# Define hyperparameter
LOWERCASE = False
VOCAB_SIZE = 30522

# Utility functions

def train_corp_iter():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]

def dummy(text):
    return text

# Creating Byte-Pair Encoding tokenizer
raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

# Adding normalization and pre_tokenizer
raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else [])
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

# Adding special tokens and creating trainer instance
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)

# Creating huggingface dataset object
dataset = Dataset.from_pandas(test[['text']])

raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)
tokenized_texts_test = []

for text in tqdm(test['text'].tolist()):
    tokenized_texts_test.append(tokenizer.tokenize(text))

tokenized_texts_train = []

for text in tqdm(train['text'].tolist()):
    tokenized_texts_train.append(tokenizer.tokenize(text))

# Vectorize the tokenized text
vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, analyzer = 'word',
    tokenizer = dummy,
    preprocessor = dummy,
    token_pattern = None, strip_accents='unicode'
                            )

vectorizer.fit(tokenized_texts_test)

# Getting vocab
vocab = vectorizer.vocabulary_
vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, vocabulary=vocab,
                            analyzer = 'word',
                            tokenizer = dummy,
                            preprocessor = dummy,
                            token_pattern = None, strip_accents='unicode'
                            )

tf_train = vectorizer.fit_transform(tokenized_texts_train)
tf_test = vectorizer.transform(tokenized_texts_test)

del vectorizer
gc.collect()

y_train = train['label'].values

# Check if it's in scoring stage
if len(test.text.values) <= 5:
    # if not, just sample submission
    sub.to_csv('submission.csv', index=False)
else:
    # otherwise, run fitting process
    clf = MultinomialNB(alpha=0.02)
    clf2 = MultinomialNB(alpha=0.01)
    sgd_model = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber")
    p6={'n_iter': 1000,'verbose': -1,'learning_rate': 0.005689066836106983, 'colsample_bytree': 0.8915976762048253, 'colsample_bynode': 0.5942203285139224, 'lambda_l1': 7.6277555139102864, 'lambda_l2': 6.6591278779517808, 'min_data_in_leaf' : 156, 'max_depth': 11, 'max_bin': 813}
    lgb=LGBMClassifier(**p6)


    cat=CatBoostClassifier(
        iterations=1000,
        verbose=0,
        l2_leaf_reg=6.6591278779517808,
        learning_rate=0.005689066836106983,
        allow_const_label=True
    )
    weights = [0.3,0.4,0.4,0.4]
    weights = [w/sum(weights) for w in weights]

    ensemble = VotingClassifier(estimators=[('mnb',clf),('sgd', sgd_model),('lgb',lgb), ('cat', cat)],
                                weights=weights, voting='soft', n_jobs=-1)
    ensemble.fit(tf_train, y_train)

    gc.collect()
    final_preds = ensemble.predict_proba(tf_test)[:,1]
    sub['generated'] = final_preds
    sub.to_csv('submission.csv', index=False)