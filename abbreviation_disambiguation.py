import csv
import random
import argparse
from time import time
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
import nltk
from nltk import word_tokenize
#nltk.download('averaged_perceptron_tagger')
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from bert_serving.client import BertClient
import tensorflow as tf
from scipy.sparse import coo_matrix, hstack
from sklearn.neural_network import MLPClassifier

"""
This script trains and evaluates a baseline feature set and model
for the clinical abbreviation/acronym disambiguation task.
The model is the NLTK NaiveBayesClassifier. The features used are:
    * The abbreviation or acronym to be disambiguated
    * Bag of words using the 2000 most frequent words in the training data.

This script should serve as a base, which you can modify to implement your own
features, models, evaluation functions, etc. Specifically, you'll likely want
to modify the main() and get_features() functions.

Questions and bugs should be sent to Jake Vasilakes (vasil024@umn.edu).
"""

# So this script produces the same result every time.
# ABSOLUTELY DO NOT CHANGE THIS!
random.seed(42)


def main(infile, outfile):
    """
    The driver function. Performs the following steps:
      * Reads the data in infile and separates the labels from the data.
      * Splits the data into training and testing folds using
        5-fold cross validation.
      * Extracts features from the data in each fold.
      * Trains and evaluates Naive Bayes' models on each fold.
      * Prints the precision, recall, and F1 score for each fold, as well as
        the averages across the folds.

    :param str infile: The path to AnonymizedClinicalAbbreviationsAndAcronymsDataSet.txt  # noqa
    :param str outfile: Where to save the results of the evaluation.
    """
    data, labels = read_abbreviation_dataset(infile, shuffle=True, n=None)
    precs = []
    recs = []
    f1s = []
    fold = 1
    print(f"Running training and evaluation on {len(data)} examples.")
    for test_start, test_end in cross_validation_folds(5, len(data)):
        print(f"Fold: {fold}  ", end='', flush=True)
        fold += 1
        test_data = data[test_start:test_end]
        test_labels = labels[test_start:test_end]
        train_data = data[:test_start] + data[test_end:]
        train_labels = labels[:test_start] + labels[test_end:]  

        train_sf_feats, test_sf_feats = get_short_form_features(train_data,test_data)  
        train_sf_feats = coo_matrix(train_sf_feats)
        test_sf_feats = coo_matrix(test_sf_feats)


        # Change get_features() to implement your own feature functions.
#        train_feats, test_feats = get_features(train_data, test_data)
#        train_feats, test_feats = get_tf_idf_features(train_data, test_data)        
#        train_feats, test_feats = get_pos_features(train_data, test_data)
        train_feats, test_feats = get_bert_embedding_features(train_data, test_data)
        
        train_feats = hstack((train_feats,train_sf_feats))
        test_feats = hstack((test_feats,test_sf_feats))
        
        print("training...", end='', flush=True)
        start = time()
        
#        train_examples = zip(train_feats, train_labels)
        # Change this line to try different models.
#        trained_classifier = nltk.NaiveBayesClassifier.train(train_examples)
        DT_model = DecisionTreeClassifier()
        trained_classifier = DT_model.fit(train_feats,train_labels)
        
#        SVM_model = svm.SVC()
#        trained_classifier = SVM_model.fit(train_feats,train_labels)
        
#        trained_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                            hidden_layer_sizes=(15,), random_state=1)
#        trained_classifier.fit(train_feats, train_labels)                
                
                        
        
        end = time()
        train_time = end - start
        print(f"{train_time:.1f} ", end='', flush=True)
        print("evaluating...", end='', flush=True)
        start = time()
#        predictions = trained_classifier.classify_many(test_feats)
        predictions = trained_classifier.predict(test_feats)
        end = time()
        prec, rec, f1 = evaluate(predictions, test_labels)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        eval_time = end - start
        print(f"{eval_time:.1f} ", end='', flush=True)
        print("done\n")

    summary = results_summary(precs, recs, f1s)
    with open(outfile, 'w') as outF:
        outF.write(summary)



def read_abbreviation_dataset(infile, shuffle=True, n=None):
    """
    DO NOT MODIFY THIS FUNCTION!

    Reads AnonymizedClinicalAbbreviationsAndAcronymsDataSet.txt and
    separates it into data and labels.

    :param str infile: AnonymizedClinicalAbbreviationsAndAcronymsDataSet.txt
    :param bool shuffle: (Default True) If True, randomly shuffle the order
                         of the examples.
    :param int n: (Default None) If an integer is specified, return that
                  many examples, after shuffling (if shuffle is True).
    :returns: data and labels
    :rtype: tuple(list, list)
    """
    data = []
    labels = []
    with open(infile, 'r', errors="ignore") as inF:
        reader = csv.reader(inF, delimiter='|', quoting=csv.QUOTE_NONE)
        for (i, line) in enumerate(reader):
            # data: [short_form, short_form_in_sentence,
            #        start_pos, end_pos, section, sample]
            data.append([line[0]] + line[2:])
            labels.append(line[1])
    if shuffle is True:
        shuffled = random.sample(list(zip(data, labels)), k=len(data))
        data = [elem[0] for elem in shuffled]
        labels = [elem[1] for elem in shuffled]
    if n is not None:
        data = data[:n]
        labels = labels[:n]
    return data, labels


#New Features   
def get_short_form_features(train_data, test_data):
    """
    Feature representation of the short form we are trying to disambiguate.

    :param list train_data: List of raw training data.
    :param list test_data: List of raw testing data.
    :returns: Features representing this short form for train_data and test_data
    :rtype: list
    """
    train_sf = [train_ex[0] for train_ex in train_data]
    test_sf = [test_ex[0] for test_ex in test_data]
    sf_feat = ([ ], [])
    for (i, example_set) in enumerate([train_sf, test_sf]):
        for sf in example_set:
            if sf == 'FISH':
                sfv = [1]
            elif sf =='MR':
                sfv = [2]
            elif sf == 'IT':
                sfv = [3]
            elif sf == 'US':
                sfv = [4]
            elif sf == 'OR':
                sfv = [5]
            elif sf == 'MOM':
                sfv = [6]
            elif sfv == 'MS':
                sfv = [7]
            sf_feat[i].append(sfv)
    train_sfv, test_sfv = sf_feat
    
    return train_sfv, test_sfv 
    
def get_tf_idf_features(train_data, test_data):
    """
    Applying TF-IDF feature function to the data
    in the train and test splits.

    :param list examples: List of raw example data.
    :returns: Features for each example.
    :rtype: list
    """     
    train_text = [train_ex[5] for train_ex in train_data]
    test_text = [test_ex[5] for test_ex in test_data]
    tfidf_vectorizer = TfidfVectorizer()
    train_tfidf_vectors = tfidf_vectorizer.fit_transform(train_text)
    test_tfidf_vectors = tfidf_vectorizer.transform(test_text)
    
    
    
    return  train_tfidf_vectors , test_tfidf_vectors

def get_pos_features(train_data, test_data):
    """
    Applying POS tagging feature function to the data
    in the train and test splits.

    :param list examples: List of raw example data.
    :returns: Features for each example.
    :rtype: list
    """     
    train_tokens = [word_tokenize(train_ex[5]) for train_ex in train_data]
    test_tokens = [word_tokenize(test_ex[5]) for test_ex in test_data]
    train_pos_features = []
    test_pos_features = [] 
    
    for sentence in train_tokens:
        POS_features = dict(nltk.pos_tag(sentence))
        train_pos_features.append(POS_features)
            
    for sentence in test_tokens:
        POS_features = dict(nltk.pos_tag(sentence))
        test_pos_features.append(POS_features)
         
    vectoriser = DictVectorizer(sparse=False)
    train_pos_vectors = vectoriser.fit_transform(train_pos_features)  
    test_pos_vectors = vectoriser.transform(test_pos_features)  


    return train_pos_vectors, test_pos_vectors

def get_bert_embedding_features(train_data, test_data):
    """
    Applying Biobert feature function to the data
    in the train and test splits.

    :param list examples: List of raw example data.
    :returns: Features for each example.
    :rtype: list
    """     
    train_text = [train_ex[5] for train_ex in train_data]
    test_text = [test_ex[5] for test_ex in test_data]
    #need to set up the bert-as-service server in your laptop.
    bc = BertClient()
    train_feat = bc.encode(train_text)
    test_feat = bc.encode(test_text)
    
    return train_feat, test_feat

# I didn't use these baseline feature functions in the end
def get_features(train_data, test_data):
    """
    Wrapper function for applying a number of feature functions to the data
    in the train and test splits.

    :param list examples: List of raw example data.
    :returns: Features for each example.
    :rtype: list
    """
    # Notice we only obtain the vocabularies from the training data.
    # Why shouldn't we get them from the test data too?
    # Avoid overfitting     
    short_form_vocab = {train_ex[0].lower() for train_ex in train_data}
    vocabulary = get_vocabulary(train_data)
    
    # (train_features, test_features)
    feature_sets = ([], [])
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer = tfidf_vectorizer.fit(train_data)
        
    for (i, example_set) in enumerate([train_data, test_data]):
        for example in example_set:
            # Add new features in this loop.
            target_sf = example[0]
            sf_feature = get_short_form_feature(target_sf, short_form_vocab)
            document = example[5]
            bow_feature = get_bag_of_words_features(document, vocabulary)
            feat = {**sf_feature, **bow_feature}
            feature_sets[i].append(feat)
    return feature_sets

def get_vocabulary(examples):
    """
    Get the set of unique words from the text data in examples.
    Used by get_bag_of_words_features().

    :param list examples: List of example data, each of which is
                          a list with plain text data at example[5].
    :returns: Set of unique words.
    :rtype: list
    """
    tokens = [word.lower() for example in examples
              for word in nltk.word_tokenize(example[5])]
    vocabulary = nltk.FreqDist(t for t in tokens)
    freqvoc = vocabulary.most_common(2000)
    return list(freqvoc)


def get_short_form_feature(short_form, all_short_forms):
    """
    Feature representation of the short form we are trying to disambiguate.

    :param str short_form: An abbreviation or acronym, e.g. "AB".
    :param list all_short_forms: The set of all unique abbreviations/acronyms.
    :returns: Feature representing this short form.
    :rtype: dict
    """
    features = {}
    for sf in all_short_forms:
        features[f"short_form({sf})"] = (sf == short_form.lower())
    # Unknown short_form. I.e. we didn't see it in the training set.
    features["UNK"] = (short_form.lower() in all_short_forms)
    return features



def get_bag_of_words_features(document, vocabulary):
    """
    Bag of words representation of the text in the specified document.

    :param str document: Plain text.
    :param list vocabulary: The unique set of words across all documents.
    :returns: Bag of words features for this document.
    :rtype: dict
    """
    document_words = set(nltk.word_tokenize(document.lower()))
    features = {}
    for word in vocabulary:
        features[f"contains({word})"] = (word in document_words)
    return features


def cross_validation_folds(num_folds, data_size):
    """
    DO NOT MODIFY THIS FUNCTION!

    Given the desired number of cross validation folds and a dataset size
    returns a generator of start, end indices for the test data partitions.

    :param int num_folds: An integer >0 specifying the number of folds.
    :param int data_size: The number of examples in the dataset.
    :returns: Generator of start, end index tuples.
    """
    fold_size = data_size // num_folds
    test_start = 0
    test_end = fold_size
    for k in range(num_folds):
        test_start = fold_size * k
        test_end = test_start + fold_size
        if (k + 1) == num_folds:
            test_end = data_size
        yield test_start, test_end


def evaluate(predictions, gold_labels):
    """
    DO NOT MODIFY THIS FUNCTION!

    Given a model's predictions and the gold standard labels,
    compute the precision, recall, and F1 score of the predictions.

    :param list predictions: Predicted labels.
    :param list gold_labels: Gold standard labels.
    :returns: precision, recall, F1
    :rtype: (float, float, float)
    """
    if len(predictions) != len(gold_labels):
        raise ValueError("Number of predictions and gold labels differ.")
    prec, rec, f1, _ = precision_recall_fscore_support(predictions,
                                                       gold_labels,
                                                       average="weighted",
                                                       zero_division=0)
    return prec, rec, f1


def results_summary(precs, recs, f1s):
    """
    Prints a table of precision, recall, and F1 scores for each
    cross validation fold, as well as the average over the folds.

    :param list precs: The precisions for each fold.
    :param list recs: The recalls for each fold.
    :param list f1s: The F1 scores for each fold.
    """
    assert len(precs) == len(recs) == len(f1s)
    n_folds = len(precs)
    folds_strs = [f"Fold {i+1: <3}" for i in range(n_folds)]
    folds_str = ' '.join(f"{fold_str: <10}" for fold_str in folds_strs)
    precs_str = ' '.join(f"{prec: <10.2f}" for prec in precs)
    precs_avg = sum(precs) / len(precs)
    recs_str = ' '.join(f"{rec: <10.2f}" for rec in recs)
    recs_avg = sum(recs) / len(recs)
    f1s_str = ' '.join(f"{f1: <10.2f}" for f1 in f1s)
    f1s_avg = sum(f1s) / len(f1s)
    outstr = ""
    outstr += f"{'': <13} " + folds_str + f"{'Average': <10}\n"
    outstr += f"{'Precision': <15}" + precs_str + f"{precs_avg: <10.2f}\n"
    outstr += f"{'Recall': <15}" + recs_str + f"{recs_avg: <10.2f}\n"
    outstr += f"{'F1 score': <15}" + f1s_str + f"{f1s_avg: <10.2f}\n"
    return outstr


def describe_data(infile):
    """
    Counts and displays the number of senses per short form in the dataset.
    Run this function using the --describe_data command line option.

    :param str infile: AnonymizedClinicalAbbreviationsAndAcronymsDataSet.txt
    """
    sf2senses = defaultdict(list)
    with open(infile, 'r', errors="ignore") as inF:
        reader = csv.reader(inF, delimiter='|', quoting=csv.QUOTE_NONE)
        for (i, line) in enumerate(reader):
            sf = line[0]
            sense = line[1]
            sf2senses[sf].append(sense)

    all_counts = {}
    for (sf, senses) in sf2senses.items():
        sense_counts = Counter(senses)
        all_counts[sf] = sense_counts
    for (sf, counts) in all_counts.items():
        print(sf)
        for (sense, count) in counts.most_common():
            print(f"  {sense}: {count}")
        print()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str,
                        help="""The file containing the input data
                                (i.e. DeidentifiedSymbolDataSet.txt)""")
    parser.add_argument("outfile", type=str,
                        help="Where to write the evaluation result.")
    parser.add_argument("--describe_data", action="store_true", default=False,
                        help="""Compute descriptive statistics about
                                the input dataset before running the
                                training/testing pipeline.""")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    if args.describe_data is True:
        describe_data(args.infile)
    main(args.infile, args.outfile)
