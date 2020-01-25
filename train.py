import codecs
import pickle
from collections import defaultdict
from random import random

import spacy
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sys import argv

from features import get_features

nlp = spacy.load('en_core_web_sm')


def write_feature_map(feature_map_file, features_map):
    with open(feature_map_file, "w") as file:
        for index in range(len(features_map)):
            file.write('{0} : {1} \n'.format(features_map[index], index))


def read_lines(file_name):
    for line in codecs.open(file_name, encoding="utf8"):
        sent_id, sent = line.strip().split("\t")
        sent = sent.replace("-LRB-", "(")
        sent = sent.replace("-RRB-", ")")
        for e in ['province', 'the']:
            if e in sent:
                sent = sent.replace(e, '').strip()
        sent_id = int(sent_id.split('sent')[1])
        yield sent_id, sent


def write_logistic_regression_model(file_name, model):
    # print to model_file
    with open(file_name, "wb") as file:
        pickle.dump(model, file, fix_imports=True)


def create_model(all_features, labels):
    vec = DictVectorizer()

    transform_of_features = vec.fit_transform(all_features)
    features_map = vec.get_feature_names()

    clf = LogisticRegression(tol=0.001, solver='saga', multi_class='multinomial').fit(transform_of_features, labels)

    return transform_of_features, features_map, (clf, vec)


def parse_annotation(annotations_file_name):
    annotations = defaultdict(dict)
    with open(annotations_file_name, 'r') as annotations_file:
        for line in annotations_file:
            sent_id, first_ent, rel, second_ent, sentence = line.strip().split("\t")
            sent_id = int(sent_id.split('sent')[1])
            if random() < 0.2:
                rel = "Noise"
            annotations[sent_id][(first_ent, second_ent)] = rel
    return annotations


def main(corpus_file_name, annotations_file_name):
    vectors_features_list = list()
    labels = list()
    annotations = parse_annotation(annotations_file_name)
    for sent_id, sent_str in read_lines(corpus_file_name):
        sent = nlp(sent_str)
        print("#id:", sent_id)
        print("#text:", sent.text)
        print()
        entities = sent.ents
        for i, first_ent in enumerate(entities):
            for second_ent in entities[:i] + entities[i + 1:]:
                pair_ent = (str(first_ent), str(second_ent))
                if pair_ent in annotations[sent_id].keys():
                    rel = annotations[sent_id][pair_ent]
                    vectors_features_list.append(get_features(first_ent, second_ent))
                    labels.append(rel)

    transform_of_features, features_map, model = create_model(vectors_features_list, labels)
    write_feature_map('C:/Users/DELL/PycharmProjects/NLP/Assignment_4/feature_map.txt', features_map)
    write_logistic_regression_model('C:/Users/DELL/PycharmProjects/NLP/Assignment_4/model_file', model)


if __name__ == '__main__':
    main(argv[1], argv[2])
