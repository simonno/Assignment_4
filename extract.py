import codecs
import pickle

import spacy
from scipy import sparse
from sys import argv

nlp = spacy.load('en_core_web_sm')


def read_logistic_regression_model(file_name):
    with open(file_name, 'rb') as model_file:
        (clf, vec) = pickle.load(model_file)
    return clf, vec


def create_features_numeric_format(features, features_map):
    col = list()
    row = list()
    data = list()
    for i in range(len(features)):
        word_features = features[i]
        for feature_key, feature_value in word_features.items():
            feature = '{0}={1}'.format(feature_key, feature_value)
            if feature in features_map:
                row.append(i)
                col.append(features_map.index(feature))
                data.append(1)
    return sparse.csr_matrix((data, (row, col)), shape=(len(features), len(features_map)))


def get_prediction(features, clf, feature_map):
    numeric_format_matrix = create_features_numeric_format(features, feature_map)
    predict = clf.predict(numeric_format_matrix)
    return predict


def entity_features(entity):
    return {'type': entity.root.ent_type_, 'root_text': entity.root.text,
            'root_dep': entity.root.dep_, 'root_head_text': entity.root.head.text}


def get_features(first_ent, second_ent):
    first_ent_features = entity_features(first_ent)
    second_ent_features = entity_features(second_ent)
    features = {'1_' + k: v for k, v in first_ent_features.items()}
    features.update({'2_' + k: v for k, v in second_ent_features.items()})
    return features


def read_lines(file_name):
    for line in codecs.open(file_name, encoding="utf8"):
        sent_id, sent = line.strip().split("\t")
        sent = sent.replace("-LRB-", "(")
        sent = sent.replace("-RRB-", ")")
        sent_id = int(sent_id.split('sent')[1])
        yield sent_id, sent


def write_predictions(file_name, annotations, predictions):
    with open(file_name, 'w') as output_file:
        for annotation, prediction in zip(annotations, predictions):
            if prediction == 'Work_For':
                output_file.write(
                    'sent{0}\t{1}\t{2}\t{3}\t{4}\n'.format(annotation[0], annotation[1], prediction, annotation[2],
                                                           annotation[3]))


def main(corpus_file_name, output_predictions_file):
    clf, vec = read_logistic_regression_model('C:/Users/DELL/PycharmProjects/NLP/Assignment_4/model_file')
    features_map = vec.get_feature_names()
    annotations = list()
    features_vectors = list()
    for sent_id, sent_str in read_lines(corpus_file_name):
        sent = nlp(sent_str)
        print("#id:", sent_id)
        print("#text:", sent.text)
        print()
        entities = sent.ents
        
        for i, first_ent in enumerate(entities):
            for second_ent in entities[:i] + entities[i + 1:]:
                annotations.append((sent_id, str(first_ent), str(second_ent), sent))
                features_vectors.append(get_features(first_ent, second_ent))

    predictions = get_prediction(features_vectors, clf, features_map)
    write_predictions(output_predictions_file, annotations, predictions)


if __name__ == '__main__':
    main(argv[1], argv[2])
