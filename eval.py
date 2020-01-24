from collections import defaultdict

from sys import argv


def read_lines(file_name):
    lines = set()
    with open(file_name, 'r') as file:
        for line in file:
            if line.find('Work_For') > -1:
                lines.add(line)
    return lines


def parse_annotation(annotations_file_name):
    annotations = defaultdict(dict)
    with open(annotations_file_name, 'r') as annotations_file:
        for line in annotations_file:
            sent_id, first_ent, rel, second_ent, sentence = line.strip().split("\t")
            sent_id = int(sent_id.split('sent')[1])
            annotations[sent_id][(first_ent, second_ent)] = rel
    return annotations


def compare_accuracy(gold, pred):
    correct = 0
    total = 0
    for sent_id, relations in pred.items():
        for entities, rel in relations.items():
            total += 1
            if sent_id in gold.keys():
                if entities in gold[sent_id]:
                    if rel == gold[sent_id][entities]:
                        correct += 1
    return correct / total


def main(gold_data, predicted_data):
    gold_annotations = parse_annotation(gold_data)
    predicted_annotations = parse_annotation(predicted_data)
    gold_lines = read_lines(gold_data)
    predicted_lines = read_lines(predicted_data)

    acc = compare_accuracy(gold_annotations, predicted_annotations)
    print("Accuracy:", acc)

    prec = len(gold_lines.intersection(predicted_lines)) / float(len(gold_lines))
    rec = len(gold_lines.intersection(predicted_lines)) / float(len(predicted_lines))
    f_measure = (prec * rec) / (prec + rec)
    print("Prec:%s Rec:%s F-measure:%s F1:%s" % (prec, rec, f_measure, 2 * f_measure))


if __name__ == '__main__':
    main(argv[1], argv[2])
