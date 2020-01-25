from collections import defaultdict

from sys import argv


def parse_annotation(annotations_file_name, rel_type):
    annotations = defaultdict(dict)
    with open(annotations_file_name, 'r') as annotations_file:
        for line in annotations_file:
            sent_id, first_ent, rel, second_ent, sentence = line.strip().split("\t")
            if rel == rel_type:
                sent_id = int(sent_id.split('sent')[1])
                annotations[sent_id][(first_ent.strip(' .'), second_ent.strip(' .'))] = rel
    return annotations


def calc(first, second):
    correct = 0
    total = 0
    for sent_id, relations in second.items():
        for entities, rel in relations.items():
            total += 1
            if sent_id in first.keys():
                if entities in first[sent_id]:
                    if rel == first[sent_id][entities]:
                        correct += 1
    return 0 if total == 0 else correct / total


def main(gold_data, predicted_data):
    gold_annotations = parse_annotation(gold_data, 'Work_For')
    predicted_annotations = parse_annotation(predicted_data, 'Work_For')

    prec = calc(gold_annotations, predicted_annotations)
    rec = calc(predicted_annotations, gold_annotations)

    f_measure = 0 if prec == 0 and rec == 0 else (prec * rec) / (prec + rec)
    print("Prec:%s\nRec:%s\nF-measure:%s\nF1:%s" % (prec, rec, f_measure, 2 * f_measure))


if __name__ == '__main__':
    main(argv[1], argv[2])
