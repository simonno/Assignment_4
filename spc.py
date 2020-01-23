import codecs

import spacy
from sys import argv

nlp = spacy.load('en_core_web_sm')


def read_lines(file_name):
    for line in codecs.open(file_name, encoding="utf8"):
        sent_id, sent = line.strip().split("\t")
        sent = sent.replace("-LRB-", "(")
        sent = sent.replace("-RRB-", ")")
        yield sent_id, sent




def main(corpus_file_name):
    for sent_id, sent_str in read_lines(corpus_file_name):
        sent = nlp(sent_str)
        print("#id:", sent_id)
        print("#text:", sent.text)
        # for word in sent:
        #     head_id = str(word.head.i + 1)  # we want ids to be 1 based
        #     if word == word.head:  # and the ROOT to be 0.
        #         assert (word.dep_ == "ROOT"), word.dep_
        #         head_id = "0"  # root
        #     print("\t".join(
        #         [str(word.i + 1), word.text, word.lemma_, word.tag_, word.pos_, head_id, word.dep_, word.ent_iob_,
        #          word.ent_type_]))
        print()
        # print "#", Noun Chunks:
        for np in sent.noun_chunks:
            print(np.text, np.root.text, np.root.dep_, np.root.head.text)
        # print "#", named entities:
        for ne in sent.ents:
            print(ne.text, ne.root.ent_type_, ne.root.text, ne.root.dep_, ne.root.head.text)


if __name__ == '__main__':
    main(argv[1])
