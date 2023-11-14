import os
import re

from nltk.corpus import framenet as fn

PATH = os.path.dirname(os.path.abspath(__file__)) + "/wikipedia/cleaned/"

FILENAMES = [
    "Ada_Lovelace.txt",
    "Alan_Turing.txt",
    "Albert_Einstein.txt",
    "Alonzo_Church.txt",
    "Charles_Babbage.txt",
    "David_Hilbert.txt",
    "Emmy_Noether.txt",
    "Frédéric_Joliot-Curie.txt",
    "Irène_Joliot-Curie.txt",
    "Marie_Curie.txt",
]


def extract_relations(text):
    relations = []

    for sentence in re.split(r"[.!?()]", text):
        for word in sentence.split():
            frames = fn.frames_by_lemma(word)

            for frame in frames:
                for role, role_info in frame["FE"].items():
                    if role_info["coreType"] == "Core":
                        relation = (frame["name"], role, word)
                        relations.append(relation)
    return relations


def nltk_test():
    for nf in FILENAMES:
        with open(PATH + nf, "r", encoding="utf-8") as f:
            text = f.read()

            relations = extract_relations(text)

            # Print the extracted relations
            for relation in relations:
                print("Frame:", relation[0])
                print("Role:", relation[1])
                print("Entity:", relation[2])
                print()


def main():
    nltk_test()


if __name__ == "__main__":
    main()
