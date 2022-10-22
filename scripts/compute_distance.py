from itertools import product
from collections import defaultdict
from unittest import result
import panphon
import panphon.distance
import numpy as np
import pandas as pd
import seaborn as sn

# dst = panphon.distance.Distance()
# print(dst.feature_edit_distance('warugo', 'warude'))
# print(dst.feature_edit_distance('warugo', 'waruko'))

class WordDistanceComputer:
    """
    TODO
    """

    def __init__(self) :
        self.distance_computer = panphon.distance.Distance()
    
    def cognates_distance(self,
                        first_cognates: str,
                        second_cognates: str) -> float:
        """
        TODO
        """
        cognates_combinations = product(first_cognates.split(", "), second_cognates.split(", "))
        distances = []
        for first_word, second_word in cognates_combinations:
            distances.append(self.distance_computer.feature_edit_distance(first_word, second_word))
        return int(np.mean(distances))
    
    def row_distances(self, df) -> dict:
        """
        TODO
        """
        results = defaultdict(lambda: defaultdict(int))
        for row in df.itertuples(index=False):
            row = row._asdict()
            for dialect_1, dialect_2 in product(row.keys(), repeat=2):
                if dialect_1 == dialect_2:
                    results[dialect_1][dialect_2] += 0.0
                    continue
                distance = self.cognates_distance(row[dialect_1], row[dialect_2])
                results[dialect_1][dialect_2] += distance
        return {dialect_1 : dict(results[dialect_1]) for dialect_1 in results}

wc = WordDistanceComputer()
df = pd.read_csv("data/words_phonemized.csv")
df = df.rename(columns={"FulfuldeNigerNiamey" : "FN",
                        "FulfuldeBurkinaFaso" : "FBF",
                        "PulaarFuutaTooro" : "PFT",
                        "PulaarFirduKantora" : "PFK",
                        "PularFuutaJalon" : "PFJ",
                        "FulfuldeMaasina" : "FM",
                        "FulfuldeAdamawa" : "FA",
                        "FulfuldeGombe" : "FG"})
results = wc.row_distances(df)
matrix = pd.DataFrame(results).T
sn.clustermap(matrix, annot=True, cmap="mako", standard_scale=None).figure.savefig("plots/clustering.png", dpi=600)

# distance1 = wc.words_distance('warugo, waːrude', 'warugo, warːude')
# print(ft.word_fts("waːᵐb"))
# print(distance1)