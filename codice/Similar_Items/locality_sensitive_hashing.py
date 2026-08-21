import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def split_vector(signature, b):

    assert len(signature) %b == 0

    r = int(len(signature)/b)

    subvecs = []

    for i in range(0, len(signature), r):
        subvecs.append(signature[i: i+r])
    return subvecs


#MOSTRARE I CANDIDATI UGUALI 
#for a_rows, b_rows in zip(band_a, band_b):
#    if a_rows == b_rows:
#        print(f"Candidate pair: {a_rows} == {b_rows}")


#FORMULA PER TROVARE LA COPPIA SE LA SIMILARITA' È PARI AD s
def probability(s, r, b):
    return 1 - (1 - s**r)**b



def is_candidate(band1, band2):
    for x,y in zip(band1,band2):
        if x == y:
            return True
    return False


# CALCOLO TEORICO DELLA PROBABILITA' PER TUTTO IL RANGE DI SIMILARITA'
def plot_lsh_probability():
    results = pd.DataFrame({
        's': [],
        'P': [],
        'r,b': []
    })
    
    #QUI SI DISEGNANO SOLO LE CURVE TEORICHE INFATTI SI UTILIZZA TUTTO L'INTERVALLO
    for s in np.arange(0.01, 1, 0.01):
        total = 100
        for b in [100, 50, 25, 20, 10, 5, 4, 2, 1]:
            r = int(total/b)
            P = probability(s, r, b)
            new_row = pd.DataFrame({
                's': [s],
                'P': [P],
                'r,b': [f"{r},{b}"]
            })
            results = pd.concat([results,new_row], ignore_index=True)

    sns.lineplot(data=results, x='s', y='P', hue='r,b')
    plt.show()