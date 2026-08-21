from random import shuffle

# CREA ORDINI CASUALI DI RIGHE
def create_hash_func(size: int):
    hash_ex = list(range(1, size+1))
    shuffle(hash_ex)
    return hash_ex


# CREA PIU' HASH UTILIZZANDO PERMUTAZIONI DIVERSE
def build_minhash_func(vocab_size: int, nbits: int):
    hashes = []
    for _ in range(nbits):
        hashes.append(create_hash_func(vocab_size))
    return hashes





def create_hash(vector: list, minhash_func: list, vocab: list):
    signature = []

    for func in minhash_func:
        # cerco la prima riga con valore 1 nella permutazione
        for i in range(1, len(vocab) + 1):

            idx = func.index(i)

            if vector[idx] == 1:
                signature.append(idx)
                break

    return signature

# QUINDI SI PRENDE IL PRIMO INDICE DELLA RIGA CHE HA VALORE = 1
# INDICI: [0,1,2,3,4] VALORI [1,0,0,1,0]
# PRIMA PERMUTAZIONE [4,1,2,3,0] --> PRIMO INDICE [3]
# SECONDA PERMUTAZIONE [2,4,0,3,1] --> PRIMO INDICE [0]
# Signature [3,0]



#SBAGLIATO NON SI POSSONO UTILIZZARE I SET PER LE FIRME DATO CHE L'ORDINE È IMPORTANTE
# E CON I SET SI RIORDINANO GLI INDICI(SIGNATURE DIVERSA)
#def jaccard(a, b):
#    a = set(a)
#    b = set(b)
#    return len(a.intersection(b)) / len(a.union(b))


def minhash_similarity(sig1, sig2):
    matches = 0

    for x,y in zip(sig1, sig2):
        if(x == y):
            matches += 1
    return matches / len(sig1)






