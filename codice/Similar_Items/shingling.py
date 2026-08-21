def shingle(text: str, k: int):
    shingle_set = []
    for i in range(len(text) - k+1):
        shingle_set.append(text[i: i+k])
    return set(shingle_set) 
    #RESTITUISCE UN INSIEME PERCHE' DEVONO ESSERE RIMOSSI I DUPLICATI


def get_vocab(*sets: set):
    vocab = set().union(*sets)
    return vocab


def get_1hot(current: set, vocab: set):
    current_1hot = [1 if x in current else 0 for x in vocab]
    #CREA UN ARRAY DEL TIPO [1,0,1,0...,0] verificando se lo shingle è presente nel vocabolario
    return current_1hot
