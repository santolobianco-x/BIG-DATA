import config as cf
import preprocessing as prp
import minhash as mh
import prediction as pre


#CARICAMENTE DATI
train, test =  prp.load_data(cf.spark)


train.show(5, truncate=100)
train.show(5, truncate=100)



#SHINGLING
train, test = prp.create_shingles(train, test)
train.select("popolarity", "title", "shingles").show(2, truncate=False)


#CREAZIONE VOCABOLARIO
vocab, shingle_to_index = prp.create_vocabulary(train)

vocab_size = len(vocab)

print("Numero di shingles: ", vocab_size)

#CREAZIONE VETTORI 
train, test = prp.create_vectors(train, test, shingle_to_index, vocab_size)

train.select(
    "popolarity",
    "title",
    "features"
).show(5, truncate=False)


#MINHASH LSH
model = mh.train_minhash(train)

train_hashed, test_hashed = mh.hash_data(model, train, test)



train_hashed.select(
    "popolarity",
    "title",
    "hashes"
).show(5, truncate=False)

#PREDIZIONE kNN
risultati = pre.predict(model, train_hashed, test_hashed, k=3)
corrette, totale, accuracy = pre.calculate_accuracy(risultati)


#ACCURACY
print("Predizioni corrette:", corrette)
print("Predizioni totali:", totale)
print("Accuracy:", accuracy)