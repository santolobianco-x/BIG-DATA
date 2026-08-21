import shingling as sh
import minhashing as mn
import locality_sensitive_hashing as lsh


#SHINGLING
a = "Il ragazzo pazzo mangia una mela rossa nel parco."
b = "Nel parco, il giovane sta camminando e mangiando una mela rossa."
c = "La macchina veloce sfreccia lungo l'autostrada al tramonto."

k = 2
print("a BEFORE SPLITTING INTO SHINGLE")
print(a)

a = sh.shingle(a,k)
b = sh.shingle(b,k)
c = sh.shingle(c,k)

print("a AFTER SPLITTING INTO SHINGLE")
print(a)



vocab = sh.get_vocab(a,b,c)

a_1hot = sh.get_1hot(a,vocab)
b_1hot = sh.get_1hot(b,vocab)
c_1hot = sh.get_1hot(c,vocab)

print("a AFTER APPLING 1HOT")
print(a_1hot)


#MIN-HASHING

minhash_func = mn.build_minhash_func(len(vocab), 20)


a_sig = mn.create_hash(a_1hot, minhash_func, vocab)
b_sig = mn.create_hash(b_1hot, minhash_func, vocab)
c_sig = mn.create_hash(c_1hot, minhash_func, vocab)

print("SIGNATURE OF a")
print(a_sig)

print()

s_ab = mn.minhash_similarity(a_sig, b_sig)
s_ac = mn.minhash_similarity(a_sig, c_sig)
s_bc = mn.minhash_similarity(b_sig, c_sig)

print(f"JACCARD SIMILARITY(a_sig b_sig): {s_ab}")
print(f"JACCARD SIMILARITY(a_sig c_sig): {s_ac}")
print(f"JACCARD SIMILARITY(b_sig c_sig): {s_bc}")


# LOCALITY SENSITIVE HASHING(LSH)

# NUMERO DI BANDE DELLA SIGNATURE
b = 10

band_a = lsh.split_vector(a_sig, b)
band_b = lsh.split_vector(b_sig, b)
band_c = lsh.split_vector(c_sig, b)



r = int(len(a_sig)/b)


print("PROBABILITY THAT TWO DOCUMENTS BECOME A CANDIDATE PAIR (AT LEAST ONE BAND IS IDENTICAL)")
p_ab = lsh.probability(s_ab, r, b)
p_ac = lsh.probability(s_ac, r, b)
p_bc = lsh.probability(s_bc, r, b)

print(f"A - B: {p_ab}")
print(f"A - C: {p_ac}")
print(f"B - C: {p_bc}")


print()
print()
print("CANDIDATE PAIRS FOUND IN THE SAME BUCKET")
print("A - B", lsh.is_candidate(band_a, band_b))
print("A - C", lsh.is_candidate(band_a, band_c))
print("B - C", lsh.is_candidate(band_b, band_c))



lsh.plot_lsh_probability()