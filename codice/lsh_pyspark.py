import os
os.environ["PYSPARK_PYTHON"] = "/opt/anaconda3/bin/python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/opt/anaconda3/bin/python"


#LIBRERIE PER PYSPARK
import pyspark
from pyspark.mllib import *
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
from pyspark.ml.feature import MinHashLSH, BucketedRandomProjectionLSH
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col



conf = pyspark.SparkConf().set("spark.ui.port", "4050")


sc = pyspark.SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()





dataA = [(0, Vectors.sparse(6, [0, 1, 2], [1.0, 1.0, 1.0])),
         (1, Vectors.sparse(6, [2, 3, 4], [1.0, 1.0, 1.0])),
         (2, Vectors.sparse(6, [0, 2, 4], [1.0, 1.0, 1.0]))]

#       Vectors.sparse(grandezza, [indici], [valori degli indici])
#       il valore che assume il vettore negli indici non indicati è 0.


dfA = spark.createDataFrame(dataA, ["id", "features"])

dataB = [(3, Vectors.sparse(6, [1, 3, 5], [1.0, 1.0, 1.0])),
         (4, Vectors.sparse(6, [2, 3, 4], [1.0, 1.0, 1.0])),
         (5, Vectors.sparse(6, [1, 2, 4], [1.0, 1.0, 1.0]))]
dfB = spark.createDataFrame(dataB, ["id", "features"])



key = Vectors.sparse(6, [1, 3], [1.0, 1.0])



# si utilizza l'algoritmo minhash lsh sulla colonna features
# e si vogliono memorizzare gli hash nella colonna hashes
mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)




#Così si allena il modello che conterrà le informazioni necessarie per applicare le funzioni hash
model = mh.fit(dfA)


print("Il set di dati con hash, "
"in cui i valori con hash sono memorizzati nella colonna 'hash:'")


model.transform(dfA).show()





#Cerca tra A e B le coppie che sono sufficientemente simili secondo la Jaccard distance
print("Cerchiamo le coppie di punti nei due dataframe con una distanza minore di 0.6:")
model.approxSimilarityJoin(dfA, dfB, 0.6, distCol="JaccardDistance")\
    .select(col("datasetA.id").alias("idA"),
            col("datasetB.id").alias("idB"),
            col("JaccardDistance")).show()






print("Cercare approssimativamente in dfA i 2 vicini più prossimi della chiave:")
model.approxNearestNeighbors(dfA, key, 2).show()



dataA = [(0, Vectors.dense([1.0, 1.0]),),
         (1, Vectors.dense([1.0, 1.0]),),
         (2, Vectors.dense([-1.0, -1.0]),),
         (3, Vectors.dense([-1.0, 1.0]),)]


dfA = spark.createDataFrame(dataA, ["id", "features"])


dataB = [(4, Vectors.dense([1.0, 0.0]),),
         (5, Vectors.dense([-1.0, 0.0]),),
         (6, Vectors.dense([0.0, 1.0]),),
         (7, Vectors.dense([0.0, 1.0]),)]

dfB = spark.createDataFrame(dataB, ["id", "features"])


key = Vectors.dense([1.0, 0.0])




# misura la vicinanza mediante la distanza euclidea
brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes",
                                  bucketLength=2.0, numHashTables=3)


model = brp.fit(dfA)

print("Il set di dati con il valore hash:")
model.transform(dfA).show()




print("Join approssimata tra dfA e dfB con una distanzaa Euclidea minore di 1.5:")
model.approxSimilarityJoin(dfA, dfB, 1.5, distCol="EuclideanDistance")\
.select(col("datasetA.id").alias("idA"),
        col("datasetB.id").alias("idB"),
        col("EuclideanDistance")).show()




print("Ricerca approssimata nel dfa per i due nearest neighbor della chiave:")
model.approxNearestNeighbors(dfA, key, 2).show()

