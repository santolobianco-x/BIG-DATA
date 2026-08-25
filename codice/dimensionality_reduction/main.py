import os
os.environ["PYSPARK_PYTHON"] = "/opt/anaconda3/bin/python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/opt/anaconda3/bin/python"

import numpy as np

from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf

from preprocessing import load_data
from pca import calculate_pca, plot_pca
from kmeans import calculate_kmeans, calculate_silhouette
from CUR import cur_decomposition


conf = SparkConf().set("spark.ui.port", "4050")

sc = SparkContext(conf=conf)

spark = SparkSession.builder.getOrCreate()

print("Master:", spark.sparkContext.master)




vectors, labels, feature_names = load_data(spark)

pca_data = calculate_pca(vectors, k=2)

plot_pca(pca_data)




model_original, predictions_original = calculate_kmeans(
    vectors,
    k=2
)

silhouette_original = calculate_silhouette(
    predictions_original
)

print("\nSilhouette dati originali:")
print(silhouette_original)




model_pca, predictions_pca = calculate_kmeans(pca_data,k=2)

silhouette_pca = calculate_silhouette(predictions_pca)

print("\nSilhouette dati PCA:")
print(silhouette_pca)


print("\n==============================")
print("CONFRONTO")
print("==============================")

print(f"Dati originali: {silhouette_original:.4f}")

print(f"Dati PCA:       {silhouette_pca:.4f}")


if silhouette_original > silhouette_pca:
    print("\nIl clustering sui dati originali è migliore.")
elif silhouette_pca > silhouette_original:
    print("\nIl clustering sui dati PCA è migliore.")
else:
    print("\nI due clustering hanno la stessa qualità.")




matrix = np.array(vectors.map(lambda v: v.toArray()).collect())
C, U, R = cur_decomposition(matrix, rank=5)



print("\nDimensioni delle matrici CUR:") 
print("Matrice originale A:", matrix.shape) 
print("Matrice C:", C.shape) 
print("Matrice U:", U.shape) 
print("Matrice R:", R.shape)



C_df = spark.createDataFrame( C.tolist() ) 
U_df = spark.createDataFrame( U.tolist() ) 
R_df = spark.createDataFrame( R.tolist() )



print("\nMatrice C:") 
C_df.show( truncate=False ) 

print("\nMatrice U:") 
U_df.show( truncate=False ) 

print("\nMatrice R:") 
R_df.show( truncate=False )



spark.stop()