import pandas as pd
import matplotlib.pyplot as plt


from pyspark.mllib.feature import PCA
from pyspark.core.rdd import RDD



def calculate_pca(vectors, k=2):
    pca = PCA(k).fit(vectors)

    pca_data = pca.transform(vectors)

    return pca_data



def plot_pca(pca_data: RDD):


    data = pca_data.collect()

    pca_pd = pd.DataFrame(
        [x.toArray() for x in data],
        columns=["PC1", "PC2"]
    )

    plt.scatter(
        pca_pd["PC1"],
        pca_pd["PC2"]
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA - Prime due componenti")

    plt.show()



