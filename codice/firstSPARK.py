import pandas as pd
from matplotlib import pyplot as plt
import pyspark

from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf




#CREA LA SESSIONE SPARK
spark = SparkSession.builder.getOrCreate()



df = spark.read.csv(
    "temperature_data/*.csv",
    header=True,
    sep=";",
    inferSchema=True
)

df.printSchema()

#CREA UNA VISTA SQL
precipitazioni = df.createOrReplaceGlobalTempView("precipitazioni")



#SOMMA TOTALE PRECIPITAZIONI PER STAZIONE
query1 = """
SELECT ID_STAZ, ROUND(sum(VALORE),2) AS TOT_PRECIPITAZIONI
FROM global_temp.precipitazioni
WHERE VALORE is not null
GROUP BY ID_STAZ
"""

calcolo_precipitazioni = spark.sql(query1)
calcolo_precipitazioni.show()




#DIVISIONE IN FASCE ORARIRE
# -hour estrapola l'ora di un tipo 'timestamp'
notte = df.filter(
    (hour(col("DATARIL"))>= 0) &
    (hour(col("DATARIL")) < 6)
)



media_n = notte.agg(
    avg("VALORE").alias("media_notte")
)
media_notte = media_n.collect()[0]["media_notte"]




mattina = df.filter(
    (hour(col("DATARIL"))>= 6) &
    (hour(col("DATARIL")) < 12)
)

media_m = mattina.agg(
    avg("VALORE").alias("media_mattina")
)
media_mattina = media_m.collect()[0]["media_mattina"]


pomeriggio = df.filter(
    (hour(col("DATARIL"))>= 12) &
    (hour(col("DATARIL")) < 18)
)

media_p = pomeriggio.agg(
    avg("VALORE").alias("media_pomeriggio")
)
media_pomeriggio = media_p.collect()[0]["media_pomeriggio"]

sera = df.filter(
    (hour(col("DATARIL"))>= 18) &
    (hour(col("DATARIL")) < 24)
)

media_s = sera.agg(
    avg("VALORE").alias("media_sera")
)
media_sera = media_s.collect()[0]["media_sera"]




print(media_mattina)

print(f"PRECIPITAZIONE MEDIA MATTINA(06:00 - 11:59): {media_mattina}")
print(f"PRECIPITAZIONE MEDIA POMERIGGIO(12:00 - 17:59): {media_pomeriggio}")
print(f"PRECIPITAZIONE MEDIA SERA(18:00 - 23:59): {media_sera}")
print(f"PRECIPITAZIONE MEDIA NOTTE(00:00 - 05:59): {media_notte}")






# STAZIONE OCN MASSIMO LIVELLO DI PRECIPITAZIONE
query2 = """
SELECT DATA_GIORNO, ID_STAZ, TOT
FROM (
    SELECT
        to_date(DATARIL) AS DATA_GIORNO,
        ID_STAZ,
        SUM(VALORE) AS TOT,
        ROW_NUMBER() OVER (
            PARTITION BY to_date(DATARIL)
            ORDER BY SUM(VALORE) DESC
        ) AS rn
    FROM global_temp.precipitazioni
    GROUP BY to_date(DATARIL), ID_STAZ
)
WHERE rn = 1
"""


stazione_max = spark.sql(query2)
stazione_max.take(5)




#ANDAMENTO GIORNO PER GIORNO PER OGNI SENSORE
df_giornaliero = df.withColumn(
    "DATA_GIORNO", to_date("DATARIL")
).groupBy("DATA_GIORNO", "ID_STAZ")\
.agg(sum("VALORE").alias("PRECIP"))



df_giornaliero_pd = df_giornaliero.toPandas()


pivot = df_giornaliero_pd.pivot(
    index="DATA_GIORNO",
    columns="ID_STAZ",
    values="PRECIP"
)


pivot.plot(figsize=(12,6))
plt.title("Precipitazioni giornaliere per sensore")
plt.xlabel("Giorno")
plt.ylabel("Precipitazioni")
plt.show()


