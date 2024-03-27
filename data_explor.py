
import os
import pyspark
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

os.environ["JAVA_HOME"] = 'C:\Program Files\Java\jre-1.8'
spark = SparkSession.builder.appName('musikita').getOrCreate()

song_charts = spark.read.csv(r'./data/charts.csv', header=True, multiLine=True)
song_lyrics = spark.read.csv(r'./data/song_lyrics.csv', header=True, multiLine=True, escape='"')

song_charts = song_charts.withColumn(
    "arts",
    F.regexp_replace(F.col("artists"), "[\['\].]", "")
)

song_charts = song_charts.drop('artists').drop('artist_genres')

song_charts = song_charts.withColumnRenamed(
    "name", 
    "title"
)

song_charts = song_charts.withColumnRenamed(
    "arts", 
    "artist"
)

song_lyrics = song_lyrics.filter(song_lyrics.lyrics != 'NULL').filter(song_lyrics.year > 1000)
song_lyrics = song_lyrics.drop('features')

merged_data = song_charts.join(song_lyrics, ['title', 'artist'], 'inner')

merged_data = merged_data.withColumn(
    "year",
    F.substring(F.col("date"), 1, 4)
)

merged_data = merged_data.withColumn(
    "month",
    F.substring(F.col("date"), 6, 2)
)

merged_data = merged_data.withColumn(
    "day",
    F.substring(F.col("date"), 9, 2)
)

merged_data = merged_data.drop('date')

country_list = merged_data.select('country').distinct().collect()

country_names = []
for country in country_list:
    country_names.append(country['country'])



#for country in country_names:
#    country_df = merged_data.filter(merged_data.country == country)
#    country_df.toPandas().to_csv(f'./data/by_country/{country}',sep=',',index=False)

#merged_data.toPandas().to_csv(r'./data/merged_data.csv',sep=',',index=False)