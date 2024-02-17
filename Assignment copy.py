import findspark

findspark.init()

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .getOrCreate()

df=spark.read.parquet('shake.parquet') 
df = df.sample(0.10)
df.show()

df.createOrReplaceTempView("df")

from pyspark.sql.functions import monotonically_increasing_id
from systemds.context import SystemDSContext
import numpy as np
import pandas as pd

def dft_systemds(signal,name):


    with SystemDSContext(8080) as sds:
        size = len(signal)
        signal = sds.from_numpy(signal.to_numpy())
        pi = sds.scalar(3.141592654)

        n = sds.seq(0,size-1)
        k = sds.seq(0,size-1)

        M = (n @ (k.t())) * (2*pi/size)
        
        Xa = M.cos() @ signal
        Xb = M.sin() @ signal

        index = (list(map(lambda x: [x],np.array(range(0, size, 1)))))
        DFT = np.hstack((index,Xa.cbind(Xb).compute()))
        DFT_df = spark.createDataFrame(DFT.tolist(),["id",name+'_sin',name+'_cos'])
        return DFT_df

x0 = df.filter("class = 1").select("X")
y0 = df.filter("class = 1").select("Y")
z0 = df.filter("class = 1").select("Z")
x1 = df.filter("class = 2").select("X")
y1 = df.filter("class = 2").select("Y")
z1 = df.filter("class = 2").select("Z")

from pyspark.sql.functions import lit

df_class_0 = dft_systemds(x0,'x') \
    .join(dft_systemds(y0,'y'), on=['id'], how='inner') \
    .join(dft_systemds(z0,'z'), on=['id'], how='inner') \
    .withColumn('class', lit(1))
    
df_class_1 = dft_systemds(x1,'x') \
    .join(dft_systemds(y1,'y'), on=['id'], how='inner') \
    .join(dft_systemds(z1,'z'), on=['id'], how='inner') \
    .withColumn('class', lit(2))

df_dft = df_class_0.union(df_class_1)

df_dft.show()

from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols=["x_sin", "x_cos", "y_sin", "y_cos", "z_sin", "z_cos"], outputCol="features")

from pyspark.ml.classification import GBTClassifier
classifier = GBTClassifier(labelCol="class", featuresCol="features", maxIter=10)

from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[vectorAssembler, classifier])
model = pipeline.fit(df_dft)
prediction = model.transform(df_dft)
prediction.show()
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
binEval = MulticlassClassificationEvaluator().setMetricName("accuracy") .setPredictionCol("prediction").setLabelCol("class")
    
binEval.evaluate(prediction) 

prediction = prediction.repartition(1)
prediction.write.json('a2_m4.json')
from rklib import zipit
zipit('a2_m4.json.zip','a2_m4.json')


