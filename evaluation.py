"""
    - restore the model of your choice from disk
    - compute clustering quality and prediction performance metrics
      for KMeans, LDA and supervised learning - MLP

"""
sc.setLogLevel('OFF')

from pyspark.sql.types import FloatType, DoubleType, StringType, StructType
from pyspark.ml import Pipeline, PipelineModel

from pyspark.ml.tuning import TrainValidationSplitModel

from pyspark.ml.classification import MultilayerPerceptronClassifier

import os


FEAT_DATA_HDFS_PATH = 'hdfs://localhost:9000/user/data/kdd_names.txt'
DATA_HDFS = 'hdfs://localhost:9000/user/data/kddcup.data_10_percent_corrected'


k = 23   #   {'best', 2, 23, 100, 150}    for other values of k train first the model with kmeans_clustering.py
verbose = False
if sc.getConf().get('spark.executor.memory') != '4g':
    # Dynamically configure spark.executor.memory of the cluster. Amount of memory to use per executor process.
    SparkContext.setSystemProperty('spark.executor.memory','4g')
    SparkContext.setSystemProperty('spark.driver.memory','6g')
    SparkContext.setSystemProperty('spark.master','local[4]')
    
try:
    col_names_rdd = sc.textFile(FEAT_DATA_HDFS_PATH)      #put names to columns
except Exception:
    print("""Specify your path to the network traffic features file on hdfs.
             Set DATA_HDFS_PATH global variable  accordingly.""")
    
lines = col_names_rdd.collect()
schema = StructType()
num_cols = []
cat_cols = []

for l in lines[1:]:
    key = l.split(':')[0].strip() 
    if l.split(':')[1].strip(".").strip() == 'continuous':
        num_cols.append(key)
        value_type = DoubleType()
    else:
        cat_cols.append(key)
        value_type = StringType()
        
    schema.add(key,value_type)
    
schema.add('label',StringType())
    
#df = spark.read.schema(schema).option('header','false').csv('hdfs://localhost:9000/user/a/anomaly_detection/data/kdd_data')  # 8 part
df = spark.read.schema(schema).csv(DATA_HDFS)    #4 part   
#df = df.repartition(100)


import pyspark.sql.functions as F

udf_strip = F.udf(lambda s : s.strip("."))
df = df.withColumn('label', udf_strip('label'))

#For splitting the dataset in train and test datasets we may either pick a randomSplit
# or better choose a stratified approach in dataset splitting.

# 1st approach

label_values = df.groupBy('label').count().select('label').collect()
label2key = dict()
key2label = dict()

for i, r in enumerate(label_values):
    label2key[r['label']] = float(i)
    key2label[i] = r['label']
    v = r['label']
    if i == 0:
        training, test = df.where(f"label = '{v}'").randomSplit([0.9, 0.1], seed = 42)
    else: 
        train_temp, test_temp = df.where(f"label = '{v}'").randomSplit([0.9, 0.1], seed = 42)
        training = training.union(train_temp)
        test = test.union(test_temp)

normal_index = label2key['normal']
# 2nd approach
#training,test = df.randomSplit([0.9, 0.1])      

training = training.cache()
test = test.cache()

training = training.withColumn('label', F.udf(lambda c : label2key[c])(training.label).cast(DoubleType()))
test = test.withColumn('label', F.udf(lambda c : label2key[c])(test.label).cast(DoubleType()))

training = training.cache()
test = test.cache()

from pyspark.ml.evaluation import BinaryClassificationEvaluator, ClusteringEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.clustering import LDA

import pyspark.sql.functions as F



# Retrieve the processing pipeline
pipe = Pipeline().load("file://" + os.getcwd() + "/pipelines/kmeans_clustering")      ############################   replace path

# Retrieve kmeans model
if k == 'best':
    model = PipelineModel.load("file://" + os.getcwd() + "/best_models/kmeans")       ############################   replace path
else: 
    model = PipelineModel.load("file://" + os.getcwd() + f"/models/kmeans/{k}") 
    
new_stages = []
for i, pm in enumerate(model.stages):
    if i != 2:
        new_stages.extend(pm.stages)
    
new_pm = PipelineModel(new_stages)
    
km = model.stages[-1].stages[-1]
k = km.getK()

maxIter = km.maxIter

km_clusts = model.transform(training)
km_preds = model.transform(test)


clustering_evaluator = ClusteringEvaluator()
binary_evaluator = BinaryClassificationEvaluator(rawPredictionCol = 'raw_prediction')
multiclass_evaluator = MulticlassClassificationEvaluator()
multiclass_evaluator_bin = MulticlassClassificationEvaluator(predictionCol = 'raw_prediction')



# evaluation of clustering
km_sil_tr = clustering_evaluator.evaluate(km_clusts)
km_sil = clustering_evaluator.evaluate(km_preds)
# evaluation of classification
# seen as binary task - normal -not normal classes
gross_clusts  = km_clusts.withColumn('raw_prediction', F.udf(lambda el : 0.0 if el == normal_index  else 1.0)(km_clusts.prediction).cast(DoubleType()))
gross_clusts = gross_clusts.withColumn('prediction',gross_clusts.prediction.cast(DoubleType()))
auc_on_train_km = binary_evaluator.evaluate(gross_clusts)
gross_preds  = km_preds.withColumn('raw_prediction', F.udf(lambda el : 0.0 if el == normal_index  else 1.0)(km_preds.prediction).cast(DoubleType()))
gross_preds = gross_preds.withColumn('prediction',gross_preds.prediction.cast(DoubleType()))
auc_km = binary_evaluator.evaluate(gross_preds)
# the binary seen as multiclassification task
f1_gross_km_clusts = multiclass_evaluator_bin.evaluate(gross_clusts)
f1_gross_km_preds = multiclass_evaluator_bin.evaluate(gross_preds)
# the multiclass seen as multiclassification task
km_clusts =km_clusts.withColumn('prediction',km_clusts.prediction.cast(DoubleType()))
f1_score_km_clusts = multiclass_evaluator.evaluate(km_clusts)
km_preds = km_preds.withColumn('prediction',km_preds.prediction.cast(DoubleType()))
f1_score_km = multiclass_evaluator.evaluate(km_preds)

print(f"km silhouette for the training data : {km_sil_tr}    km silhouette : {km_sil}    km auc for the training : {auc_on_train_km }    km auc : {auc_km} ")
print(f"km f1_b for the training data  : {f1_gross_km_clusts}    km f1_b : {f1_gross_km_preds}    km_f1 for the training : { f1_score_km_clusts}    f1_score_km : {f1_score_km} ")

#the same analysis follows for the rest clustering methods used

# lda
lda = LDA(k = k, maxIter = 20, seed = 42)
pipe_lda = Pipeline().setStages(pipe.getStages()[:-1] + [Pipeline().setStages([lda])])
pm_lda = pipe_lda.fit(training)
lda_clusts = pm_lda.transform(training)
lda_preds = pm_lda.transform(test)

lda_clusts = lda_clusts.withColumn('prediction', F.udf(lambda el : [i for i,entry in enumerate(el.toArray()) if entry == max(el.toArray())][0])(lda_clusts.topicDistribution).cast(DoubleType()))
lda_preds = lda_preds.withColumn('prediction', F.udf(lambda el : [i for i,entry in enumerate(el.toArray()) if entry == max(el.toArray())][0])(lda_preds.topicDistribution).cast(DoubleType()))

lda_sil_tr = clustering_evaluator.evaluate(lda_clusts)
lda_sil = clustering_evaluator.evaluate(lda_preds)

gross_lda_clusts  = lda_clusts.withColumn('raw_prediction', F.udf(lambda el : 0.0 if el == normal_index else 1.0)(lda_clusts.prediction).cast(DoubleType()))
auc_on_train_lda = binary_evaluator.evaluate(gross_lda_clusts)
gross_lda_preds  = lda_preds.withColumn('raw_prediction', F.udf(lambda el : 0.0 if el ==  normal_index else 1.0)(lda_preds.prediction).cast(DoubleType()))
auc_lda = binary_evaluator.evaluate(gross_lda_preds)

gross_lda_clusts = gross_lda_clusts.withColumn('prediction',gross_lda_clusts.prediction.cast(DoubleType()))
f1_gross_lda_clusts = multiclass_evaluator_bin.evaluate(gross_lda_clusts)
gross_lda_preds = gross_lda_preds.withColumn('prediction',gross_lda_preds.prediction.cast(DoubleType()))
f1_gross_lda_preds = multiclass_evaluator_bin.evaluate(gross_lda_preds)

lda_clusts = lda_clusts.withColumn('prediction',lda_clusts.prediction.cast(DoubleType()))
f1_score_lda_clusts = multiclass_evaluator.evaluate(lda_clusts)
lda_preds = lda_preds.withColumn('prediction',lda_preds.prediction.cast(DoubleType()))
f1_score_lda = multiclass_evaluator.evaluate(lda_preds)

print(f"lda silhouette for the training data : {lda_sil_tr}    lda silhouette : {lda_sil}    lda auc for the training : {auc_on_train_lda }    lda auc : {auc_lda} ")
print(f"lda f1_b for the training data  : {f1_gross_lda_clusts}    lda f1_b : {f1_gross_lda_preds}    lda_f1 for the training : { f1_score_lda_clusts}    f1_score_lda : {f1_score_lda} ")

training_2 = training.withColumn('label',F.udf(lambda l : 1.0 if l != normal_index else 0.0)(training.label).cast(DoubleType()))
training_2 = training_2.cache()
test_2 = test.withColumn('label',F.udf(lambda l : 1.0 if l != normal_index else 0.0)(test.label).cast(DoubleType()))
test_2 = test_2.cache()


mlp_2 = MultilayerPerceptronClassifier(maxIter=100, layers=[121, 100, 2], blockSize=128, seed=42)
pipe_mlp_2 = Pipeline().setStages(pipe.getStages()[:-1] + [Pipeline().setStages([mlp_2])])
mlp_2_m = pipe_mlp_2.fit(training_2)
mlp_2_preds = mlp_2_m.transform(test_2)

auc_mlp2 = binary_evaluator.evaluate(mlp_2_preds.withColumnRenamed('prediction','raw_prediction'))
print(f"Anomaly detection using supervised learning - mlp///")
print(f"AUC : {auc_mlp2} ")


mlp_23 = MultilayerPerceptronClassifier(maxIter=100, layers=[121, 100, 23], blockSize=128, seed=42)
pipe_mlp_23 = Pipeline().setStages(pipe.getStages()[:-1] + [Pipeline().setStages([mlp_23])])
mlp_23_m = pipe_mlp_23.fit(training)
mlp_23_preds = mlp_23_m.transform(test)

f1_mlp23 = multiclass_evaluator.evaluate(mlp_23_preds) 
f1_mlp2 = multiclass_evaluator.evaluate(mlp_2_preds) 
print(f"Anomaly detection using supervised learning - mlp///")
print(f"F1-score  : {f1_mlp2} ")
print("Classification using supervised learning - mlp ///")
print(f"F1-score : {f1_mlp23 }")