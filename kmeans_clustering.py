"""
    - read the data
    - splitting the data to train and test data set
    - one-hot encoder for categorical values
    - standard scaling of the features
    - KMeans for various k
    - hyperparameter tuning on k and the distance measure
    - save the pipeline, the models for various k, the best model
    
"""
sc.setLogLevel('OFF')
from pyspark.sql.types import FloatType, DoubleType, StringType, StructType
from pyspark.sql import Row
import os


FEAT_DATA_HDFS_PATH = 'hdfs://localhost:9000/user/data/kdd_names.txt'
DATA_HDFS = 'hdfs://localhost:9000/user/data/kddcup.data_10_percent_corrected'

verbose = True
if sc.getConf().get('spark.executor.memory') != '5g':
    # Dynamically configure spark.executor.memory of the cluster. Amount of memory to use per executor process.
    SparkContext.setSystemProperty('spark.executor.memory','4g')
    SparkContext.setSystemProperty('spark.driver.memory','6g')
    SparkContext.setSystemProperty('spark.master','local[4]')
    

col_names_rdd = sc.textFile(FEAT_DATA_HDFS_PATH)      #put names to columns
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
    
df = spark.read.schema(schema).csv(DATA_HDFS)    #4 part   
#df = df.repartition(100)
df.count()    #  494021
len(df.columns)    # 42

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


# Data exploration
print("\n")
print("Data Exploration")	
print("======================\n")	
print("Distribution of the labels")				 
#training.groupBy('label').count().orderBy(F.desc('count')).select(['label','count']).show()
df.groupBy('label').count().orderBy(F.desc('count')).withColumn('label_key',F.udf(lambda r : label2key[r])(F.col('label'))).select(['label','label_key','count']).show(23)

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline

print("\n")
print("Training")
print("================")

# Construct and Fit the pipeline

# preprocessing for categorical features
from pyspark.ml.feature import StringIndexer, OneHotEncoder
def one_hot_encoding_pipeline(cols):
    stages = []
    new_cols = [col+"_enc" for col in cols]
    for col in cols:
        si = StringIndexer( stringOrderType = "frequencyDesc",handleInvalid="keep" ).setInputCol(col).setOutputCol(col+"_new")
        encoder = OneHotEncoder(inputCol = col+"_new", outputCol = col+"_enc")
        if len(stages) == 0:
            stages = [si,encoder]
        else:
            stages.append(si)
            stages.append(encoder)
   	
    pipe = Pipeline().setStages(stages)

    return pipe, new_cols
    

pipe_cat, new_cols = one_hot_encoding_pipeline(cat_cols)


# preprocessing for numeric data
van = VectorAssembler().setInputCols(num_cols + new_cols).setOutputCol('num_features')
scaler = StandardScaler(inputCol = 'num_features', outputCol= 'features')
pipe_num = Pipeline().setStages([van, scaler])
# assembling of all the preprocessed features and kmeans clustering
km = KMeans(k = 2, seed = 42)
pipe_final =  Pipeline().setStages([km])
pipe = Pipeline().setStages([pipe_cat, pipe_num, pipe_final])
pm = pipe.fit(training)
cd = pm.transform(training)
cdt = pm.transform(test)

from  pyspark.ml.evaluation import ClusteringEvaluator
import pandas as pd
# It calculates the Silouette measure for the validation of consistency between the clusters created
# [-1,1], 1 when points assigned to a cluster are close to points assigned to the same cluster and 
# in distance with those assigned to other clusters.
evaluator = ClusteringEvaluator(predictionCol = 'prediction', featuresCol = 'features')
k_list = [2, 23,100,150]

for k in k_list:
    km = KMeans(k = k, seed = 42)
    pipe_final =  Pipeline().setStages([km])
    pipe = Pipeline().setStages([pipe_cat, pipe_num, pipe_final])
    pm = pipe.fit(training)
    #save the model
    if os.path.exists("file://" + os.getcwd() + f"/models/kmeans/{k}") == False:
        os.makedirs("file://" + os.getcwd() + f"/models/kmeans/{k}")
    pm.write().overwrite().save("file://" + os.getcwd() + f"/models/kmeans/{k}")      #  it saves the model to the current working dir. Check at your current wd.
  
    if verbose :
        cd = pm.transform(training)
        cd.cache()
        cdt = pm.transform(test)
        cdt = cdt.cache()
        print(key2label)
        print(f'Training data clustered, k = {k}')
        score = evaluator.evaluate(cdt)
        print(f'Score = {score}')
        print("prediction     |     [[label, count from that label classified to the predicted class]]")
        cdn = cd.select(['prediction','label']).groupBy(['prediction','label']).agg(F.collect_list(F.col('label')).alias('label_list')).withColumn('label_list', F.udf(lambda l :  len(l))(F.col('label_list')))
        # caching the data when the memory is not large enough has proved to be a bad idea !
        #cdn = cdn.cache()
        cdn = cdn.rdd.map(lambda r : Row(prediction = r['prediction'], label_pair = (r['label'],r['label_list']))).toDF(['prediction','label_counts'])
        cdf = cdn.groupBy('prediction').agg(F.collect_list('label_counts').alias('label_dist')).orderBy(F.asc('prediction'))
        cdf.withColumn('label_dist', F.udf(lambda l : sorted([[k,v] for k,v in l], key = lambda t : int(t[1]), reverse = True ))('label_dist')).orderBy(F.asc('prediction')).show(150,truncate = False)
    
        print(f'Test data clustered, k = {k}')
        cdn = cdt.select(['prediction','label']).groupBy(['prediction','label']).agg(F.collect_list(F.col('label')).alias('label_list')).withColumn('label_list', F.udf(lambda l :  len(l))(F.col('label_list')))
        cdn = cdn.rdd.map(lambda r : Row(prediction = r['prediction'], label_pair = (r['label'],r['label_list']))).toDF(['prediction','label_counts'])
        cdf = cdn.groupBy('prediction').agg(F.collect_list('label_counts').alias('label_dist')).orderBy(F.asc('prediction'))
        cdf.withColumn('label_dist', F.udf(lambda l : sorted([[k,v] for k,v in l], key = lambda t : int(t[1]), reverse = True ))('label_dist')).orderBy(F.asc('prediction')).show(150,truncate = False)
    
        print(f"Parameters for the clustering, k = {k} :\n")
        km_model = pm.stages[-1].stages[-1]
        parameters = pd.DataFrame.from_dict({key.name : value for key, value in km_model.extractParamMap().items()}, orient = 'index', columns = ['value'])
    
        print(parameters)
    
cd.unpersist()
cdt.unpersist()

print("\n")
print("Hyperparameter tuning")
print("========================")

from pyspark.ml.tuning import ParamGridBuilder


parameter_grid = ParamGridBuilder().addGrid(km.distanceMeasure, ['euclidean','cosine']).addGrid(km.k, [2,23,50,100]).build()

from pyspark.ml.tuning import TrainValidationSplit

#tvs = TrainValidationSplit(estimator = pipe, estimatorParamMaps = parameter_grid, evaluator = evaluator, seed = 42, parallelism = 1).setTrainRatio(0.9)
tvs = TrainValidationSplit(estimator = pipe, estimatorParamMaps = parameter_grid, evaluator = evaluator, seed = 42, parallelism = 1, ).setTrainRatio(0.9)
tvs_model = tvs.fit(training)


best_model = tvs_model.bestModel

validation_metrics = tvs_model.validationMetrics

best_predictions = best_model.transform(test)

score_best = evaluator.evaluate(best_predictions)
									 
print('Best Model')
print('=================\n')

if verbose :
    print('Training data clustered')
    bcd = best_model.transform(training)
    cdn = bcd.select(['prediction','label']).groupBy(['prediction','label']).agg(F.collect_list(F.col('label')).alias('label_list')).withColumn('label_list', F.udf(lambda l :  len(l))(F.col('label_list')))
    cdn = cdn.rdd.map(lambda r : Row(prediction = r['prediction'], label_pair = (r['label'],r['label_list']))).toDF(['prediction','label_counts'])
    cdf = cdn.groupBy('prediction').agg(F.collect_list('label_counts').alias('label_dist'))
    cdf.withColumn('label_dist', F.udf(lambda l : sorted([[k,v] for k,v in l], key = lambda t : int(t[1]), reverse = True ))('label_dist')).orderBy(F.asc('prediction')).show(150, truncate = False)
    print('Test data clustered')
    cdn = best_predictions.select(['prediction','label']).groupBy(['prediction','label']).agg(F.collect_list(F.col('label')).alias('label_list')).withColumn('label_list', F.udf(lambda l :  len(l))(F.col('label_list')))
    cdn = cdn.rdd.map(lambda r : Row(prediction = r['prediction'], label_pair = (r['label'],r['label_list']))).toDF(['prediction','label_counts'])
    cdf = cdn.groupBy('prediction').agg(F.collect_list('label_counts').alias('label_dist')).orderBy(F.asc('prediction'))
    cdf.withColumn('label_dist', F.udf(lambda l : sorted([[k,v] for k,v in l], key = lambda t : int(t[1]), reverse = True ))('label_dist')).orderBy(F.asc('prediction')).show(150, truncate = False)
    
    print("Parameters for the clustering :\n")
    newkm = best_model.stages[-1].stages[-1]
    best_parameters = pd.DataFrame.from_dict({key.name : value for key, value in newkm.extractParamMap().items()}, orient = 'index', columns = ['value'])
    print(best_parameters)
    
print(f'Best model score : {score_best}')

#save the pipeline
if os.path.exists("file://" + os.getcwd() + "/pipelines/kmeans_clustering") == False:
    os.makedirs("file://" + os.getcwd() + "/pipelines/kmeans_clustering")
pipe.write().overwrite().save("file://" + os.getcwd() + "/pipelines/kmeans_clustering")   

#save the best model
if os.path.exists("file://" + os.getcwd() + "/best_models/kmeans") == False:
    os.makedirs("file://" + os.getcwd() + "/best_models/kmeans")
best_model.write().overwrite().save("file://" + os.getcwd() + "/best_models/kmeans")     

