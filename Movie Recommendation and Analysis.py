from pyspark.sql import SparkSession

spark = SparkSession.builder \
        .master("local[4]") \
        .appName("Movies Rating") \
        .config("spark.local.dir","/fastdata/acp21zgs") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")

sc = spark.sparkContext
# Here I AM IMPORTING Libraries
import pyspark.sql.functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.functions import lit
from pyspark.sql.functions import desc
from pyspark.ml.linalg import Vectors
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.types import DoubleType
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ratingss = spark.read.load("/data/acp21zgs/ScalableML/Data/ratings.csv", format = 'csv', inferSchema = "true", header = "true").cache()
ratings = ratingss.na.drop()
ratings = ratings.drop("timestamp")
ratings.show(20,False)
myseed=6012

tag = spark.read.load("/data/acp21zgs/ScalableML/Data/tags.csv", format = 'csv', inferSchema = "true", header = "true").cache()
tag.show()

spark.conf.set("spark.sql.pivotMaxValues", 100000)
# Here I am making empty list so that I can append later
als1RMSEHot = []
als1RMSECool = []
als2RMSEHot = []
als2RMSECool = []

Top_Movie_Tags = []
Bottom_Movie_Tags = []
# Here I am making two different ALS models 
alsA = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop")
alsB = ALS(maxIter = 10, regParam = 0.01, userCol = 'userId', itemCol = 'movieId', ratingCol = 'rating',
             coldStartStrategy = 'drop')
# Here I am making Two RMSE Evaluator
rmseAEvaluator = RegressionEvaluator(metricName = 'rmse', labelCol = 'rating', predictionCol = 'prediction')
rmseBEvaluator = RegressionEvaluator(metricName = 'rmse', labelCol = 'rating', predictionCol = 'prediction')

numberOfFolds = 5
splitData = ratings.randomSplit([0.2, 0.2, 0.2, 0.2, 0.2], (numberOfFolds*numberOfFolds))
# Here i am using for loop to do it for every fold and every split
for fold in range(numberOfFolds):
    print("Begin Fold: " + str(fold + 1))

    # Select the test data from the 5 pre-split data sets
    test = splitData[fold]
    # Create a blank Data Frame with same schema as test and the original data frame
    #  will be used to store rest of pre-split data sets
    train = spark.createDataFrame(sc.emptyRDD(), ratings.schema)

    # Iterate through each dataset in pre-split datasets
    for split in splitData:
        # If the dataset isnt the set set then it is part of the training set
        if split != test:
            train = train.union(split)
        
          
        #x = train.groupBy("userId").count().orderBy(desc("count")).limit(c)
        #c = round(0.1 * train.count())
    x = train.groupBy("userId").count().orderBy(desc("count"))
    cx = round(0.1 * x.count())
    x = x.limit(cx)
    x_new = x.withColumnRenamed("userId", "HotUsers").withColumnRenamed("count", "Hot_Count")
    x_new.show()
        #y = train.groupBy("userId").count().orderBy("count").limit(c)
    y = train.groupBy("userId").count().orderBy("count")
    cy = round(0.1 * y.count())
    y = y.limit(cy)
    y_new = y.withColumnRenamed("userId", "CoolUsers").withColumnRenamed("count", "Cool_Count")
    y_new.show()
        #ratings.createOrReplaceTempView("ratings")
    #test.createOrReplaceTempView("test")
    y_new.createOrReplaceTempView("y_new")
    x_new.createOrReplaceTempView("x_new")
    #coolness = spark.sql("SELECT * FROM test where userId in (select CoolUsers from y_new)")
    #Hotness = spark.sql("SELECT * FROM test where userId in (select HotUsers from x_new)")
    test.createOrReplaceTempView("test")
    coolness_test = spark.sql("SELECT * FROM test where userId in (select CoolUsers from y_new)")
    Hotness_test = spark.sql("SELECT * FROM test where userId in (select HotUsers from x_new)")

    print("Fit ALS 1 on Train")
    alsAModel = alsA.fit(train)
    
    #print("Fit ALS 1 on CoolUsers")
    #alsAModelCool = alsA.fit(coolness)
    print("Fit ALS 2 on Train")
    alsBModel = alsB.fit(train)
    #print("Fit ALS 2 on CoolUsers")
    #alsBModelCool = alsB.fit(coolness)

    print("Begin ALS 1 Evaluation for HotUsers")
    alsAPredictionsHot = alsAModel.transform(Hotness_test)
    alsARMSEEvalHot = rmseAEvaluator.evaluate(alsAPredictionsHot)

    print("Begin ALS 1 Evaluation for CoolUsers")
    alsAPredictionsCool = alsAModel.transform(coolness_test)
    alsARMSEEvalCool = rmseAEvaluator.evaluate(alsAPredictionsCool)

    print("Begin ALS 2 Evaluation for HotUsers")
    alsBPredictionsHot = alsBModel.transform(Hotness_test)
    alsBRMSEEvalHot = rmseBEvaluator.evaluate(alsBPredictionsHot)

    print("Begin ALS 2 Evaluation for CoolUsers")
    alsBPredictionsCool = alsBModel.transform(coolness_test)
    alsBRMSEEvalCool = rmseBEvaluator.evaluate(alsBPredictionsCool)
    
    print(alsARMSEEvalHot)
    print(alsARMSEEvalCool)
    print(alsBRMSEEvalHot)
    print(alsBRMSEEvalCool)

    als1RMSEHot.append(alsARMSEEvalHot)
    als1RMSECool.append(alsARMSEEvalCool)
    als2RMSEHot.append(alsBRMSEEvalHot)
    als2RMSECool.append(alsBRMSEEvalCool)
    
    #Question 2.B
    # Here I am starting Question 2 
    data_frame = alsAModel.itemFactors
    n = data_frame.select('features')
    data_frame.show()
    # Here I am defining K means 
    kmeans = KMeans(k=10, seed=1)
    model = kmeans.fit(n)
    prediction = model.transform(n)
    prediction.show()
    
    p = model.summary.clusterSizes
    p
    # Converting list to dictionary
    def Convert(lst):
        res_dct = {lst[i]: i for i in range(0, len(lst))}
        return res_dct         
    # Driver code
    #print(Convert(p))
    x = Convert(p)
    print(x)
    
    l = {k: v for k, v in sorted(x.items(), key=lambda item: item[0])}
    l
    
    s = l.values()
    f = list(s)
    f
    h = f[-2:]
    
    rt = prediction.filter((prediction.prediction == h[0]) | (prediction.prediction == h[1]))
    rt.show()
    

    

    
    rt.createOrReplaceTempView("rt")
    data_frame.createOrReplaceTempView("data_frame")
    tag.createOrReplaceTempView("tag")
    rt_count = rt.groupBy('prediction').count()
    rt_count = rt_count.sort('count', asecnding=False).limit(2)
    for row in rt_count.collect(): 
      print(row) 
      j = spark.sql(f"SELECT rt.features, rt.prediction, data_frame.id FROM rt INNER JOIN data_frame ON rt.features = data_frame.features Where rt.prediction={row['prediction']}")
      j.show()
      j.createOrReplaceTempView("j")
      merge = spark.sql("SELECT tag.tag, tag.movieId FROM tag INNER JOIN j ON tag.movieId = j.id")
      merge.show()
      BottomClustertag = merge.groupBy("tag").count()
      Bottom = BottomClustertag.sort("count", ascending=True).limit(1)
      Bottom.show()
      mvv = Bottom.select("tag").rdd.flatMap(lambda x: x).collect()
      type(mvv)
      Top = BottomClustertag.sort('count', ascending=False).limit(1)
      Top.show()
      mvvs = Top.select("tag").rdd.flatMap(lambda x: x).collect()
      mvvs
      for tag_row in Bottom.collect():
        Bottom_Movie_Tags.append(tag_row['tag'])
      for tag_row in Top.collect():
        Top_Movie_Tags.append(tag_row['tag'])
    
print(Top_Movie_Tags)
print(Bottom_Movie_Tags)
plt.gcf().set_size_inches(22.5, 13.5)
plt.plot(alsARMSEEvalHot, ls = '-', marker='x')
plt.plot(alsARMSEEvalCool, ls = '-', marker='o')
plt.plot(alsBRMSEEvalHot, ls = '-', marker='x')
plt.plot(alsBRMSEEvalCool, ls = '-', marker='o')
plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
plt.ylabel('Root Mean Square Error')
plt.title('ALS Hot and Cool')
plt.legend(labels=['AlsA v1 Hot','AlsA v2 Cool', 'AlsB v2 Hot', 'AlsB v1 Cool'])
plt.savefig("../Output/Q2_figA.png",bbox_inches = "tight")    