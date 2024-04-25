# Importing necessary libraries

import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.rdd import reduce



# Creating a spark session 
spark = SparkSession.builder.master("local[*]").getOrCreate()

# Checking for commandline arguments passed
# If a file is passed via commandline, use that for validation of the model
# Else use the default validation file present inside the container

try:
        if len(sys.argv) == 2:
                filepn = str(sys.argv[1])
                data_test = spark.read.option("delimiter", ";").csv(filepn, header=True, inferSchema=True)
                print("\n ***********************************************************************")
                print(" Argument passed is :", str(sys.argv[1]))
                print(" ***********************************************************************")
        else:
                data_test = spark.read.option("delimiter", ";").csv('ValidationDataset.csv', header=True, inferSchema=True)
except:
         print("\n ***********************************************************************")
         print(" Make sure TrainDataset.csv and ValidationDataset.csv are present in the same folder as the pythonfile")
         print(" ***********************************************************************")
         exit(0)

# Reading the training dataset locally stored in the container
data_train = spark.read.option("delimiter", ";").csv('TrainingDataset.csv', header=True, inferSchema=True)


#To clean out CSV headers if quotes are present
old_column_name = data_train.schema.names
print(data_train.schema)
clean_column_name = []

for name in old_column_name:
    clean_column_name.append(name.replace('"',''))

data_train = reduce(lambda data_train, idx: data_train.withColumnRenamed(old_column_name[idx], clean_column_name[idx]), range(len(clean_column_name)), data_train)
data_test = reduce(lambda data_test, idx: data_test.withColumnRenamed(old_column_name[idx], clean_column_name[idx]), range(len(clean_column_name)), data_test)
print(data_train.schema)

# Dropping rows with quality equal to 3 because it contains very little data
data_train_new = data_train.filter(data_train['quality'] != "3")


# Selecting all columns except quality as feature columns from our train dataset
feature_cols = [x for x in data_train_new.columns if x != "quality"]

# Using a vector assembler for processing features
vect_assembler = VectorAssembler(inputCols=feature_cols, outputCol="feature")

# Using a standard scaler to remove mean and scale to unit variance for better model analysis
Scaler = StandardScaler().setInputCol('feature').setOutputCol('Scaled_feature')

# Using a logistic regression model for training 
logr = LogisticRegression(labelCol = "quality", featuresCol = "Scaled_feature")

# Creating a pipeline object of stages: [VectorAssmbler, StandardScaler, LogisticRegression]
Pipe = Pipeline(stages=[vect_assembler, Scaler, logr])

# Training the model with train dataset via Pipeline stages
PipeModel = Pipe.fit(data_train_new)

# Saving the Pipeline model results 
PipeModel.write().overwrite().save("Modelfile")

# PGenerating predictions for Train and Validation datasets
try: 
	train_prediction = PipeModel.transform(data_train)
	test_prediction = PipeModel.transform(data_test)
except:
	print("\n ***********************************************************************")
	print(" Please check CSV file(Training & Validation). Labels may be improper ")
	print(" ***********************************************************************"); exit(0)


# Creating a evaluator classification object to genertae metrics for predictions
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol = "prediction")


# Calculating the F1 score of the model with Train and Validation datasets
train_F1score = evaluator.evaluate(train_prediction, {evaluator.metricName: "f1"})
test_F1score = evaluator.evaluate(test_prediction, {evaluator.metricName: "f1"})

# Calculating the Accuracy of the model with Train and Validation datasets
train_accuracy = evaluator.evaluate(train_prediction, {evaluator.metricName: "accuracy"})
test_accuracy = evaluator.evaluate(test_prediction, {evaluator.metricName: "accuracy"})


# Priting the metrics for the user to see
print("\n ***********************************************************************")
print(" ++++++++++++++++++++++++++++++ Metrics ++++++++++++++++++++++++++++++++")
print(" ***********************************************************************")
print(" [Train] F1 score = ", train_F1score)
print(" [Train] Accuracy = ", train_accuracy)
print(" ***********************************************************************")
print(" [Validation] F1 score = ", test_F1score)
print(" [Validation] Accuracy = ", test_accuracy)
print("***********************************************************************")

# Save the results onto a Text File called results.txt
fp = open("results.txt", "w")
fp.write("***********************************************************************\n")
fp.write("[Train] F1 score =  %s\n" %train_F1score)
fp.write("[Train] Accuracy = %s\n" %train_accuracy)
fp.write("***********************************************************************\n")
fp.write("[Test] F1 score =  %s\n" %test_F1score)
fp.write("[Test] Accuracy =  %s\n" %test_accuracy)
fp.write("***********************************************************************\n")

# Closing the file
fp.close()