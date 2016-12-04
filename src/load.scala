import spark.implicits._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.unix_timestamp
import org.apache.spark.sql.functions.{concat, lit}
import org.apache.spark.sql.functions.udf

val project = "/project"
val archive = "2000"

// Read csv file with headers from hdfs
var flightsDF = spark.read.format("csv").option("header", "true").load("hdfs://"+project+"/"+archive+".csv")

// Print schema
flightsDF.printSchema

// Forbidden variables
val forbiddenVariables = Array("ArrTime", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay", "TaxiIn", "Diverted", "ActualElapsedTime", "AirTime")

// Creating allowed variables extracting the forbidden variables from the original ones.
val allowedVariables = flightsDF.columns.filter(x => !forbiddenVariables.contains(x))
// Convert them into column format.
val allowedColumns = allowedVariables.map(x => col(x))

// Selecting the just the allowed columns.
flightsDF = flightsDF.select(allowedColumns:_*)

//Drop rows with null values in the target variable
flightsDF = flightsDF.na.drop(Array("ArrDelay"))

//Drop 

// Let us convert these variable into TimeStamp
//flightsDF = flightsDF.withColumn("DepTime", flightsDF("DepTime").cast(DoubleType))
flightsDF = flightsDF.withColumn("CRSDepTime", flightsDF("CRSDepTime").cast(DoubleType))
flightsDF = flightsDF.withColumn("CRSArrtime", flightsDF("CRSArrTime").cast(DoubleType))

// Given a dataframe and the name of a column (string) which has time in format hhmm, it creates a new column based on the day, month, year, hour and minute. 
def toTimeStamp(df: org.apache.spark.sql.DataFrame, a: String) : org.apache.spark.sql.DataFrame = {
	return df.withColumn(a, unix_timestamp(concat($"DayOfMonth", lit("/"), $"Month", lit("/"), $"Year", lit(" "), col(a)), "dd/MM/yyyy HHmm"))
}

//flightsDF = toTimeStamp(flightsDF, "DepTime")
flightsDF = toTimeStamp(flightsDF, "CRSDepTime")
flightsDF = toTimeStamp(flightsDF, "CRSArrTime")

// Transforming the type of variables.
flightsDF = flightsDF.withColumn("Year", flightsDF("Year").cast(DoubleType))
flightsDF = flightsDF.withColumn("Month", flightsDF("Month").cast(DoubleType))
flightsDF = flightsDF.withColumn("DayOfMonth", flightsDF("DayOfMonth").cast(DoubleType))
flightsDF = flightsDF.withColumn("DayOfWeek", flightsDF("DayOfWeek").cast(DoubleType))
flightsDF = flightsDF.withColumn("CRSElapsedTime", flightsDF("CRSElapsedTime").cast(DoubleType))
flightsDF = flightsDF.withColumn("ArrDelay", flightsDF("ArrDelay").cast(DoubleType))
flightsDF = flightsDF.withColumn("DepDelay", flightsDF("DepDelay").cast(DoubleType))
flightsDF = flightsDF.withColumn("Distance", flightsDF("Distance").cast(DoubleType))
flightsDF = flightsDF.withColumn("TaxiOut", flightsDF("TaxiOut").cast(DoubleType))
flightsDF = flightsDF.withColumn("Cancelled", flightsDF("Cancelled").cast(DoubleType))
flightsDF = flightsDF.withColumn("CRSDepTime", flightsDF("CRSDepTime").cast(DoubleType))
flightsDF = flightsDF.withColumn("CRSArrTime", flightsDF("CRSArrTime").cast(DoubleType))

// We remove the cancelled flights since they do not contain information about the target variable (ArrDelay).
flightsDF = flightsDF.filter(col("Cancelled") === false)
// We remove such column and the CancellationCode
flightsDF = flightsDF.drop("Cancelled").drop("CancellationCode")

val toDayOfWeek: Double => String =  _ match {
case 1 => "Monday"
case 2 => "Tuesday"
case 3 => "Wednesday"
case 4 => "Thursday"
case 5 => "Friday"
case 6 => "Saturday"
case 7 => "Sunday"
}

val toDayOfWeekDF = udf(toDayOfWeek)
//flightsDF = flightsDF.withColumn("DayOfWeek", toDayOfWeekDF($"DayOfWeek"))

val toMonth: Double => String =  _ match {
case 1 => "January"
case 2 => "February"
case 3 => "March"
case 4 => "April"
case 5 => "May"
case 6 => "June"
case 7 => "July"
case 8 => "August"
case 9 => "September"
case 10 => "October"
case 11 => "November"
case 12 => "December"
}

val toMonthDF = udf(toMonth)
//flightsDF = flightsDF.withColumn("Month", toMonthDF($"Month"))

// Normalize UNIX time, we take as reference point the earliest year in the database.
val timeStampReference = unix_timestamp(lit("01/01/1987"), "dd/MM/yy")
flightsDF = flightsDF.withColumn("CRSDepTime", $"CRSDepTime" - timeStampReference)
flightsDF = flightsDF.withColumn("CRSArrTime", $"CRSArrTime" - timeStampReference)

// Eliminate DepTime since DepTime = CRSDepTime + DepDelay by definition
flightsDF = flightsDF.drop("DepTime")

//Stats
flightsDF.describe().show()
flightsDF.stat.corr("ArrDelay","Distance")

// ML 


import org.apache.spark.ml.feature.VectorAssembler


flightsDF = flightsDF.drop("DayOfWeek").drop("Month").drop("UniqueCarrier").drop("FlightNum").drop("TailNum").drop("Origin").drop("Dest").drop("DayOfMonth").drop("Year")

flightsDF = flightsDF.na.drop()

// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = flightsDF.limit(100000).randomSplit(Array(0.7, 0.3))

val assembler = new VectorAssembler()
 .setInputCols(flightsDF.drop("ArrDelay").columns)
 .setOutputCol("features")

val output = assembler.transform(trainingData)

//Regression

import org.apache.spark.ml.regression.LinearRegression
val lr = new LinearRegression()
 .setFeaturesCol("features")
 .setLabelCol("ArrDelay")
 .setMaxIter(100)
 .setElasticNetParam(0.8)

val lrModel = lr.fit(output)


val trainingSummary = lrModel.summary
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"r2: ${trainingSummary.r2}")

//Test
val output2 = assembler.transform(testData)
lrModel.evaluate(output2).rootMeanSquaredError //To evaluate a test set
//lrModel.transform(output2) //To predict


//Random Trees

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.feature.VectorAssembler


flightsDF = flightsDF.drop("UniqueCarrier").drop("FlightNum").drop("TailNum").drop("Origin").drop("Dest")
flightsDF = flightsDF.na.drop()


// Automatically identify categorical features, and index them.
// Set maxCategories so features with > 15 distinct values are treated as continuous.
val featureIndexer = new VectorAssembler()
  .setInputCols(flightsDF.drop("ArrDelay").columns)
  .setOutputCol("features")


// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = flightsDF.limit(100000).randomSplit(Array(0.7, 0.3))

// Train a RandomForest model.
val rf = new RandomForestRegressor()
  .setLabelCol("ArrDelay")
  .setFeaturesCol("features")

val tr = featureIndexer.transform(trainingData)

val indexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexed")
  .setMaxCategories(15)

val indexerModel = indexer.fit(tr)


val trc = indexerModel.transform(tr)

// Train model. This also runs the indexer.
val model = rf.fit(trc)

val tst = featureIndexer.transform(testData)

// Make predictions.
val tstc = indexerModel.transform(tst)

val predictions = model.transform(tstc)


val evaluator = new RegressionEvaluator()
  .setLabelCol("ArrDelay")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

val rmse = evaluator.evaluate(predictions)

//Boosting trees

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.feature.VectorAssembler

flightsDF = flightsDF.drop("UniqueCarrier").drop("FlightNum").drop("TailNum").drop("Origin").drop("Dest")
flightsDF = flightsDF.na.drop()


val featureIndexer = new VectorAssembler()
  .setInputCols(flightsDF.drop("ArrDelay").columns)
  .setOutputCol("features")

// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = flightsDF.limit(100000).randomSplit(Array(0.7, 0.3))

// Train a GBT model.
val gbt = new GBTRegressor()
  .setLabelCol("ArrDelay")
  .setFeaturesCol("features")
  .setMaxIter(10)

val tr = featureIndexer.transform(trainingData)

val indexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexed")
  .setMaxCategories(15)

val indexerModel = indexer.fit(tr)


val trc = indexerModel.transform(tr)


// Train model. This also runs the indexer.
val model = gbt.fit(trc)

val tst = featureIndexer.transform(testData)

// Make predictions.

val tstc = indexerModel.transform(tst)
val predictions = model.transform(tstc)

val evaluator = new RegressionEvaluator()
  .setLabelCol("ArrDelay")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

val rmse = evaluator.evaluate(predictions)

//Pipeline example with boosting


val pipeline = new Pipeline().setStages(Array(featureIndexer, indexerModel, gbt))

val pModel = pipeline.fit(trainingData)
val predictions = pModel.transform(testData)


//VectorIndexer to transform categorical variables and label them

import org.apache.spark.ml.feature.VectorIndexer

val indexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexed")
  .setMaxCategories(15)

val indexerModel = indexer.fit(output)


val indexedData = indexerModel.transform(output)

//StringIndexer to transform strings to integers for categorical variables

import org.apache.spark.ml.feature.StringIndexer	

val sIndexer = new StringIndexer().setInputCol("UniqueCarrier").setOutputCol("UniqueCarrierInt")

flightsDF=sIndexer.fit(flightsDF).transform(flightsDF)

//val sIndexer2 = new StringIndexer().setInputCol("Dest").setOutputCol("DestInt")

//flightsDF=sIndexer2.fit(flightsDF).transform(flightsDF)

//OneHotEncoder to create dummy variables for carrier, month and day of the week

import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}

val encoder = new OneHotEncoder().setInputCol("DayOfWeek").setOutputCol("dummyDayOfWeek")

val encoder2 = new OneHotEncoder().setInputCol("Month").setOutputCol("dummyMonth")

val encoder3 = new OneHotEncoder().setInputCol("UniqueCarrierInt").setOutputCol("dummyUniqueCarrier")

flightsDF = encoder.transform(flightsDF)

flightsDF = encoder2.transform(flightsDF)

flightsDF = encoder3.transform(flightsDF)

flightsDF=flightsDF.drop("DayOfWeek").drop("Month").drop("UniqueCarrierInt")
