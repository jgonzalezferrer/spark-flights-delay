package upm.bd

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.Pipeline


object App {

  /* 
   * Transform a date in "dd/MM/yyyy HHmm" format to Unix TimeStamp 
   * Input: columnName -> name of the column to transform
   * Return: new dataframe
   */
   def dateToTimeStamp(df: org.apache.spark.sql.DataFrame, columnName: String) : org.apache.spark.sql.DataFrame = {
   	return df.withColumn(columnName, 
   		unix_timestamp(concat(col("DayOfMonth"), lit("/"), col("Month"), lit("/"), col("Year"), lit(" "), col(columnName)), 
   			"dd/MM/yyyy HHmm"))
   }  

   def main(args : Array[String]) {

	// Disabling debug option. 
	Logger.getRootLogger().setLevel(Level.WARN)

	val spark = SparkSession
	.builder()
	.appName("Spark Flights Delay")
	.getOrCreate()

	import spark.implicits._

	// TODO: Change it as a program parameter.
	val project = "/project"

	// Read all csv files with headers from hdfs.
	// The valid columns are selected, casting them (the default type is String).
	val flightsOriginalDF = spark.read
	.format("com.databricks.spark.csv")
	.option("header", "true")
	.load("hdfs://"+args(0)+"*.csv")
	.select(col("Year").cast(StringType),
		col("Month").cast(StringType),
		col("DayOfMonth").cast(StringType),
		col("DayOfWeek").cast(DoubleType),
		col("DepTime").cast(DoubleType),
		col("CRSDepTime").cast(StringType),
		col("CRSArrtime").cast(StringType),
		col("UniqueCarrier").cast(StringType),
		col("FlightNum").cast(StringType),
		col("TailNum").cast(StringType),
		col("CRSElapsedTime").cast(DoubleType),
		col("ArrDelay").cast(DoubleType),
		col("DepDelay").cast(DoubleType),
		col("Origin").cast(StringType),
		col("Dest").cast(StringType),
		col("Distance").cast(DoubleType),
		col("TaxiOut").cast(DoubleType),
		col("Cancelled").cast(BooleanType),
		col("CancellationCode").cast(StringType))

	///////////// Data Manipulation ////////////	

	var flightsDF = flightsOriginalDF

	/* Removing data points */

	//Drop rows with null values in the target variable	
	flightsDF = flightsDF.na.drop(Array("ArrDelay"))
	
	/* Removing variables */
	flightsDF = flightsDF.drop("DepTime").drop("Cancelled").drop("CancellationCode")	


	/* Modifying variables */
	
	//Convert scheduled departure and arrival time to TimeStamp
	flightsDF = dateToTimeStamp(flightsDF, "CRSDepTime")
	flightsDF = dateToTimeStamp(flightsDF, "CRSArrTime")

	// Normalize UNIX time, we take as reference point the earliest year in the database.
	val timeStampReference = unix_timestamp(lit("01/01/1987"), "dd/MM/yy")
	flightsDF = flightsDF.withColumn("CRSDepTime", $"CRSDepTime" - timeStampReference)
	flightsDF = flightsDF.withColumn("CRSArrTime", $"CRSArrTime" - timeStampReference)

	//Cast variables to their original types
	flightsDF=flightsDF.withColumn("DayOfMonth", col("DayOfMonth").cast(DoubleType))
	flightsDF=flightsDF.withColumn("CRSDepTime", col("CRSDepTime").cast(DoubleType))
	flightsDF=flightsDF.withColumn("CRSArrTime", col("CRSArrTime").cast(DoubleType))
	flightsDF=flightsDF.withColumn("Year", col("Year").cast(DoubleType))
	flightsDF=flightsDF.withColumn("Month", col("Month").cast(DoubleType))
	

	/* Adding new variables */ //TODO: Maybe not necessary??
	
	// TODO: Change it as a program parameter. 
	val airportsDF = spark.read
	.format("com.databricks.spark.csv")
	.option("header", "true")
	.load("hdfs://"+args(1))
	.select(col("iata"), col("state"))


	// New column: State of the Origin airports.
	flightsDF = flightsDF.join(airportsDF, flightsDF("Origin") === airportsDF("iata"), "left_outer").withColumnRenamed("state", "OriginState").drop("iata")
	// New column: State of the Dest airports.
	flightsDF = flightsDF.join(airportsDF, flightsDF("Dest") === airportsDF("iata"), "left_outer").withColumnRenamed("state", "DestState").drop("iata")
	

	flightsDF.show


	// Machine learning pipes:

	//StringIndexer to transform the UniqueCarrier string to integer for using it as a categorical variable

	val sIndexer = new StringIndexer().setInputCol("UniqueCarrier").setOutputCol("UniqueCarrierInt")
	flightsDF=sIndexer.fit(flightsDF).transform(flightsDF)

	//Remove variables we do not consider appropriate for the ML algorithms (also the string version of UniqueCarrier)

	flightsDF = flightsDF.drop("FlightNum").drop("TailNum").drop("Origin").drop("Dest").drop("DayOfMonth").drop("Year").drop("UniqueCarrier")

	//Remove rows with null values for the remaining variables
	flightsDF = flightsDF.na.drop()

	//Prepare the assembler that will transform the remaining variables to a feature vector for the ML algorithms

 	val assembler = new VectorAssembler()
 	.setInputCols(flightsDF.drop("ArrDelay").columns)
 	.setOutputCol("features")

	// Split the data into training and test sets (30% held out for testing). Will be used in both tree algorithms
	val Array(trainingData, testData) = flightsDF.randomSplit(Array(0.7, 0.3))


//Linear regression

 //OneHotEncoder to create dummy variables for carrier, month and day of the week 
 //Linear regression needs them to handle those categorical variables properly
 val encoder = new OneHotEncoder().setInputCol("DayOfWeek").setOutputCol("dummyDayOfWeek")
 val encoder2 = new OneHotEncoder().setInputCol("Month").setOutputCol("dummyMonth")
 val encoder3 = new OneHotEncoder().setInputCol("UniqueCarrierInt").setOutputCol("dummyUniqueCarrier")
 var flightsDFReg = encoder.transform(flightsDF)
 flightsDFReg = encoder2.transform(flightsDFReg)
 flightsDFReg = encoder3.transform(flightsDFReg)
 //Remove the original variables not to use them in regression
 flightsDFReg = flightsDFReg.drop("DayOfWeek").drop("Month").drop("UniqueCarrierInt") 
// Split the data into training and test sets (30% held out for testing).
val Array(trainingDataR, testDataR) = flightsDFReg.randomSplit(Array(0.7, 0.3))

//We use different data name for this algorithm because of the dummy variables, they are different for the tree models.

//Defining the model

val lr = new LinearRegression()
.setFeaturesCol("features")
.setLabelCol("ArrDelay")
.setMaxIter(100)
.setElasticNetParam(0.8)

//Preparing the pipeline

val regressionPipeline = new Pipeline().setStages(Array(encoder, encoder2, encoder3, assembler, lr))
//Fitting the model to our data
val rModel = regressionPipeline.fit(trainingDataR)
//Making predictions
var predictions  = rModel.transform(testDataR)

//Evaluating the result

var evaluator = new RegressionEvaluator()
  .setLabelCol("ArrDelay")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

val rmseRegression = evaluator.evaluate(predictions)

//Random Trees

//Vector Indexer to indicate that some variables are categorical, so they are treated properly by the algorithms
//In our case, DayOfWeek, Month, UniqueCarrier have less than 15 different classes, so they will be marked as categorical, as we want
 var indexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexed")
  .setMaxCategories(15)
 
//Defining the model

val rf = new RandomForestRegressor()
	.setLabelCol("ArrDelay")
	.setFeaturesCol("features")


//Pipeline of random forest: 

val randomTreesPipeline = new Pipeline().setStages(Array(assembler, indexer, rf))
//Fitting the model to our data
val RTModel = randomTreesPipeline.fit(trainingData)
//Making predictions
predictions = RTModel.transform(testData)

//Ealuating the result of the predictions
evaluator = new RegressionEvaluator()
  .setLabelCol("ArrDelay")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

val rmseRandom = evaluator.evaluate(predictions)

//Boosting trees

//Same as before, we mark the variables that we want as categorical so they are treated properly by the algorithm.
	indexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexed")
  .setMaxCategories(15)

//Defining the model

val gbt = new GBTRegressor()
  .setLabelCol("ArrDelay")
  .setFeaturesCol("features")
  .setMaxIter(10)

//Pipeline to train and test the data with the boosting algorithm

val pipeline = new Pipeline().setStages(Array(assembler, indexer, gbt))
//Training using the pipeline
val BModel = pipeline.fit(trainingData)
//Predictions 
predictions = BModel.transform(testData)

//Evaluating the performance of the predictions
evaluator = new RegressionEvaluator()
  .setLabelCol("ArrDelay")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

val rmseBoosintg = evaluator.evaluate(predictions)

println("rmse for different algorithms: ")
println("Linear regression = "+rmseRegression)
println("Random forests = "+rmseRandom)
println("Boosting trees = "+rmseBoosintg)

  }

}
