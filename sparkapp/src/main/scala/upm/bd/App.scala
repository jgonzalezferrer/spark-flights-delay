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
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

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
	//.load("hdfs://"+args(0)+"*.csv")
	.load("hdfs:///project/flights/*.csv")
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
	.load("hdfs:///project/extra/airports.csv")
	.select(col("iata"), col("state"))


	// New column: State of the Origin airports.
	flightsDF = flightsDF.join(airportsDF, flightsDF("Origin") === airportsDF("iata")).withColumnRenamed("state", "OriginState").drop("iata")
	// New column: State of the Dest airports.
	flightsDF = flightsDF.join(airportsDF, flightsDF("Dest") === airportsDF("iata")).withColumnRenamed("state", "DestState").drop("iata")
	

	// Machine learning pipes:

	//Linear regression
	flightsDF=flightsDF.sample(false, 0.0025, 100) // Last parameter is the seed

	//StringIndexer to transform the UniqueCarrier string to integer for using it as a categorical variable

	val sIndexer = new StringIndexer().setInputCol("UniqueCarrier").setOutputCol("UniqueCarrierInt")
	flightsDF=sIndexer.fit(flightsDF).transform(flightsDF)

	//val oIndexer = new  StringIndexer().setInputCol("OriginState").setOutputCol("OriginStateInt")
	//flightsDF=oIndexer.fit(flightsDF).transform(flightsDF)
	
	//val dIndexer = new  StringIndexer().setInputCol("DestState").setOutputCol("DestStateInt")
        //flightsDF=dIndexer.fit(flightsDF).transform(flightsDF)



	//Remove variables we do not consider appropriate for the ML algorithms (also the string version of UniqueCarrier)

	flightsDF = flightsDF.drop("FlightNum").drop("TailNum").drop("Origin").drop("Dest").drop("DayOfMonth").drop("Year").drop("UniqueCarrier").drop("DestState").drop("OriginState")
	//Remove rows with null values for the remaining variables
	flightsDF = flightsDF.na.drop()



 //OneHotEncoder to create dummy variables for carrier, month and day of the week 
 //Linear regression needs them to handle those categorical variables properly
var flightsDFReg=flightsDF

 val encoder = new OneHotEncoder().setInputCol("DayOfWeek").setOutputCol("dummyDayOfWeek")
 val encoder2 = new OneHotEncoder().setInputCol("Month").setOutputCol("dummyMonth")
 val encoder3 = new OneHotEncoder().setInputCol("UniqueCarrierInt").setOutputCol("dummyUniqueCarrier")
 //val encoder4 = new OneHotEncoder().setInputCol("OriginStateInt").setOutputCol("dummyOriginState")
 //val encoder5 = new OneHotEncoder().setInputCol("DestStateInt").setOutputCol("dummyDestState") 
flightsDFReg = encoder.transform(flightsDFReg)
 flightsDFReg = encoder2.transform(flightsDFReg)
 flightsDFReg = encoder3.transform(flightsDFReg)
//flightsDFReg = encoder4.transform(flightsDFReg)
//flightsDFReg = encoder5.transform(flightsDFReg)

 //Remove the original variables not to use them in regression
 flightsDFReg = flightsDFReg.drop("DayOfWeek").drop("Month").drop("UniqueCarrierInt").drop("OriginStateInt").drop("DestStateInt") 
// Split the data into training and test sets (30% held out for testing).
val Array(trainingDataR, testDataR) = flightsDFReg.randomSplit(Array(0.7, 0.3), 100) // last parameter is the seed

//We use different data name for this algorithm because of the dummy variables, they are different for the tree models.

 val assemblerReg = new VectorAssembler()
 	.setInputCols(flightsDFReg.drop("ArrDelay").columns)
 	.setOutputCol("features")

//Defining the model


val lr = new LinearRegression()
.setFeaturesCol("features")
.setLabelCol("ArrDelay")
.setMaxIter(100)
.setElasticNetParam(1)
//Preparing the pipeline

val regressionPipeline = new Pipeline().setStages(Array(assemblerReg, lr))

//Evaluating the result

var evaluator = new RegressionEvaluator()
  .setLabelCol("ArrDelay")
  .setPredictionCol("prediction")
  .setMetricName("rmse")
//To tune the parameters of the model

val paramGrid = new ParamGridBuilder()
  //.addGrid(lr.getParam("elasticNetParam"), Array(0.0,0.5,1.0))
  .addGrid(lr.getParam("regParam"), Array(0.1, 1.0,10.0))
  .build()

val cv = new CrossValidator()
  .setEstimator(regressionPipeline)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(3)

//Fitting the model to our data
val rModel = cv.fit(trainingDataR)
//Making predictions
var predictions  = rModel.transform(testDataR)


val rmseRegression = evaluator.evaluate(predictions)


//Trees

//Prepare the assembler that will transform the remaining variables to a feature vector for the ML algorithms

val assembler = new VectorAssembler()
 .setInputCols(flightsDF.drop("ArrDelay").columns)
 .setOutputCol("features")

// Split the data into training and test sets (30% held out for testing). Will be used in both tree algorithms
val Array(trainingData, testData) = flightsDF.randomSplit(Array(0.7, 0.3))

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
spark.stop()
  }

}
