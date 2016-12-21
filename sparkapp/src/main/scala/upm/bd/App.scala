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

	/* Discarding data points */
	//Drop rows with null values in the target variable	
	flightsDF = flightsDF.na.drop(Array("ArrDelay"))

	/* Transformation of variables */

	//Convert scheduled departure and arrival time to TimeStamp
	flightsDF = dateToTimeStamp(flightsDF, "CRSDepTime")
	flightsDF = dateToTimeStamp(flightsDF, "CRSArrTime")

	// Normalize UNIX time, we take as reference point the earliest date in the database.
	val timeStampReference = unix_timestamp(lit("01/01/1987"), "dd/MM/yy")
	flightsDF = flightsDF.withColumn("CRSDepTime", $"CRSDepTime" - timeStampReference)
	flightsDF = flightsDF.withColumn("CRSArrTime", $"CRSArrTime" - timeStampReference)

	//Cast variables to Double due to machine learning methods restrictions.
	flightsDF = flightsDF.withColumn("DayOfMonth", col("DayOfMonth").cast(DoubleType))
	flightsDF = flightsDF.withColumn("CRSDepTime", col("CRSDepTime").cast(DoubleType))
	flightsDF = flightsDF.withColumn("CRSArrTime", col("CRSArrTime").cast(DoubleType))
	flightsDF = flightsDF.withColumn("Year", col("Year").cast(DoubleType))
	flightsDF = flightsDF.withColumn("Month", col("Month").cast(DoubleType))

	//StringIndexer to transform the UniqueCarrier string to integer for using it as a categorical variable.
	val sIndexer = new StringIndexer().setInputCol("UniqueCarrier").setOutputCol("UniqueCarrierInt")
	flightsDF = sIndexer.fit(flightsDF).transform(flightsDF)

	//OneHotEncoder to create dummy variables for carrier, month and day of the week 
	//Linear regression needs them to handle those categorical variables properly.
	val dayEncoder = new OneHotEncoder().setInputCol("DayOfWeek").setOutputCol("dummyDayOfWeek")
	val monthEncoder = new OneHotEncoder().setInputCol("Month").setOutputCol("dummyMonth")
	val carrierEncoder = new OneHotEncoder().setInputCol("UniqueCarrierInt").setOutputCol("dummyUniqueCarrier")
	
	flightsDF = dayEncoder.transform(flightsDFReg)
	flightsDF = monthEncoder.transform(flightsDFReg)
	flightsDF = carrierEncoder.transform(flightsDFReg)


	/* Adding new variables */
	val airportsDF = spark.read
		.format("com.databricks.spark.csv")
		.option("header", "true")
		.load("hdfs:///project/extra/airports.csv")
		.select(col("iata"),
				col("lat").cast(DoubleType),
				col("long").cast(DoubleType))


	// New columns: lat and long of the Origin airports.
	flightsDF = flightsDF.join(airportsDF, flightsDF("Origin") === airportsDF("iata"))
				.withColumnRenamed("lat", "OriginLat")
				.withColumnRenamed("long", "OriginLong")
				.drop("iata")

	flightsDF = flightsDF.join(airportsDF, flightsDF("Dest") === airportsDF("iata"))
				.withColumnRenamed("lat", "DestLat")
				.withColumnRenamed("long", "DestLong")
				.drop("iata")


	/* Discarding unused variables */
	flightsDF = flightsDF.drop("DepTime").drop("Cancelled")
						 .drop("CancellationCode").drop("FlightNum")
						 .drop("TailNum")drop("DayOfWeek")
						 .drop("Month").drop("UniqueCarrierInt")
						 .drop("Origin").drop("Dest") 

	/* Null treatment */
	// We discard all the rows with at least one null value since they represent a reasonably low amount (<1%).
	flightsDF = flightsDF.na.drop()

	//////////////// Machine learning pipes ////////////////

	// TODO: remove this
	
	flightsDF=flightsDF.sample(false, 0.005, 100) // Last parameter is the seed
	//Linear regression	

	//Check the stardard deviation and mean of the target variable
	val dMean =flightsDF.select(mean("ArrDelay")).take(1)(0)(0)
	val dStDev=flightsDF.select(stddev("ArrDelay")).take(1)(0)(0)

	
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

	var paramGrid = new ParamGridBuilder()
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
	val Array(trainingData, testData) = flightsDF.randomSplit(Array(0.7, 0.3), 100) //Last parameter is the seed

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

	println("Mean of arrival delay = "+dMean)
	println("Standard deviation of arrival delay = "+dStDev)
	println("rmse for different algorithms: ")
	println("Linear regression = "+rmseRegression)
	println("Random forests = "+rmseRandom)
	println("Boosting trees = "+rmseBoosintg)
	spark.stop()
}

