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
	def main(args : Array[String]) {

	// Disabling debug option. 
	Logger.getRootLogger().setLevel(Level.WARN)

	val spark = SparkSession
			.builder()
			.appName("Spark Flights Delay")
			.getOrCreate()
 
	import spark.implicits._

	val targetVariable = "ArrDelay"
	var flights = new Flights(spark, targetVariable)
	// TODO: pass as a programme parameter
	flights.load("hdfs:///project/flights/*.csv")
	
	/* Discarding data points */
	//Drop rows with null values in the target variable	
	flights.df = flights.df.na.drop(Array("ArrDelay"))	

	/* Transformation of variables */
	flights.variablesTransformation()

	/* Adding new variables */	
	val airportsDF = spark.read
		.format("com.databricks.spark.csv")
		.option("header", "true")
		.load("hdfs:///project/extra/airports.csv")
		.select(col("iata"),
				col("lat").cast(DoubleType),
				col("long").cast(DoubleType))

	// New columns: lat and long of the Origin airports.
	flights.df = flights.df.join(airportsDF, flights.df("Origin") === airportsDF("iata"))
				.withColumnRenamed("lat", "OriginLat")
				.withColumnRenamed("long", "OriginLong")
				.drop("iata")

				
	flights.df = flights.df.join(airportsDF, flights.df("Dest") === airportsDF("iata"))
				.withColumnRenamed("lat", "DestLat")
				.withColumnRenamed("long", "DestLong")
				.drop("iata")
	
	// TODO: remove this
	flights.df = flights.df.sample(false, 0.5, 100) // Last parameter is the seed

	//Discarding unused variables 
	flights.df = flights.df.drop("DepTime").drop("Cancelled")
						.drop("CancellationCode").drop("FlightNum")
						.drop("TailNum").drop("UniqueCarrier")
						.drop("Year").drop("DayOfMonth")
						.drop("Origin").drop("Dest")


	/* Null treatment */
	// We discard all the rows with at least one null value since they represent a reasonably low amount (<1%).
	flights.df = flights.df.na.drop()

	// We will take the standard deviation to use it as a baseline.
	// We will compare the other methods againts this naive method.
	val dStDev = flights.df.select(stddev("ArrDelay")).take(1)(0)(0)
	
	/* Modification of variable to use them in the regression part */
	//OneHotEncoder to create dummy variables for carrier, month and day of the week 
	val dayEncoder = new OneHotEncoder().setInputCol("DayOfWeek").setOutputCol("dummyDayOfWeek")
	val monthEncoder = new OneHotEncoder().setInputCol("Month").setOutputCol("dummyMonth")
	val carrierEncoder = new OneHotEncoder().setInputCol("UniqueCarrierInt").setOutputCol("dummyUniqueCarrier")

	// Just for regression
	//Linear regression needs these variables to handle those categorical variables properly.
	flights.df = dayEncoder.transform(flights.df)
	flights.df = monthEncoder.transform(flights.df)
	flights.df = carrierEncoder.transform(flights.df)

	/* Training and Test dataset */
	// Split the data into training and test sets (30% held out for testing).
	var Array(trainingData, testData) = flights.df.randomSplit(Array(0.7, 0.3), 100) // last parameter is the seed

	/* Linear Regression */
	// For regression, we will use the OneHotEncoder variables, for Random Forests and Boosting Trees we will use the original ones.
	var trainingDataR = trainingData
	var testDataR = testData

	// Drop old variables.
	trainingDataR = trainingDataR.drop("DayOfWeek")
							.drop("Month").drop("UniqueCarrierInt")
	testDataR = testDataR.drop("DayOfWeek")
							.drop("Month").drop("UniqueCarrierInt")

	// Training the model
	flights.linearRegression(trainingDataR, 100, 1, 3, Array(0.1, 1.0, 10.0))	
	val lrModel = flights.linearRegressionModel.fit(trainingDataR)
	val lrPredictions = lrModel.transform(testDataR)
	val rmseRegression = flights.evaluator.evaluate(lrPredictions)


	// For Random forest and bosting trees, we discard the OneHotEncoder variables, keeping the old ones.
	trainingData = trainingData.drop("dummyDayOfWeek")
							.drop("dummyMonth").drop("dummyUniqueCarrierInt")
	testData = testData.drop("dummyDayOfWeek")
							.drop("dummyMonth").drop("dummyUniqueCarrierInt")

	/* Random Forest */
	flights.randomForest(trainingData, 15)
	val rfModel = flights.randomForestModel.fit(trainingData)
	val rfPredictions = rfModel.transform(testData)
	val rmseRandom = flights.evaluator.evaluate(rfPredictions)

	/* Boosting trees */
	flights.boostingTrees(trainingData, 15, 10)
	val btModel = flights.boostingTreesModel.fit(trainingData)
	val btPredictions = btModel.transform(testData)
	val rmseBoosting = flights.evaluator.evaluate(btPredictions)

	println("Standard deviation of arrival delay = "+dStDev)
	println("rmse for different algorithms: ")
	println("Linear regression = "+rmseRegression)
	println("Random forests = "+rmseRandom)
	println("Boosting trees = "+rmseBoosting)

	}
}
