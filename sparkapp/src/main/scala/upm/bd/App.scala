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

	var flights = new Flights(spark)
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

				/*
	flights.df = flights.df.join(airportsDF, flights.df("Dest") === airportsDF("iata"))
				.withColumnRenamed("lat", "DestLat")
				.withColumnRenamed("long", "DestLong")
				.drop("iata")
	

	 Discarding unused variables */
	flights.df = flights.df.drop("DepTime").drop("Cancelled")
						.drop("CancellationCode").drop("FlightNum")
						.drop("TailNum").drop("DayOfWeek")
						.drop("Month").drop("UniqueCarrier")
						.drop("UniqueCarrierInt")
						.drop("Origin").drop("Dest")


	/* Null treatment */
	// We discard all the rows with at least one null value since they represent a reasonably low amount (<1%).
	flights.df = flights.df.na.drop()

	// TODO: remove this
	flights.df = flights.df.sample(false, 0.005, 100) // Last parameter is the seed

	/* Machine learning part */

	// Split the data into training and test sets (30% held out for testing).
	val Array(trainingData, testData) = flights.df.randomSplit(Array(0.7, 0.3), 100) // last parameter is the seed

	val targetVariable = "ArrDelay"

	//Prepare the assembler that will transform the remaining variables to a feature vector for the ML algorithms
	val assembler = new VectorAssembler()
			.setInputCols(flights.df.drop(targetVariable).columns)
			.setOutputCol("features")	
	
	flights.setAssembler(assembler)

	//Evaluating the result
	val evaluator = new RegressionEvaluator()
			.setLabelCol(targetVariable)
			.setPredictionCol("prediction")
			.setMetricName("rmse")

	flights.setEvaluator(evaluator)


	// Linear Regression
	flights.linearRegression(targetVariable, 100, 1, 3, Array(0.1, 1.0, 10.0))	
	val lrModel = flights.linearRegressionModel.fit(trainingData)
	val lrPredictions = lrModel.transform(testData)
	val rmseRegression = flights.evaluator.evaluate(lrPredictions)
	
	// Random Forest
	flights.randomForest(targetVariable, 15)
	val rfModel = flights.randomForestModel.fit(trainingData)
	val rfPredictions = rfModel.transform(testData)
	val rmseRandom = flights.evaluator.evaluate(rfPredictions)

	//Boosting trees
	flights.boostingTrees(targetVariable, 15, 10)
	val btModel = flights.boostingTreesModel.fit(trainingData)
	val btPredictions = btModel.transform(testData)
	val rmseBoosting = flights.evaluator.evaluate(btPredictions)

	println("rmse for different algorithms: ")
	println("Linear regression = "+rmseRegression)
	println("Random forests = "+rmseRandom)
	println("Boosting trees = "+rmseBoosting)

	}
}
