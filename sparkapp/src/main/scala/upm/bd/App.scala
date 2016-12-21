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

	flights.df = flights.df.join(airportsDF, flights.df("Dest") === airportsDF("iata"))
				.withColumnRenamed("lat", "DestLat")
				.withColumnRenamed("long", "DestLong")
				.drop("iata")


	/* Discarding unused variables */
	flights.df = flights.df.drop("DepTime").drop("Cancelled")
						.drop("CancellationCode").drop("FlightNum")
						.drop("TailNum").drop("DayOfWeek")
						.drop("Month").drop("UniqueCarrier")
						.drop("UniqueCarrierInt")
						.drop("Origin").drop("Dest") 

	/* Null treatment */
	// We discard all the rows with at least one null value since they represent a reasonably low amount (<1%).
	flights.df = flights.df.na.drop()

	flights.df = flights.df.sample(false, 0.0005, 100) // Last parameter is the seed
	/* Machine learning part */

	// Split the data into training and test sets (30% held out for testing).
	//val Array(trainingData, testData) = flights.df.randomSplit(Array(0.7, 0.3), 100) // last parameter is the seed

	//flights.linearRegression(trainingData, testData, "ArrDelay", 100, 1, 2, Array(0.1))
	var flightsDFReg = flights.df
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
	.addGrid(lr.getParam("regParam"), Array(0.1))
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
	println(rmseRegression)
	
 }
}
