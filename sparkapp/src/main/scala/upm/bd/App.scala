package upm.bd

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.PipelineModel

/** Spark application that creates and compares machine learning models from MLlib for predicting the arrival delay of USA commercial flights.
 * 
 *  The development of this application follows the usual data science pipeline:
 *	- Load and process input data.
 * 	- Select, transform and manipulate the input variables, to prepare them for the ML models. 
 *	- Create and compare machine learning model that predicts the arrival delay time.
 *  - Validate the created model and provide some measure of its accuracy.
 *
 *	@param: the HDFS location of the dataset(s).
 *	@author: Antonio Javier González Ferrer
 *	@author: Aitor Palacios Cuesta
 */

object App {
	def main(args : Array[String]) {

	/* Configuration 
	 *
	 * Defining some Spark configuration parameters.
	 *
	 */

	// Disabling debug option. 
	Logger.getRootLogger().setLevel(Level.WARN)
	// Entry point to programming Spark with the Dataset and DataFrame API.
	val spark = SparkSession
			.builder()
			.appName("Spark Flights Delay")
			.getOrCreate() 
 	// Implicit methods available in Scala for converting Scala objects into DataFrames.
	import spark.implicits._


	/* Loading data
	 *
	 * We load the data from the HDFS, a highly fault-tolerant distributed file system. 
	 *
	 */

	val targetVariable = "ArrDelay"
	val datasetsPath = args(0)
	// Create a Flights object, where the DataFrame and the Machine Learning methods will be stored.
	var flights = new Flights(spark, targetVariable)	
	flights.load("hdfs://"+datasetsPath+"*.csv")
	

	/* Data Manipulation
	 *
	 * The dataset contains 29 initial variables. Some of these variables are not allowed to be used in this ML problem.
	 * Furthermore, some of the varibles need to be transformated, cleaned or discarded.
	 *
	 */
	//Drop rows with null values in the target variable	(not allowed in supervised learning).
	flights.df = flights.df.na.drop(Array("ArrDelay"))	

	// Transformation of variables for the learning phase. 
	flights.variablesTransformation()

	// Adding new varibles: lat and long of each airport.	
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

	// New columns: lat and long of the Dest airports.	
	flights.df = flights.df.join(airportsDF, flights.df("Dest") === airportsDF("iata"))
				.withColumnRenamed("lat", "DestLat")
				.withColumnRenamed("long", "DestLong")
				.drop("iata")

	//Discarding unused variables 
	flights.df = flights.df.drop("DepTime").drop("Cancelled")
						.drop("CancellationCode").drop("FlightNum")
						.drop("TailNum").drop("UniqueCarrier")
						.drop("Year").drop("DayOfMonth")
						.drop("Origin").drop("Dest")


	// Null treatment.
	// We discard all the rows with at least one null value since they represent a reasonably low amount (<1%).
	flights.df = flights.df.na.drop()

	// We will take the standard deviation to use it as a baseline.
	// We will compare the other methods against this naive method.
	val dStDev = flights.df.select(stddev("ArrDelay")).take(1)(0)(0)
	
	// Linear Regression method needs to define a special transformation for categorical variables.
	//OneHotEncoder to create dummy variables for carrier, month and day of the week 
	val dayEncoder = new OneHotEncoder().setInputCol("DayOfWeek").setOutputCol("dummyDayOfWeek")
	val monthEncoder = new OneHotEncoder().setInputCol("Month").setOutputCol("dummyMonth")
	val carrierEncoder = new OneHotEncoder().setInputCol("UniqueCarrierInt").setOutputCol("dummyUniqueCarrier")

	flights.df = dayEncoder.transform(flights.df)
	flights.df = monthEncoder.transform(flights.df)
	flights.df = carrierEncoder.transform(flights.df)


	/* Machine Learning methods
	 *
	 * We will choose three supervised machine learning to predict the arrival delay:
	 * - Linear Regression
	 * - Random Forest
	 * - Boosting Trees
	 * 
	 */

	// Training and Test datasets
	// Split the data into training and test sets (30% held out for testing).
	var Array(trainingData, testData) = flights.df.randomSplit(Array(0.7, 0.3))

	// We need to modify the variables in Linear Regression, converting the categorical into OneHotEncoder.
	// For regression, we will use the OneHotEncoder variables, for Random Forests and Boosting Trees we will use the original ones.
	var trainingDataR = trainingData
	var testDataR = testData

	// Drop old variables.
	trainingDataR = trainingDataR.drop("DayOfWeek")
							.drop("Month").drop("UniqueCarrierInt")
	testDataR = testDataR.drop("DayOfWeek")
							.drop("Month").drop("UniqueCarrierInt")

	// For Random forest and bosting trees, we discard the OneHotEncoder variables, keeping the old ones.
	trainingData = trainingData.drop("dummyDayOfWeek")
							.drop("dummyMonth").drop("dummyUniqueCarrierInt")
	testData = testData.drop("dummyDayOfWeek")
							.drop("dummyMonth").drop("dummyUniqueCarrierInt")


	/* Linear Regression Model
	 *	
	 * We would like to tune the regularisaiton hyperparameter in order to avoid overfitting. We perform a grid search
     * of three parameters using 3-fold cross-validation to find out the best possible combination of parameters.
	 *
	 * On the other hand, the max number of iterations will be 100 since it usually converges quite fast, 
	 * and the elastic net parameter is set to 1 (Lasso regularisation).
	 */
	val lrMaxNumIterations = 100
	val elasticNetParameter = 1
	val k = 3
	val regularisations = Array(0.1, 1.0, 10.0)
	flights.linearRegression(trainingDataR, lrMaxNumIterations, elasticNetParameter, k, regularisations)

	// Training the model	
	val lrModel = flights.linearRegressionModel.fit(trainingDataR)

	// Retrieving the best model of tuning selection.
	val pipeline = lrModel.bestModel.asInstanceOf[PipelineModel]
	val bestRegularizer = pipeline.stages(1).asInstanceOf[LinearRegressionModel].getRegParam

	// Validation
	val lrPredictions = lrModel.transform(testDataR)
	val rmseRegression = flights.evaluator.evaluate(lrPredictions)

	/* Random Forest Model
	 * 
	 * Usually works well in the presence of categorical variables. 
	 *
	 * The max number of categories to be considered as a categorical variable is set to 15.
	 *
	 */
	
	val rfMaxCategories = 15
	flights.randomForest(trainingData, rfMaxCategories)

	// Training the model
	val rfModel = flights.randomForestModel.fit(trainingData)

	// Validation
	val rfPredictions = rfModel.transform(testData)
	val rmseRandom = flights.evaluator.evaluate(rfPredictions)

	/* Boosting Trees Model
	 *
	 * We do not choose random selection of the feature to split the data node, as in Random Forest.
	 *
	 * The max number of categories to be considered a categorical variable is set to 15.
	 * The max number of iterations is 10.
	 */

	val btMaxCategories = 15
	val btMaxNumIterations = 10
	flights.boostingTrees(trainingData, btMaxCategories, btMaxNumIterations)

	// Training the model
	val btModel = flights.boostingTreesModel.fit(trainingData)

	// Validating the model
	val btPredictions = btModel.transform(testData)
	val rmseBoosting = flights.evaluator.evaluate(btPredictions)


	/* Validation
	 *
	 * We decide to use the root-mean-square error (RMSE) as the accuracy measure.
	 *
	 */

	// Baseline to improve
	println("Standard deviation of arrival delay = "+dStDev)

	// RMSE of the different ML techniques.
	println("rmse for different algorithms: ")
	println("Linear regression with best regularizer: "+bestRegularizer+" = "+rmseRegression)
	println("Random forests = "+rmseRandom)
	println("Boosting trees = "+rmseBoosting)

	}
}
