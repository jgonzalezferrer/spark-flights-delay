package upm.bd

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

/** This class models and manipulates the USA commercial flights dataset to train some machine learning methods.
 * 
 *  The Flights class stores a DataFrame of flights, which can be modified in order to use it as input of machine learning methods.
 *	Three machine learning methods are currently supported provided by the MLlib library:
 *	- Linear Regression Model
 *  - Random Forests
 *  - Boosting Trees
 *
 *	@param: spark, a Spark session 
 *  @param: targetVariable, the target variable to predict.
 *	@author: Antonio Javier González Ferrer
 *	@author: Aitor Palacios Cuesta
 */

class Flights(spark: SparkSession, targetVariable: String) {

	import spark.implicits._

	var df: DataFrame = null
	var evaluator: RegressionEvaluator = null
	var linearRegressionModel: CrossValidator = null	
	var randomForestModel: Pipeline = null
	var boostingTreesModel: Pipeline = null

	//Evaluator defined for the ML methods using the RMSE metric.
	evaluator = new RegressionEvaluator()
				.setLabelCol(targetVariable)
				.setPredictionCol("prediction")
				.setMetricName("rmse")

	/* Read all csv files with headers from HDFS.
	 *
	 * The valid columns are selected, casting them to their correct type (the default type is String).
	 * 
	 * @param: hdfsPath, the hdfs path of the datasets.
	 */
	def load(hdfsPath: String){
		df = spark.read
			.format("com.databricks.spark.csv")
			.option("header", "true")
			.load(hdfsPath)
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
	}

	/* Transform a column date in "dd/MM/yyyy HHmm" format to a Unix TimeStamp column
	 * 
	 * @param: df, the original dataframe
	 * @param: columnName, the column name to transform.
	 * @return: new dataframe with the TimeStamp column.
	 */
	def dateToTimeStamp(df: org.apache.spark.sql.DataFrame, columnName: String) : org.apache.spark.sql.DataFrame = {
	return df.withColumn(columnName, 
		unix_timestamp(concat(col("DayOfMonth"), lit("/"), col("Month"), lit("/"), col("Year"), lit(" "), col(columnName)), 
			"dd/MM/yyyy HHmm"))
	}  

	/* Transformation of initial variables to be suitable for the learning phase.
	 *
	 * - CRSDepTime and CRSArrTime are converted to TimeStamp.
	 * - Numerical variables changed to Double variables due to regression models limitations.
	 * - UniqueCarrier variable from String to Categorical using StringIndexer.
	 *
	 */
	def variablesTransformation(){
		//Convert scheduled departure and arrival time to TimeStamp
		df = dateToTimeStamp(df, "CRSDepTime")
		df = dateToTimeStamp(df, "CRSArrTime")

		// Normalize UNIX time, we take as reference point the earliest date in the database.
		val timeStampReference = unix_timestamp(lit("01/01/1987"), "dd/MM/yy")
		df = df.withColumn("CRSDepTime", $"CRSDepTime" - timeStampReference)
		df = df.withColumn("CRSArrTime", $"CRSArrTime" - timeStampReference)

		//Cast variables to Double due to machine learning methods restrictions.
		df = df.withColumn("DayOfMonth", col("DayOfMonth").cast(DoubleType))
		df = df.withColumn("CRSDepTime", col("CRSDepTime").cast(DoubleType))
		df = df.withColumn("CRSArrTime", col("CRSArrTime").cast(DoubleType))
		df = df.withColumn("Year", col("Year").cast(DoubleType))
		df = df.withColumn("Month", col("Month").cast(DoubleType))

		//StringIndexer to transform the UniqueCarrier string to integer for using it as a categorical variable.
		val sIndexer = new StringIndexer().setInputCol("UniqueCarrier").setOutputCol("UniqueCarrierInt")
		df = sIndexer.fit(df).transform(df)
	}

	/* Linear Regression method.
	 *
	 * @param: trainingData, the training data.
	 * @param: maxIter, max. number of iterations (if it does not converge before).
	 * @param: elasticNetParameter, the elastic net parameter from 0 to 1.
	 * @param: k, the number of folds in cross validation.
	 * @param: hyperaparameters, set of regularizer variables to be tune by cross validation.
	 */
	def linearRegression(trainingData: DataFrame, maxIter: Int, elasticNetParameter: Int, k: Int, hyperparameters: Array[Double]){ 	
		//Prepare the assembler that will transform the remaining variables to a feature vector for the ML algorithms
		val assembler = new VectorAssembler()
				.setInputCols(trainingData.drop(targetVariable).columns)
				.setOutputCol("features")

		// Defininf the model.
		val lr = new LinearRegression()
			.setFeaturesCol("features")
			.setLabelCol(targetVariable)
			.setMaxIter(maxIter)
			.setElasticNetParam(elasticNetParameter)

		//Preparing the pipeline to train and test the data with the regression algorithm.
		val regressionPipeline = new Pipeline().setStages(Array(assembler, lr))

		//To tune the parameters of the model.
		var paramGrid = new ParamGridBuilder()
			.addGrid(lr.getParam("regParam"), hyperparameters)
			.build()

		// Cross validation to select the best hyperparameter.
		linearRegressionModel = new CrossValidator()
			.setEstimator(regressionPipeline)
			.setEvaluator(evaluator)
			.setEstimatorParamMaps(paramGrid)
			.setNumFolds(k)
	}

	/* Random Forest method.
	 *
	 * @param: trainingData, the training data.
	 * @param: maxCategories, the max number of different categories to be considered as a categorical variable.
	 */
	def randomForest(trainingData: DataFrame, maxCategories: Int){

		//Prepare the assembler that will transform the remaining variables to a feature vector for the ML algorithms.
		val assembler = new VectorAssembler()
				.setInputCols(trainingData.drop(targetVariable).columns)
				.setOutputCol("features")

		//Vector Indexer to indicate that some variables are categorical, so they are treated properly by the algorithms.
		//In our case, DayOfWeek, Month, UniqueCarrier have less than 15 different classes, so they will be marked as categorical, as we want.
		var indexer = new VectorIndexer()
			.setInputCol("features")
			.setOutputCol("indexed")
			.setMaxCategories(maxCategories)

		//Defining the model
		val rf = new RandomForestRegressor()
			.setLabelCol(targetVariable)
			.setFeaturesCol("features")

		//Pipeline to train and test the data with the random forest.
		randomForestModel = new Pipeline().setStages(Array(assembler, indexer, rf))	
	}

	/* Boosting Trees method.
	 *
	 * @param: trainingData, the training data.
	 * @param: maxCategories, the max number of different categories to be considered as a categorical variable.
	 * @param: maxIter, the max number of iterations.
	 */
	def boostingTrees(trainingData: DataFrame, maxCategories: Int, maxIter: Int){
		//Prepare the assembler that will transform the remaining variables to a feature vector for the ML algorithms
		val assembler = new VectorAssembler()
				.setInputCols(trainingData.drop(targetVariable).columns)
				.setOutputCol("features")

		var indexer = new VectorIndexer()
				.setInputCol("features")
				.setOutputCol("indexed")
				.setMaxCategories(maxCategories)

		val gbt = new GBTRegressor()
				.setLabelCol("ArrDelay")
				.setFeaturesCol("features")
				.setMaxIter(maxIter)

		//Pipeline to train and test the data with the boosting algorithm
		boostingTreesModel = new Pipeline().setStages(Array(assembler, indexer, gbt))
	}
}