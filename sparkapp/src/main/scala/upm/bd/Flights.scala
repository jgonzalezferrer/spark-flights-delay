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
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

class Flights(spark: SparkSession, targetVariable: String) {

	import spark.implicits._

	var df: DataFrame = null
	var linearRegressionModel: CrossValidator = null
	var evaluator: RegressionEvaluator = null
	var assembler: VectorAssembler = null
	var randomForestModel: Pipeline = null
	var boostingTreesModel: Pipeline = null


	// Read all csv files with headers from hdfs.
	// The valid columns are selected, casting them (the default type is String).
	def load(hdfsPath: String){
		df = spark.read
			.format("com.databricks.spark.csv")
			.option("header", "true")
			//.load("hdfs://"+args(0)+"*.csv")
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

		//Prepare the assembler that will transform the remaining variables to a feature vector for the ML algorithms
		assembler = new VectorAssembler()
				.setInputCols(df.drop(targetVariable).columns)
				.setOutputCol("features")

		//Evaluating the result
		evaluator = new RegressionEvaluator()
				.setLabelCol(targetVariable)
				.setPredictionCol("prediction")
				.setMetricName("rmse")
	}

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

	def variablesTransformation(){
		//Convert scheduled departure and arrival time to TimeStamp
		df = dateToTimeStamp(df, "CRSDepTime")
		df = dateToTimeStamp(df, "CRSArrTime")

		// Normalize UNIX time, we take as reference point the earliest date in the database.
		val timeStampReference = unix_timestamp(lit("01/01/1987"), "dd/MM/yy")
		df = df.withColumn("CRSDepTime", $"CRSDepTime" - timeStampReference)
		df = df.withColumn("CRSArrTime", $"CRSArrTime" - timeStampReference)

		//Cast variables to Double deu to machine learning methods restrictions.
		df = df.withColumn("DayOfMonth", col("DayOfMonth").cast(DoubleType))
		df = df.withColumn("CRSDepTime", col("CRSDepTime").cast(DoubleType))
		df = df.withColumn("CRSArrTime", col("CRSArrTime").cast(DoubleType))
		df = df.withColumn("Year", col("Year").cast(DoubleType))
		df = df.withColumn("Month", col("Month").cast(DoubleType))

		//StringIndexer to transform the UniqueCarrier string to integer for using it as a categorical variable.
		val sIndexer = new StringIndexer().setInputCol("UniqueCarrier").setOutputCol("UniqueCarrierInt")
		df = sIndexer.fit(df).transform(df)

	}

	def linearRegression(maxIter: Int, elasticNetParameter: Int, k: Int, hyperparameters: Array[Double]){ 
	
		//Defining the model

		val lr = new LinearRegression()
			.setFeaturesCol("features")
			.setLabelCol(targetVariable)
			.setMaxIter(maxIter)
			.setElasticNetParam(elasticNetParameter)

		//Preparing the pipeline
		val regressionPipeline = new Pipeline().setStages(Array(assembler, lr))

		//To tune the parameters of the model
		var paramGrid = new ParamGridBuilder()
			.addGrid(lr.getParam("regParam"), hyperparameters)
			.build()

		linearRegressionModel = new CrossValidator()
			.setEstimator(regressionPipeline)
			.setEvaluator(evaluator)
			.setEstimatorParamMaps(paramGrid)
			.setNumFolds(k)
	}
	
	def randomForest(maxCategories: Int){
		//Vector Indexer to indicate that some variables are categorical, so they are treated properly by the algorithms
		//In our case, DayOfWeek, Month, UniqueCarrier have less than 15 different classes, so they will be marked as categorical, as we want
		var indexer = new VectorIndexer()
			.setInputCol("features")
			.setOutputCol("indexed")
			.setMaxCategories(maxCategories)

		//Defining the model
		val rf = new RandomForestRegressor()
			.setLabelCol(targetVariable)
			.setFeaturesCol("features")

		randomForestModel = new Pipeline().setStages(Array(assembler, indexer, rf))
		
	}

	def boostingTrees(maxCategories: Int, maxIter: Int){

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

