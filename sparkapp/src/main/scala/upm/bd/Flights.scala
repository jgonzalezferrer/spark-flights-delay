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

class Flights(spark: SparkSession) {

	import spark.implicits._

	var df: DataFrame = null
	var linearRegressionModel: LinearRegression = null
	var linearRegressionEvaluator: RegressionEvaluator = null

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

		//OneHotEncoder to create dummy variables for carrier, month and day of the week 
		//Linear regression needs them to handle those categorical variables properly.
		val dayEncoder = new OneHotEncoder().setInputCol("DayOfWeek").setOutputCol("dummyDayOfWeek")
		val monthEncoder = new OneHotEncoder().setInputCol("Month").setOutputCol("dummyMonth")
		val carrierEncoder = new OneHotEncoder().setInputCol("UniqueCarrierInt").setOutputCol("dummyUniqueCarrier")
		df = dayEncoder.transform(df)
		df = monthEncoder.transform(df)
		df = carrierEncoder.transform(df)
	}

	def linearRegression(trainingData: DataFrame, testData: DataFrame, targetVariable: String, maxIter: Int, elasticNetParameter: Int, k: Int, hyperparameters: Array[Double]){ 

		val assemblerReg = new VectorAssembler()
			.setInputCols(df.drop(targetVariable).columns)
			.setOutputCol("features")		

		//Defining the model

		val lr = new LinearRegression()
			.setFeaturesCol("features")
			.setLabelCol(targetVariable)
			.setMaxIter(maxIter)
			.setElasticNetParam(elasticNetParameter)

		//Preparing the pipeline
		val regressionPipeline = new Pipeline().setStages(Array(assemblerReg, lr))

		//Evaluating the result
		linearRegressionEvaluator = new RegressionEvaluator()
			.setLabelCol("ArrDelay")
			.setPredictionCol("prediction")
			.setMetricName("rmse")

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


}

