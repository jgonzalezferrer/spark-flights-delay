package upm.bd

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame

class Flights(spark: SparkSession) {

	import spark.implicits._

	var flights: DataFrame = null

	// Read all csv files with headers from hdfs.
	// The valid columns are selected, casting them (the default type is String).
	def load(hdfsPath: String){
		flights = spark.read
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
}

