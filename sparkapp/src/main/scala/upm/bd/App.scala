package upm.bd

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.unix_timestamp
import org.apache.spark.sql.functions.{concat, lit}
import org.apache.spark.sql.functions.udf
import org.apache.log4j.{Level, Logger}

object App {
  def main(args : Array[String]) { 
        
        Logger.getRootLogger().setLevel(Level.WARN)
	
        val spark = SparkSession
		.builder()
		.appName("Spark Flights Delay")
		.getOrCreate()
   	
	import spark.implicits._

	val project = "/project"
	val archive = "2000"

	// Read csv file with headers from hdfs
	var flightsDF = spark.read.format("com.databricks.spark.csv").option("header", "true").load("hdfs://"+project+"/"+archive+".csv")

	// Print schema
	flightsDF.printSchema


  }

}
