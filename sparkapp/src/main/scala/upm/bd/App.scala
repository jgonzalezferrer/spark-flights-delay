package upm.bd

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.log4j.{Level, Logger}

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

  
  /*
   * Transform a day in number format to its corresponding String.
   */
  def toDayOfWeek: Double => String =  _ match {
	case 1 => "Monday"
	case 2 => "Tuesday"
	case 3 => "Wednesday"
	case 4 => "Thursday"
	case 5 => "Friday"
	case 6 => "Saturday"
	case 7 => "Sunday"
  }  
  
  /*
   * Transform a month in number format to its corresponding String.
   */
  def toMonth: Double => String =  _ match {
	case 1 => "January"
	case 2 => "February"
	case 3 => "March"
	case 4 => "April"
	case 5 => "May"
	case 6 => "June"
	case 7 => "July"
	case 8 => "August"
	case 9 => "September"
	case 10 => "October"
	case 11 => "November"
	case 12 => "December"
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
		.load("hdfs://"+project+"/*.csv")
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

	/* Removing data points */

	//Drop rows with cancelled flights since they do not contain information about the target variable.
	flightsDF = flightsDF.filter(col("Cancelled") === false)
	//Drop rows with null values in the target variable	
	flightsDF = flightsDF.na.drop(Array("ArrDelay"))
	
	/* Removing variables */
	flightsDF = flightsDF.drop("DepTime").drop("Cancelled").drop("CancellationCode")	

	/* Modifying variables */
	
	//Convert scheduled departure and arrival time to TimeStamp
	flightsDF = dateToTimeStamp(flightsDF, "CRSDepTime")
	flightsDF = dateToTimeStamp(flightsDF, "CRSArrTime")

	// Normalize UNIX time, we take as reference point the earliest year in the database.
	val timeStampReference = unix_timestamp(lit("01/01/1987"), "dd/MM/yy")
	flightsDF = flightsDF.withColumn("CRSDepTime", $"CRSDepTime" - timeStampReference)
	flightsDF = flightsDF.withColumn("CRSArrTime", $"CRSArrTime" - timeStampReference)

	 //Convert day of week to strings.
        val toDayOfWeekDF = udf(toDayOfWeek)
        flightsDF = flightsDF.withColumn("DayOfWeek", toDayOfWeekDF($"DayOfWeek"))
        //Convert month to strings.
        val toMonthDF = udf(toMonth)
        flightsDF = flightsDF.withColumn("Month", toMonthDF($"Month"))

	//Cast variables to their original types
        flightsDF.withColumn("DayOfMonth", col("DayOfMonth").cast(DoubleType))
        flightsDF.withColumn("CRSDepTime", col("CRSDepTime").cast(DoubleType))
        flightsDF.withColumn("CRSArrTime", col("CRSArrTime").cast(DoubleType))
	flightsDF.withColumn("Year", col("Year").cast(DoubleType))
	
	/* Adding new variables */
	
	// TODO: Change it as a program parameter. 
	val airportFile = "file:///root/javier/spark-flights-delay/data/extra/airports.csv"
	val airportsDF = spark.read
                .format("com.databricks.spark.csv")
                .option("header", "true")
                .load(airportFile)
                .select(col("iata"), col("state"))


	// New column: State of the Origin airports.
	flightsDF = flightsDF.join(airportsDF, flightsDF("Origin") === airportsDF("iata"), "left_outer").withColumnRenamed("state", "OriginState").drop("iata")
	// New column: State of the Dest airports.
	flightsDF = flightsDF.join(airportsDF, flightsDF("Dest") === airportsDF("iata"), "left_outer").withColumnRenamed("state", "DestState").drop("iata")
	

	flightsDF.show

  }

}
