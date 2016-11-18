import spark.implicits._
import org.apache.spark.sql.types._

val project = "/project"
val archive = "2000"

// Read csv file with headers from hdfs
var flightsDF = spark.read.format("csv").option("header", "true").load("hdfs://"+project+"/"+archive+".csv")

// Print schema
flightsDF.printSchema

// Forbidden variables
val forbiddenVariables = Array("ArrTime", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay", "TaxiIn", "Diverted", "ActualElapsedTime", "AirTime")

// Creating allowed variables extracting the forbidden variables from the original ones.
val allowedVariables = flightsDF.columns.filter(x => !forbiddenVariables.contains(x))
// Convert them into column format.
val allowedColumns = allowedVariables.map(x => col(x))

// Selecting the just the allowed columns.
flightsDF = flightsDF.select(allowedColumns:_*)

// Transforming the type of variables.
flightsDF = flightsDF.withColumn("Year", flightsDF("Year").cast(IntegerType))
flightsDF = flightsDF.withColumn("Month", flightsDF("Month").cast(IntegerType))
flightsDF = flightsDF.withColumn("DayOfMonth", flightsDF("DayOfMonth").cast(IntegerType))
flightsDF = flightsDF.withColumn("DayOfWeek", flightsDF("DayOfWeek").cast(IntegerType))
flightsDF = flightsDF.withColumn("CRSElapsedTime", flightsDF("CRSElapsedTime").cast(IntegerType))
flightsDF = flightsDF.withColumn("ArrDelay", flightsDF("ArrDelay").cast(IntegerType))
flightsDF = flightsDF.withColumn("DepDelay", flightsDF("DepDelay").cast(IntegerType))
flightsDF = flightsDF.withColumn("Distance", flightsDF("Distance").cast(IntegerType))
flightsDF = flightsDF.withColumn("TaxiOut", flightsDF("TaxiOut").cast(IntegerType))
flightsDF = flightsDF.withColumn("Cancelled", flightsDF("Cancelled").cast(BooleanType))

