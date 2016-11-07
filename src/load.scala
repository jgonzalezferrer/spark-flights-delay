import spark.implicits._

val project = "/project"
val archive = "2000"

// Read csv file with headers from hdfs
var flightsDF = spark.read.format("csv").option("header", "true").load("hdfs://"+project+"/"+archive+".csv")

// Print schema
flightsDF.printSchema

// Forbidden variables
val forbiddenVariables = Array("ArrTime", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay")

// Creating allowed variables extracting the forbidden variables from the original ones.
val allowedVariables = flightsDF.columns.filter(x => !forbiddenVariables.contains(x))
// Convert them into column format.
val allowedColumns = allowedVariables.map(x => col(x))

// Selecting the just the allowed columns.
flightsDF = flightsDF.select(allowedColumns:_*)

