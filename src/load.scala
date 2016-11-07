import spark.implicits._

val project = "/project"
val archive = "2000"

// Read csv file with headers from hdfs
val flightsDF = spark.read.format("csv").option("header", "true").load("hdfs://"+project+"/"+archive+".csv")

// Print schema
flightsDF.printSchema
