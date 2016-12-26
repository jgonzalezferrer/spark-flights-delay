# spark-flights-delay: Predicting the arrival delay of US commercial flights using Spark and MLlib.

The goal of this project is to develop a Big Data application using Spark and the library MLlib in order to predict the arrival delay of commercial fights. This task is not easy and a good amount of structured information is necessary to obtain a good performance. The [US Department of Transportation](http://stat-computing.org/dataexpo/2009/the-data.html) provides all the data regarding the US commercial flights from 1987 to 2008.

Instructions on how to execute
----------- 
First of all, it is necessary to execute the <i>script.sh</i> provided in the main folder of the project. It will download
and place in HDFS the external [airports.csv](ttp://stat-computing.org/dataexpo/2009/supplemental-data.html) dataset. This file is used internally in the project to create extra
variables.

The whole Spark application has been developed and tested as a Maven project. The way to easily compile it is to be located in the sparkapp folder and execute the mvn programme with the option package. Once compiled, run the Spark application using the spark-submit script located in the bin folder of your Spark module. The script must be called using one parameter: the hdfs location of the datasets.

````
$ mvn package
$ spark-submit --class upm.bd.App target/sparkapp-1.0-SNAPSHOT.jar /hdfs/path/to/datasets/
````

