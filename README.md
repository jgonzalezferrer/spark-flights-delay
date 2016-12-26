# spark-flights-delay: Predicting the arrival delay of US commercial flights using Spark and MLlib.

The goal of this project is to develop a Big Data application using Spark and the library MLlib in order to predict the arrival delay of commercial fights. This task is not easy and a good amount of structured information is necessary to obtain a good performance. The [US Department of Transportation](http://stat-computing.org/dataexpo/2009/the-data.html) provides all the data regarding the US commercial flights from 1987 to 2008.

Notice that the original datasets contain a total of 29 different variables. Since the goal of the model is to predict the 
flight arrival delay, we are not allowed to use variables that contain information only known after the take off. 

Instructions on how to execute
----------- 
First of all, it is necessary to execute the ````script.sh```` located in the data folder of the project. It will download
and place in HDFS the external [airports.csv](ttp://stat-computing.org/dataexpo/2009/supplemental-data.html) dataset. This file is used internally in the project to create extra
variables.

The whole Spark application has been developed and tested as a Maven project. The way to easily compile it is to be located in the project main folder and execute the mvn programme with the option package. Once compiled, run the Spark application using the spark-submit script located in the bin folder of your Spark module. The script must be called using one parameter: the hdfs location of the datasets.

````
$ mvn package
$ spark-submit --class upm.bd.App target/sparkapp-1.0-SNAPSHOT.jar /hdfs/path/to/datasets/
````

Machine Learning models
----------- 

We have decided to implement three Machine Learning regression models. MLlib has a large set of machine learning methods and also provides the possibility to define [Pipelines](https://spark.apache.org/docs/2.0.2/ml-pipeline.html), which easily represent learning workflows composed of preprocessing steps, inference algorithms and validation of the data.

* [Linear regression](https://spark.apache.org/docs/2.0.2/ml-classification-regression.html#linear-regression): which can be a good
idea when dealing with a signfficant amount of continuous variables.

* [Random Forests regression](https://spark.apache.org/docs/2.0.2/ml-classification-regression.html#random-forest-regression): gives the advantage to partition the input space and treat regression differently for certain subspaces of it according to some categorical splits of the data.

* [Gradient-Bosted tree regression](https://spark.apache.org/docs/2.0.2/ml-classification-regression.html#gradient-boosted-tree-regression): it is also an ensemble technique of Decision Trees. However, in this case, there is no random selection of the feature to split the data at each node. These trees are built placing more importance to some training examples that have been badly predicted by previously trained trees.




