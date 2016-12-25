folder=/project

wget -nc http://stat-computing.org/dataexpo/2009/airports.csv

hdfs dfs -mkdir -p $folder 
hdfs dfs -mkdir -p $folder/extra 
hdfs dfs -put airports.csv $folder/extra
