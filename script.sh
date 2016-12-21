archive=2000
folder=/project

mkdir -p data
mkdir -p data/flights
mkdir -p data/extra
cd data/flights

wget -nc http://stat-computing.org/dataexpo/2009/$archive.csv.bz2 
bzip2 -dk $archive.csv.bz2

cd ..
cd extra
wget -nc http://stat-computing.org/dataexpo/2009/airports.csv
cd ..

hdfs dfs -mkdir -p $folder 
hdfs dfs -mkdir -p $folder/flights 
hdfs dfs -mkdir -p $folder/extra 
hdfs dfs -put ./flights/$archive.csv $folder/flights
hdfs dfs -put ./extra/airports.csv $folder/extra
