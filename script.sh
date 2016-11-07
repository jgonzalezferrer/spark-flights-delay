archive=2000
folder=project

mkdir -p data
cd data

wget -nc http://stat-computing.org/dataexpo/2009/$archive.csv.bz2 
bzip2 -dk $archive.csv.bz2

hdfs -dfs -mkdir -p $folder
hdfs -dfs -put $archive.csv $project

cd ..
