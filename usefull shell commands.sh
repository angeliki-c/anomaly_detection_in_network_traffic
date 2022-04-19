# assign the desired permission to the folder of the project for accessing code + data
sudo chmod 700 -R the/location/of/your/project/anomaly_detection_in_network_traffic
# start the ssh client and server
sudo service ssh --full-restart
# start hadoop
start-dfs.sh
# copy the data to hadoop file system
hdfs dfs -put anomaly_detection_in_network_traffic/data/ hdfs://localhost:9000/user/

pyspark