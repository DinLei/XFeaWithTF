export HADOOP_HOME=/home/bigdata/software/hadoop/
export HADOOP_HDFS_HOME=$HADOOP_HOME
export PATH=$PATH:$HADOOP_HOME/bin
export JAVA_HOME=/usr/java/jdk1.8.0_201
export HADOOP_HDFS_HOME=/home/bigdata/software/hadoop/
export PATH=$PATH:$HADOOP_HDFS_HOME/libexec/hadoop-config.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$JAVA_HOME/jre/lib/amd64/server
export PATH=$PATH:$HADOOP_HDFS_HOME/bin:$HADOOP_HDFS_HOME/sbin
export CLASSPATH="$(hadoop classpath --glob)"

rm -rf checkpoints_dir_v2/*
python RankModel_v2.py --task=train --checkpoints_dir=checkpoints_dir_v2
#python RankModel_v2.py
