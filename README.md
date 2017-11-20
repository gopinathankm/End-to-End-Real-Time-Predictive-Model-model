# End-to-End-Real-Time-Predictive-Model-model
End-to-End-Real Time Predictive Modelling using Scikit-Learn, JPMML and Openscoring Rest Webservice

This initiative is an attempt to achieve real time prediction of power output given a set of environmental readings from various sensors in a natural gas-fired power generation plant. Linear Regression model is inspired and adpated from the study of Dr. Fisseha Berhane, PhD, Using the Scikit-learn library  and Python 3 and perform Exploratory Data Analysis (EDA) on a real-world dataset, and then apply non-regularized linear regression to solve a supervised regression problem on the dataset. (http://datascience-enthusiast.com/R/ML_python_R_part1.html) 
Further attempted to  making this model to a real time predictive model.

1. Persisted Linear Regression model and generated a pickled file using Scikit-Learn and Python 3.

2. Converted serialized model to PMML format using JPMML.

3. Using Kafka producer to  generate random real time data power plant data and written to Kafka topic.

4. Using a Kafka Consumer which is actually an Openscoring Python client reads data from Kafka topic and predicts power generation Online using Openscoring REST Web service.

Getting Started

Prerequisites, or environment what is used.Softwares, Version and Download & Installation

1 Ubuntu
16.04 LTS
https://www.ubuntu.com/

2 Java
1.8.0_152
http://www.oracle.com/technetwork/java/javase/downloads/jre8-downloads-2133155.html

3 Python 3
3.6.3, Anaconda 5.0.1
https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh

4 Apache Maven
3.3.9
https://maven.apache.org/

5 Apache Zookeeper
3.4.10
https://zookeeper.apache.or

6 Apache Kafka
2_11.0.10.0.0
https://kafka.apache.org

7 NumPy
1.13.3
Conda install

8 SciPy
0.19.1
Conda install

9 Pandas
0.21.0
Conda install

10 Matplotlib
2.1.0 
Conda install

11 Seaborn
0.8.0 
Conda install

12 Scikit-Learn
0.19.1
Conda install

13 Sklearn-pandas 
1.6.0
Pip install

14 Sklearn-pmml
0.1.2
Pip install

15 SkLearn2pmml
0.26.0
Pip install

16 Kafka-python
1.3.5
Pip installation

17 JPMML
0.26.0
https://github.com/jpmml/jpmml-sklearn

18 Openscoring-python REST web service
https://github.com/openscoring/openscoring-python

19 xlrd
1.1.0
Pip installation

Getting started with. 

Step 1. Ensure all softwares are installed and verify the versions.

Step 2. Start Zookeeper server as instructed in installation document.

Step 3. Start Kafka server as instructed in installation document.

Step 4. Create a topic, “power” in Kafka broker.

Step 5. Make an separate environment in conda and activate the same.

Step 6. Run the python program: 

        $python ccppOpencore.py
        
        This will create a Linear Regression model and serialize using pickling.
        Adapted and modified from:  http://datascience-enthusiast.com/R/ML_python_R_part1.html
        On successfull execution it will generate a pickled file, “ccpp.pkl.z”.

Step 7. Go to JPMML installed directory, run the command: $java -jar target/converter-executable-1.4-SNAPSHOT.jar --pkl-input	      /home/gopinathankm/work/workspace/machine-learning/ccpp.pkl.z --pmml-output ccpp.pmml

        (convert the directory structure as applicable to your environment.)

        This will create a PMML file, called “ccpp.pmml”.

Step 8. Start the REST web service, Go to Openscoring-python installed directory, change to openscoring-server folder, run the            
        following command to start the openscoring server and keep the terminal as it is.
        
        $ java -jar target/server-executable-1.3-SNAPSHOT.jar

Step 9. On another terminal, Run the KafkaProducer program, kafka-producer.py 
  
        $ python kafka-producer.py
	      
        This will generate randomly the real time power plant data  and send to topic, power as  byte string.

Step 10. On another terminal , run the KafkaConsumer program, kafka-consumer.py  

        $ python kafka-consumer.py
	      
        This will read the topic, power and converts converts byte string to string and then to a python 	dictionary which                          
        contains the arguments for Openscoring REST web service. It invokes the REST web service using the arguments and get the              
        prediction for the power generated as	response. Results can be watched at kafka-consumer and Openscoring server     
        terminals.

(my_env36) gopinathankm@gopinathankm-Lenovo-G400 ~/anaconda3/envs/my_env36

$ python kafka_consumer.py

{'AT': 35.82, 'V': 46.56, 'AP': 1028.39, 'RH': 37.9}

{'PE': 443.19161716627207}

{'AT': 20.42, 'V': 66.18, 'AP': 1002.93, 'RH': 51.5}

{'PE': 442.0365251361153}

{'AT': 28.18, 'V': 33.93, 'AP': 1014.36, 'RH': 47.95}

{'PE': 457.8909955630907}

{'AT': 4.14, 'V': 76.24, 'AP': 1023.84, 'RH': 30.82}

{'PE': 451.47549183834303}

{'AT': 21.54, 'V': 29.88, 'AP': 1030.68, 'RH': 32.79}

{'PE': 467.7256185768422}

{'AT': 32.06, 'V': 28.13, 'AP': 1019.2, 'RH': 26.78}

{'PE': 456.8347823391881}

Openscoring Server terminal


Nov 20, 2017 9:44:20 AM org.openscoring.service.ModelResource evaluate

INFO: Received EvaluationRequest{id=null, arguments={AT=23.07, V=31.24, AP=1005.53, RH=50.48}}

Nov 20, 2017 9:44:20 AM org.openscoring.service.ModelResource evaluate

INFO: Returned EvaluationResponse{id=null, result={PE=463.4972706962021}}

Nov 20, 2017 9:44:21 AM org.openscoring.service.ModelResource evaluate

INFO: Received EvaluationRequest{id=null, arguments={AT=16.37, V=51.97, AP=1016.94, RH=35.16}}

Nov 20, 2017 9:44:21 AM org.openscoring.service.ModelResource evaluate

INFO: Returned EvaluationResponse{id=null, result={PE=455.88452677259056}}

Nov 20, 2017 9:44:22 AM org.openscoring.service.ModelResource evaluate

INFO: Received EvaluationRequest{id=null, arguments={AT=4.6, V=64.38, AP=1020.06, RH=97.28}}

Nov 20, 2017 9:44:22 AM org.openscoring.service.ModelResource evaluate

INFO: Returned EvaluationResponse{id=null, result={PE=464.9272268034214}}

Nov 20, 2017 9:44:23 AM org.openscoring.service.ModelResource evaluate

INFO: Received EvaluationRequest{id=null, arguments={AT=3.92, V=78.51, AP=1004.72, RH=87.27}}

Nov 20, 2017 9:44:23 AM org.openscoring.service.ModelResource evaluate

INFO: Returned EvaluationResponse{id=null, result={PE=452.8512764613673}}

Nov 20, 2017 9:44:24 AM org.openscoring.service.ModelResource evaluate

INFO: Received EvaluationRequest{id=null, arguments={AT=36.01, V=63.47, AP=1002.85, RH=42.26}}

Nov 20, 2017 9:44:24 AM org.openscoring.service.ModelResource evaluate

INFO: Returned EvaluationResponse{id=null, result={PE=428.34784854311886}}


Authors

Gopinathan Munappy

Balakrishnan Sreekumar Maliekal

K. Ajith Kumar 

George Mani
.

License
This project is licensed under the MIT License - see the LICENSE.md file for details

Acknowledgments
Hat tip to anyone who's code was used:  Dr. Fisseha Berhane, PhD
Inspiration : http://datascience-enthusiast.com/R/ML_python_R_part1.html
