import ast
from kafka import KafkaConsumer
import openscoring
import os

os = openscoring.Openscoring("http://localhost:8080/openscoring")

kwargs = {"auth" : ("admin", "adminadmin")}

os.deploy("CCPP", "/home/gopinathankm/jpmml-sklearn-master/ccpp.pmml", **kwargs)

consumer = KafkaConsumer('power', bootstrap_servers='localhost:9092')

for message in consumer:
    arguments =message.value 
    
    argsdict = arguments.decode("utf-8") 
    dict = ast.literal_eval(argsdict)
    print(dict)

    result = os.evaluate("CCPP", dict)
    print(result)
