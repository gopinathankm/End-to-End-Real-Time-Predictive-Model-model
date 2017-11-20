import random
import time

from kafka import KafkaProducer
from kafka.errors import KafkaError
 
producer = KafkaProducer(bootstrap_servers='localhost:9092')
topic = "power"

for i in range(1000):
    AT = "19.651231" 
    V  = "54.305804"
    AP = "1013.259078"
    RH = "73.308978"

#                 AT            V           AP           RH
# min       1.810000    25.360000   992.890000    25.560000
# max      37.110000    81.560000  1033.300000   100.160000
    def getAT():
        return  str(round(random.uniform(2.0, 38.0),2))
     
    def getV():
        return str(round(random.uniform(26.0, 81.5),2)) 
         
    def getAP():
        return str(round(random.uniform(993.0, 1033.0),2))  

    def getRH():
        return str(round(random.uniform(26.0, 101.0),2))  

    
    message = "{\"AT\" : " + getAT() + "," + "\"V\" : " +getV() + "," + "\"AP\" : " +getAP() + "," + "\"RH\" : " + getRH() + "}" 
    producer.send(topic, key=str.encode('key_{}'.format(i)), value=(message.encode('utf-8')))

    time.sleep(1)
 
producer.close()
