#https://ucla.box.com/s/nv9td9kvvfvg3tya0dlvbs1kn5o87gmv
#Download tweet data from above mentioned url and folder in tweet_data in project directory

import os
import json
import time
from Project4.Functions import *

start_time = time.time()

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in

hashtags = ["gohawks","gopatriots","nfl","patriots","sb49","superbowl"]

#Q.1
print("--------------Q1----------------")
for hashtag in hashtags:
    getHashTagStats(hashtag)

print("--- %s seconds ---" % (time.time() - start_time))
