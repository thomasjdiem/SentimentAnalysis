from train import Predict
import json
from math import e
from collections import defaultdict

with open("weights.json","r") as infile:
    weights = json.loads(infile.read())

weights = defaultdict(int,weights)

while True:
    text = input("Enter text (Q to quit): ")
    if text.lower() == "q":
        break
    else:
        x = Predict(text,weights)
        certainty = 1/(1+e**-x)
        if certainty > 0.5:
            print(f"Positive {certainty*100:0.2f}%")
        else:
            print(f"Negative {(1-certainty)*100:0.2f}%")