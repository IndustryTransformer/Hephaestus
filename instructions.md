# Transition to Encoder/Decoder Architecture

## Objective

I have an anomaly detection dataset. I currently have a decoder only dataset that I use
for planets.py but for the dataset in anomaly_detection.py this is no longer a good architecture.
I want to make this an encoder decoder model that takes the input variables as the encoder part
and then uses causal masking to classify the `class` of the anomaly.

To do this you will probably need to make a new data data set class and create a new model
architecture.

Please think through this carefully.
