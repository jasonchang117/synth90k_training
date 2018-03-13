# synth90k_training

This repo shows an example that how to train synth-90k dataset using Keras and reach 90.8% of training accruacy.
The main architecture is combining CNN and LSTM layer together instead of naive CNN and fully-connected layer.

## Preprocessing
First, use "chmod +x ./*.py" to change all the python file to executable file.
Then run preprocessing.py
Please wait for all the job complete and you can go to the next step.
Note that you will need about 250GB free space on your hard disk.


## Train the model
Run build_archi.py
This code will generate the model and then you can start training on the next step.

Run train_model.py
This code will start training the whole model.
It takes about 18 hours of training time on a single NVIDIA GTX 1060 6GB video card.
The terminal will show the final training result and the accuracy.
