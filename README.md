# synth90k_training

This repo shows an example that how to train synth-90k dataset using Keras and reach 90.8% of training accruacy.
The main architecture is combining CNN and LSTM layer together instead of naive CNN and fully-connected layer.

## Preprocessing
`chmod +x ./*.py`

Use the above command to change all the python file to executable file.

`./preprocessing.py`

Run preprocessing.py
Please wait for all the job complete and you can go to the next step.
Note that you will need about 250GB of free space on your hard disk.


## Train the model
`./build_archi.py`

This step will generate the model and then you can start training on the next step.

`./train_model.py`

You are now starting to train the whole model.
It takes about 18 hours of training time on a single NVIDIA GTX 1060 6GB video card.
The terminal will show the final training result and the accuracy after the training is complete.
