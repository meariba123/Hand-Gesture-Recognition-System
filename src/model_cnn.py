# week 2 - my CNN from scratch 
from tensorflow.keras import layers, models

#needs explaining 
def create_cnn(num_classes=4): 
    model = models.Sequential([ #layers are stacked after one another 
        layers.Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)), #number of filters are 32,3,3 a small 
        layers.MaxPooling2D((2,2)), #pooling reduces image size by taking max value line in every 2x2 area

        layers.Conv2D(64, (3,3), activation="relu"), 
        layers.MaxPooling2D((2,2)), #this is important because it helps the model to focus on important features like the hands and ignore noise - reduces noise 

        layers.Flatten(), #converts the 2D feauture maps into a 1D vector.
        layers.Dense(128, activation="relu"), #decision making layer combining all learned features.
        layers.Dropout(0.5), #prevents overfitting by 50% of neauron during training so we dont rely too much on one specific neuron.
        layers.Dense(num_classes, activation="softmax") #num classes is 3 for now 3 output neurons - rock paper scissors  one per class. and softmax turns them into probabilities
    ])
    model.compile(optimizer="adam", #adjusts weights efficiently to minimise loss 
                  loss="categorical_crossentropy", #used for multi-class classification
                  metrics=["accuracy"]) #we’ll track accuracy during training.
    return model #returns the built and compiled CNN model so you can train it
