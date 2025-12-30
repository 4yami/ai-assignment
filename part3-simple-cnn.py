import tensorflow as tf

def build_simple_cnn(input_shape=(64,64,1), num_classes=4):
    model = tf.keras.Sequential()
    
    # Conv Layer 1
    model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    
    # Conv Layer 2
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    
    # Conv Layer 3
    model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    
    # Flatten + Dense
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    
    # Output layer
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    return model

# Instantiate
simple_cnn = build_simple_cnn()
simple_cnn.summary()
