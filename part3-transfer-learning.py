import tensorflow as tf

input_tensor = tf.keras.Input(shape=(64,64,1))
x = tf.keras.layers.Conv2D(3, (3,3), padding='same')(input_tensor)

base_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights=None,  # train from scratch
    input_tensor=x
)

base_model.trainable = False

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(4, activation='softmax')(x)
transfer_model = tf.keras.Model(inputs=input_tensor, outputs=output)
transfer_model.summary()
