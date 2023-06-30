import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values
train_images = train_images / 255.0
test_images = test_images / 255.0

# Data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1
)
datagen.fit(train_images.reshape(-1, 28, 28, 1))

# Train the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(datagen.flow(train_images.reshape(-1, 28, 28, 1), train_labels, batch_size=32), epochs=5)

# Save the model
model.save('mnist_model.h5')

# Function to display an image
def display(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

# User input for drawing a digit
user_digit = np.zeros((28, 28), dtype=np.uint8)  # Initialize an empty digit image

def draw_digit():
    fig, ax = plt.subplots()
    ax.imshow(user_digit, cmap='gray')
    ax.set_title("Draw a digit")
    ax.axis('off')
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_press_event', on_button_press)
    plt.show()

def on_motion(event):
    if event.inaxes:
        x, y = int(event.xdata), int(event.ydata)
        user_digit[y, x] = 255
        event.inaxes.imshow(user_digit, cmap='gray')
        plt.draw()

def on_button_press(event):
    if event.button == 1:  # Left mouse button
        plt.close()

draw_digit()

# Recognize the drawn digit
prediction = model.predict(np.expand_dims(user_digit / 255.0, axis=0))
predicted_label = np.argmax(prediction)
print("Predicted digit:", predicted_label)
