import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

n_epochs = 500
batch_size_train = 1000
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 2

tf.random.set_seed(random_seed)

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_images = train_images.reshape(-1, 28, 28, 1).astype("float32") / 255.0
test_images = test_images.reshape(-1, 28, 28, 1).astype("float32") / 255.0

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size_train)

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = test_dataset.batch(batch_size_test)

model = keras.Sequential(
    [
        layers.Conv2D(10, kernel_size=5, activation="relu"),
        layers.Dropout(0.5),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(20, kernel_size=5, activation="relu"),
        layers.Dropout(0.5),
        layers.MaxPooling2D(pool_size=2),
        layers.Flatten(),
        layers.Dense(50, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

model.load_weights("model_weights.h5")
optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
optimizer.load_weights("optimizer_weights.h5")
train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_dataset) for i in range(n_epochs + 1)]


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        output = model(images, training=True)
        loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(labels, output))

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def train(epoch):
    for batch_idx, (images, labels) in enumerate(train_dataset):
        loss = train_step(images, labels)

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * batch_size_train,
                    len(train_dataset) * batch_size_train,
                    100.0 * batch_idx / len(train_dataset),
                    loss,
                )
            )
            train_losses.append(loss)
            train_counter.append(batch_idx * batch_size_train + (epoch - 1) * len(train_dataset))

    model.save_weights("model_weights.h5")
    optimizer.save_weights("optimizer_weights.h5")


@tf.function
def test_step(images, labels):
    output = model(images, training=False)
    loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(labels, output))
    return loss, output


def test():
    test_loss = 0.0
    correct = 0

    for images, labels in test_dataset:
        loss, output = test_step(images, labels)
        test_loss += loss * len(images)
        correct += tf.reduce_sum(tf.cast(tf.argmax(output, axis=1) == labels, dtype=tf.int32))

    test_loss /= len(test_images)
    test_losses.append(test_loss)

    print(
        "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_images), 100.0 * correct / len(test_images)
        )
    )


test()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()
