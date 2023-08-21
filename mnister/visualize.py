from pathlib import Path
import random
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import numpy as np
from matplotlib.animation import FuncAnimation

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

test_images = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255


def get_saliency_map(model, image, true_class):
    image_tensor = tf.convert_to_tensor(image)
    image_tensor = tf.Variable(image_tensor, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        prediction = model(image_tensor)
        class_output = prediction[:, true_class]

    gradients = tape.gradient(class_output, image_tensor)
    saliency = tf.norm(gradients, axis=-1)
    return saliency.numpy()


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    if len(img_array.shape) != 4:
        img_array = tf.expand_dims(img_array, 0)
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def integrated_visualization(saliency, grad_cam):
    # Normalize both visualizations to [0, 1]
    saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency))
    grad_cam = (grad_cam - np.min(grad_cam)) / (np.max(grad_cam) - np.min(grad_cam))

    # Integrate
    integrated = (saliency + grad_cam) / 2.0
    return integrated


def update(frame, axes, model):
    for digit in range(10):
        index_list = np.where(test_labels == digit)[0]
        index = index_list[frame % len(index_list)]
        image = test_images[index]
        true_class = np.argmax(test_labels[index])

        saliency = get_saliency_map(model, np.expand_dims(image, axis=0), true_class)
        grad_cam = make_gradcam_heatmap(image, model, "conv2d_1")

        axes[digit, 0].imshow(image.squeeze(), cmap="gray")
        axes[digit, 0].imshow(saliency.squeeze(), cmap="hot", alpha=0.75)
        # axes[digit, 0].set_title("Saliency Map")
        axes[digit, 0].axis("off")

        axes[digit, 1].imshow(image.squeeze(), cmap="gray")
        grad_cam = cv2.resize(grad_cam, (28, 28))  # grad_cam.squeeze()
        axes[digit, 1].imshow(grad_cam, cmap="hot", alpha=0.75)
        # axes[digit, 1].set_title("Grad-CAM")
        axes[digit, 1].axis("off")

        integrated_vis = integrated_visualization(saliency.squeeze(), grad_cam)
        axes[digit, 2].imshow(image.squeeze(), cmap="gray")
        axes[digit, 2].imshow(integrated_vis, cmap="hot", alpha=0.75)
        axes[digit, 2].axis("off")

    plt.tight_layout()


def main():
    # load model
    if Path("mnist_classifier.keras").exists():
        model = tf.keras.models.load_model("mnist_classifier.keras")
    else:
        raise Exception("No model found, please run __main__.py and train one first.")

    # Create a grid of 10x3 to plot the images
    fig, axes = plt.subplots(10, 3)

    ani = FuncAnimation(
        fig, lambda f: update(f, axes=axes, model=model), frames=60, repeat=True
    )  # 974 total frames possible
    # ani.save("mnist.gif", writer="imagemagick", fps=8)

    plt.show()


if __name__ == "__main__":
    main()
