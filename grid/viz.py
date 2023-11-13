import torch
import cv2
import numpy as np


class GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_image, model_output, target_class):
        if target_class is None:
            target_class = np.argmax(model_output.cpu().data.numpy())

        self.model.zero_grad()

        # resize to match batch size
        batch_size = input_image.size(0)
        one_hot_out = torch.zeros(
            (batch_size, model_output.size()[-1]), device=model_output.device
        )
        one_hot_out[:, target_class] = 1

        model_output.backward(gradient=one_hot_out, retain_graph=True)

        # rn main model is on cuda, but we want to do numpy stuff
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_image.shape[2:])
        cam -= np.min(cam)
        cam /= np.max(cam)

        return cam
