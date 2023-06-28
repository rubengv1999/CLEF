import onnxruntime
import numpy as np

ort_session = onnxruntime.InferenceSession("Convnext_Base_All_95.onnx")
input_name = ort_session.get_inputs()[0].name
ort_inputs = {
    input_name: np.random.randn(1, 3, 224, 224).astype(np.float32)
}  # Convert to float32
ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs)


from PIL import Image
import torchvision.transforms as transforms
import torch

# Load the image
image_path = "DF20_300/2238546328-30620.jpg"
image = Image.open(image_path).convert("RGB")

# Define the transformation
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Apply the transformation and convert the image to a tensor
image_tensor = transform(image)

# Add the batch dimension if needed (e.g., for passing the tensor to a model)
image_tensor = image_tensor.unsqueeze(0)

print(image_tensor.shape)
import math

torch.set_printoptions(threshold=math.inf)
print(image_tensor)

image_np = image_tensor.numpy().astype(np.float32)

ort_inputs = {input_name: image_np}  # Convert to float32
ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs)
