import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


image_path = '../data/ScanNet/scannet/scannet_2d/scene0000_00/color/2460.jpg'
image = Image.open(image_path).convert('RGB')
transform = transforms.Compose([
    transforms.ToTensor(),
])
image_tensor = transform(image)

height, width = image_tensor.shape[-2:]

def create_circular_mask(height, width, radius):
    center = (height // 2, width // 2)
    y, x = np.ogrid[:height, :width]
    mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
    return torch.tensor(mask, dtype=torch.float32)

def create_high_pass_filter(height, width, radius):
    low_pass_filter = create_circular_mask(height, width, radius)
    high_pass_filter = 1 - low_pass_filter
    return high_pass_filter

radius = 24
low_pass_mask = create_circular_mask(height, width, 10)
high_pass_mask = create_high_pass_filter(height, width, 30)

low_pass_mask = low_pass_mask.unsqueeze(0)  # (1, height, width)
high_pass_mask = high_pass_mask.unsqueeze(0)  # (1, height, width)

def frequency_transform(x, mask):
    x_freq = torch.fft.fft2(x)
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))
    amplitude = torch.abs(x_freq)
    phase = torch.angle(x_freq)
    amplitude_masked = amplitude * mask
    x_freq_masked = amplitude_masked * torch.exp(1j * phase)
    #x_freq_masked = x_freq * mask
    x_freq_masked = torch.fft.ifftshift(x_freq_masked, dim=(-2, -1))
    x_corrupted = torch.fft.ifft2(x_freq_masked).real
    x_corrupted = torch.clamp(x_corrupted, min=0., max=1.)
    return x_corrupted

def apply_filter(image_tensor, mask):
    r, g, b = image_tensor[0], image_tensor[1], image_tensor[2]
    r_transformed = frequency_transform(r.unsqueeze(0), mask).squeeze(0)
    g_transformed = frequency_transform(g.unsqueeze(0), mask).squeeze(0)
    b_transformed = frequency_transform(b.unsqueeze(0), mask).squeeze(0)
    return torch.stack([r_transformed, g_transformed, b_transformed])

low_pass_filtered = apply_filter(image_tensor, low_pass_mask)
normalized_low = low_pass_filtered.reshape(3, -1)
normalized_low = (normalized_low - normalized_low.min(1).values[:, None])/(normalized_low.max(1).values - normalized_low.min(1).values)[:, None]
normalized_low = normalized_low.reshape(3, low_pass_filtered.shape[1], low_pass_filtered.shape[2])
output_low_pass_image = transforms.ToPILImage()(normalized_low)
output_low_pass_image.save('rgb_low_pass_2460.png')

high_pass_filtered = apply_filter(image_tensor, high_pass_mask)
normalized_high = high_pass_filtered.reshape(3, -1)
normalized_high = (normalized_high - normalized_high.min(1).values[:, None])/(normalized_high.max(1).values - normalized_high.min(1).values)[:, None]
normalized_high = normalized_high*2
normalized_high = normalized_high.reshape(3, high_pass_filtered.shape[1], high_pass_filtered.shape[2])
output_high_pass_image = transforms.ToPILImage()(normalized_high)
output_high_pass_image.save('rgb_high_pass_2460.png')

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

axs[0].imshow(image)
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(output_low_pass_image)
axs[1].set_title('Low Pass Filtered Image')
axs[1].axis('off')

axs[2].imshow(output_high_pass_image)
axs[2].set_title('High Pass Filtered Image')
axs[2].axis('off')

plt.show()
