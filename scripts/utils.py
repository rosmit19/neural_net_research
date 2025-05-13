import torch

# Coordinates for Fourier-based spatial labelling from Wu et al. (2023)
label_coordinates = [
    [14, 8], [12, 14], [14, 10], [14, 12], [8, 14],
    [10, 14], [13, 15], [12, 16], [12, 12], [13, 13]
]

def apply_fourier_label(image_tensor, label):
    device = image_tensor.device
    coord = label_coordinates[label]

    work = torch.zeros((28, 28), device=device)
    work[14, 14] = 256
    work[coord[0], coord[1]] = 1

    wave = abs(torch.fft.ifft2(torch.fft.fftshift(work)))
    wave = (wave - wave.min()) / (wave.max() - wave.min())

    labeled = image_tensor + 0.5 * wave
    return labeled
