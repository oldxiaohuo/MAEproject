import matplotlib.pyplot as plt

with torch.no_grad():
    pred, target, mask = model(images, mask_ratio=0.75)
    recon_img = model.unpatchify(pred)

    plt.subplot(1, 3, 1); plt.imshow(images[0, 0].cpu(), cmap='gray'); plt.title("Original")
    plt.subplot(1, 3, 2); plt.imshow(recon_img[0, 0].cpu(), cmap='gray'); plt.title("Reconstructed")
    plt.subplot(1, 3, 3); plt.imshow(mask[0].cpu().reshape(64,32), cmap='gray'); plt.title("Mask")
    plt.show()