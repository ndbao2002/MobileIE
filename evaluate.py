import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import glob

def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] 
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(255.0 * 255.0 / np.mean(np.square(diff)))
    return psnr

def evaluate_folders(
    target_dir,
    ref_dir,
    resize_mode="error",
    convert_to_y=False
):
    """
    Evaluate mean SSIM and PSNR over two folders using provided implementations.

    Parameters
    ----------
    target_dir : str or Path
        Ground-truth images
    ref_dir : str or Path
        Predicted images
    resize_mode : {"error", "resize"}
        How to handle size mismatch
    convert_to_y : bool
        If True, compute metrics on Y channel (BT.601)

    Returns
    -------
    mean_ssim : float
    mean_psnr : float
    """

    target_dir = Path(target_dir)
    ref_dir = Path(ref_dir)

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    def index_by_stem(folder):
        return {
            f.stem: f
            for f in folder.iterdir()
            if f.suffix.lower() in valid_exts
        }

    target_files = index_by_stem(target_dir)
    ref_files = index_by_stem(ref_dir)

    common_keys = sorted(target_files.keys() & ref_files.keys())
    if not common_keys:
        raise ValueError("No matching filenames found between folders.")

    ssim_vals = []
    psnr_vals = []

    for key in common_keys:
        img_t = Image.open(target_files[key]).convert("RGB")
        img_r = Image.open(ref_files[key]).convert("RGB")

        if img_t.size != img_r.size:
            if resize_mode == "resize":
                img_r = img_r.resize(img_t.size, Image.BICUBIC)
            else:
                raise ValueError(f"Size mismatch for image: {key}")

        img_t = np.asarray(img_t, dtype=np.float64)
        img_r = np.asarray(img_r, dtype=np.float64)

        if convert_to_y:
            # BT.601 luminance
            img_t = 0.299 * img_t[..., 0] + 0.587 * img_t[..., 1] + 0.114 * img_t[..., 2]
            img_r = 0.299 * img_r[..., 0] + 0.587 * img_r[..., 1] + 0.114 * img_r[..., 2]

        ssim_vals.append(calculate_ssim(img_t, img_r))
        psnr_vals.append(calculate_psnr(img_t, img_r))

    return float(np.mean(ssim_vals)), float(np.mean(psnr_vals))

# Arg parser for command line execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate SSIM and PSNR between two folders of images."
    )
    parser.add_argument("target_dir", type=str, help="Ground-truth images folder")
    parser.add_argument("ref_dir", type=str, help="Predicted images folder")
    parser.add_argument(
        "--resize_mode",
        type=str,
        choices=["error", "resize"],
        default="error",
        help="How to handle size mismatch",
    )
    parser.add_argument(
        "--convert_to_y",
        action="store_true",
        help="If set, compute metrics on Y channel (BT.601)",
    )

    args = parser.parse_args()

    mean_ssim, mean_psnr = evaluate_folders(
        args.target_dir,
        args.ref_dir,
        resize_mode=args.resize_mode,
        convert_to_y=args.convert_to_y,
    )

    print("Evaluation on folders:")
    print("GT folder: ", args.target_dir)
    print("Pred folder: ", args.ref_dir)
    print("Evaluation Results:")
    print(f"Mean SSIM: {mean_ssim:.4f}")
    print(f"Mean PSNR: {mean_psnr:.2f} dB")