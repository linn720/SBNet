import os
import argparse
import cv2
import numpy as np


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def is_img(name):
    return name.lower().endswith(IMG_EXTS)


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def edge_aware_denoise(img_bgr, h=5, h_color=6, blend=0.55, edge_power=1.5):
    """
    对增强结果进行边缘保护型去噪。
    平坦区域去噪强，边缘区域去噪弱，避免图像过糊。
    """

    img = img_bgr.astype(np.float32) / 255.0

    # 1. 非局部均值彩色去噪
    den = cv2.fastNlMeansDenoisingColored(
        img_bgr,
        None,
        h,
        h_color,
        7,
        21
    ).astype(np.float32) / 255.0

    # 2. 计算边缘强度
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)

    edge = edge / (edge.max() + 1e-6)
    edge = np.clip(edge, 0, 1)

    # 边缘越强，去噪权重越低
    smooth_weight = (1.0 - edge) ** edge_power
    smooth_weight = np.clip(smooth_weight * blend, 0.0, 1.0)
    smooth_weight = smooth_weight[..., None]

    out = img * (1.0 - smooth_weight) + den * smooth_weight
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--h", type=float, default=5.0)
    parser.add_argument("--h_color", type=float, default=6.0)
    parser.add_argument("--blend", type=float, default=0.55)
    parser.add_argument("--edge_power", type=float, default=1.5)

    args = parser.parse_args()

    mkdir(args.output_dir)

    names = sorted([n for n in os.listdir(args.input_dir) if is_img(n)])

    for name in names:
        in_path = os.path.join(args.input_dir, name)
        out_path = os.path.join(args.output_dir, name)

        img = cv2.imread(in_path, cv2.IMREAD_COLOR)
        if img is None:
            print("skip:", name)
            continue

        out = edge_aware_denoise(
            img,
            h=args.h,
            h_color=args.h_color,
            blend=args.blend,
            edge_power=args.edge_power
        )

        cv2.imwrite(out_path, out)
        print("saved:", out_path)

    print("Done:", args.output_dir)


if __name__ == "__main__":
    main()