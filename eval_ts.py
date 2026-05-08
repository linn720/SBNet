import os
import argparse
import csv
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from net.teacher_student import MultiTeacherSBNet


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def is_img(name):
    return name.lower().endswith(IMG_EXTS)


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def pad_to_multiple(x, multiple=8):
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h == 0 and pad_w == 0:
        return x, h, w

    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, h, w


def save_tensor_img(tensor, path):
    tensor = torch.clamp(tensor.detach().cpu().squeeze(0), 0.0, 1.0)
    img = transforms.ToPILImage()(tensor)
    img.save(path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output/SBNet_TS")

    parser.add_argument("--expert_color", type=str, default="./weights/experts/expert_color.pth")
    parser.add_argument("--expert_denoise", type=str, default="./weights/experts/expert_denoise.pth")
    parser.add_argument("--expert_balance", type=str, default="./weights/experts/expert_balance.pth")

    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--color_threshold", type=float, default=0.010)
    parser.add_argument("--green_threshold", type=float, default=0.006)
    parser.add_argument("--red_threshold", type=float, default=0.006)
    parser.add_argument("--dark_threshold", type=float, default=0.08)
    parser.add_argument("--noise_threshold", type=float, default=0.12)

    parser.add_argument("--save_all_experts", action="store_true")

    args = parser.parse_args()

    mkdir(args.output_dir)

    if args.save_all_experts:
        mkdir(os.path.join(args.output_dir, "expert_color"))
        mkdir(os.path.join(args.output_dir, "expert_denoise"))
        mkdir(os.path.join(args.output_dir, "expert_balance"))

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = MultiTeacherSBNet(
        color_weight=args.expert_color,
        denoise_weight=args.expert_denoise,
        balance_weight=args.expert_balance,
        device=device,
        strict=False,
        color_threshold=args.color_threshold,
        green_threshold=args.green_threshold,
        red_threshold=args.red_threshold,
        dark_threshold=args.dark_threshold,
        noise_threshold=args.noise_threshold
    ).to(device)
    model.eval()

    to_tensor = transforms.ToTensor()

    names = sorted([n for n in os.listdir(args.input_dir) if is_img(n)])
    report_path = os.path.join(args.output_dir, "route_report.csv")

    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "route",
            "mean_luma",
            "dark_ratio",
            "red_ratio",
            "green_ratio",
            "noise_score"
        ])

        for name in names:
            img_path = os.path.join(args.input_dir, name)
            img = Image.open(img_path).convert("RGB")
            x = to_tensor(img).unsqueeze(0).to(device)
            x, h, w = pad_to_multiple(x, multiple=8)

            with torch.no_grad():
                out, route_names, feats = model(x, return_info=True)
                out = out[:, :, :h, :w]

            save_tensor_img(out, os.path.join(args.output_dir, name))

            route = route_names[0]
            row = [
                name,
                route,
                float(feats["mean_luma"][0].detach().cpu()),
                float(feats["dark_ratio"][0].detach().cpu()),
                float(feats["red_ratio"][0].detach().cpu()),
                float(feats["green_ratio"][0].detach().cpu()),
                float(feats["noise_score"][0].detach().cpu())
            ]
            writer.writerow(row)

            print(name, "=>", route)

            if args.save_all_experts:
                with torch.no_grad():
                    y_color = model.teacher_color(x)[:, :, :h, :w]
                    y_denoise = model.teacher_denoise(x)[:, :, :h, :w]
                    y_balance = model.teacher_balance(x)[:, :, :h, :w]

                save_tensor_img(y_color, os.path.join(args.output_dir, "expert_color", name))
                save_tensor_img(y_denoise, os.path.join(args.output_dir, "expert_denoise", name))
                save_tensor_img(y_balance, os.path.join(args.output_dir, "expert_balance", name))

    print("Done.")
    print("Output:", args.output_dir)
    print("Route report:", report_path)


if __name__ == "__main__":
    main()