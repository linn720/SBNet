import torch
import torch.nn as nn
import torch.nn.functional as F

from net.CIDNet import CIDNet


class ImageTypeStudent(nn.Module):
    """
    轻量学生路由模块。
    不直接生成图像，只根据输入低光图像的统计特征选择教师专家。
    """

    def __init__(
        self,
        color_threshold=0.010,
        green_threshold=0.006,
        red_threshold=0.006,
        dark_threshold=0.08,
        noise_threshold=0.12
    ):
        super(ImageTypeStudent, self).__init__()
        self.color_threshold = float(color_threshold)
        self.green_threshold = float(green_threshold)
        self.red_threshold = float(red_threshold)
        self.dark_threshold = float(dark_threshold)
        self.noise_threshold = float(noise_threshold)

    def extract_features(self, x):
        """
        x: [B, 3, H, W], range [0, 1]
        返回每张图的亮度、暗区比例、红色比例、绿色比例、噪声估计。
        """
        r = x[:, 0:1, :, :]
        g = x[:, 1:2, :, :]
        b = x[:, 2:3, :, :]

        luma = 0.299 * r + 0.587 * g + 0.114 * b
        mean_luma = luma.mean(dim=(1, 2, 3))
        dark_ratio = (luma < 0.10).float().mean(dim=(1, 2, 3))

        rgb_sum = r + g + b + 1e-6
        rn = r / rgb_sum
        gn = g / rgb_sum
        bn = b / rgb_sum

        valid = luma > 0.015

        red_mask = valid & (rn > 0.40) & (rn > gn + 0.04) & (rn > bn + 0.04)
        green_mask = valid & (gn > 0.36) & (gn > rn + 0.025) & (gn > bn + 0.025)

        red_ratio = red_mask.float().mean(dim=(1, 2, 3))
        green_ratio = green_mask.float().mean(dim=(1, 2, 3))

        smooth = F.avg_pool2d(luma, kernel_size=3, stride=1, padding=1)
        noise_score = torch.abs(luma - smooth).mean(dim=(1, 2, 3)) / (mean_luma + 0.03)

        feats = {
            "mean_luma": mean_luma,
            "dark_ratio": dark_ratio,
            "red_ratio": red_ratio,
            "green_ratio": green_ratio,
            "noise_score": noise_score
        }
        return feats

    def forward(self, x):
        """
        route:
            0: color expert
            1: denoise expert
            2: balance expert
        """
        feats = self.extract_features(x)

        mean_luma = feats["mean_luma"]
        red_ratio = feats["red_ratio"]
        green_ratio = feats["green_ratio"]
        noise_score = feats["noise_score"]

        bsz = x.size(0)
        route = torch.full((bsz,), 2, dtype=torch.long, device=x.device)

        # 暗且噪声重：走去噪教师
        denoise_cond = (mean_luma < self.dark_threshold) & (noise_score > self.noise_threshold)
        route[denoise_cond] = 1

        # 有明显红/绿目标：优先走颜色教师
        color_cond = (
            (red_ratio > self.red_threshold)
            | (green_ratio > self.green_threshold)
            | ((red_ratio + green_ratio) > self.color_threshold)
        )
        route[color_cond] = 0

        return route, feats


class MultiTeacherSBNet(nn.Module):
    """
    多教师 SBNet。
    三个教师都是 CIDNet，但加载不同 checkpoint。
    学生模块只做路由选择。
    """

    def __init__(
        self,
        color_weight,
        denoise_weight,
        balance_weight,
        device="cuda",
        color_params=None,
        denoise_params=None,
        balance_params=None,
        strict=False,
        color_threshold=0.010,
        green_threshold=0.006,
        red_threshold=0.006,
        dark_threshold=0.08,
        noise_threshold=0.12
    ):
        super(MultiTeacherSBNet, self).__init__()
        self.device = device

        color_params = color_params or {}
        denoise_params = denoise_params or {}
        balance_params = balance_params or {}

        self.teacher_color = CIDNet(**color_params).to(device)
        self.teacher_denoise = CIDNet(**denoise_params).to(device)
        self.teacher_balance = CIDNet(**balance_params).to(device)

        self._load_teacher(self.teacher_color, color_weight, strict)
        self._load_teacher(self.teacher_denoise, denoise_weight, strict)
        self._load_teacher(self.teacher_balance, balance_weight, strict)

        self.teacher_color.eval()
        self.teacher_denoise.eval()
        self.teacher_balance.eval()

        for p in self.teacher_color.parameters():
            p.requires_grad = False
        for p in self.teacher_denoise.parameters():
            p.requires_grad = False
        for p in self.teacher_balance.parameters():
            p.requires_grad = False

        self.student = ImageTypeStudent(
            color_threshold=color_threshold,
            green_threshold=green_threshold,
            red_threshold=red_threshold,
            dark_threshold=dark_threshold,
            noise_threshold=noise_threshold
        )

        self.route_names = {
            0: "color",
            1: "denoise",
            2: "balance"
        }

    def _load_teacher(self, model, weight_path, strict):
        state = torch.load(weight_path, map_location=self.device)
        missing, unexpected = model.load_state_dict(state, strict=strict)
        print("Load teacher:", weight_path)
        print("  missing keys:", len(missing))
        print("  unexpected keys:", len(unexpected))

    @torch.no_grad()
    def forward(self, x, return_info=False):
        route, feats = self.student(x)

        outputs = []
        route_name_list = []

        for i in range(x.size(0)):
            xi = x[i:i+1]

            if route[i].item() == 0:
                yi = self.teacher_color(xi)
            elif route[i].item() == 1:
                yi = self.teacher_denoise(xi)
            else:
                yi = self.teacher_balance(xi)

            outputs.append(yi)
            route_name_list.append(self.route_names[route[i].item()])

        out = torch.cat(outputs, dim=0)

        if return_info:
            return out, route_name_list, feats

        return out