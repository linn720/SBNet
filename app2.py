import os
import traceback
import torch
import gradio as gr
import torchvision.transforms as transforms
import torch.nn.functional as F
from net.CIDNet import CIDNet

# ===== device =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== weight =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT_PATH = os.path.join(BASE_DIR, "weights", "train", "epoch_120.pth")  # 改成你想用的
print("Loading weight:", WEIGHT_PATH)
assert os.path.exists(WEIGHT_PATH), f"Weight not found: {WEIGHT_PATH}"

# ===== model =====
model = CIDNet().to(device)
state = torch.load(WEIGHT_PATH, map_location=device)
model.load_state_dict(state, strict=True)
model.trans.gated = True
model.trans.gated2 = True
model.eval()

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

def enhance_only(input_img, gamma, alpha_s, alpha_i):
    try:
        print("[DEBUG] enhance_only called")
        if input_img is None:
            print("[DEBUG] input_img is None")
            return None, "No image uploaded."

        input_img = input_img.convert("RGB")
        x = to_tensor(input_img).to(device)

        factor = 8
        h, w = x.shape[1], x.shape[2]
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        x = F.pad(x.unsqueeze(0), (0, padw, 0, padh), 'reflect')

        with torch.no_grad():
            model.trans.alpha_s = float(alpha_s)
            model.trans.alpha = float(alpha_i)
            y = model(x ** float(gamma))
            y = torch.clamp(y, 0, 1)
            y = y[:, :, :h, :w].squeeze(0).cpu()

        out = to_pil(y)
        print("[DEBUG] success")
        return out, "Done."
    except Exception as e:
        traceback.print_exc()
        return None, f"Error: {e}"

with gr.Blocks(title="HVI-CIDNet Debug") as demo:
    gr.Markdown("## HVI-CIDNet Local Demo")
    with gr.Row():
        in_img = gr.Image(type="pil", image_mode="RGB", label="Input")
        out_img = gr.Image(type="pil", label="Output")
    gamma = gr.Slider(0.1, 5, value=1.0, step=0.01, label="gamma")
    alpha_s = gr.Slider(0.0, 2.0, value=1.0, step=0.01, label="alpha_s")
    alpha_i = gr.Slider(0.1, 2.0, value=1.0, step=0.01, label="alpha_i")
    run_btn = gr.Button("Enhance")
    status = gr.Textbox(label="Status")

    run_btn.click(
        fn=enhance_only,
        inputs=[in_img, gamma, alpha_s, alpha_i],
        outputs=[out_img, status]
    )

demo.launch(server_port=7862, debug=True, show_error=True)