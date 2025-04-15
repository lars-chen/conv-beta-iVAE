import gradio as gr
import torch
import numpy as np
from PIL import Image
import yaml
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.datasets as Datasets

# Project modules
import helpers as hf
from RES_VAE_Dynamic import VAE

# Configuration setup
run = 4
path = f"Runs/Run_{run}/config.yml"
config = hf.read_config(path=path)
label_idxs = config[0]
t_idx = config[1]
image_size = 128

with open(path, "r") as file:
    labels = yaml.safe_load(file)["model_labels"]

attribute_names = labels.keys()

# use_cuda = torch.cuda.is_available()
device = torch.device("cpu")  # device_index if use_cuda else

celeb_transform = transforms.Compose(
    [
        transforms.CenterCrop(150),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ]
)

# Load test data
test_dataset = Datasets.CelebA(
    "../../../../../groups/kempter/chen/data",
    transform=celeb_transform,
    download=False,
    split="valid",
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=1, num_workers=16, shuffle=True
)


# Create AE network.
vae_net = VAE(
    channel_in=3,
    label_dim=len(label_idxs),
    image_size=image_size,
    ch=config[7],
    blocks=config[-2],
    latent_channels=config[10],
    num_res_blocks=config[8],
    norm_type=config[2],
    deep_model=config[11],
).to(device)

checkpoint = torch.load(
    f"Runs/Run_4/epoch8_step_44999/model_128.pt",
    map_location="cpu",
)
print("-Checkpoint loaded!")
vae_net.load_state_dict(checkpoint["model_state_dict"])
vae_net.eval()


def post_process(imgs):
    imgs = np.array(imgs.detach().cpu().permute(2, 3, 1, 0)).copy()

    for i in range(imgs.shape[3]):
        img = imgs[:, :, :, i]
        img_norm = (img - img.min()) / (img.max() - img.min())
        imgs[:, :, :, i] = img_norm
    return 255 * imgs


def get_random_test_sample():
    dataiter = iter(test_loader)
    test_images, label_vals = next(dataiter)

    # Convert labels to checkbox format
    binary_labels = label_vals[0, label_idxs].cpu().numpy()
    selected_attributes = [
        list(labels.keys())[i] for i, val in enumerate(binary_labels) if val == 1
    ]

    # Get treatment value
    treatment_value = float(label_vals[0, t_idx].cpu().numpy())

    # Process image
    pil_image = Image.fromarray(post_process(test_images).squeeze().astype(np.uint8))

    original_data = {
        "original_tensor": test_images,
        "label_tensor": label_vals[:, label_idxs].to(device),
        "treatment": label_vals[:, t_idx].to(device),
    }

    return (pil_image, selected_attributes, treatment_value, original_data)


def predict(scale, stored_data):
    # Unpack stored data
    image_tensor = stored_data["original_tensor"]
    label_tensor = stored_data["label_tensor"]
    original_treatment = stored_data["treatment"]

    # Adjust treatment
    new_treatment = (
        original_treatment
        + (((original_treatment + 1) % 2) - original_treatment) * scale
    )

    # Generate image
    with torch.no_grad():
        imgs, mu, log_var, _ = vae_net(
            image_tensor.to(device),
            label_tensor.to(device),
            new_treatment.to(device),
        )

    processed_img = Image.fromarray(post_process(imgs).squeeze().astype(np.uint8))
    return processed_img


with gr.Blocks(theme=gr.themes.Soft(), css="custom.css") as app:
    gr.Markdown("# Facial Image Generator with Learned Prior")
    stored_data = gr.State()

    with gr.Row():
        with gr.Column():
            get_random_btn = gr.Button("Get Random Sample", variant="primary")
            with gr.Row():
                with gr.Column():
                    original_img = gr.Image(label="Original Image", type="pil")
                with gr.Column():
                    generated_img = gr.Image(label="Generated Image", type="pil")
            treatment_slider = gr.Slider(
                minimum=-5.0,
                maximum=5.0,
                step=0.1,
                value=0.0,
                label="Treatment Adjustment Scale",
                interactive=True,
            )
            labels_display = gr.CheckboxGroup(
                choices=attribute_names,
                label="Attributes",
                interactive=False,
            )
        with gr.Column():
            pass

    # Event handling with error logging
    get_random_btn.click(
        fn=get_random_test_sample,
        outputs=[original_img, labels_display, treatment_slider, stored_data],
    )

    treatment_slider.change(
        fn=predict, inputs=[treatment_slider, stored_data], outputs=generated_img
    )

if __name__ == "__main__":
    app.launch(show_api=True, share=True)
