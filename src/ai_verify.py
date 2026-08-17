from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms


BASE_DIR = Path(
    __file__
).resolve().parent.parent


MODEL_PATH = (
    BASE_DIR
    / "results"
    / "drift_sense_cnn_v2.pth"
)


IMAGE_SIZE = 64

# ============================================================
# CNN MODEL
# ============================================================

class DriftSenseCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64 * 8 * 8,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                64,
                2
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# LOAD MODEL
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model = DriftSenseCNN().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.5],
        [0.5]
    )

])


# ============================================================
# VERIFY CANDIDATE
# ============================================================

def verify_candidate(
    image,
    confidence_threshold=0.5
):

    if image is None:

        raise ValueError(
            "Candidate image is empty."
        )


    tensor = transform(
        image
    )

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.to(
        device
    )


    with torch.no_grad():

        output = model(
            tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )


    negative_probability = (
        probabilities[0][0].item()
    )

    positive_probability = (
        probabilities[0][1].item()
    )


    accepted = (
        positive_probability
        >= confidence_threshold
    )


    return {

        "accepted": accepted,

        "positive_probability":
            positive_probability,

        "negative_probability":
            negative_probability

    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("DRIFT-SENSE AI VERIFIER")
    print("=" * 60)

    print(
        f"Model : {MODEL_PATH}"
    )

    print(
        f"Device: {device}"
    )

    print()
    print(
        "AI verifier loaded successfully."
    )