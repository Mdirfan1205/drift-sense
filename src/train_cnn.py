import random
from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    random_split,
    WeightedRandomSampler
)

from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    "data/ai_dataset/v3_training"
)

MODEL_PATH = Path(
    "results/drift_sense_cnn_v3.pth"
)

IMAGE_SIZE = 64

BATCH_SIZE = 16

EPOCHS = 30

LEARNING_RATE = 0.001

VALIDATION_RATIO = 0.20

RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("DRIFT-SENSE CNN V3 TRAINING")
print("=" * 70)

print(
    f"Device : {device}"
)

print(
    f"Dataset: {DATASET_DIR}"
)

print(
    f"Model  : {MODEL_PATH}"
)


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=1
    ),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )

])


# ============================================================
# LOAD DATASET
# ============================================================

dataset = datasets.ImageFolder(
    DATASET_DIR,
    transform=transform
)


print()
print(
    f"Total samples : {len(dataset)}"
)

print(
    f"Classes       : {dataset.classes}"
)


# ============================================================
# CHECK EXPECTED CLASSES
# ============================================================

if dataset.classes != [
    "negative",
    "positive"
]:

    raise RuntimeError(
        "Expected classes "
        "['negative', 'positive'] "
        f"but found {dataset.classes}"
    )


# ============================================================
# COUNT CLASSES
# ============================================================

class_counts = [
    0 for _ in dataset.classes
]

for _, label in dataset.samples:

    class_counts[label] += 1


print()
print(
    "Class distribution"
)

for index, class_name in enumerate(
    dataset.classes
):

    print(
        f"  {class_name:<10} : "
        f"{class_counts[index]}"
    )


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

validation_size = int(
    len(dataset)
    *
    VALIDATION_RATIO
)

training_size = (
    len(dataset)
    -
    validation_size
)


train_dataset, validation_dataset = (
    random_split(
        dataset,
        [
            training_size,
            validation_size
        ],
        generator=torch.Generator().manual_seed(
            RANDOM_SEED
        )
    )
)


print()
print(
    f"Training samples   : "
    f"{len(train_dataset)}"
)

print(
    f"Validation samples : "
    f"{len(validation_dataset)}"
)


# ============================================================
# TRAINING LABELS
# ============================================================

train_labels = []

for index in train_dataset.indices:

    label = dataset.samples[index][1]

    train_labels.append(
        label
    )


# ============================================================
# TRAINING CLASS DISTRIBUTION
# ============================================================

train_class_counts = [
    0 for _ in dataset.classes
]

for label in train_labels:

    train_class_counts[label] += 1


print()
print(
    "Training class distribution"
)

for index, class_name in enumerate(
    dataset.classes
):

    print(
        f"  {class_name:<10} : "
        f"{train_class_counts[index]}"
    )


# ============================================================
# VALIDATION LABELS
# ============================================================

validation_labels = []

for index in validation_dataset.indices:

    label = dataset.samples[index][1]

    validation_labels.append(
        label
    )


validation_class_counts = [
    0 for _ in dataset.classes
]

for label in validation_labels:

    validation_class_counts[label] += 1


print()
print(
    "Validation class distribution"
)

for index, class_name in enumerate(
    dataset.classes
):

    print(
        f"  {class_name:<10} : "
        f"{validation_class_counts[index]}"
    )


# ============================================================
# SAFETY CHECK
# ============================================================

if train_class_counts[1] == 0:

    raise RuntimeError(
        "Training split contains no positive samples."
    )

if validation_class_counts[1] == 0:

    raise RuntimeError(
        "Validation split contains no positive samples."
    )


# ============================================================
# BALANCED TRAINING SAMPLER
# ============================================================

class_weights_for_sampler = []

for count in train_class_counts:

    if count == 0:

        class_weights_for_sampler.append(
            0.0
        )

    else:

        class_weights_for_sampler.append(
            1.0 / count
        )


sample_weights = [
    class_weights_for_sampler[label]
    for label in train_labels
]


sampler = WeightedRandomSampler(

    weights=torch.DoubleTensor(
        sample_weights
    ),

    num_samples=len(
        train_dataset
    ),

    replacement=True
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    sampler=sampler
)


validation_loader = DataLoader(

    validation_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False
)


# ============================================================
# CNN MODEL
# ============================================================

class DriftSenseCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # ------------------------------------------------
            # Convolution Layer 1
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            # ------------------------------------------------
            # Convolution Layer 2
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            # ------------------------------------------------
            # Convolution Layer 3
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            )

        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64 * 8 * 8,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                64,
                2
            )
        )


    def forward(self, x):

        x = self.features(
            x
        )

        x = self.classifier(
            x
        )

        return x


# ============================================================
# CREATE MODEL
# ============================================================

model = DriftSenseCNN().to(
    device
)


print()
print(
    "Model:"
)

print(
    model
)


# ============================================================
# LOSS
# ============================================================

# The sampler already balances the training batches.
# We therefore use standard CrossEntropyLoss here rather
# than applying both sampling and large class weights,
# which could over-correct the imbalance.

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# BEST MODEL TRACKING
# ============================================================

best_f1 = -1.0

best_accuracy = -1.0

best_epoch = 0


# ============================================================
# CREATE RESULT DIRECTORY
# ============================================================

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(
    1,
    EPOCHS + 1
):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    train_loss_total = 0.0

    train_correct = 0

    train_total = 0


    for images, labels in train_loader:

        images = images.to(
            device
        )

        labels = labels.to(
            device
        )

        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()


        train_loss_total += (
            loss.item()
            *
            images.size(0)
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        train_correct += (
            predictions
            ==
            labels
        ).sum().item()


        train_total += (
            labels.size(0)
        )


    train_loss = (
        train_loss_total
        /
        train_total
    )

    train_accuracy = (
        train_correct
        /
        train_total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    validation_loss_total = 0.0

    validation_total = 0

    validation_predictions = []

    validation_targets = []


    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )


            validation_loss_total += (
                loss.item()
                *
                images.size(0)
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            validation_predictions.extend(
                predictions.cpu().tolist()
            )

            validation_targets.extend(
                labels.cpu().tolist()
            )


            validation_total += (
                labels.size(0)
            )


    validation_loss = (
        validation_loss_total
        /
        validation_total
    )


    validation_accuracy = accuracy_score(
        validation_targets,
        validation_predictions
    )


    validation_precision = precision_score(
        validation_targets,
        validation_predictions,
        pos_label=1,
        zero_division=0
    )


    validation_recall = recall_score(
        validation_targets,
        validation_predictions,
        pos_label=1,
        zero_division=0
    )


    validation_f1 = f1_score(
        validation_targets,
        validation_predictions,
        pos_label=1,
        zero_division=0
    )


    # ========================================================
    # PRINT EPOCH RESULTS
    # ========================================================

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: "
        f"{train_accuracy * 100:6.2f}% | "
        f"Val Loss: {validation_loss:.4f} | "
        f"Val Acc: "
        f"{validation_accuracy * 100:6.2f}% | "
        f"Precision: "
        f"{validation_precision * 100:6.2f}% | "
        f"Recall: "
        f"{validation_recall * 100:6.2f}% | "
        f"F1: "
        f"{validation_f1 * 100:6.2f}%"
    )


    # ========================================================
    # SAVE BEST MODEL
    #
    # F1 is more useful than raw accuracy here because
    # positive samples are rare.
    # ========================================================

    is_better = (

        validation_f1
        >
        best_f1

    ) or (

        validation_f1
        ==
        best_f1

        and

        validation_accuracy
        >
        best_accuracy
    )


    if is_better:

        best_f1 = validation_f1

        best_accuracy = validation_accuracy

        best_epoch = epoch

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            f"  >>> BEST MODEL SAVED "
            f"(epoch {epoch}, "
            f"F1={validation_f1 * 100:.2f}%)"
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# ============================================================
# FINAL VALIDATION
# ============================================================

final_predictions = []

final_targets = []


with torch.no_grad():

    for images, labels in validation_loader:

        images = images.to(
            device
        )

        outputs = model(
            images
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        final_predictions.extend(
            predictions.cpu().tolist()
        )

        final_targets.extend(
            labels.tolist()
        )


# ============================================================
# FINAL METRICS
# ============================================================

final_accuracy = accuracy_score(
    final_targets,
    final_predictions
)

final_precision = precision_score(
    final_targets,
    final_predictions,
    pos_label=1,
    zero_division=0
)

final_recall = recall_score(
    final_targets,
    final_predictions,
    pos_label=1,
    zero_division=0
)

final_f1 = f1_score(
    final_targets,
    final_predictions,
    pos_label=1,
    zero_division=0
)

final_matrix = confusion_matrix(
    final_targets,
    final_predictions
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("DRIFT-SENSE CNN V3 FINAL RESULTS")
print("=" * 70)

print(
    f"Best epoch : {best_epoch}"
)

print(
    f"Accuracy   : "
    f"{final_accuracy * 100:.2f}%"
)

print(
    f"Precision  : "
    f"{final_precision * 100:.2f}%"
)

print(
    f"Recall     : "
    f"{final_recall * 100:.2f}%"
)

print(
    f"F1 Score   : "
    f"{final_f1 * 100:.2f}%"
)

print()
print(
    "Confusion Matrix"
)

print(
    "                 Predicted"
)

print(
    "                 Negative  Positive"
)

if final_matrix.shape == (2, 2):

    print(
        f"Actual Negative "
        f"{final_matrix[0, 0]:10d}"
        f"{final_matrix[0, 1]:10d}"
    )

    print(
        f"Actual Positive "
        f"{final_matrix[1, 0]:10d}"
        f"{final_matrix[1, 1]:10d}"
    )

print()
print(
    f"Best model saved to: "
    f"{MODEL_PATH}"
)

print("=" * 70)