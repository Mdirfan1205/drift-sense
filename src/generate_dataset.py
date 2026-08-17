import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# DRIFT-SENSE SYNTHETIC DATASET GENERATOR
# ============================================================
#
# Generates:
#   - High-magnification reference image
#   - Lower-magnification 1000x1000 search image
#   - Independent noise in both images
#   - SEM-like edge brightening
#   - Blur
#   - Rotation
#   - Small scale variation
#   - Ground-truth center coordinates
#
# Architecture:
#   DRAM-style
#   Horizontal word-lines
#   Vertical bit-lines
#   Contact/via at intersections
#
# ============================================================


SEARCH_SIZE = 1000
REFERENCE_SIZE = 100

RNG_SEED = 42


# ------------------------------------------------------------
# 1. Create a DRAM-style grid
# ------------------------------------------------------------

def create_dram_grid(
    height,
    width,
    pitch,
    line_width,
    via_radius,
    brightness=220,
):
    """
    Create a synthetic DRAM-like periodic structure.

    Horizontal lines represent word-lines.
    Vertical lines represent bit-lines.
    Small circles represent contacts/vias.
    """

    image = np.full(
        (height, width),
        25,
        dtype=np.float32
    )

    # Small random offset makes different samples less identical.
    x_offset = pitch // 2
    y_offset = pitch // 2

    # --------------------------------------------------------
    # Vertical bit-lines
    # --------------------------------------------------------

    for x in range(x_offset, width, pitch):

        cv2.line(
            image,
            (x, 0),
            (x, height - 1),
            brightness,
            line_width,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Horizontal word-lines
    # --------------------------------------------------------

    for y in range(y_offset, height, pitch):

        cv2.line(
            image,
            (0, y),
            (width - 1, y),
            brightness,
            line_width,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Contacts / vias
    # --------------------------------------------------------

    for y in range(y_offset, height, pitch):

        for x in range(x_offset, width, pitch):

            cv2.circle(
                image,
                (x, y),
                via_radius,
                min(255, brightness + 20),
                -1,
                cv2.LINE_AA
            )

    return np.clip(image, 0, 255).astype(np.uint8)


# ------------------------------------------------------------
# 2. Add SEM-like edge brightening
# ------------------------------------------------------------

def add_edge_brightening(image, strength=35):
    """
    Brighten strong feature edges.

    This is a simplified phenomenological model of SEM
    edge contrast rather than a full electron-transport
    simulation.
    """

    image_float = image.astype(np.float32)

    # Calculate image gradients.
    grad_x = cv2.Sobel(
        image_float,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    grad_y = cv2.Sobel(
        image_float,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(
        grad_x,
        grad_y
    )

    # Normalize edge strength.
    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        1,
        cv2.NORM_MINMAX
    )

    # Add controlled brightness around edges.
    result = image_float + strength * magnitude

    return np.clip(result, 0, 255).astype(np.uint8)


# ------------------------------------------------------------
# 3. Add independent sensor noise
# ------------------------------------------------------------

def add_sensor_noise(
    image,
    rng,
    gaussian_sigma=5.0,
    poisson_scale=20.0
):
    """
    Add independent noise to one image.

    Gaussian component:
        Represents electronic/readout-like variation.

    Poisson component:
        Represents counting/statistical variation.

    A new random realization is generated every time this
    function is called.
    """

    image_float = image.astype(np.float32)

    # Gaussian noise
    gaussian_noise = rng.normal(
        loc=0.0,
        scale=gaussian_sigma,
        size=image.shape
    )

    # Poisson-like noise
    normalized = np.clip(
        image_float / 255.0,
        0,
        1
    )

    poisson_noise = (
        rng.poisson(
            normalized * poisson_scale
        ) - normalized * poisson_scale
    )

    # Combine noise sources.
    noisy = (
        image_float
        + gaussian_noise
        + poisson_noise
    )

    return np.clip(
        noisy,
        0,
        255
    ).astype(np.uint8)


# ------------------------------------------------------------
# 4. Rotate an image while preserving dimensions
# ------------------------------------------------------------

def rotate_image(image, angle):

    height, width = image.shape

    center = (
        width / 2,
        height / 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    return rotated


# ------------------------------------------------------------
# 5. Create the reference image
# ------------------------------------------------------------

def create_reference(rng):

    # Reference is high magnification.
    #
    # A pitch of ~18-22 pixels gives us several visible
    # periodic structures inside the 100x100 crop.

    pitch = int(
        rng.integers(
            18,
            23
        )
    )

    line_width = int(
        rng.integers(
            2,
            4
        )
    )

    via_radius = int(
        rng.integers(
            1,
            3
        )
    )

    brightness = int(
        rng.integers(
            205,
            235
        )
    )

    reference = create_dram_grid(
        REFERENCE_SIZE,
        REFERENCE_SIZE,
        pitch,
        line_width,
        via_radius,
        brightness
    )

    # Small rotation variation.
    angle = float(
        rng.uniform(
            -2.0,
            2.0
        )
    )

    reference = rotate_image(
        reference,
        angle
    )

    # Reference is the sharper / cleaner image.
    reference = cv2.GaussianBlur(
        reference,
        (3, 3),
        0.4
    )

    reference = add_edge_brightening(
        reference,
        strength=25
    )

    # IMPORTANT:
    # Noise is generated independently for this image.
    reference = add_sensor_noise(
        reference,
        rng,
        gaussian_sigma=3.0,
        poisson_scale=10.0
    )

    return reference

# ------------------------------------------------------------
# 6. Create the search image
# ------------------------------------------------------------

def create_search(
    reference,
    rng,
    target_x,
    target_y
):
    """
    Create a 1000 x 1000 lower-magnification search image.

    The search background uses a much finer DRAM pitch than
    the high-magnification reference.

    Reference pitch:
        approximately 18-22 pixels

    Search pitch:
        approximately 2-3 pixels

    This gives approximately a 10x spatial scale difference
    between the two periodic structures.
    """

    # ========================================================
    # SEARCH BACKGROUND
    # ========================================================

    # IMPORTANT:
    #
    # Earlier V1 used approximately 18-24 pixels here.
    # That made the reference and search structures nearly
    # identical in pixel scale.
    #
    # For V2 we use approximately 2-3 pixels.
    #
    # 20 / 2 = 10x
    #
    # This models the lower-magnification search image.
    # ========================================================

    search_pitch = int(
        rng.integers(
            2,
            4
        )
    )

    # At this very fine pitch we use one-pixel lines.
    search_line_width = 1

    # Small contacts.
    search_via_radius = 0

    search_brightness = int(
        rng.integers(
            150,
            190
        )
    )

    # Create the fine DRAM background.
    search = create_dram_grid(
        SEARCH_SIZE,
        SEARCH_SIZE,
        search_pitch,
        search_line_width,
        search_via_radius,
        search_brightness
    )

    # ========================================================
    # BACKGROUND BLUR
    # ========================================================

    # At lower magnification the fine periodic structures
    # should not remain perfectly sharp.
    #
    # A small blur prevents the 2-3 pixel structure from
    # becoming unrealistically crisp.

    background_sigma = float(
        rng.uniform(
            0.15,
            0.35
        )
    )

    search = cv2.GaussianBlur(
        search,
        (3, 3),
        background_sigma
    )

    # ========================================================
    # TARGET
    # ========================================================

    # Keep the target approximately 100 x 100 pixels.
    #
    # This is the region we will ask our localization
    # algorithm to find.

    scale = float(
        rng.uniform(
            0.70,
            1.30
        )
    )

    target_size = int(
        REFERENCE_SIZE * scale
    )

    target_size = max(
        70,
        min(
            130,
            target_size
        )
    )

    target = cv2.resize(
        reference,
        (
            target_size,
            target_size
        ),
        interpolation=cv2.INTER_AREA
    )

    # ========================================================
    # TARGET ROTATION
    # ========================================================

    target_angle = float(
        rng.uniform(
            -4.0,
            4.0
        )
    )

    target = rotate_image(
        target,
        target_angle
    )

    # ========================================================
    # TARGET BLUR
    # ========================================================

    target_sigma = float(
        rng.uniform(
            0.5,
            1.8
        )
    )

    target = cv2.GaussianBlur(
        target,
        (5, 5),
        target_sigma
    )

    # ========================================================
    # TARGET EDGE BRIGHTENING
    # ========================================================

    target = add_edge_brightening(
        target,
        strength=float(
            rng.uniform(
                25,
                40
            )
        )
    )

    # ========================================================
    # INDEPENDENT TARGET NOISE
    # ========================================================

    # IMPORTANT:
    #
    # The target receives a new noise realization.
    # We do NOT copy noise from the reference image.

    target = add_sensor_noise(
        target,
        rng,
        gaussian_sigma=float(
            rng.uniform(
                4.0,
                12.0
            )
        ),
        poisson_scale=float(
            rng.uniform(
                20.0,
                35.0
            )
        )
    )

    # ========================================================
    # INSERT TARGET
    # ========================================================

    target_height, target_width = target.shape

    search[
        target_y:target_y + target_height,
        target_x:target_x + target_width
    ] = target

    # ========================================================
    # SEARCH-IMAGE NOISE
    # ========================================================

    # Add another independent noise realization to the
    # complete search image.

    search = add_sensor_noise(
        search,
        rng,
        gaussian_sigma=float(
            rng.uniform(
                3.0,
                9.0
            )
        ),
        poisson_scale=float(
            rng.uniform(
                15.0,
                25.0
            )
        )
    )

    # ========================================================
    # FINAL CONTRAST NORMALIZATION
    # ========================================================

    search = cv2.normalize(
        search,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return (
        search,
        target_width,
        target_height
    )








# ------------------------------------------------------------
# 7. Generate one image pair
# ------------------------------------------------------------

def generate_pair(
    pair_id,
    output_dir,
    rng
):

    # --------------------------------------------------------
    # Reference
    # --------------------------------------------------------

    reference = create_reference(rng)

    # --------------------------------------------------------
    # Choose target location.
    #
    # Keep the target inside the search image with margin.
    # --------------------------------------------------------

    target_size = int(
        rng.integers(
            90,
            111
        )
    )

    margin = 120

    target_x = int(
        rng.integers(
            margin,
            SEARCH_SIZE - target_size - margin
        )
    )

    target_y = int(
        rng.integers(
            margin,
            SEARCH_SIZE - target_size - margin
        )
    )

    # --------------------------------------------------------
    # Search image
    # --------------------------------------------------------

    search, target_width, target_height = create_search(
        reference,
        rng,
        target_x,
        target_y
    )

    # --------------------------------------------------------
    # Ground-truth center.
    # --------------------------------------------------------

    center_x = target_x + target_width / 2.0
    center_y = target_y + target_height / 2.0

    # --------------------------------------------------------
    # Save files.
    # --------------------------------------------------------

    pair_dir = (
        output_dir /
        f"pair_{pair_id:03d}"
    )

    pair_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    reference_path = (
        pair_dir /
        "reference.png"
    )

    search_path = (
        pair_dir /
        "search.png"
    )

    cv2.imwrite(
        str(reference_path),
        reference
    )

    cv2.imwrite(
        str(search_path),
        search
    )

    return {
        "pair_id": pair_id,
        "reference": str(reference_path),
        "search": str(search_path),
        "center_x": round(center_x, 2),
        "center_y": round(center_y, 2),
        "target_width": target_width,
        "target_height": target_height,
    }


# ------------------------------------------------------------
# 8. Generate complete dataset
# ------------------------------------------------------------

def generate_dataset(
    num_pairs,
    output_dir,
    seed=RNG_SEED
):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    rng = np.random.default_rng(seed)

    metadata = []

    for pair_id in range(1, num_pairs + 1):

        sample = generate_pair(
            pair_id,
            output_dir,
            rng
        )

        metadata.append(sample)

        print(
            f"Generated pair {pair_id:03d}: "
            f"center=({sample['center_x']}, "
            f"{sample['center_y']})"
        )

    # --------------------------------------------------------
    # Save ground-truth CSV.
    # --------------------------------------------------------

    csv_path = (
        output_dir /
        "ground_truth.csv"
    )

    with open(
        csv_path,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "pair_id",
                "reference",
                "search",
                "center_x",
                "center_y",
                "target_width",
                "target_height",
            ]
        )

        writer.writeheader()

        writer.writerows(
            metadata
        )

    print()
    print(
        f"Dataset generation complete."
    )
    print(
        f"Pairs generated: {num_pairs}"
    )
    print(
        f"Output directory: {output_dir}"
    )
    print(
        f"Ground truth: {csv_path}"
    )


# ------------------------------------------------------------
# 9. Command-line interface
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic DRAM-style "
            "Drift-Sense image pairs."
        )
    )

    parser.add_argument(
        "--architecture",
        choices=["DRAM", "dram"],
        default="DRAM",
        help="Architecture style. Currently DRAM."
    )

    parser.add_argument(
        "--pairs",
        type=int,
        default=30,
        help="Number of image pairs to generate."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/dataset",
        help="Output directory."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )

    args = parser.parse_args()

    if args.pairs < 30:
        print(
            "Warning: The hackathon requires "
            "at least 30 randomized pairs."
        )

    generate_dataset(
        num_pairs=args.pairs,
        output_dir=Path(args.output),
        seed=args.seed
    )


if __name__ == "__main__":
    main()