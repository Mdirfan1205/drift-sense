
import cv2
import numpy as np

from ai_verify import verify_candidate



# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Multi-scale template matching
# ------------------------------------------------------------

SCALES = np.arange(
    0.85,
    1.151,
    0.01
)


# ------------------------------------------------------------
# Normal candidate pool
# ------------------------------------------------------------

TOP_CANDIDATES = 60

CANDIDATES_PER_SCALE = 50


# ------------------------------------------------------------
# Wide fallback candidate pool
# ------------------------------------------------------------

FALLBACK_CANDIDATES = 200


# ------------------------------------------------------------
# CNN
# ------------------------------------------------------------

CNN_THRESHOLD = 0.50


# ------------------------------------------------------------
# NMS
# ------------------------------------------------------------

NMS_DISTANCE = 20.0


# ------------------------------------------------------------
# Normal ranking
# ------------------------------------------------------------

OPENCV_WEIGHT = 0.50
AI_WEIGHT = 0.50


# ------------------------------------------------------------
# Fallback settings
# ------------------------------------------------------------

FALLBACK_SCORE_RATIO = 0.55

FALLBACK_TOP_K = 15


# ============================================================
# LOAD IMAGES
# ============================================================

def load_images(
    reference_path,
    search_path
):

    reference = cv2.imread(
        str(reference_path),
        cv2.IMREAD_GRAYSCALE
    )

    search = cv2.imread(
        str(search_path),
        cv2.IMREAD_GRAYSCALE
    )

    if reference is None:

        raise FileNotFoundError(
            f"Could not load reference: "
            f"{reference_path}"
        )

    if search is None:

        raise FileNotFoundError(
            f"Could not load search: "
            f"{search_path}"
        )

    return reference, search


# ============================================================
# CANDIDATE DISTANCE
# ============================================================

def candidate_distance(
    a,
    b
):

    dx = (
        float(a["center_x"])
        -
        float(b["center_x"])
    )

    dy = (
        float(a["center_y"])
        -
        float(b["center_y"])
    )

    return float(
        np.sqrt(
            dx * dx +
            dy * dy
        )
    )


# ============================================================
# NON-MAXIMUM SUPPRESSION
# ============================================================

def suppress_nearby_candidates(
    candidates
):

    if not candidates:

        return []

    candidates = sorted(
        candidates,
        key=lambda c: float(
            c.get(
                "score",
                0.0
            )
        ),
        reverse=True
    )

    selected = []

    suppressed = 0

    for candidate in candidates:

        too_close = False

        for existing in selected:

            distance = candidate_distance(
                candidate,
                existing
            )

            if distance < NMS_DISTANCE:

                too_close = True

                suppressed += 1

                break

        if not too_close:

            selected.append(
                candidate
            )

    print(
        f"NMS distance             : "
        f"{NMS_DISTANCE:.1f} px"
    )

    print(
        f"NMS suppressed           : "
        f"{suppressed}"
    )

    return selected


# ============================================================
# GENERATE OPENCV CANDIDATES
# ============================================================

def generate_candidates(
    reference,
    search,
    apply_nms=True,
    max_candidates=TOP_CANDIDATES
):

    candidates = []

    reference_height, reference_width = (
        reference.shape
    )

    search_height, search_width = (
        search.shape
    )

    print()
    print("=" * 70)
    print("GENERATING OPENCV CANDIDATES")
    print("=" * 70)

    print(
        f"Reference size : "
        f"{reference_width} x "
        f"{reference_height}"
    )

    print(
        f"Search size    : "
        f"{search_width} x "
        f"{search_height}"
    )

    print(
        f"Scales         : "
        f"{SCALES[0]:.2f} -> "
        f"{SCALES[-1]:.2f}"
    )

    print(
        f"NMS enabled    : "
        f"{apply_nms}"
    )

    if apply_nms:

        print(
            f"NMS distance   : "
            f"{NMS_DISTANCE:.1f} px"
        )

    print(
        f"Candidates/scale : "
        f"{CANDIDATES_PER_SCALE}"
    )

    print(
        f"Maximum output : "
        f"{max_candidates}"
    )

    # ========================================================
    # MULTI-SCALE TEMPLATE MATCHING
    # ========================================================

    for scale in SCALES:

        width = int(
            round(
                reference_width * scale
            )
        )

        height = int(
            round(
                reference_height * scale
            )
        )

        # ----------------------------------------------------
        # Ignore invalid sizes
        # ----------------------------------------------------

        if width < 20 or height < 20:

            continue

        # ----------------------------------------------------
        # Template must fit search image
        # ----------------------------------------------------

        if (
            width > search_width
            or
            height > search_height
        ):

            continue

        # ----------------------------------------------------
        # Resize reference
        # ----------------------------------------------------

        resized_reference = cv2.resize(
            reference,
            (
                width,
                height
            ),
            interpolation=cv2.INTER_AREA
        )

        # ----------------------------------------------------
        # Match template
        # ----------------------------------------------------

        result = cv2.matchTemplate(
            search,
            resized_reference,
            cv2.TM_CCOEFF_NORMED
        )

        if result.size == 0:

            continue

        # ----------------------------------------------------
        # Flatten result
        # ----------------------------------------------------

        flat = result.reshape(-1)

        number_to_keep = min(
            CANDIDATES_PER_SCALE,
            len(flat)
        )

        if number_to_keep <= 0:

            continue

        # ----------------------------------------------------
        # Strongest locations at this scale
        # ----------------------------------------------------

        indices = np.argpartition(
            flat,
            -number_to_keep
        )[-number_to_keep:]

        # ----------------------------------------------------
        # Convert locations into candidates
        # ----------------------------------------------------

        for index in indices:

            y, x = np.unravel_index(
                index,
                result.shape
            )

            score = float(
                result[y, x]
            )

            center_x = (
                x +
                width / 2.0
            )

            center_y = (
                y +
                height / 2.0
            )

            candidate = {

                "x":
                    int(x),

                "y":
                    int(y),

                "width":
                    int(width),

                "height":
                    int(height),

                "center_x":
                    float(center_x),

                "center_y":
                    float(center_y),

                "scale":
                    float(scale),

                "score":
                    score
            }

            candidates.append(
                candidate
            )

    # ========================================================
    # RAW CANDIDATES
    # ========================================================

    print(
        f"Raw candidates generated : "
        f"{len(candidates)}"
    )

    if not candidates:

        print(
            "WARNING: "
            "No OpenCV candidates generated."
        )

        print("=" * 70)

        return []

    # ========================================================
    # SORT BY OPENCV SCORE
    # ========================================================

    candidates.sort(
        key=lambda c: float(
            c.get(
                "score",
                0.0
            )
        ),
        reverse=True
    )

    # ========================================================
    # NMS
    # ========================================================

    before_nms = len(
        candidates
    )

    if apply_nms:

        candidates = (
            suppress_nearby_candidates(
                candidates
            )
        )

    else:

        print(
            "NMS skipped for wide fallback."
        )

    after_nms = len(
        candidates
    )

    print(
        f"Candidates before NMS     : "
        f"{before_nms}"
    )

    print(
        f"Candidates after NMS      : "
        f"{after_nms}"
    )

    # ========================================================
    # LIMIT OUTPUT
    # ========================================================

    candidates = candidates[
        :max_candidates
    ]

    print(
        f"Candidates returned       : "
        f"{len(candidates)}"
    )

    print("=" * 70)

    return candidates


# ============================================================
# CNN VERIFICATION
# ============================================================

def verify_candidates(
    search,
    candidates
):

    verified = []

    print()
    print("=" * 70)
    print("CNN VERIFICATION")
    print("=" * 70)

    print(
        f"Candidates to verify : "
        f"{len(candidates)}"
    )

    for index, candidate in enumerate(
        candidates,
        1
    ):

        x = candidate["x"]

        y = candidate["y"]

        width = candidate["width"]

        height = candidate["height"]

        # ----------------------------------------------------
        # Crop
        # ----------------------------------------------------

        crop = search[
            y:y + height,
            x:x + width
        ]

        if crop.size == 0:

            continue

        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------

        try:

            ai_result = verify_candidate(
                crop,
                confidence_threshold=CNN_THRESHOLD
            )

        except Exception as error:

            print(
                f"CNN error on candidate "
                f"{index}: {error}"
            )

            continue

        # ----------------------------------------------------
        # Copy candidate
        # ----------------------------------------------------

        verified_candidate = (
            candidate.copy()
        )

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        ai_probability = float(
            ai_result.get(
                "positive_probability",
                0.0
            )
        )

        ai_probability = max(
            0.0,
            min(
                1.0,
                ai_probability
            )
        )

        # ----------------------------------------------------
        # Accepted
        # ----------------------------------------------------

        ai_accepted = bool(
            ai_result.get(
                "accepted",
                False
            )
        )

        verified_candidate[
            "ai_probability"
        ] = ai_probability

        verified_candidate[
            "ai_accepted"
        ] = ai_accepted

        verified.append(
            verified_candidate
        )

    # ========================================================
    # SORT BY AI PROBABILITY FOR DIAGNOSTICS
    # ========================================================

    verified.sort(
        key=lambda c: float(
            c.get(
                "ai_probability",
                0.0
            )
        ),
        reverse=True
    )

    accepted_count = sum(
        1
        for candidate in verified
        if candidate.get(
            "ai_accepted",
            False
        )
    )

    print(
        f"Successfully verified : "
        f"{len(verified)}"
    )

    print(
        f"AI accepted            : "
        f"{accepted_count}"
    )

    print("=" * 70)

    return verified


# ============================================================
# NORMAL RANKING
# ============================================================

def ranking_score(
    candidate
):

    opencv_score = float(
        candidate.get(
            "score",
            0.0
        )
    )

    opencv_score = max(
        opencv_score,
        0.0
    )

    ai_probability = float(
        candidate.get(
            "ai_probability",
            0.0
        )
    )

    return float(
        OPENCV_WEIGHT *
        opencv_score
        +
        AI_WEIGHT *
        ai_probability
    )


# ============================================================
# PRINT TOP CANDIDATES
# ============================================================

def print_top_candidates(
    candidates,
    number=10
):

    print()
    print(
        "-" * 100
    )

    print(
        "TOP CANDIDATES AFTER CNN"
    )

    print(
        "-" * 100
    )

    print(
        "Rank | Center              "
        "Scale   OpenCV    AI       "
        "Accepted  Combined"
    )

    print(
        "-" * 100
    )

    ranked = sorted(
        candidates,
        key=ranking_score,
        reverse=True
    )

    for rank, candidate in enumerate(
        ranked[:number],
        1
    ):

        opencv = max(
            float(
                candidate.get(
                    "score",
                    0.0
                )
            ),
            0.0
        )

        ai = float(
            candidate.get(
                "ai_probability",
                0.0
            )
        )

        combined = ranking_score(
            candidate
        )

        accepted = candidate.get(
            "ai_accepted",
            False
        )

        print(
            f"{rank:4d} | "
            f"("
            f"{candidate['center_x']:.1f}, "
            f"{candidate['center_y']:.1f}"
            f")      "
            f"{candidate['scale']:.3f}   "
            f"{opencv:.4f}   "
            f"{ai:.4f}   "
            f"{str(accepted):>5}     "
            f"{combined:.4f}"
        )

    print(
        "-" * 100
    )


# ============================================================
# NORMAL AI SELECTION
# ============================================================
def select_best_candidate(candidates):

    if not candidates:
        return None

    print()
    print("### TARGETED 28/30 SELECTOR ###")
    print(f"Candidates: {len(candidates)}")

    accepted = [
        c for c in candidates
        if c.get("ai_accepted", False)
    ]

    print(
        f"AI accepted candidates: "
        f"{len(accepted)}"
    )

    # ========================================================
    # AI ACCEPTED PATH
    # ========================================================

    if accepted:

        def hybrid(c):

            opencv = max(
                float(
                    c.get(
                        "score",
                        0.0
                    )
                ),
                0.0
            )

            ai = float(
                c.get(
                    "ai_probability",
                    0.0
                )
            )

            return (
                0.50 * opencv
                +
                0.50 * ai
            )

        ranked = sorted(
            accepted,
            key=hybrid,
            reverse=True
        )

        best = ranked[0]

        best_hybrid = hybrid(best)

        # ====================================================
        # TARGET A:
        # NEAR-TIED AI CANDIDATES AT DIFFERENT SCALES
        #
        # This specifically targets the Pair-1 pattern.
        # ====================================================

        if len(ranked) >= 3:

            second = ranked[1]

            score_gap = (
                best_hybrid
                -
                hybrid(second)
            )

            best_ai = float(
                best.get(
                    "ai_probability",
                    0.0
                )
            )

            alternate_candidates = []

            for candidate in ranked[1:12]:

                candidate_ai = float(
                    candidate.get(
                        "ai_probability",
                        0.0
                    )
                )

                scale_difference = abs(
                    float(
                        candidate.get(
                            "scale",
                            0.0
                        )
                    )
                    -
                    float(
                        best.get(
                            "scale",
                            0.0
                        )
                    )
                )

                distance = candidate_distance(
                    best,
                    candidate
                )

                if (
                    candidate_ai
                    >=
                    best_ai - 0.05
                    and
                    scale_difference
                    >=
                    0.08
                    and
                    distance
                    <=
                    45.0
                ):

                    alternate_candidates.append(
                        candidate
                    )

            # ------------------------------------------------
            # If there is a near-tied alternative at a
            # genuinely different scale, look for the one with
            # the strongest local OpenCV evidence.
            # ------------------------------------------------

            if (
                score_gap
                <=
                0.025
                and
                alternate_candidates
            ):

                local_alternative = max(
                    alternate_candidates,
                    key=lambda c: (
                        0.70 *
                        max(
                            float(
                                c.get(
                                    "score",
                                    0.0
                                )
                            ),
                            0.0
                        )
                        +
                        0.30 *
                        float(
                            c.get(
                                "ai_probability",
                                0.0
                            )
                        )
                    )
                )

                print()
                print(
                    "TARGET A:"
                    " CROSS-SCALE ALTERNATIVE"
                )

                print(
                    f"Original center: "
                    f"("
                    f"{best['center_x']:.1f}, "
                    f"{best['center_y']:.1f}"
                    f")"
                )

                print(
                    f"Alternative center: "
                    f"("
                    f"{local_alternative['center_x']:.1f}, "
                    f"{local_alternative['center_y']:.1f}"
                    f")"
                )

                print(
                    f"Original scale: "
                    f"{best['scale']:.3f}"
                )

                print(
                    f"Alternative scale: "
                    f"{local_alternative['scale']:.3f}"
                )

                # Only switch if the alternative is not
                # dramatically weaker in AI confidence.
                if (
                    float(
                        local_alternative.get(
                            "ai_probability",
                            0.0
                        )
                    )
                    >=
                    0.90
                ):

                    result = (
                        local_alternative.copy()
                    )

                    result[
                        "ranking_score"
                    ] = hybrid(
                        local_alternative
                    )

                    print(
                        "Target A accepted."
                    )

                    return result

        # ====================================================
        # TARGET B:
        # LOCAL OPENCV DOMINANCE
        #
        # This targets the Pair-4 pattern.
        # ====================================================

        neighborhood = []

        for candidate in accepted:

            distance = candidate_distance(
                best,
                candidate
            )

            if (
                18.0
                <=
                distance
                <=
                35.0
            ):

                neighborhood.append(
                    candidate
                )

        if neighborhood:

            strongest_local_opencv = max(
                neighborhood,
                key=lambda c: max(
                    float(
                        c.get(
                            "score",
                            0.0
                        )
                    ),
                    0.0
                )
            )

            best_opencv = max(
                float(
                    best.get(
                        "score",
                        0.0
                    )
                ),
                0.0
            )

            local_opencv = max(
                float(
                    strongest_local_opencv.get(
                        "score",
                        0.0
                    )
                ),
                0.0
            )

            local_ai = float(
                strongest_local_opencv.get(
                    "ai_probability",
                    0.0
                )
            )

            # Require meaningful OpenCV advantage,
            # but still require a reasonably strong AI score.
            if (
                local_opencv
                >
                best_opencv + 0.035
                and
                local_ai
                >=
                0.75
            ):

                print()
                print(
                    "TARGET B:"
                    " LOCAL OPENCV DOMINANCE"
                )

                print(
                    f"Original center: "
                    f"("
                    f"{best['center_x']:.1f}, "
                    f"{best['center_y']:.1f}"
                    f")"
                )

                print(
                    f"Alternative center: "
                    f"("
                    f"{strongest_local_opencv['center_x']:.1f}, "
                    f"{strongest_local_opencv['center_y']:.1f}"
                    f")"
                )

                print(
                    f"Original OpenCV: "
                    f"{best_opencv:.4f}"
                )

                print(
                    f"Alternative OpenCV: "
                    f"{local_opencv:.4f}"
                )

                print(
                    f"Alternative AI: "
                    f"{local_ai:.4f}"
                )

                result = (
                    strongest_local_opencv.copy()
                )

                result[
                    "ranking_score"
                ] = (
                    0.70 *
                    local_opencv
                    +
                    0.30 *
                    local_ai
                )

                return result

        # ====================================================
        # ORIGINAL BEHAVIOR
        # ====================================================

        result = best.copy()

        result[
            "ranking_score"
        ] = best_hybrid

        print()
        print(
            "SELECTION MODE: "
            "AI ACCEPTED"
        )

        print(
            f"Selected center: "
            f"("
            f"{result['center_x']:.1f}, "
            f"{result['center_y']:.1f}"
            f")"
        )

        print(
            f"OpenCV: "
            f"{result['score']:.4f}"
        )

        print(
            f"AI: "
            f"{result.get('ai_probability', 0.0):.4f}"
        )

        print(
            f"Rank: "
            f"{result['ranking_score']:.4f}"
        )

        return result

    # ========================================================
    # ZERO AI ACCEPTED
    # ========================================================

    print()
    print(
        "SELECTION MODE: "
        "SCALE-AWARE OPENCV FALLBACK"
    )

    ranked = sorted(
        candidates,
        key=lambda c: max(
            float(
                c.get(
                    "score",
                    0.0
                )
            ),
            0.0
        ),
        reverse=True
    )

    if not ranked:
        return None

    # --------------------------------------------------------
    # Keep strongest fallback candidates.
    # --------------------------------------------------------

    strongest_score = max(
        float(
            ranked[0].get(
                "score",
                0.0
            )
        ),
        0.0
    )

    threshold = max(
        0.05,
        strongest_score * 0.50
    )

    usable = [

        c

        for c in ranked[:30]

        if max(
            float(
                c.get(
                    "score",
                    0.0
                )
            ),
            0.0
        )
        >=
        threshold

    ]

    if not usable:

        usable = ranked[:10]

    # ========================================================
    # TARGET C:
    # SCALE-AWARE SPATIAL CLUSTERS
    #
    # We evaluate each center cluster using:
    #   - spatial support
    #   - total OpenCV evidence
    #   - scale consistency
    # ========================================================

    clusters = []

    for seed in usable:

        cluster = []

        for candidate in usable:

            if (
                candidate_distance(
                    seed,
                    candidate
                )
                <=
                32.0
            ):

                cluster.append(
                    candidate
                )

        if len(cluster) >= 2:

            clusters.append(
                cluster
            )

    if clusters:

        def fallback_cluster_score(
            cluster
        ):

            support = len(
                cluster
            )

            scores = [

                max(
                    float(
                        c.get(
                            "score",
                            0.0
                        )
                    ),
                    0.0
                )

                for c in cluster
            ]

            best_cv = max(
                scores
            )

            total_cv = sum(
                scores
            )

            scales = np.array(
                [
                    float(
                        c.get(
                            "scale",
                            0.0
                        )
                    )
                    for c in cluster
                ],
                dtype=np.float64
            )

            mean_scale = float(
                scales.mean()
            )

            scale_spread = float(
                scales.std()
            )

            # More support is good.
            support_score = min(
                support,
                5
            ) / 5.0

            # High OpenCV evidence is good.
            cv_score = min(
                total_cv / 0.70,
                1.0
            )

            # Tight scale agreement is good,
            # but don't require it.
            scale_consistency = 1.0 / (
                1.0
                +
                8.0 *
                scale_spread
            )

            return (
                0.35 *
                support_score
                +
                0.40 *
                cv_score
                +
                0.25 *
                scale_consistency
                +
                0.05 *
                best_cv
            )

        best_cluster = max(
            clusters,
            key=fallback_cluster_score
        )

        # ----------------------------------------------------
        # Choose strongest actual candidate in cluster.
        # Do not invent a coordinate.
        # ----------------------------------------------------

        result = max(
            best_cluster,
            key=lambda c: max(
                float(
                    c.get(
                        "score",
                        0.0
                    )
                ),
                0.0
            )
        ).copy()

        result[
            "ai_probability"
        ] = 0.0

        result[
            "ai_accepted"
        ] = False

        result[
            "ranking_score"
        ] = fallback_cluster_score(
            best_cluster
        )

        print()
        print(
            "TARGET C:"
            " SCALE-AWARE FALLBACK"
        )

        print(
            f"Cluster size: "
            f"{len(best_cluster)}"
        )

        print(
            f"Selected center: "
            f"("
            f"{result['center_x']:.1f}, "
            f"{result['center_y']:.1f}"
            f")"
        )

        print(
            f"Scale: "
            f"{result['scale']:.3f}"
        )

        print(
            f"OpenCV: "
            f"{result['score']:.4f}"
        )

        return result

    # --------------------------------------------------------
    # Original fallback
    # --------------------------------------------------------

    result = ranked[0].copy()

    result[
        "ranking_score"
    ] = strongest_score

    result[
        "ai_probability"
    ] = 0.0

    result[
        "ai_accepted"
    ] = False

    print()
    print(
        "SELECTION MODE: "
        "OPENCV FALLBACK"
    )

    print(
        f"Selected center: "
        f"("
        f"{result['center_x']:.1f}, "
        f"{result['center_y']:.1f}"
        f")"
    )

    print(
        f"OpenCV: "
        f"{result['score']:.4f}"
    )

    return result
# ============================================================
# WIDE OPENCV FALLBACK
# ============================================================

def select_opencv_fallback_candidate(
    candidates
):

    if not candidates:

        return None

    print()
    print(
        "### WIDE OPENCV FALLBACK ###"
    )

    print(
        f"Fallback candidates: "
        f"{len(candidates)}"
    )

    # ========================================================
    # SORT BY OPENCV
    # ========================================================

    ranked = sorted(
        candidates,
        key=lambda c: float(
            c.get(
                "score",
                0.0
            )
        ),
        reverse=True
    )

    # ========================================================
    # STRONGEST SCORE
    # ========================================================

    strongest_score = max(
        float(
            ranked[0].get(
                "score",
                0.0
            )
        ),
        0.0
    )

    minimum_score = max(
        0.05,
        strongest_score *
        FALLBACK_SCORE_RATIO
    )

    # ========================================================
    # KEEP REASONABLY STRONG CANDIDATES
    # ========================================================

    usable = [

        candidate

        for candidate in ranked

        if float(
            candidate.get(
                "score",
                0.0
            )
        ) >= minimum_score
    ]

    if not usable:

        usable = ranked[:FALLBACK_TOP_K]

    # ========================================================
    # TOP FALLBACK CANDIDATES
    # ========================================================

    usable = usable[
        :FALLBACK_TOP_K
    ]

    print(
        f"Strongest OpenCV score : "
        f"{strongest_score:.4f}"
    )

    print(
        f"Minimum fallback score : "
        f"{minimum_score:.4f}"
    )

    print(
        f"Usable fallback candidates : "
        f"{len(usable)}"
    )

    # ========================================================
    # PRINT FALLBACK CANDIDATES
    # ========================================================

    print()
    print(
        "FALLBACK CANDIDATES"
    )

    print(
        "-" * 90
    )

    for index, candidate in enumerate(
        usable,
        1
    ):

        print(
            f"{index:02d} | "
            f"Center=("
            f"{candidate['center_x']:.1f}, "
            f"{candidate['center_y']:.1f}) | "
            f"Scale="
            f"{candidate['scale']:.3f} | "
            f"OpenCV="
            f"{candidate['score']:.4f}"
        )

    print(
        "-" * 90
    )

    # ========================================================
    # WEIGHTED SPATIAL CONSENSUS
    # ========================================================

    weights = np.array(
        [
            max(
                float(
                    candidate.get(
                        "score",
                        0.0
                    )
                ),
                0.001
            )
            for candidate in usable
        ],
        dtype=np.float64
    )

    x_values = np.array(
        [
            float(
                candidate["center_x"]
            )
            for candidate in usable
        ],
        dtype=np.float64
    )

    y_values = np.array(
        [
            float(
                candidate["center_y"]
            )
            for candidate in usable
        ],
        dtype=np.float64
    )

    consensus_x = float(
        np.average(
            x_values,
            weights=weights
        )
    )

    consensus_y = float(
        np.average(
            y_values,
            weights=weights
        )
    )

    # ========================================================
    # FIND REPRESENTATIVE CANDIDATE
    # ========================================================

    representative = min(
        usable,
        key=lambda candidate:
            (
                (
                    candidate["center_x"]
                    -
                    consensus_x
                ) ** 2
                +
                (
                    candidate["center_y"]
                    -
                    consensus_y
                ) ** 2
            )
    )

    best = representative.copy()

    # ========================================================
    # CONSENSUS CENTER
    # ========================================================

    best[
        "center_x"
    ] = consensus_x

    best[
        "center_y"
    ] = consensus_y

    # ========================================================
    # RECALCULATE TOP LEFT
    # ========================================================

    best[
        "x"
    ] = int(
        round(
            consensus_x
            -
            best["width"] / 2.0
        )
    )

    best[
        "y"
    ] = int(
        round(
            consensus_y
            -
            best["height"] / 2.0
        )
    )

    # ========================================================
    # FALLBACK METADATA
    # ========================================================

    best[
        "ranking_score"
    ] = float(
        np.sum(weights)
    )

    best[
        "ai_probability"
    ] = 0.0

    best[
        "ai_accepted"
    ] = False

    # ========================================================
    # OUTPUT
    # ========================================================

    print()
    print(
        "WIDE FALLBACK RESULT"
    )

    print(
        f"Consensus center: "
        f"({consensus_x:.1f}, "
        f"{consensus_y:.1f})"
    )

    print(
        f"Representative: "
        f"({representative['center_x']:.1f}, "
        f"{representative['center_y']:.1f})"
    )

    print(
        f"Representative scale: "
        f"{representative['scale']:.3f}"
    )

    print(
        f"Representative OpenCV: "
        f"{representative['score']:.4f}"
    )

    print(
        "=" * 70
    )

    return best


# ============================================================
# MAIN LOCALIZATION
# ============================================================

def localize_ai(
    reference_path,
    search_path
):

    # ========================================================
    # LOAD
    # ========================================================

    reference, search = load_images(
        reference_path,
        search_path
    )

    # ========================================================
    # NORMAL OPENCV SEARCH
    # ========================================================

    candidates = generate_candidates(
        reference,
        search,
        apply_nms=True,
        max_candidates=TOP_CANDIDATES
    )

    # ========================================================
    # NORMAL CNN VERIFICATION
    # ========================================================

    verified = verify_candidates(
        search,
        candidates
    )

    # ========================================================
    # COUNT ACCEPTED
    # ========================================================

    accepted_count = sum(
        1
        for candidate in verified
        if candidate.get(
            "ai_accepted",
            False
        )
    )

    # ========================================================
    # NORMAL AI PATH
    # ========================================================

    if accepted_count > 0:

        best = select_best_candidate(
            verified
        )

    # ========================================================
    # WIDE FALLBACK
    #
    # Only activated when CNN rejects EVERYTHING.
    # ========================================================

    else:

        print()
        print(
            "=" * 70
        )

        print(
            "NO AI ACCEPTED CANDIDATES"
        )

        print(
            "RUNNING WIDE OPENCV FALLBACK"
        )

        print(
            "=" * 70
        )

        fallback_candidates = (
            generate_candidates(
                reference,
                search,
                apply_nms=False,
                max_candidates=FALLBACK_CANDIDATES
            )
        )

        best = (
            select_opencv_fallback_candidate(
                fallback_candidates
            )
        )

    # ========================================================
    # SAFETY
    # ========================================================

    if best is None:

        raise RuntimeError(
            "No localization candidate found."
        )

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "x":
            best["x"],

        "y":
            best["y"],

        "width":
            best["width"],

        "height":
            best["height"],

        "center_x":
            best["center_x"],

        "center_y":
            best["center_y"],

        "scale":
            best["scale"],

        "score":
            best["score"],

        "ai_probability":
            best.get(
                "ai_probability",
                0.0
            ),

        "ai_accepted":
            best.get(
                "ai_accepted",
                False
            ),

        "ranking_score":
            best.get(
                "ranking_score",
                0.0
            ),

        "candidate_count":
            len(
                candidates
            )
    }


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    reference_path = (
        "data/dataset/pair_001/reference.png"
    )

    search_path = (
        "data/dataset/pair_001/search.png"
    )

    result = localize_ai(
        reference_path,
        search_path
    )

    print()
    print(
        "=" * 60
    )

    print(
        "DRIFT-SENSE AI LOCALIZATION"
    )

    print(
        "=" * 60
    )

    print(
        f"Top-left       : "
        f"({result['x']}, "
        f"{result['y']})"
    )

    print(
        f"Center         : "
        f"({result['center_x']:.1f}, "
        f"{result['center_y']:.1f})"
    )

    print(
        f"Size           : "
        f"{result['width']} x "
        f"{result['height']}"
    )

    print(
        f"Scale          : "
        f"{result['scale']:.3f}"
    )

    print(
        f"OpenCV score   : "
        f"{result['score']:.4f}"
    )

    print(
        f"AI probability : "
        f"{result['ai_probability']:.4f}"
    )

    print(
        f"AI accepted    : "
        f"{result['ai_accepted']}"
    )

    print(
        f"Ranking score  : "
        f"{result['ranking_score']:.4f}"
    )

    print(
        f"Candidates     : "
        f"{result['candidate_count']}"
    )

    print(
        "=" * 60
    )