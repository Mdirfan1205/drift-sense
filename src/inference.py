from pathlib import Path
import contextlib
import io
import sys


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent.parent

SRC_DIR = (
    BASE_DIR / "src"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) != 3:

        print(
            "Usage: "
            "python inference.py "
            "<reference_image> "
            "<search_image>",
            file=sys.stderr
        )

        raise SystemExit(2)

    reference_path = Path(
        sys.argv[1]
    )

    search_path = Path(
        sys.argv[2]
    )

    if not reference_path.exists():

        print(
            f"Reference image not found: "
            f"{reference_path}",
            file=sys.stderr
        )

        raise SystemExit(1)

    if not search_path.exists():

        print(
            f"Search image not found: "
            f"{search_path}",
            file=sys.stderr
        )

        raise SystemExit(1)

    # --------------------------------------------------------
    # Import after argument validation.
    # This avoids loading unnecessary code for bad paths.
    # --------------------------------------------------------

    try:

        from localize_ai import localize_ai

    except Exception as error:

        print(
            f"Could not load localization module: "
            f"{error}",
            file=sys.stderr
        )

        raise SystemExit(1)

    # --------------------------------------------------------
    # Suppress the diagnostic print statements coming from
    # localize_ai.py.
    # --------------------------------------------------------

    try:

        silent_output = io.StringIO()

        with contextlib.redirect_stdout(
            silent_output
        ):

            result = localize_ai(
                reference_path,
                search_path
            )

    except Exception as error:

        print(
            f"Inference failed: "
            f"{error}",
            file=sys.stderr
        )

        raise SystemExit(1)

    # --------------------------------------------------------
    # CRITICAL SUBMISSION OUTPUT
    #
    # Exactly one line:
    # x y
    # --------------------------------------------------------

    print(
        f"{float(result['center_x']):.2f} "
        f"{float(result['center_y']):.2f}"
    )


if __name__ == "__main__":
    main()