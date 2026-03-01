import argparse
import cv2
from filters import detect_fire_regions


def parse_args():
    parser = argparse.ArgumentParser(description="Run fire-focused image processing on one image.")
    parser.add_argument("--input", required=True, help="Path to input image.")
    parser.add_argument("--output", default="output_result.jpg", help="Path to save result image.")
    parser.add_argument("--min-area", type=int, default=300, help="Minimum blob area to keep.")
    parser.add_argument(
        "--fire-ratio-threshold",
        type=float,
        default=0.02,
        help="Threshold for FIRE DETECTED decision.",
    )
    parser.add_argument("--show", action="store_true", help="Show result windows.")
    return parser.parse_args()


def main():
    args = parse_args()
    image = cv2.imread(args.input)
    if image is None:
        raise SystemExit(f"Failed to read image: {args.input}")

    refined, boxes, fire_ratio = detect_fire_regions(image, min_area=args.min_area)
    result = image.copy()

    for x, y, w, h, area in boxes:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            result,
            f"area={int(area)}",
            (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    label = "FIRE DETECTED" if fire_ratio >= args.fire_ratio_threshold else "NO FIRE"
    color = (0, 0, 255) if label == "FIRE DETECTED" else (0, 255, 0)
    cv2.putText(
        result,
        f"{label} ratio={fire_ratio:.3f}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )

    ok = cv2.imwrite(args.output, result)
    if not ok:
        raise SystemExit(f"Failed to write output image: {args.output}")

    print(f"Saved: {args.output}")
    print(f"Fire ratio: {fire_ratio:.4f}")
    print(f"Regions kept: {len(boxes)}")

    if args.show:
        cv2.imshow("Input", image)
        cv2.imshow("Refined Fire Mask", refined)
        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
