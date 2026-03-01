import argparse
import cv2
from filters import detect_fire_regions, fire_color_mask, glare_mask

def find_cameras(max_index=5):
    found = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        ok = cap.isOpened()
        if ok:
            ret, frame = cap.read()
            if ret and frame is not None:
                found.append((i, frame.shape[1], frame.shape[0]))
        cap.release()
    return found


def choose_camera(cameras, forced_index=None, prefer_external=True):
    if forced_index is not None:
        for cam in cameras:
            if cam[0] == forced_index:
                return forced_index
        raise SystemExit(f"Camera index {forced_index} was not found in detected cameras: {cameras}")

    if prefer_external:
        external = [cam for cam in cameras if cam[0] != 0]
        if external:
            # Prefer the lowest non-zero index (common USB camera pattern on Windows).
            return sorted(external, key=lambda c: c[0])[0][0]

    return cameras[0][0]


def open_camera_with_fallback(index):
    # Try common Windows backends first, then fallback to default backend.
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap, backend
        cap.release()
    return None, None


def build_parser():
    parser = argparse.ArgumentParser(description="Camera preview with USB webcam preference.")
    parser.add_argument("--camera-index", type=int, default=None, help="Force a specific camera index.")
    parser.add_argument("--max-index", type=int, default=8, help="Highest camera index to scan.")
    parser.add_argument(
        "--prefer-laptop-cam",
        action="store_true",
        help="Prefer laptop camera behavior (index 0 first) instead of USB camera.",
    )
    parser.add_argument(
        "--show-masks",
        action="store_true",
        help="Show fire and glare masks in separate windows.",
    )
    parser.add_argument(
        "--fire-ratio-threshold",
        type=float,
        default=0.02,
        help="Fire-mask area ratio threshold for showing FIRE DETECTED text.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.camera_index is not None:
        camera_index = args.camera_index
        print(f"Trying forced camera index: {camera_index}")
    else:
        cams = find_cameras(args.max_index)
        print("Found cameras:", cams)
        if not cams:
            raise SystemExit("No working camera found.")
        camera_index = choose_camera(
            cams,
            forced_index=None,
            prefer_external=not args.prefer_laptop_cam,
        )
        print(f"Using camera index: {camera_index}")

    cap, backend = open_camera_with_fallback(camera_index)
    if cap is None:
        raise SystemExit(f"Failed to open camera {camera_index} with available backends.")
    print(f"Camera backend: {backend}")

    print("Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to read frame from camera.")
                break

            refined, boxes, fire_ratio = detect_fire_regions(frame, min_area=300)

            view = frame.copy()
            for x, y, w, h, _ in boxes:
                cv2.rectangle(view, (x, y), (x + w, y + h), (0, 0, 255), 2)

            if fire_ratio >= args.fire_ratio_threshold:
                cv2.putText(
                    view,
                    f"FIRE DETECTED ({fire_ratio:.3f})",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    view,
                    f"No fire ({fire_ratio:.3f})",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Camera Preview", view)
            if args.show_masks:
                fire = fire_color_mask(frame)
                glare = glare_mask(frame)
                cv2.imshow("Fire Mask", fire)
                cv2.imshow("Glare Mask", glare)
                cv2.imshow("Refined Fire Mask", refined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
