import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import serial
except ImportError:
    serial = None

from record import Gamepad, detect_arduino_port, detect_camera


def parse_args():
    parser = argparse.ArgumentParser(description="Autonomous driving: model.onnx + Arduino (format: *steering*, *throttle*)")
    parser.add_argument("--model", default="", help="Path to model.onnx (default: next to the script)")
    parser.add_argument("-p", "--port", default="", help="Arduino port (auto-detect when empty, 'none' disables sending)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("-c", "--camera", type=int, default=-1, help="Camera index (auto-detect when -1)")
    parser.add_argument("--width", type=int, default=160, help="Network input width")
    parser.add_argument("--height", type=int, default=120, help="Network input height")
    parser.add_argument("--send-fps", type=float, default=20.0, help="Command send rate")
    parser.add_argument("--throttle-limit", type=float, default=1.0, help="Maximum throttle (safety limit)")
    parser.add_argument("--manual-throttle", action="store_true", help="Driver controls throttle via gamepad, AI steers only")
    parser.add_argument("--pad", type=int, default=0, help="Gamepad index (used with --manual-throttle)")
    parser.add_argument("--steer-axis", type=int, default=3, help="Steering axis (gamepad, used with --manual-throttle)")
    parser.add_argument("--gas-axis", type=int, default=1, help="Gas axis (left joystick Y, used with --manual-throttle)")
    parser.add_argument("--deadzone", type=float, default=0.05)
    return parser.parse_args()


def preprocess(frame, width, height):
    img = cv2.resize(frame, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.ascontiguousarray(img.transpose(2, 0, 1))[None]


def clamp(value, low, high):
    return max(low, min(high, value))


def draw_overlay(frame, steer, throttle, driving, sending, fps, throttle_src):
    view = frame.copy()
    lines = [
        f"Steer: {steer:+.3f} [AI]",
        f"Throttle: {throttle:.3f} [{throttle_src}]",
        f"Driving: {'YES' if driving else 'NO'}  [SPACE]",
        f"Inference: {fps:.1f} FPS   [Q] quit",
    ]
    if sending:
        lines.insert(0, "SENDING TO ARDUINO ACTIVE")
    color = (0, 0, 255) if driving else (0, 255, 0)
    y = 18
    for line in lines:
        cv2.putText(view, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        y += 15
    return view


def main():
    args = parse_args()
    base = Path(__file__).resolve().parent
    model_path = Path(args.model) if args.model else base / "model.onnx"

    if ort is None:
        print("Missing onnxruntime library. Install: pip install onnxruntime")
        sys.exit(1)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    pad = None
    if args.manual_throttle:
        try:
            pad = Gamepad(args.pad, args.steer_axis, args.gas_axis, args.deadzone)
            print("MANUAL THROTTLE: gas comes from gamepad, AI steers only.")
        except RuntimeError as e:
            print(e)
            sys.exit(1)

    port = args.port.strip()
    ser = None

    if port.lower() == "none":
        print("Sending to Arduino disabled (--port none) - preview mode.")
    else:
        if serial is None:
            print("Missing pyserial library. Install: pip install pyserial")
            if pad is not None:
                pad.quit()
            sys.exit(1)

        if not port:
            print("Detecting Arduino port...")
            port = detect_arduino_port(args.baud) or ""
            print(f"Found Arduino: {port}" if port else "Arduino not found - preview mode.")
        else:
            print(f"Arduino: {port}")

        if port:
            try:
                ser = serial.Serial(port, args.baud, timeout=1)
            except Exception as e:
                print(f"Cannot open {port}: {e}")
                if pad is not None:
                    pad.quit()
                sys.exit(1)

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cam_index = args.camera

    if cam_index < 0:
        print("Detecting camera...")
        cam_index = detect_camera(backend)
        if cam_index is None:
            print("No camera found.")
            if ser is not None:
                ser.close()
            if pad is not None:
                pad.quit()
            sys.exit(1)
        print(f"Found camera: index {cam_index}")

    cap = cv2.VideoCapture(cam_index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Cannot open camera.")
        if ser is not None:
            ser.close()
        if pad is not None:
            pad.quit()
        sys.exit(1)

    send_interval = 1.0 / max(args.send_fps, 0.1)
    last_send = 0.0
    infer_fps = 0.0
    steer = 0.0
    throttle = 0.0
    driving = False

    print("SPACE - start/stop driving, Q - quit")

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                time.sleep(0.01)
                continue

            t0 = time.time()

            preds = session.run(
                None,
                {input_name: preprocess(frame, args.width, args.height)}
            )[0][0]

            steer = clamp(float(preds[0]), -1.0, 1.0)

            throttle_src = "AI"

            if pad is not None:
                _pad_steer, pad_gas, _axes = pad.poll()
                throttle = clamp(pad_gas, 0.0, args.throttle_limit)
                throttle_src = "PAD"
            else:
                throttle = clamp(float(preds[1]), 0.0, args.throttle_limit)

            dt = time.time() - t0
            if dt > 0:
                current_fps = 1.0 / dt
                infer_fps = 0.9 * infer_fps + 0.1 * current_fps if infer_fps else current_fps

            now = time.time()
            sent = False

            if driving and ser is not None and (now - last_send) >= send_interval:
                last_send = now
                cmd = f"*{steer:.3f}*, *{throttle:.3f}*\n"

                try:
                    ser.write(cmd.encode("ascii"))
                    sent = True
                except Exception as e:
                    print(f"Send error: {e}")
                    try:
                        ser.close()
                    except Exception:
                        pass
                    ser = None
                    driving = False

            cv2.imshow(
                "Drive",
                draw_overlay(
                    frame,
                    steer,
                    throttle,
                    driving,
                    sent or driving,
                    infer_fps,
                    throttle_src
                )
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):
                driving = not driving

                if not driving and ser is not None:
                    try:
                        ser.write(b"*0.000*, *0.000*\n")
                    except Exception:
                        pass

                last_send = 0.0

    except KeyboardInterrupt:
        pass

    finally:
        if ser is not None:
            try:
                ser.write(b"*0.000*, *0.000*\n")
            except Exception:
                pass
            ser.close()

        cap.release()

        if pad is not None:
            pad.quit()

        cv2.destroyAllWindows()
        print("Finished.")


if __name__ == "__main__":
    main()

