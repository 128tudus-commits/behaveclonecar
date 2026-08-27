import argparse
import csv
import random
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import pygame
except ImportError:
    pygame = None

try:
    import serial
except ImportError:
    serial = None

VALUE_RE = re.compile(r"\*\s*([-+]?\d+(?:\.\d+)?)\s*\*\s*,\s*\*([-+]?\d+(?:\.\d+)?)\s*\*")


def clamp(value, low, high):
    return max(low, min(high, value))


def detect_arduino_port(baudrate=115200, per_port_timeout=2.0, max_ports=6):
    if serial is None:
        return None
    from serial.tools import list_ports

    keywords = ("arduino", "ch340", "ch341", "cp210", "ftdi", "usb-serial", "usb serial", "silab", "wch")
    ports = list(list_ports.comports())
    preferred = [p for p in ports if any(k in (p.description or "").lower() or k in (p.hwid or "").lower() for k in keywords)]
    others = [p for p in ports if p not in preferred]
    ordered = [p.device for p in (preferred + others)][:max_ports]

    fallback = None
    for device in ordered:
        try:
            ser = serial.Serial(device, baudrate, timeout=0.2)
        except Exception:
            continue
        if fallback is None:
            fallback = device
        deadline = time.time() + per_port_timeout
        buf = b""
        while time.time() < deadline:
            try:
                buf += ser.read(256)
            except Exception:
                break
            if VALUE_RE.search(buf.decode("utf-8", errors="ignore")):
                ser.close()
                return device
        ser.close()
    return fallback


def detect_camera(backend, max_index=10, tries=5):
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend)
        ok = False
        if cap.isOpened():
            for _ in range(tries):
                ok, frame = cap.read()
                if ok and frame is not None:
                    break
        cap.release()
        if ok:
            return i
    return None


class Gamepad:
    def __init__(self, index=0, steer_axis=0, gas_axis=1, deadzone=0.05):
        if pygame is None:
            raise RuntimeError("Missing pygame library. Install: pip install pygame")
        pygame.init()
        pygame.joystick.init()
        count = pygame.joystick.get_count()
        if count == 0:
            pygame.quit()
            raise RuntimeError("No gamepad detected.")
        if index >= count:
            pygame.quit()
            raise RuntimeError(f"Found {count} gamepad(s), index {index} does not exist.")
        self.js = pygame.joystick.Joystick(index)
        self.js.init()
        self.steer_axis = steer_axis
        self.gas_axis = gas_axis
        self.deadzone = deadzone
        print(f"Gamepad: {self.js.get_name()} | axes: {self.js.get_numaxes()} | buttons: {self.js.get_numbuttons()}")

    def poll(self):
        pygame.event.pump()
        axes = [self.js.get_axis(i) for i in range(self.js.get_numaxes())]

        steer = 0.0
        if self.steer_axis < len(axes):
            steer = axes[self.steer_axis]
            if abs(steer) < self.deadzone:
                steer = 0.0

        gas = 0.0
        if self.gas_axis < len(axes):
            gas = clamp(-axes[self.gas_axis], 0.0, 1.0)
            if abs(axes[self.gas_axis]) < self.deadzone:
                gas = 0.0

        return clamp(steer, -1.0, 1.0), gas, axes

    def quit(self):
        pygame.quit()


def aug_flip(img, steer):
    return cv2.flip(img, 1), -steer


def aug_bright_low(img, steer):
    return cv2.convertScaleAbs(img, alpha=random.uniform(0.55, 0.80), beta=0), steer


def aug_bright_high(img, steer):
    return cv2.convertScaleAbs(img, alpha=random.uniform(1.20, 1.50), beta=10), steer


def aug_noise(img, steer):
    noise = np.random.normal(0, 12, img.shape).astype(np.float32)
    out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out, steer


def aug_blur(img, steer):
    k = random.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), 0), steer


def aug_shadow(img, steer):
    h, w = img.shape[:2]
    mask = np.ones((h, w), dtype=np.float32)
    x_start = random.randint(0, w // 2)
    width = random.randint(w // 4, w // 2)
    end = min(w, x_start + width)
    mask[:, x_start:end] *= np.linspace(1.0, random.uniform(0.30, 0.55), end - x_start, dtype=np.float32)
    out = np.clip(img.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)
    return out, steer


RANDOM_AUGS = [
    ("bright_low", aug_bright_low),
    ("bright_high", aug_bright_high),
    ("noise", aug_noise),
    ("blur", aug_blur),
    ("shadow", aug_shadow),
]


def next_index(images_dir):
    indices = []
    for p in images_dir.glob("*.jpg"):
        head = p.stem.split("_")[0]
        if head.isdigit():
            indices.append(int(head))
    return (max(indices) + 1) if indices else 0


def save_sample(frame, steer, throttle, idx, images_dir, writer, extra):
    stem = f"{idx:06d}"
    flip_img, flip_steer = aug_flip(frame, steer)
    variants = [
        (stem + "_orig.jpg", frame, steer, "orig"),
        (stem + "_flip.jpg", flip_img, flip_steer, "flip"),
    ]
    used_names = set()
    for _ in range(extra):
        name, fn = random.choice(RANDOM_AUGS)
        img, s = fn(frame, steer)
        fname = f"{stem}_{name}.jpg"
        if fname in used_names:
            fname = f"{stem}_{name}_{random.randint(100, 999)}.jpg"
        used_names.add(fname)
        variants.append((fname, img, s, name))

    for fname, img, s, tag in variants:
        cv2.imwrite(str(images_dir / fname), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        writer.writerow([fname, f"{s:.4f}", f"{throttle:.4f}", tag])
    return len(variants)


def draw_overlay(frame, steer, gas, recording, saved, axes_line, pad_name):
    view = frame.copy()
    lines = [
        f"Steer : {steer:+.2f}",
        f"Gas   : {gas:.2f}",
        f"Recording: {'YES' if recording else 'NO'}  [SPACE]",
        f"Saved samples: {saved}   [Q] quit",
        f"{pad_name}: {axes_line}",
    ]
    color = (0, 0, 255) if recording else (0, 255, 0)
    y = 18
    for line in lines:
        cv2.putText(view, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        y += 15
    return view


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset collector: camera + gamepad (steer from axis, gas from trigger); also bridges gamepad -> Arduino")
    parser.add_argument("--pad", type=int, default=0, help="Gamepad index")
    parser.add_argument("--steer-axis", type=int, default=3, help="Steering axis")
    parser.add_argument("--gas-axis", type=int, default=1, help="Gas axis (left joystick Y)")
    parser.add_argument("--deadzone", type=float, default=0.05)
    parser.add_argument("-p", "--port", default="", help="Arduino port (auto-detect when empty, 'none' disables sending)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--send-fps", type=float, default=20.0, help="Command send rate to Arduino")
    parser.add_argument("-c", "--camera", type=int, default=-1, help="Camera index (auto-detect when -1)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=20.0, help="Frame saving rate")
    parser.add_argument("--extra", type=int, default=3, help="Number of random augmentations per frame")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        pad = Gamepad(args.pad, args.steer_axis, args.gas_axis, args.deadzone)
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    port = args.port.strip()
    ser = None
    if port.lower() == "none":
        print("Sending to Arduino disabled (--port none).")
    else:
        if serial is None:
            print("Missing pyserial library. Install: pip install pyserial")
            pad.quit()
            sys.exit(1)
        if not port:
            print("Detecting Arduino port...")
            port = detect_arduino_port(args.baud) or ""
            print(f"Found Arduino: {port}" if port else "Arduino not found - gamepad will not drive the car.")
        else:
            print(f"Arduino: {port}")
        if port:
            try:
                ser = serial.Serial(port, args.baud, timeout=1)
            except Exception as e:
                print(f"Cannot open {port}: {e}")
                pad.quit()
                sys.exit(1)

    send_interval = 1.0 / max(args.send_fps, 0.1)
    last_send = 0.0

    dataset_dir = Path(__file__).resolve().parent / "dataset"
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    csv_path = dataset_dir / "labels.csv"
    new_csv = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="", encoding="utf-8-sig")
    writer = csv.writer(csv_file)
    if new_csv:
        writer.writerow(["filename", "steering", "throttle", "augmentation"])

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cam_index = args.camera
    if cam_index < 0:
        print("Detecting camera...")
        cam_index = detect_camera(backend)
        if cam_index is None:
            print("No camera found.")
            pad.quit()
            sys.exit(1)
        print(f"Found camera: index {cam_index}")
    else:
        print(f"Camera: index {cam_index}")

    cap = cv2.VideoCapture(cam_index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print("Cannot open camera.")
        csv_file.close()
        pad.quit()
        sys.exit(1)

    idx = next_index(images_dir)
    recording = False
    last_save = 0.0
    saved = 0
    pad_name = pad.js.get_name()[:22]

    print(f"Dataset: {dataset_dir}")
    print(f"Steer: axis {args.steer_axis}, gas: axis {args.gas_axis} (up = throttle, center/down = idle)")
    print("SPACE - start/stop recording, Q - quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            steer, gas, axes = pad.poll()
            axes_line = " ".join(f"{v:+.2f}" for v in axes[:8])

            now = time.time()
            if ser is not None and (now - last_send) >= send_interval:
                last_send = now
                cmd = f"*{steer:.3f}*, *{gas:.3f}*\n"
                try:
                    ser.write(cmd.encode("ascii"))
                except Exception as e:
                    print(f"Send error: {e}")
                    try:
                        ser.write(b"*0.000*, *0.000*\n")
                    except Exception:
                        pass
                    ser.close()
                    ser = None

            if recording and (now - last_save) >= (1.0 / args.fps):
                last_save = now
                saved += save_sample(frame, steer, gas, idx, images_dir, writer, args.extra)
                csv_file.flush()
                idx += 1

            cv2.imshow("Dataset collector", draw_overlay(frame, steer, gas, recording, saved, axes_line, pad_name))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                recording = not recording
                last_save = 0.0
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
        csv_file.close()
        pad.quit()
        cv2.destroyAllWindows()
        print(f"Done. Total files (with augmentation): {saved}")
        print(f"Labels: {csv_path}")


if __name__ == "__main__":
    main()

