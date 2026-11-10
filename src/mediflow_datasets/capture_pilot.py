"""Preview and record photos by condition for the camera pilot on Windows."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

DOMAINS = {
    "web_skin": (1280, 720),
    "skin": (640, 480),
    "hair": (640, 480),
}
CONDITIONS = {"기준": "ref", "변경": "alt"}
BACKENDS = {"DirectShow": cv2.CAP_DSHOW, "Media Foundation": cv2.CAP_MSMF}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIRECTSHOW_LIST = PROJECT_ROOT / "scripts" / "list_directshow_cameras.ps1"


def directshow_camera_name(index: int) -> str:
    """Resolve a DirectShow index to the current Windows device name."""
    if os.name != "nt":
        return ""
    try:
        result = subprocess.run(
            [
                "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                DIRECTSHOW_LIST.read_text(encoding="utf-8"),
            ],
            capture_output=True,
            check=True,
            encoding="utf-8",
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        names = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return ""
    if isinstance(names, list) and 0 <= index < len(names) and isinstance(names[index], str):
        return names[index]
    return ""


def save_capture(frame, directory: Path, record: dict) -> Path:
    """Save a full-size PNG and its matching metadata without overwriting either."""
    ok, png = cv2.imencode(".png", frame)
    if not ok:
        raise OSError("PNG로 변환할 수 없습니다.")

    now = datetime.now().astimezone()
    id_parts = [record["participant"], record["session"], record["domain"]]
    if record.get("pair"):
        id_parts.append(record["pair"])
    id_parts.extend([record["condition"], now.strftime("%Y%m%dT%H%M%S%f")])
    photo_id = "_".join(id_parts)
    height, width = frame.shape[:2]
    photo = directory / f"{photo_id}.png"
    metadata = photo.with_suffix(".json")
    details = {
        **record,
        "photo_id": photo_id,
        "file": photo.name,
        "captured_at": now.isoformat(timespec="milliseconds"),
        "width": width,
        "height": height,
        "label_status": "unverified",
    }
    directory.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with photo.open("xb") as stream:
            created.append(photo)
            stream.write(png.tobytes())
        with metadata.open("x", encoding="utf-8") as stream:
            created.append(metadata)
            json.dump(details, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except OSError:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return photo


class PilotCapture:
    def __init__(self, root: tk.Tk, output: Path):
        self.root = root
        self.output = output
        self.camera = None
        self.frame = None
        self.preview_settings = None
        self.preview_image = None

        root.title("MediFlow 시험 촬영")
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)
        root.protocol("WM_DELETE_WINDOW", self.close)
        form = ttk.Frame(root, padding=12)
        form.grid(row=0, column=0, sticky="ns")
        self.fields = {}

        def field(key, label, value, choices=None):
            row = len(self.fields)
            variable = tk.StringVar(value=value)
            self.fields[key] = variable
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=3)
            if choices is None:
                widget = ttk.Entry(form, textvariable=variable, width=24)
            else:
                widget = ttk.Combobox(
                    form, textvariable=variable, values=choices, state="readonly", width=22
                )
            widget.grid(row=row, column=1, sticky="ew", padx=(10, 0))
            return widget

        domain = field("domain", "촬영 분야", "web_skin", tuple(DOMAINS))
        index = field("index", "카메라 번호", "", tuple(str(i) for i in range(8)))
        backend = field("backend", "연결 방식", "DirectShow", tuple(BACKENDS))
        field("camera_name", "카메라 이름 (선택)", "")
        field("participant", "참여자 번호", "P01")
        field("session", "촬영 세션", "S01")
        field("body_site", "촬영 부위", "정면 얼굴")
        field("condition", "촬영 조건", "기준", tuple(CONDITIONS))
        field("lighting", "조명 메모 (선택)", "")
        field("distance", "거리·배율 (선택)", "")
        field("camera_controls", "초점·노출·색 (선택)", "")
        domain.bind("<<ComboboxSelected>>", self.change_domain)
        index.bind("<<ComboboxSelected>>", self.change_camera)
        backend.bind("<<ComboboxSelected>>", self.change_camera)

        controls = ttk.Frame(form)
        controls.grid(row=len(self.fields), column=0, columnspan=2, sticky="ew", pady=12)
        ttk.Button(controls, text="미리보기 시작", command=self.start_preview).pack(side="left")
        ttk.Button(controls, text="중지", command=self.stop_preview).pack(side="left", padx=4)
        self.save_button = ttk.Button(
            controls, text="사진 저장", command=self.capture, state="disabled"
        )
        self.save_button.pack(side="left")

        view = ttk.Frame(root, padding=(0, 12, 12, 12))
        view.grid(row=0, column=1, sticky="nsew")
        self.preview = ttk.Label(
            view, text="카메라 번호를 선택하고 미리보기를 시작하세요.", anchor="center", width=58
        )
        self.preview.pack(fill="both", expand=True)
        self.resolution = tk.StringVar(value="")
        ttk.Label(view, textvariable=self.resolution).pack(pady=(6, 0))
        self.status = tk.StringVar(value="미리보기 중에는 사진이 저장되지 않습니다.")
        ttk.Label(root, textvariable=self.status, wraplength=790).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 6)
        )
        ttk.Label(root, text=f"저장 위치  {output}", wraplength=790).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 12)
        )
        root.after(40, self.refresh)

    def change_domain(self, _event):
        self.change_camera(_event)
        domain = self.fields["domain"].get()
        self.fields["body_site"].set("정면 얼굴" if domain == "web_skin" else "")
        self.fields["lighting"].set("")
        self.fields["distance"].set("")
        self.fields["camera_controls"].set("")
        self.fields["condition"].set("기준")

    def change_camera(self, _event):
        self.stop_preview()
        self.fields["camera_name"].set("")

    def current_settings(self):
        return tuple(self.fields[key].get() for key in ("domain", "index", "backend"))

    def start_preview(self):
        self.stop_preview()
        domain, index, backend = self.current_settings()
        if not index:
            self.status.set("카메라 번호를 먼저 선택하세요.")
            return
        width, height = DOMAINS[domain]
        camera = cv2.VideoCapture(int(index), BACKENDS[backend])
        if not camera.isOpened():
            camera.release()
            self.status.set(f"카메라 {index} 연결 실패. 번호나 연결 방식을 바꿔 보세요.")
            return
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.camera = camera
        self.preview_settings = self.current_settings()
        self.fields["camera_name"].set(
            directshow_camera_name(int(index)) if backend == "DirectShow" else ""
        )
        self.status.set(f"카메라 {index} 연결됨. 화면과 실제 해상도를 확인하세요.")

    def stop_preview(self):
        if self.camera is not None:
            self.camera.release()
        self.camera = None
        self.frame = None
        self.preview_settings = None
        self.preview_image = None
        self.preview.configure(image="", text="카메라 번호를 선택하고 미리보기를 시작하세요.")
        self.resolution.set("")
        self.save_button.configure(state="disabled")
        self.status.set("미리보기 중지")

    def refresh(self):
        if self.camera is not None:
            if self.preview_settings != self.current_settings():
                self.stop_preview()
                self.status.set("선택이 바뀌었습니다. 미리보기를 다시 시작하세요.")
            else:
                ok, frame = self.camera.read()
                if ok and frame is not None:
                    self.frame = frame
                    height, width = frame.shape[:2]
                    self.resolution.set(f"실제 영상 해상도  {width}×{height}")
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    image.thumbnail((600, 440), Image.Resampling.BILINEAR)
                    self.preview_image = ImageTk.PhotoImage(image)
                    self.preview.configure(image=self.preview_image, text="")
                    self.save_button.configure(state="normal")
                else:
                    self.stop_preview()
                    self.status.set("영상 수신 실패. 다른 카메라 앱을 닫고 다시 시작하세요.")
        self.root.after(40, self.refresh)

    def capture(self):
        if self.frame is None or self.preview_settings != self.current_settings():
            return
        record = {key: field.get().strip() for key, field in self.fields.items()}
        record["participant"] = record["participant"].upper()
        record["session"] = record["session"].upper()
        if not (re.fullmatch(r"P[0-9]{2,4}", record["participant"])
                and re.fullmatch(r"S[0-9]{2,4}", record["session"])):
            messagebox.showinfo("촬영 기록", "참여자는 P01, 세션은 S01 형식으로 입력하세요.")
            return
        if not record["body_site"]:
            messagebox.showinfo("촬영 기록", "촬영 부위를 입력하세요.")
            return
        record["condition"] = CONDITIONS[record["condition"]]
        directory = self.output / record["domain"] / record["participant"] / record["session"]
        try:
            photo = save_capture(self.frame, directory, record)
        except (OSError, cv2.error) as exc:
            messagebox.showerror("저장 실패", str(exc))
            return
        height, width = self.frame.shape[:2]
        self.status.set(f"저장 완료  {photo.name}  ·  {width}×{height}  ·  촬영 기록 JSON 포함")

    def close(self):
        self.stop_preview()
        self.root.destroy()


def main():
    root = tk.Tk()
    PilotCapture(root, PROJECT_ROOT / "camera_capture")
    root.mainloop()


if __name__ == "__main__":
    main()
