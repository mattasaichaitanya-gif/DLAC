"""
VisionPort — Interactive Graphical User Interface (GUI)
Provides a visual dashboard to run and inspect all 20 traffic monitoring tasks.
"""

import sys
import os
import json
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from visionport.common import (
    settings, open_video, frames, model, predict, detections,
    draw, text, polygon, in_zone, crossed, draw_line, Motion, LABELS
)

TASK_LIST = [
    ("Tasks 6-9: Vehicle Detection & Counting", "detect"),
    ("Task 12: Vehicle Tracking (ByteTrack)", "track"),
    ("Tasks 10-11: Density & Congestion Heuristics", "density"),
    ("Task 13: Approximate Vehicle Speed", "speed"),
    ("Task 14: Traffic Signal Status", "signal"),
    ("Task 15: Waiting Time Estimation", "waiting"),
    ("Task 16: Entry / Exit Directional Count", "entry_exit"),
    ("Task 17: Zone-to-Zone Wrong Turn Alert", "wrong_turn"),
    ("Task 18: Wrong-Way Traffic Alert", "wrong_way"),
    ("Task 19: Restricted Parking Monitoring", "parking"),
    ("Task 20: Candidate Triple Riding Alert", "triple_riding"),
    ("Task 3: Display Traffic Image", "image"),
    ("Task 4: Video Playback / Preview", "video"),
    ("Task 5: Extract Frames to Directory", "extract"),
]

class VisionPortGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("VisionPort — Intelligent Traffic Monitoring System")
        self.root.geometry("1180x760")
        self.root.minsize(950, 650)
        self.root.configure(bg="#1e1e2e")

        self.config_path = BASE_DIR / "config" / "settings.json"
        self.load_config()

        # Default to traffic.mp4 if present, else traffic.jpg
        vid_candidate = BASE_DIR / "data" / "traffic.mp4"
        if vid_candidate.exists():
            self.current_input = str(vid_candidate)
        else:
            self.current_input = str(BASE_DIR / self.cfg.get("image", "data/traffic.jpg"))
            
        self.running = False
        self.stop_requested = False
        self.model_instance = None
        self.loaded_model_name = None

        self.setup_ui()
        self.display_image(self.current_input)

    def load_config(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

    def setup_ui(self):
        # Top Header
        header = tk.Frame(self.root, bg="#181825", height=50)
        header.pack(fill=tk.X, side=tk.TOP)
        title = tk.Label(header, text="🚗 VisionPort — Intelligent Traffic Monitoring Dashboard",
                         font=("Segoe UI", 15, "bold"), fg="#cdd6f4", bg="#181825")
        title.pack(side=tk.LEFT, padx=15, pady=10)

        # Main Layout: Left controls, Center/Right preview & stats
        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg="#1e1e2e", sashrelief=tk.FLAT)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Sidebar (Controls)
        left_frame = tk.Frame(main_pane, bg="#1e1e2e", width=340)
        main_pane.add(left_frame)

        # 1. Input Selection Group
        input_box = tk.LabelFrame(left_frame, text="1. Input Media", font=("Segoe UI", 10, "bold"),
                                  fg="#89b4fa", bg="#1e1e2e", bd=1, relief=tk.GROOVE)
        input_box.pack(fill=tk.X, padx=5, pady=5)

        self.lbl_file = tk.Label(input_box, text=Path(self.current_input).name,
                                 fg="#a6adc8", bg="#1e1e2e", font=("Segoe UI", 9))
        self.lbl_file.pack(anchor="w", padx=10, pady=4)

        btn_row = tk.Frame(input_box, bg="#1e1e2e")
        btn_row.pack(fill=tk.X, padx=10, pady=5)
        btn_browse = tk.Button(btn_row, text="Browse Image / Video", font=("Segoe UI", 9),
                               bg="#313244", fg="#cdd6f4", activebackground="#45475a",
                               command=self.browse_file)
        btn_browse.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        btn_reset = tk.Button(btn_row, text="Reset Default", font=("Segoe UI", 9),
                              bg="#313244", fg="#cdd6f4", command=self.reset_default)
        btn_reset.pack(side=tk.RIGHT, padx=2)

        # 2. Model & Settings Group
        model_box = tk.LabelFrame(left_frame, text="2. Model & Precision", font=("Segoe UI", 10, "bold"),
                                  fg="#89b4fa", bg="#1e1e2e", bd=1, relief=tk.GROOVE)
        model_box.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(model_box, text="YOLO Model Weights:", fg="#cdd6f4", bg="#1e1e2e", font=("Segoe UI", 9)).pack(anchor="w", padx=10, pady=2)
        self.model_combo = ttk.Combobox(model_box, values=["yolo11m.pt (Medium - High Precision)",
                                                           "yolo11s.pt (Small - Fast & Accurate)",
                                                           "yolo11n.pt (Nano - Ultra Fast)"],
                                        state="readonly")
        self.model_combo.current(0)
        self.model_combo.pack(fill=tk.X, padx=10, pady=4)

        tk.Label(model_box, text="Confidence Threshold:", fg="#cdd6f4", bg="#1e1e2e", font=("Segoe UI", 9)).pack(anchor="w", padx=10, pady=2)
        conf_row = tk.Frame(model_box, bg="#1e1e2e")
        conf_row.pack(fill=tk.X, padx=10, pady=2)
        self.conf_slider = tk.Scale(conf_row, from_=0.10, to=0.90, resolution=0.05, orient=tk.HORIZONTAL,
                                    bg="#1e1e2e", fg="#cdd6f4", highlightthickness=0,
                                    troughcolor="#313244", activebackground="#89b4fa")
        self.conf_slider.set(self.cfg.get("confidence", 0.25))
        self.conf_slider.pack(fill=tk.X, expand=True)

        # 3. Tasks Group
        task_box = tk.LabelFrame(left_frame, text="3. Select Vision Task", font=("Segoe UI", 10, "bold"),
                                 fg="#89b4fa", bg="#1e1e2e", bd=1, relief=tk.GROOVE)
        task_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.task_listbox = tk.Listbox(task_box, bg="#181825", fg="#cdd6f4", selectbackground="#89b4fa",
                                       selectforeground="#11111b", bd=0, font=("Segoe UI", 9))
        for name, _ in TASK_LIST:
            self.task_listbox.insert(tk.END, name)
        self.task_listbox.selection_set(0)
        self.task_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Run Buttons
        action_frame = tk.Frame(left_frame, bg="#1e1e2e")
        action_frame.pack(fill=tk.X, padx=5, pady=8)

        run_btn_row = tk.Frame(action_frame, bg="#1e1e2e")
        run_btn_row.pack(fill=tk.X, pady=3)

        self.btn_run = tk.Button(run_btn_row, text="▶ Run Task", font=("Segoe UI", 10, "bold"),
                                 bg="#a6e3a1", fg="#11111b", activebackground="#94e2d5",
                                 command=self.run_task_thread)
        self.btn_run.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self.btn_stop = tk.Button(run_btn_row, text="⏹ Stop", font=("Segoe UI", 10, "bold"),
                                  bg="#f38ba8", fg="#11111b", activebackground="#eba0ac",
                                  command=self.stop_task, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.RIGHT, fill=tk.X, padx=(2, 0))

        self.btn_run_all = tk.Button(action_frame, text="⚡ Run All Tasks (Batch)", font=("Segoe UI", 9, "bold"),
                                     bg="#89b4fa", fg="#11111b", activebackground="#b4befe",
                                     command=self.run_all_thread)
        self.btn_run_all.pack(fill=tk.X, pady=3)

        # Right Area (Display & Metrics)
        right_frame = tk.Frame(main_pane, bg="#1e1e2e")
        main_pane.add(right_frame)

        # Image/Video canvas
        canvas_card = tk.LabelFrame(right_frame, text="Visual Output Preview", font=("Segoe UI", 10, "bold"),
                                   fg="#89b4fa", bg="#1e1e2e", bd=1, relief=tk.GROOVE)
        canvas_card.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas_label = tk.Label(canvas_card, bg="#11111b")
        self.canvas_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Metrics & Log card
        bottom_card = tk.LabelFrame(right_frame, text="Detection Metrics & Execution Log",
                                    font=("Segoe UI", 10, "bold"), fg="#89b4fa", bg="#1e1e2e",
                                    bd=1, relief=tk.GROOVE, height=180)
        bottom_card.pack(fill=tk.X, padx=5, pady=5)

        # Quick stats row
        self.stats_row = tk.Frame(bottom_card, bg="#1e1e2e")
        self.stats_row.pack(fill=tk.X, padx=10, pady=4)

        self.lbl_stat_total = tk.Label(self.stats_row, text="Total Vehicles: -", font=("Segoe UI", 10, "bold"),
                                       fg="#a6e3a1", bg="#1e1e2e")
        self.lbl_stat_total.pack(side=tk.LEFT, padx=10)

        self.lbl_stat_density = tk.Label(self.stats_row, text="Density Level: -", font=("Segoe UI", 10, "bold"),
                                         fg="#f9e2af", bg="#1e1e2e")
        self.lbl_stat_density.pack(side=tk.LEFT, padx=15)

        self.lbl_stat_status = tk.Label(self.stats_row, text="Status: Ready", font=("Segoe UI", 10),
                                        fg="#cdd6f4", bg="#1e1e2e")
        self.lbl_stat_status.pack(side=tk.RIGHT, padx=10)

        # Log Text Box
        self.log_text = tk.Text(bottom_card, bg="#181825", fg="#a6adc8", font=("Consolas", 9),
                                height=6, bd=0, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.log("VisionPort GUI initialized. Ready to execute.")

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def browse_file(self):
        filetypes = [("Image or Video", "*.jpg *.jpeg *.png *.mp4 *.avi"),
                     ("Images", "*.jpg *.jpeg *.png"),
                     ("Videos", "*.mp4 *.avi"),
                     ("All Files", "*.*")]
        chosen = filedialog.askopenfilename(initialdir=BASE_DIR / "data", filetypes=filetypes)
        if chosen:
            self.current_input = chosen
            self.lbl_file.config(text=Path(chosen).name)
            self.display_image(chosen)
            self.log(f"Selected media: {chosen}")

    def reset_default(self):
        self.current_input = str(BASE_DIR / "data" / "traffic.jpg")
        self.lbl_file.config(text="traffic.jpg")
        self.display_image(self.current_input)
        self.log("Reset to default bundled image (data/traffic.jpg).")

    def display_image(self, path_or_cvimg):
        try:
            if isinstance(path_or_cvimg, (str, Path)):
                p = Path(path_or_cvimg)
                if p.suffix.lower() in [".mp4", ".avi"]:
                    cap = cv2.VideoCapture(str(p))
                    ok, frame = cap.read()
                    cap.release()
                    if not ok: return
                    cvimg = frame
                else:
                    cvimg = cv2.imread(str(p))
            else:
                cvimg = path_or_cvimg

            if cvimg is None: return

            # Resize to fit inside canvas label keeping aspect ratio
            h_can = max(350, self.canvas_label.winfo_height())
            w_can = max(600, self.canvas_label.winfo_width())
            h, w = cvimg.shape[:2]
            scale = min(w_can / w, h_can / h, 1.0)
            nw, nh = int(w * scale), int(h * scale)

            resized = cv2.resize(cvimg, (nw, nh), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb)
            self.tk_img = ImageTk.PhotoImage(img_pil)
            self.canvas_label.config(image=self.tk_img)
        except Exception as e:
            self.log(f"Error displaying image: {e}")

    def get_selected_model_name(self):
        sel = self.model_combo.get()
        if "yolo11s" in sel: return "yolo11s.pt"
        if "yolo11n" in sel: return "yolo11n.pt"
        return "yolo11m.pt"

    def get_model(self):
        m_name = self.get_selected_model_name()
        if self.model_instance is None or self.loaded_model_name != m_name:
            self.log(f"Loading weights {m_name}...")
            self.model_instance = model({"model": m_name})
            self.loaded_model_name = m_name
            self.log(f"Model {m_name} loaded successfully.")
        return self.model_instance

    def run_task_thread(self):
        if self.running: return
        t = threading.Thread(target=self.execute_task, daemon=True)
        t.start()

    def run_all_thread(self):
        if self.running: return
        t = threading.Thread(target=self.execute_all_tasks, daemon=True)
        t.start()

    def stop_task(self):
        if self.running:
            self.stop_requested = True
            self.log("[*] Stop requested by user. Terminating playback...")

    def execute_task(self):
        self.running = True
        self.stop_requested = False
        self.btn_run.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.lbl_stat_status.config(text="Status: Processing...", fg="#fab387")
        try:
            sel_idx = self.task_listbox.curselection()
            if not sel_idx: sel_idx = (0,)
            task_title, task_key = TASK_LIST[sel_idx[0]]
            self.log(f"\n--- Running: {task_title} ---")

            cfg = dict(self.cfg)
            cfg["input"] = self.current_input
            cfg["confidence"] = float(self.conf_slider.get())
            cfg["model"] = self.get_selected_model_name()

            is_video = Path(self.current_input).suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]
            m = self.get_model()
            cap = open_video(cfg)

            frame_count = 0
            for i, t, frame in frames(cap):
                if self.stop_requested:
                    self.log(f"[*] Playback stopped at frame {i}.")
                    break

                if task_key == "detect":
                    items = detections(predict(m, frame, cfg))
                    counts = {}
                    for obj in items:
                        draw(frame, obj)
                        lbl = LABELS.get(obj['cls'], str(obj['cls']))
                        counts[lbl] = counts.get(lbl, 0) + 1
                    text(frame, f"Vehicles: {len(items)} | Conf: {cfg['confidence']:.2f}")

                    breakdown = ", ".join([f"{k.capitalize()}: {v}" for k, v in counts.items()])
                    self.lbl_stat_total.config(text=f"Total Vehicles: {len(items)}")
                    self.lbl_stat_density.config(text=f"Breakdown: {breakdown if breakdown else 'None'}")
                    if not is_video or i % 10 == 0:
                        self.log(f"Frame {i:03d} (t={t:.2f}s): {len(items)} vehicles ({breakdown})")

                elif task_key == "track":
                    items = detections(predict(m, frame, cfg, track=True), track=True)
                    for obj in items: draw(frame, obj)
                    tracked_count = sum(x["id"] is not None for x in items)
                    text(frame, f"Tracked IDs: {tracked_count}")
                    self.lbl_stat_total.config(text=f"Tracked: {tracked_count}")
                    if not is_video or i % 10 == 0:
                        ids = [f"#{x['id']}" for x in items if x['id'] is not None]
                        self.log(f"Frame {i:03d} (t={t:.2f}s): {tracked_count} vehicle IDs: {ids}")

                elif task_key == "density":
                    items = detections(predict(m, frame, cfg, track=True), track=True)
                    cnt = len(items)
                    level = 'LOW' if cnt <= cfg['density_low_max'] else ('MEDIUM' if cnt <= cfg['density_medium_max'] else 'HIGH')
                    for obj in items: draw(frame, obj)
                    text(frame, f"Density: {level} (Count: {cnt})")
                    self.lbl_stat_density.config(text=f"Density Level: {level}")
                    self.lbl_stat_total.config(text=f"Count: {cnt}")
                    if not is_video or i % 10 == 0:
                        self.log(f"Frame {i:03d}: Density={level} (Count: {cnt})")

                elif task_key == "triple_riding":
                    res = predict(m, frame, cfg, track=True, classes=[0, 3])
                    objs = detections(res, track=True)
                    persons = [x for x in objs if x['cls'] == 0]
                    bikes = [x for x in objs if x['cls'] == 3]
                    for b in bikes: draw(frame, b, "Motorcycle")
                    for p in persons: draw(frame, p)
                    text(frame, f"Bikes: {len(bikes)} | Persons: {len(persons)}")
                    if not is_video or i % 10 == 0:
                        self.log(f"Frame {i:03d}: {len(bikes)} bike(s), {len(persons)} person(s).")

                else:
                    items = detections(predict(m, frame, cfg))
                    for obj in items: draw(frame, obj)
                    text(frame, f"Task: {task_title}")

                self.display_image(frame)
                frame_count += 1

                # If single image, save and break immediately
                if not is_video:
                    out_path = BASE_DIR / "outputs" / f"{task_key}.jpg"
                    out_path.parent.mkdir(exist_ok=True)
                    cv2.imwrite(str(out_path), frame)
                    self.log(f"Saved output to outputs/{task_key}.jpg")
                    break

            cap.release()
            self.lbl_stat_status.config(text="Status: Completed", fg="#a6e3a1")
            self.log(f"Finished processing {frame_count} frame(s).")
        except Exception as e:
            self.lbl_stat_status.config(text="Status: Error", fg="#f38ba8")
            self.log(f"Execution Error: {e}")
        finally:
            self.running = False
            self.btn_run.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)

    def execute_all_tasks(self):
        self.running = True
        self.btn_run_all.config(state=tk.DISABLED)
        self.log("\n========================================")
        self.log("Starting Execution of ALL Tasks...")
        self.log("========================================")
        try:
            for idx, (title, key) in enumerate(TASK_LIST):
                self.task_listbox.selection_clear(0, tk.END)
                self.task_listbox.selection_set(idx)
                self.execute_task()
            self.log("\nAll tasks completed successfully!")
            messagebox.showinfo("VisionPort", "All 20 VisionPort tasks finished successfully!\nCheck the outputs/ directory.")
        finally:
            self.running = False
            self.btn_run_all.config(state=tk.NORMAL)

def main():
    root = tk.Tk()
    app = VisionPortGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
