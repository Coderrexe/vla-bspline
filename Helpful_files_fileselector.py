#this is a file that can help you select a folder of images and play them as a sequence, with options to mark a range and extract it to another folder. You will need to install the Pillow library for image handling.
import os
import re
import shutil
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
 
from PIL import Image, ImageTk
 
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
 
 
def natural_key(s):
    """Sort strings containing numbers in human ('natural') order,
    so frame2.png comes before frame10.png."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", s)]
 
 
class ImageSequencePlayer(tk.Tk):
    def __init__(self, initial_folder=None):
        super().__init__()
        self.title("Image Sequence Player")
        self.geometry("1000x750")
        self.minsize(700, 550)
 
        self.folder = None
        self.files = []
        self.index = 0
        self.playing = False
        self.after_id = None
        self.fps = tk.DoubleVar(value=8.0)
        self.mark_start = None
        self.mark_end = None
        self.photo = None
 
        self._build_ui()
 
        if initial_folder:
            self.load_folder(initial_folder)
 
    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(side="top", fill="x", padx=8, pady=6)
 
        ttk.Button(top, text="Open Folder…", command=self.choose_folder).pack(side="left")
        self.folder_label = ttk.Label(top, text="No folder loaded", foreground="#555")
        self.folder_label.pack(side="left", padx=10)
 
        ttk.Label(top, text="FPS:").pack(side="left", padx=(20, 2))
        ttk.Spinbox(top, from_=0.5, to=60, increment=0.5,
                    textvariable=self.fps, width=6).pack(side="left")
 
        self.canvas = tk.Canvas(self, bg="black")
        self.canvas.pack(side="top", fill="both", expand=True, padx=8, pady=4)
        self.canvas.bind("<Configure>", lambda e: self._render_current())
 
        info = ttk.Frame(self)
        info.pack(side="top", fill="x", padx=8)
        self.filename_label = ttk.Label(info, text="", font=("TkDefaultFont", 11, "bold"))
        self.filename_label.pack(side="left")
        self.index_label = ttk.Label(info, text="")
        self.index_label.pack(side="right")
 
        self.slider = ttk.Scale(self, from_=0, to=0, orient="horizontal",
                                 command=self._on_slider_move)
        self.slider.pack(side="top", fill="x", padx=8, pady=4)
 
        transport = ttk.Frame(self)
        transport.pack(side="top", pady=4)
        ttk.Button(transport, text="⏮ Prev", command=self.prev_frame).grid(row=0, column=0, padx=3)
        self.play_btn = ttk.Button(transport, text="▶ Play", command=self.toggle_play)
        self.play_btn.grid(row=0, column=1, padx=3)
        ttk.Button(transport, text="Next ⏭", command=self.next_frame).grid(row=0, column=2, padx=3)
 
        marks = ttk.Frame(self)
        marks.pack(side="top", pady=6)
        ttk.Button(marks, text="Mark Start Here [S]", command=self.set_mark_start).grid(row=0, column=0, padx=4)
        ttk.Button(marks, text="Mark End Here [E]", command=self.set_mark_end).grid(row=0, column=1, padx=4)
        ttk.Button(marks, text="Clear Marks", command=self.clear_marks).grid(row=0, column=2, padx=4)
        ttk.Button(marks, text="Extract Range…", command=self.extract_range).grid(row=0, column=3, padx=12)
 
        self.marks_label = ttk.Label(self, text="Start: — End: — Range: —", foreground="#333")
        self.marks_label.pack(side="top", pady=(0, 6))
 
        self.bind("<space>", lambda e: self.toggle_play())
        self.bind("<Left>", lambda e: self.prev_frame())
        self.bind("<Right>", lambda e: self.next_frame())
        self.bind("s", lambda e: self.set_mark_start())
        self.bind("e", lambda e: self.set_mark_end())
 
    def choose_folder(self):
        folder = filedialog.askdirectory(title="Choose folder of images")
        if folder:
            self.load_folder(folder)
 
    def load_folder(self, folder):
        files = [
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        ]
        files.sort(key=lambda p: natural_key(os.path.basename(p)))
 
        if not files:
            messagebox.showwarning("No images", "No supported image files found in that folder.")
            return
 
        self.stop_playing()
        self.folder = folder
        self.files = files
        self.index = 0
        self.mark_start = None
        self.mark_end = None
 
        self.folder_label.config(text=f"{folder}  ({len(files)} images)")
        self.slider.config(from_=0, to=len(files) - 1)
        self.slider.set(0)
        self._update_marks_label()
        self._render_current()
 
    def _render_current(self):
        if not self.files:
            return
        path = self.files[self.index]
        try:
            img = Image.open(path)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
        except Exception as exc:
            self.filename_label.config(text=f"[Error opening {os.path.basename(path)}: {exc}]")
            return
 
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        iw, ih = img.size
        scale = min(cw / iw, ch / ih) if iw and ih else 1.0
        scale = min(scale, 1.0) if scale > 0 else 1.0
        new_size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
        img = img.resize(new_size, Image.LANCZOS)
 
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.photo, anchor="center")
 
        self.filename_label.config(text=os.path.basename(path))
        self.index_label.config(text=f"{self.index + 1} / {len(self.files)}")
        if abs(self.slider.get() - self.index) >= 1:
            self.slider.set(self.index)
 
    def toggle_play(self):
        if not self.files:
            return
        self.playing = not self.playing
        self.play_btn.config(text="⏸ Pause" if self.playing else "▶ Play")
        if self.playing:
            self._play_step()
        else:
            self.stop_playing(keep_state=True)
 
    def stop_playing(self, keep_state=False):
        if self.after_id:
            self.after_cancel(self.after_id)
            self.after_id = None
        if not keep_state:
            self.playing = False
            if hasattr(self, "play_btn"):
                self.play_btn.config(text="▶ Play")
 
    def _play_step(self):
        if not self.playing:
            return
        self.next_frame(loop=True)
        delay = max(10, int(1000 / max(self.fps.get(), 0.1)))
        self.after_id = self.after(delay, self._play_step)
 
    def next_frame(self, loop=False):
        if not self.files:
            return
        if self.index < len(self.files) - 1:
            self.index += 1
        elif loop:
            self.index = 0  # wrap around during playback
        self._render_current()
 
    def prev_frame(self):
        if not self.files:
            return
        self.stop_playing()
        self.index = max(0, self.index - 1)
        self._render_current()
 
    def _on_slider_move(self, value):
        if not self.files:
            return
        idx = int(float(value))
        if idx != self.index:
            self.stop_playing()
            self.index = idx
            self._render_current()
 
    def set_mark_start(self):
        if not self.files:
            return
        self.mark_start = self.index
        self._update_marks_label()
 
    def set_mark_end(self):
        if not self.files:
            return
        self.mark_end = self.index
        self._update_marks_label()
 
    def clear_marks(self):
        self.mark_start = None
        self.mark_end = None
        self._update_marks_label()
 
    def _update_marks_label(self):
        s, e = self.mark_start, self.mark_end
        if s is None and e is None:
            text = "Start: — End: — Range: —"
        else:
            lo, hi = self._ordered_marks()
            count = (hi - lo + 1) if (lo is not None and hi is not None) else "—"
            s_name = os.path.basename(self.files[s]) if s is not None else "—"
            e_name = os.path.basename(self.files[e]) if e is not None else "—"
            text = f"Start: {s_name}   End: {e_name}   Range: {count} images"
        self.marks_label.config(text=text)
 
    def _ordered_marks(self):
        if self.mark_start is None or self.mark_end is None:
            return None, None
        return (self.mark_start, self.mark_end) if self.mark_start <= self.mark_end \
            else (self.mark_end, self.mark_start)
 
    def extract_range(self):
        if not self.files:
            return
        lo, hi = self._ordered_marks()
        if lo is None:
            messagebox.showinfo("No range marked",
                                 "Mark a start and end frame first (buttons or S / E keys).")
            return
 
        dest = filedialog.askdirectory(title="Choose destination folder for extracted images")
        if not dest:
            return
 
        subset = self.files[lo:hi + 1]
        out_dir = os.path.join(dest, f"extracted_{lo:04d}_{hi:04d}")
        os.makedirs(out_dir, exist_ok=True)
 
        for path in subset:
            shutil.copy2(path, os.path.join(out_dir, os.path.basename(path)))
 
        messagebox.showinfo("Done", f"Copied {len(subset)} images to:\n{out_dir}")
 
 
def main():
    initial_folder = sys.argv[1] if len(sys.argv) > 1 else None
    app = ImageSequencePlayer(initial_folder)
    app.mainloop()
 
 
if __name__ == "__main__":
    main()
 
