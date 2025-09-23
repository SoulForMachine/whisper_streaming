import sys
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, filedialog


def list_unique_input_devices(preferred_apis=None):
    """
    Return a dict of unique input devices, picking the best API
    if duplicates exist.
    """
    if preferred_apis is None:
        if sys.platform.startswith("win"):
            preferred_apis = ["Windows WASAPI", "Windows DirectSound", "MME"]
        elif sys.platform == "darwin":
            preferred_apis = ["Core Audio"]
        else:  # Linux
            preferred_apis = ["ALSA", "PulseAudio", "JACK"]

    devices = sd.query_devices()
    apis = sd.query_hostapis()

    unique = {}
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            api_name = apis[dev['hostapi']]['name']
            key = dev['name']
            entry = (i, api_name, dev)

            if key not in unique:
                unique[key] = entry
            else:
                # Prefer device with "better" API
                existing = unique[key]
                try:
                    if preferred_apis.index(api_name) < preferred_apis.index(existing[1]):
                        unique[key] = entry
                except ValueError:
                    # if API not in preferred list, keep existing
                    pass

    return unique

def default_input_device_index():
    return sd.default.device[0]

def str_to_float(s):
    try:
        f = float(s)
        return f, True  # conversion successful
    except ValueError:
        return None, False  # invalid float


class ZoomCaptionerUI:
    def __init__(self):
        root = tk.Tk()
        self.root = root
        self.root.title("Zoom Captioner")
        self.root.resizable(False, False)  # make window non-resizable
        self.row_i = 0  # to keep track of grid row index

        # --- Zoom URL ---
        row_idx = self.next_row()
        tk.Label(root, text="Zoom URL").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.zoom_url_var = tk.StringVar(value="")
        self.zoom_url_entry = ttk.Entry(root, textvariable=self.zoom_url_var)
        self.zoom_url_entry.grid(row=row_idx, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        self.clear_btn = ttk.Button(root, text="Clear", command=lambda: self.zoom_url_var.set(""))
        self.clear_btn.grid(row=row_idx, column=3, sticky="ew", padx=5, pady=5)

        # --- Audio device ---
        row_idx = self.next_row()
        self.device_map = list_unique_input_devices()
        device_list = list(self.device_map.keys())
        default_device_index = default_input_device_index()

        tk.Label(root, text="Audio device").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.audio_device_combo = ttk.Combobox(root, values=device_list, state="readonly")
        self.audio_device_combo.grid(row=row_idx, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        self.audio_device_combo.current(default_device_index if default_device_index < len(device_list) else 0)

        # --- Whisper model ---
        row_idx = self.next_row()
        tk.Label(root, text="Whisper model").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.model_var = tk.StringVar(value="small.en")
        model_options = [
            "tiny.en", "tiny", "base.en", "base", "small.en", "small",
            "medium.en", "medium",
            "large-v1", "large-v2", "large-v3", "large", "large-v3-turbo"
        ]
        self.model_combo = ttk.Combobox(root, textvariable=self.model_var, values=model_options, state="readonly")
        self.model_combo.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)

        # --- Whisper device ---
        row_idx = self.next_row()
        tk.Label(root, text="Whisper device").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.whisper_device_var = tk.StringVar(value="cuda")
        self.whisper_device_combo = ttk.Combobox(root, textvariable=self.whisper_device_var, values=["cuda", "cpu"], state="readonly")
        self.whisper_device_combo.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)

        # --- Whisper compute type ---
        row_idx = self.next_row()
        tk.Label(root, text="Whisper compute type").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.whisper_compute_type_var = tk.StringVar(value="int8")
        self.whisper_compute_type_combo = ttk.Combobox(root, textvariable=self.whisper_compute_type_var, values=["int8", "int8_float16", "float16", "float32"], state="readonly")
        self.whisper_compute_type_combo.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)

        # --- Speech language ---
        row_idx = self.next_row()
        tk.Label(root, text="Speech language").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.lang_var = tk.StringVar(value="English")
        self.lang_combo = ttk.Combobox(root, textvariable=self.lang_var, values=["English", "Serbian"], state="readonly")
        self.lang_combo.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)

        # --- Target language ---
        row_idx = self.next_row()
        tk.Label(root, text="Target language").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.target_lang_var = tk.StringVar(value="Serbian")
        self.target_lang_combo = ttk.Combobox(root, textvariable=self.target_lang_var, values=["Serbian", "English"], state="readonly")
        self.target_lang_combo.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)
        self.enable_translation_var = tk.BooleanVar(value=True)
        self.enable_translation_check = ttk.Checkbutton(root, text="Enable translation", variable=self.enable_translation_var, command=self.on_enable_translation_toggle)
        self.enable_translation_check.grid(row=row_idx, column=2, sticky="w", padx=5, pady=5)

        # --- Warmup file ---
        row_idx = self.next_row()
        tk.Label(root, text="Warmup file").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.warmup_var = tk.StringVar(value="samples_jfk.wav")
        self.warmup_entry = ttk.Entry(root, textvariable=self.warmup_var)
        self.warmup_entry.grid(row=row_idx, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        self.warmup_btn = ttk.Button(root, text="Browse...", command=self.browse_file)
        self.warmup_btn.grid(row=row_idx, column=3, sticky="ew", padx=5, pady=5)

        # --- Non-speech probability threshold ---
        row_idx = self.next_row()
        tk.Label(root, text="Non-speech probability\nthreshold", justify="left", anchor="w").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.threshold_var = tk.StringVar(value="1.0")
        self.threshold_entry = ttk.Entry(root, textvariable=self.threshold_var)
        self.threshold_entry.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)

        # --- Minimum chunk size ---
        row_idx = self.next_row()
        tk.Label(root, text="Minimum chunk size (sec)", justify="left", anchor="w").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.min_chunk_size_var = tk.StringVar(value="0.5")
        self.min_chunk_size_entry = ttk.Entry(root, textvariable=self.min_chunk_size_var)
        self.min_chunk_size_entry.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)

        # --- Checkboxes ---
        row_idx = self.next_row()
        self.vac_var = tk.BooleanVar(value=True)
        self.vad_var = tk.BooleanVar(value=True)
        self.vac_check = ttk.Checkbutton(root, text="Voice activity controller", variable=self.vac_var)
        self.vad_check = ttk.Checkbutton(root, text="Voice activity detection", variable=self.vad_var)
        self.vac_check.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        row_idx = self.next_row()
        self.vad_check.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)

        # --- Buttons ---
        row_idx = self.next_row()
        self.start_btn = ttk.Button(root, text="Start", command=self.start)
        self.quit_btn = ttk.Button(root, text="Quit", command=root.quit)
        self.start_btn.grid(row=row_idx, column=2, sticky="e", padx=5, pady=10)
        self.quit_btn.grid(row=row_idx, column=3, sticky="w", padx=5, pady=10)

        # make second column expand
        root.columnconfigure(1, weight=1)

    def browse_file(self):
        filename = filedialog.askopenfilename(title="Select warmup file")
        if filename:
            self.warmup_var.set(filename)

    def start(self):
        # Collect all values for debugging/demo purposes
        print("Zoom URL:", self.zoom_url_var.get())
        print("Audio device:", self.audio_device_combo.get())
        print("Whisper model:", self.model_var.get())
        print("Whisper device:", self.whisper_device_var.get())
        print("Whisper compute type:", self.whisper_compute_type_var.get())
        print("Language:", self.lang_var.get())
        print("Enable translation:", self.enable_translation_var.get())
        print("Target language:", self.target_lang_var.get())
        print("Warmup file:", self.warmup_var.get())
        print("Threshold:", self.threshold_var.get())
        print("Minimum chunk size:", self.min_chunk_size_var.get())
        print("VAC enabled:", self.vac_var.get())
        print("VAD enabled:", self.vad_var.get())

        print("Running Zoom captioner...")
        self.run_zoom_captioner()

    def on_enable_translation_toggle(self):
        if self.enable_translation_var.get():
            self.target_lang_combo.config(state="readonly")
        else:
            self.target_lang_combo.config(state="disabled")

    # Helper to manage grid row indices
    def next_row(self):
        r = self.row_i
        self.row_i += 1
        return r

    def get_selected_device_index(self):
        sel_name = self.audio_device_combo.get()
        # Find the index in input_devices
        dev_index = None
        for i, name in enumerate(self.device_map.keys()):
            if name == sel_name:
                dev_index = i
                break
        return dev_index
    
    def run_zoom_captioner(self):
        import zoom_captioner

        dev_index = self.get_selected_device_index()

        threshold, valid = str_to_float(self.threshold_var.get())
        if not valid or not (0.0 <= threshold <= 1.0):
            print("Error: Non-speech probability threshold must be a number between 0.0 and 1.0. Setting to default: 1.0")
            threshold = 1.0

        min_chunk_size, valid = str_to_float(self.min_chunk_size_var.get())
        if not valid:
            print("Error: Invalid minimum chunk size. Setting to default: 0.5")
            min_chunk_size = 0.5

        zoom_url = self.zoom_url_var.get().strip()

        args = [
            "--audio-input-device", str(dev_index) if dev_index is not None else "1",
            "--model", self.model_var.get(),
            "--whisper-device", self.whisper_device_var.get(),
            "--whisper-compute-type", self.whisper_compute_type_var.get(),
            "--language", self.lang_var.get() == "English" and "en" or "sr",
            #"--enable_translation", str(self.enable_translation_var.get()),
            #"--target_language", self.target_lang_var.get(),
            "--warmup-file", self.warmup_var.get(),
            "--nsp-threshold", str(threshold),
            "--min-chunk-size", str(min_chunk_size),
            "-l", "INFO"
        ]

        if self.vac_var.get():
            args.append("--vac")
        if self.vad_var.get():
            args.append("--vad")

        if zoom_url:
            args.append("--zoom-url")
            args.append(zoom_url)

        return zoom_captioner.main(args)

    def run_gui(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = ZoomCaptionerUI()
    app.run_gui()
