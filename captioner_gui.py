import sys
import socket
import queue
import threading
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, filedialog
import logging
import captioner_common as ccmn

logger = logging.getLogger(__name__)

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

def str_to_int(s):
    try:
        i = int(s)
        return i, True  # conversion successful
    except ValueError:
        return None, False  # invalid int

class CaptionerUI:
    def __init__(self):
        self.setup_ui()
        self.is_running = False
        self.audio_thread = None
        self.audio_listener = None
        self.audio_queue = None
        self.whisper_client = None
        self.whisper_client_thread = None
        self.captions_overlay = None
        self.captions_thread = None

    def setup_ui(self):
        root = tk.Tk()
        self.root_wnd = root
        self.root_wnd.title("Zoom Captioner")
        self.root_wnd.resizable(False, False)  # make window non-resizable
        self.row_i = 0  # to keep track of grid row index

        # --- Whisper Server URL ---
        row_idx = self.next_row()
        tk.Label(root, text="Whisper Server URL").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.server_url_var = tk.StringVar(value="localhost")
        self.server_url_entry = ttk.Entry(root, textvariable=self.server_url_var)
        self.server_url_entry.grid(row=row_idx, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        self.server_port_var = tk.StringVar(value="5000")
        self.server_port_entry = ttk.Entry(root, textvariable=self.server_port_var, width=6)
        self.server_port_entry.grid(row=row_idx, column=3, sticky="ew", padx=5, pady=5)

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
        self.model_var = tk.StringVar(value="tiny.en")
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
        self.whisper_compute_type_var = tk.StringVar(value="float32")
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
        self.threshold_var = tk.StringVar(value="0.95")
        self.threshold_entry = ttk.Entry(root, textvariable=self.threshold_var)
        self.threshold_entry.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)

        # --- Minimum chunk size ---
        row_idx = self.next_row()
        tk.Label(root, text="Minimum chunk size (sec)", justify="left", anchor="w").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.min_chunk_size_var = tk.StringVar(value="0.6")
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
        if self.is_running:
            self.stop_captioner()
            self.start_btn.config(text="Start")
            self.is_running = False
            print("Captioner stopped.")
        else:
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

            print("Running Captioner...")
            self.run_captioner()
            self.start_btn.config(text="Stop")
            self.is_running = True

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

    def run_captioner(self):
        self.audio_queue = queue.Queue()
        self.resuts_queue = queue.Queue()
        self.connect_to_server()
        self.run_audio_listener()

        if not self.zoom_url_var.get().strip():
            self.run_captions_overlay()

    def connect_to_server(self):
        threshold, valid = str_to_float(self.threshold_var.get())
        if not valid or not (0.0 <= threshold <= 1.0):
            print("Error: Non-speech probability threshold must be a number between 0.0 and 1.0. Setting to default: 1.0")
            threshold = 1.0

        min_chunk_size, valid = str_to_float(self.min_chunk_size_var.get())
        if not valid:
            print("Error: Invalid minimum chunk size. Setting to default: 0.6")
            min_chunk_size = 0.6

        port, valid = str_to_int(self.server_port_var.get())
        if not valid or not (0 < port < 65536):
            print("Error: Invalid port number. Setting to default: 5000")
            port = 5000

        params = {
            "zoom_url": self.zoom_url_var.get().strip(),
            "model": self.model_var.get(),
            "whisper_device": self.whisper_device_var.get(),
            "whisper_compute_type": self.whisper_compute_type_var.get(),
            "language": "en" if self.lang_var.get() == "English" else "sr",
            "enable_translation": bool(self.enable_translation_var.get()),
            "target_language": self.target_lang_var.get(),
            "warmup_file": self.warmup_var.get(),
            "nsp_threshold": threshold,
            "min_chunk_size": min_chunk_size,
            "log_level": "INFO",
            "vac": self.vac_var.get(),
            "vad": self.vad_var.get(),
        }

        self.whisper_client = WhisperClient(self.server_url_var.get().strip(), port, params, self.audio_queue, self.resuts_queue)
        self.whisper_client_thread = threading.Thread(target=self.whisper_client.run, daemon=True)
        self.whisper_client_thread.start()

    def run_captions_overlay(self):
        self.captions_overlay = CaptionsOverlay(self.root_wnd, self.resuts_queue)
        self.captions_thread = threading.Thread(target=self.captions_overlay.run, daemon=True)
        self.captions_thread.start()

    def run_audio_listener(self):
        dev_index = self.get_selected_device_index()

        if dev_index is not None:
            audio_input_device = dev_index 
            audio_input_device_name = self.audio_device_combo.get()
        else:
            audio_input_device = self.default_input_device_index()
            audio_input_device_name = self.device_map.keys()[audio_input_device]

        min_chunk_size, valid = str_to_float(self.min_chunk_size_var.get())
        if not valid:
            print("Error: Invalid minimum chunk size. Setting to default: 0.6")
            min_chunk_size = 0.6

        self.audio_listener = AudioListener(min_chunk_size, audio_input_device, audio_input_device_name, self.audio_queue)
        self.audio_thread = threading.Thread(target=self.audio_listener.run, daemon=True)
        self.audio_thread.start()

    def stop_captioner(self):
        if self.audio_thread:
            self.audio_listener.stop()
            self.audio_thread.join()
            self.audio_thread = None
            self.audio_listener = None

        if self.whisper_client:
            self.whisper_client.stop()
            self.whisper_client_thread.join()
            self.whisper_client = None
            self.whisper_client_thread = None

        if self.captions_overlay:
            self.captions_overlay.stop()
            self.captions_thread.join()
            self.captions_overlay = None
            self.captions_thread = None

        self.audio_queue = None
        self.resuts_queue = None

    def run_gui(self):
        self.root_wnd.mainloop()


class AudioListener:
    def __init__(self, min_chunk_size: float, audio_input_device: int, audio_input_device_name: str, result_queue: queue.Queue):
        self.min_chunk_size = min_chunk_size
        self.audio_input_device = audio_input_device
        self.audio_input_device_name = audio_input_device_name
        self.last_end = None
        self.is_first = True

        self.audio_queue = queue.Queue()
        self.result_queue = result_queue

        self.stop_event = threading.Event()

        # setting whisper object by args 
        self.SAMPLING_RATE = 16000

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio callback status: {status}")
        pcm16 = (indata * 32767).astype(np.int16)
        audio_float = pcm16.astype(np.float32) / 32767.0
        self.audio_queue.put(audio_float.copy())

    def receive_audio_chunk(self):
        out = []
        minlimit = int(self.min_chunk_size * self.SAMPLING_RATE)
        while sum(len(x) for x in out) < minlimit:
            try:
                # Get a chunk from audio queue. Timeout is slightly longer than minimum chunk duration.
                chunk = self.audio_queue.get(timeout=self.min_chunk_size * 1.1)
            except queue.Empty:
                break
            out.append(chunk)

        if not out:
            return None
        conc = np.concatenate(out)
        if self.is_first and len(conc) < minlimit:
            return None
        self.is_first = False
        return conc

    def format_output_transcript(self, o):
        if o[0] is not None:
            beg, end = o[0]*1000, o[1]*1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)
            self.last_end = end
            print("%1.0f %1.0f %s" % (beg,end,o[2]), flush=True)
        else:
            logger.debug("No text in this segment")

    def run(self):
        with sd.InputStream(
            samplerate=self.SAMPLING_RATE,
            channels=1,
            dtype="float32",
            blocksize=int(self.SAMPLING_RATE * 0.5),
            callback=self.audio_callback,
            device=self.audio_input_device
        ):
            print(f"Listening to {self.audio_input_device_name}.", flush=True)
            while not self.stop_event.is_set():
                chunk = self.receive_audio_chunk()
                if chunk is None or len(chunk) == 0:
                    continue

                self.result_queue.put(chunk)

    def stop(self):
        self.stop_event.set()


class WhisperClient:
    def __init__(self, server_url: str, port: int, params: map, audio_queue: queue.Queue, results_queue: queue.Queue = None):
        self.server_url = server_url
        self.port = port
        self.params = params
        self.audio_queue = audio_queue
        self.results_queue = results_queue
        self.connected_event = threading.Event()
        self.results_thread = None

    def run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.server_url, self.port))
        except Exception as e:
            print(f"Could not connect to the whisper server: {e}")
            return

        # Step 1: send params JSON
        ccmn.send_json(sock, self.params)

        # Step 2: start a thread to receive results
        self.results_thread = threading.Thread(target=self.listen_for_results, args=(sock,), daemon=True)
        self.results_thread.start()

        # Step 3: stream audio
        try:
            while True:
                chunk = self.audio_queue.get()
                if chunk is None:
                    # Tell the server we're done
                    ccmn.send_ndarray(sock, np.array([], dtype=np.float32))
                    self.results_thread.join()
                    self.results_thread = None
                    break
                ccmn.send_ndarray(sock, chunk)
        except (BrokenPipeError, ConnectionResetError) as e:
            print(f"Server connection closed while streaming audio: {e}")

        sock.close()

    def listen_for_results(self, sock):
        # Receive status and translation messages from the server.
        while True:
            try:
                msg = ccmn.recv_json(sock)
            except (ConnectionResetError, OSError) as e:
                print(f"Connection lost: {e}.")
                break

            if msg is None:
                continue

            if msg["type"] == "status":
                if msg["value"] == "shutdown":
                    break
            elif msg["type"] == "translation":
                #print(f"[{msg['lang']}]{' (complete)' if msg['complete'] else ''}: {msg['text']}")
                if self.results_queue:
                    self.results_queue.put((msg['text'], msg['complete']))
            else:
                print("Unknown message: ", msg)

    def stop(self):
        self.audio_queue.put(None)


class CaptionsOverlay:
    font_size = 18

    def __init__(self, root_wnd, source_queue):
        self.source_queue = source_queue
        # Tkinter window for captions
        self.overlay_wnd = tk.Toplevel(root_wnd)
        self.overlay_wnd.title("Live Captions")
        self.overlay_wnd.configure(bg="black")
        self.overlay_wnd.attributes("-topmost", True)
        self.overlay_wnd.attributes("-alpha", 0.8)
        self.overlay_wnd.overrideredirect(True)   # remove title bar and border

        self.label = tk.Label(self.overlay_wnd, text="", fg="white", bg="black", font=("Arial", self.font_size), wraplength=800, anchor="w", justify="left")
        self.label.pack(padx=20, pady=20)

        self.button_frame = tk.Frame(self.overlay_wnd)
        self.button_frame.pack(side="bottom", padx=0, pady=0)

        # Three buttons centered
        font_inc = tk.Button(self.button_frame, text="⇧", fg="white", bg="black", command=self.increase_font_size)
        font_dec = tk.Button(self.button_frame, text="⇩", fg="white", bg="black", command=self.decrease_font_size)
        font_inc.pack(side="left", padx=0)
        font_dec.pack(side="left", padx=0)

        self.sentences = []
        self.incomplete_sentence = ""
        self.displ_text = ""

        self.overlay_wnd.bind("<Button-1>", self.start_move)
        self.overlay_wnd.bind("<B1-Motion>", self.do_move)
        self.start_x = 0
        self.start_y = 0

        btn = tk.Button(self.button_frame, text="X", fg="white", bg="black", command=self.overlay_wnd.destroy)
        #btn.place(relx=1.0, y=0, anchor="s") # anchor to top-right corner
        btn.pack(side="left", padx=0)

        # get window and screen sizes
        self.overlay_wnd.update_idletasks()
        win_w = self.overlay_wnd.winfo_width()
        win_h = self.overlay_wnd.winfo_height()
        scr_w = self.overlay_wnd.winfo_screenwidth()
        scr_h = self.overlay_wnd.winfo_screenheight()

        # distance from bottom (pixels)
        margin_bottom = 100  

        # calculate position
        x = (scr_w // 2) - (win_w // 2)    # center horizontally
        y = scr_h - win_h - margin_bottom  # margin from bottom

        self.overlay_wnd.geometry(f"+{x}+{y}")

    def start_move(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def do_move(self, event):
        x = self.overlay_wnd.winfo_x() + (event.x - self.start_x)
        y = self.overlay_wnd.winfo_y() + (event.y - self.start_y)
        self.overlay_wnd.geometry(f"+{x}+{y}")

    def run(self):
        while True:
            text, complete = self.source_queue.get()
            #print(f"[DEBUG] raw from queue: {text!r}, complete={complete}", flush=True)
            if text is None:
                break
            elif not text.strip():
                continue

            #print(f"Overlay Caption: {text}", flush=True)

            if complete:
                self.sentences.append(text) # add the complete sentence
                self.incomplete_sentence = ""
            else:
                self.incomplete_sentence = text  # store the incomplete sentence

            self.sentences = self.sentences[-3:]  # keep only last 3 sentences
            self.displ_text = "\n".join(self.sentences)
            if self.incomplete_sentence:
                if self.displ_text:
                    self.displ_text += "\n"
                self.displ_text += self.incomplete_sentence

            if self.displ_text:
                self.overlay_wnd.after(0, lambda: self.update_label(text=self.displ_text))

    def stop(self):
        self.source_queue.put((None, True))  # signal to stop
        self.overlay_wnd.destroy()

    def get_anchor_pt(self):
        # get current bottom center point
        x = self.overlay_wnd.winfo_x()
        y = self.overlay_wnd.winfo_y()
        win_w = self.overlay_wnd.winfo_width()
        win_h = self.overlay_wnd.winfo_height()
        bc_pt_x = x + win_w // 2
        bc_pt_y = y + win_h
        return (bc_pt_x, bc_pt_y)
    
    def anchor_wnd_to_pt(self, pt_x, pt_y):
        # keep bottom center point fixed
        new_win_w = self.overlay_wnd.winfo_width()
        new_win_h = self.overlay_wnd.winfo_height()
        new_x = pt_x - new_win_w // 2
        new_y = pt_y - new_win_h
        self.overlay_wnd.geometry(f"+{new_x}+{new_y}")

    def update_label(self, text):
        bc_pt_x, bc_pt_y = self.get_anchor_pt()

        self.label.config(text=text)
        self.overlay_wnd.update_idletasks()

        self.anchor_wnd_to_pt(bc_pt_x, bc_pt_y)

    def increase_font_size(self):
        if self.font_size < 42:
            bc_pt_x, bc_pt_y = self.get_anchor_pt()

            self.font_size += 2
            self.label.config(font=("Arial", self.font_size))
            self.overlay_wnd.update_idletasks()

            self.anchor_wnd_to_pt(bc_pt_x, bc_pt_y)

    def decrease_font_size(self):
        if self.font_size > 12:
            bc_pt_x, bc_pt_y = self.get_anchor_pt()

            self.font_size -= 2
            self.label.config(font=("Arial", self.font_size))
            self.overlay_wnd.update_idletasks()

            self.anchor_wnd_to_pt(bc_pt_x, bc_pt_y)


if __name__ == "__main__":
    app = CaptionerUI()
    app.run_gui()
