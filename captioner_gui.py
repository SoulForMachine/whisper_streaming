import sys
import socket
import queue
import threading
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk
import logging
import captioner_common as ccmn
from collections import defaultdict

logger = logging.getLogger(__name__)

class InputDeviceCaps:
    def __init__(self, channels: int, samplerate: float, min_latency: float, max_latency: float, index: int):
        self.channels = channels
        self.samplerate = samplerate
        self.min_latency = min_latency
        self.max_latency = max_latency
        self.index = index

    def __repr__(self):
        return f"InputDeviceCaps(channels={self.channels}, samplerate={self.samplerate}, index={self.index})"

class InputDeviceInfo:
    def __init__(self, name: str, api: str, latency: float, caps: InputDeviceCaps):
        self.name = name
        self.api = api
        self.channels = caps.channels
        self.samplerate = caps.samplerate
        self.min_latency = caps.min_latency
        self.max_latency = caps.max_latency
        self.latency = latency
        self.index = caps.index

def sort_api_by_preference(api_list):
    if sys.platform.startswith("win"):
        preferred_apis = ["MME", "Windows DirectSound", "Windows WDM-KS", "Windows WASAPI"]
    elif sys.platform == "darwin":
        preferred_apis = ["Core Audio"]
    else:  # Linux
        preferred_apis = ["ALSA", "PulseAudio", "JACK"]

    order = {item: i for i, item in enumerate(preferred_apis)}

    # Use a tuple as key: (index if in preferred_apis, else very large number to push to end)
    return sorted(api_list, key=lambda x: order.get(x, float('inf')))


def list_unique_input_devices():
    # Outer dict: device_name -> hostapi_name -> InputDeviceCaps
    devices_dict = defaultdict(dict)

    all_devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    input_devices = []
    input_devices_mme = []

    for dev in all_devices:
        if dev['max_input_channels'] > 0:
            hostapi_name = hostapis[dev['hostapi']]['name']
            if hostapi_name == "MME":
                input_devices_mme.append(dev)
            else:
                input_devices.append(dev)

    # Convert to DeviceList
    input_devices = sd.DeviceList(input_devices)
    input_devices_mme = sd.DeviceList(input_devices_mme)

    for dev in input_devices:
        hostapi_name = hostapis[dev['hostapi']]['name']

        devices_dict[dev['name']][hostapi_name] = InputDeviceCaps(
            channels=dev['max_input_channels'],
            samplerate=dev['default_samplerate'],
            min_latency=dev['default_low_input_latency'],
            max_latency=dev['default_high_input_latency'],
            index=dev['index']
        )

    for dev in input_devices_mme:
        # If we already saved a device whose name starts with this devices name,
        # we assume that it is the same device - only with a cut-off name - and we
        # just add device caps for MME API to the saved device's API map.
        merged = False
        for saved_name, saved_api_map in devices_dict.items():
            if saved_name.startswith(dev['name']):
                if not saved_api_map.get("MME"):
                    saved_api_map["MME"] = InputDeviceCaps(
                        channels=dev['max_input_channels'],
                        samplerate=dev['default_samplerate'],
                        min_latency=dev['default_low_input_latency'],
                        max_latency=dev['default_high_input_latency'],
                        index=dev['index']
                    )
                    merged = True
                    break

        if not merged:
            devices_dict[dev['name']]["MME"] = InputDeviceCaps(
                channels=dev['max_input_channels'],
                samplerate=dev['default_samplerate'],
                min_latency=dev['default_low_input_latency'],
                max_latency=dev['default_high_input_latency'],
                index=dev['index']
            )

    return devices_dict


def default_input_device(device_map):
    def_dev_index = sd.default.device[0]
    for name, api_map in device_map.items():
        for api, caps in api_map.items():
            if caps.index == def_dev_index:
                return (name, api)
    return None

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
        self.audio_listener = None
        self.audio_listener_2 = None
        self.audio_mixer = None
        self.audio_temp_queue_1 = None
        self.audio_temp_queue_2 = None
        self.audio_queue = None
        self.resuts_queue = None
        self.whisper_client = None
        self.whisper_client_thread = None
        self.captions_overlay = None

        self.gui_queue = queue.Queue()
        threading.Thread(target=self.update_gui, daemon=True).start()

    def update_gui(self):
        while True:
            try:
                lmbd = self.gui_queue.get()
                if lmbd is not None:
                    lmbd()
            except Exception:
                pass

    def setup_ui(self):
        root = tk.Tk()
        self.root_wnd = root
        self.root_wnd.title("Captioner")
        self.root_wnd.resizable(False, False)  # make window non-resizable
        self.row_i = 0  # to keep track of grid row index

        # --- Whisper Server URL ---
        row_idx = self.next_row()
        tk.Label(root, text="Whisper Server URL").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.server_url_var = tk.StringVar(value="galloway")
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

        # --- Audio device 1 ---
        self.device_map = list_unique_input_devices()
        device_list = list(self.device_map.keys())
        dev_name, api_name = default_input_device(self.device_map)

        row_idx = self.next_row()
        tk.Label(root, text="Audio device 1").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.audio_device_combo_1 = ttk.Combobox(root, values=device_list, state="readonly")
        self.audio_device_combo_1.grid(row=row_idx, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

        row_idx = self.next_row()
        tk.Label(root, text="Host API").grid(row=row_idx, column=0, sticky="e", padx=5, pady=5)
        self.audio_device_host_api_combo_1 = ttk.Combobox(root, values=[], state="readonly")
        self.audio_device_host_api_combo_1.grid(row=row_idx, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        self.audio_device_host_api_combo_1.bind("<<ComboboxSelected>>", self.on_audio_device_host_api_1_selection_change)

        ch_resample_frame = tk.Frame(root)
        ch_resample_frame.grid(row=row_idx, column=3)

        self.audio_device_channels_var_1 = tk.StringVar(value="default ch")
        self.audio_device_channels_combo_1 = ttk.Combobox(ch_resample_frame, values=["default ch", "2ch"], state="readonly", textvariable=self.audio_device_channels_var_1, width=10)
        self.audio_device_channels_combo_1.pack(side="left", padx=(5, 5))

        self.audio_device_resample_var_1 = tk.BooleanVar(value=True)
        self.audio_device_resample_check_1 = ttk.Checkbutton(ch_resample_frame, text="Resample", variable=self.audio_device_resample_var_1)
        self.audio_device_resample_check_1.pack(side="left", padx=(5, 5))

        # Use a Label widget for multiline read-only display
        row_idx = self.next_row()
        tk.Label(root, text="").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.input_dev_info_label_1 = tk.Label(root, text="", bd=1, relief="solid", justify="left", anchor="nw")
        self.input_dev_info_label_1.grid(row=row_idx, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

        # For latency
        latency_frame = tk.Frame(root)
        latency_frame.grid(row=row_idx, column=3, padx=5, pady=5, sticky="ew")
        latency_frame.columnconfigure(0, weight=1)
        latency_frame.rowconfigure(0, weight=1)
        latency_frame.rowconfigure(1, weight=1)

        # Label
        self.audio_device_latency_label_1 = ttk.Label(latency_frame, text="Latency: \nBlock size: ", relief="solid", justify="left", anchor="w")
        self.audio_device_latency_label_1.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        # Slider
        self.audio_device_latency_slider_var_1 = tk.DoubleVar(value=0.5)
        self.audio_device_latency_slider_1 = tk.Scale(latency_frame, from_=0.1, to=1.0, orient="horizontal", resolution=0.01, variable=self.audio_device_latency_slider_var_1, command=self.on_audio_device_1_latency_change)
        self.audio_device_latency_slider_1.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        # Now set the combo selection changed callback and select the default audio device.
        self.audio_device_combo_1.set(dev_name)
        self.audio_device_combo_1.bind("<<ComboboxSelected>>", self.on_audio_device_1_selection_change)
        self.audio_device_combo_1.event_generate("<<ComboboxSelected>>")

        # --- Audio device 2 ---
        row_idx = self.next_row()
        tk.Label(root, text="Audio device 2").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.audio_device_combo_2 = ttk.Combobox(root, values=device_list, state="disabled")
        self.audio_device_combo_2.grid(row=row_idx, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

        # Checkbox for enabling the second device
        self.use_second_audio_dev_var = tk.BooleanVar(value=False)
        self.use_second_audio_dev_check = ttk.Checkbutton(root, text="Use this device", variable=self.use_second_audio_dev_var, command=self.on_enable_second_audio_device)
        self.use_second_audio_dev_check.grid(row=row_idx, column=3, sticky="w", padx=5, pady=5)

        row_idx = self.next_row()
        tk.Label(root, text="Host API").grid(row=row_idx, column=0, sticky="e", padx=5, pady=5)
        self.audio_device_host_api_combo_2 = ttk.Combobox(root, values=[], state="disabled")
        self.audio_device_host_api_combo_2.grid(row=row_idx, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        self.audio_device_host_api_combo_2.bind("<<ComboboxSelected>>", self.on_audio_device_host_api_2_selection_change)

        ch_resample_frame_2 = tk.Frame(root)
        ch_resample_frame_2.grid(row=row_idx, column=3)

        self.audio_device_channels_var_2 = tk.StringVar(value="default ch")
        self.audio_device_channels_combo_2 = ttk.Combobox(ch_resample_frame_2, values=["default ch", "2ch"], state="disabled", textvariable=self.audio_device_channels_var_2, width=10)
        self.audio_device_channels_combo_2.pack(side="left", padx=(5, 5))

        self.audio_device_resample_var_2 = tk.BooleanVar(value=True)
        self.audio_device_resample_check_2 = ttk.Checkbutton(ch_resample_frame_2, text="Resample", variable=self.audio_device_resample_var_2, state="disabled")
        self.audio_device_resample_check_2.pack(side="left", padx=(5, 5))

        # Use a Label widget for multiline read-only display
        row_idx = self.next_row()
        tk.Label(root, text="").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.input_dev_info_label_2 = tk.Label(root, text="", bd=1, relief="solid", justify="left", anchor="nw")
        self.input_dev_info_label_2.grid(row=row_idx, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

        # For latency
        latency_frame_2 = tk.Frame(root)
        latency_frame_2.grid(row=row_idx, column=3, padx=5, pady=5, sticky="ew")
        latency_frame_2.columnconfigure(0, weight=1)
        latency_frame_2.rowconfigure(0, weight=1)
        latency_frame_2.rowconfigure(1, weight=1)

        # Label
        self.audio_device_latency_label_2 = ttk.Label(latency_frame_2, text="Latency: \nBlock size: ", relief="solid", justify="left", anchor="w")
        self.audio_device_latency_label_2.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        # Slider
        self.audio_device_latency_slider_var_2 = tk.DoubleVar(value=0.5)
        self.audio_device_latency_slider_2 = tk.Scale(latency_frame_2, from_=0.1, to=1.0, orient="horizontal", resolution=0.01, variable=self.audio_device_latency_slider_var_2, command=self.on_audio_device_2_latency_change)
        self.audio_device_latency_slider_2.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        # Now set the combo selection changed callback and select the default audio device.
        self.audio_device_combo_2.set(dev_name)
        self.audio_device_combo_2.bind("<<ComboboxSelected>>", self.on_audio_device_2_selection_change)
        self.audio_device_combo_2.event_generate("<<ComboboxSelected>>")

        # --- Whisper model ---
        row_idx = self.next_row()
        tk.Label(root, text="Whisper model").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.model_var = tk.StringVar(value="base.en")
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
        buttons_frame = tk.Frame(root)
        buttons_frame.grid(row=row_idx, rowspan=2, column=3, columnspan=2)

        row_idx = self.next_row()
        self.start_btn = ttk.Button(buttons_frame, text="Start", command=self.start)
        self.quit_btn = ttk.Button(buttons_frame, text="Quit", command=self.quit)
        self.start_btn.grid(row=1, column=0, sticky="e", padx=5, pady=10)
        self.quit_btn.grid(row=1, column=1, sticky="e", padx=5, pady=10)

        # Update latency labels
        self.on_audio_device_1_latency_change(None)
        self.on_audio_device_2_latency_change(None)

        # make second column expand
        root.columnconfigure(1, weight=1)

        root.protocol("WM_DELETE_WINDOW", self.quit)

    def show_modal_message(self, title, message, parent):
        # Create a modal Toplevel window
        win = tk.Toplevel(parent)
        win.title(title)
        win.transient(parent)   # Keep on top of parent
        win.grab_set()          # Make modal
        win.resizable(False, False)  # Disable resizing

        # Message text
        label = tk.Label(win, text=message, padx=20, pady=20)
        label.pack()

        # OK button
        btn = tk.Button(win, text="OK", command=win.destroy, width=10)
        btn.pack(pady=10)

        # Center relative to parent
        win.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() // 2) - (win.winfo_width() // 2)
        y = parent.winfo_rooty() + (parent.winfo_height() // 2) - (win.winfo_height() // 2)
        win.geometry(f"+{x}+{y}")

        parent.wait_window(win)  # Wait until this window is closed

    def start(self):
        if self.is_running:
            self.stop_captioner()
            self.start_btn.config(text="Start")
            self.is_running = False
            print("Captioner stopped.")
        else:
            if self.use_second_audio_dev_var.get() and (self.audio_device_combo_1.get() == self.audio_device_combo_2.get()):
                self.show_modal_message("Error", "Please select two different audio devices.", self.root_wnd)
                return

            # Collect all values for debugging/demo purposes
            print("Zoom URL:", self.zoom_url_var.get())
            print("Audio device:", self.audio_device_combo_1.get())
            if self.use_second_audio_dev_var.get():
                print("Audio device 2:", self.audio_device_combo_2.get())
            print("Whisper model:", self.model_var.get())
            print("Whisper device:", self.whisper_device_var.get())
            print("Whisper compute type:", self.whisper_compute_type_var.get())
            print("Language:", self.lang_var.get())
            print("Enable translation:", self.enable_translation_var.get())
            print("Target language:", self.target_lang_var.get())
            print("Threshold:", self.threshold_var.get())
            print("Minimum chunk size:", self.min_chunk_size_var.get())
            print("VAC enabled:", self.vac_var.get())
            print("VAD enabled:", self.vad_var.get())

            print("Running Captioner...")
            self.run_captioner()
            self.start_btn.config(text="Stop")
            self.is_running = True

    def quit(self):
        if self.is_running:
            self.stop_captioner()
        self.root_wnd.quit()
        self.root_wnd.destroy()

    def on_enable_translation_toggle(self):
        if self.enable_translation_var.get():
            self.target_lang_combo.config(state="readonly")
        else:
            self.target_lang_combo.config(state="disabled")

    def on_enable_second_audio_device(self):
        if self.use_second_audio_dev_var.get():
            self.audio_device_combo_2.config(state="readonly")
            self.audio_device_host_api_combo_2.config(state="readonly")
            self.audio_device_channels_combo_2.config(state="readonly")
            self.audio_device_resample_check_2.config(state="readonly")
        else:
            self.audio_device_combo_2.config(state="disabled")
            self.audio_device_host_api_combo_2.config(state="disabled")
            self.audio_device_channels_combo_2.config(state="disabled")
            self.audio_device_resample_check_2.config(state="disabled")

    def on_audio_device_1_selection_change(self, event):
        dev_name = self.audio_device_combo_1.get()
        api_map = self.device_map[dev_name]
        self.audio_device_host_api_combo_1['values'] = sort_api_by_preference(api_map.keys())
        self.audio_device_host_api_combo_1.current(0)
        self.audio_device_host_api_combo_1.event_generate("<<ComboboxSelected>>")

    def on_audio_device_host_api_1_selection_change(self, event):
        dev_name = self.audio_device_combo_1.get()
        host_api = self.audio_device_host_api_combo_1.get()
        caps = self.device_map[dev_name][host_api]
        info_text = (
            f"Index: {caps.index}\n"
            f"Channels: {caps.channels}\n"
            f"Samplerate: {int(caps.samplerate)}\n"
            f"Latency range: {caps.min_latency} - {caps.max_latency}"
        )
        self.input_dev_info_label_1.config(text=info_text)

    def on_audio_device_2_selection_change(self, event):
        dev_name = self.audio_device_combo_2.get()
        api_map = self.device_map[dev_name]
        self.audio_device_host_api_combo_2['values'] = sort_api_by_preference(api_map.keys())
        self.audio_device_host_api_combo_2.current(0)
        self.audio_device_host_api_combo_2.event_generate("<<ComboboxSelected>>")

    def on_audio_device_host_api_2_selection_change(self, event):
        dev_name = self.audio_device_combo_2.get()
        host_api = self.audio_device_host_api_combo_2.get()
        caps = self.device_map[dev_name][host_api]
        info_text = (
            f"Index: {caps.index}\n"
            f"Channels: {caps.channels}\n"
            f"Samplerate: {int(caps.samplerate)}\n"
            f"Latency range: {caps.min_latency} - {caps.max_latency}"
        )
        self.input_dev_info_label_2.config(text=info_text)

    def on_audio_device_1_latency_change(self, value):
        dev_info = self.get_selected_device_info(1)
        block_size = dev_info.samplerate * dev_info.latency
        info_text = (
            f"Latency: {dev_info.latency: .2f} s\n"
            f"Block size: {int(block_size)} frames"
        )
        self.audio_device_latency_label_1.config(text=info_text)

    def on_audio_device_2_latency_change(self, value):
        dev_info = self.get_selected_device_info(2)
        block_size = dev_info.samplerate * dev_info.latency
        info_text = (
            f"Latency: {dev_info.latency: .2f} s\n"
            f"Block size: {int(block_size)} frames"
        )
        self.audio_device_latency_label_2.config(text=info_text)

    # Helper to manage grid row indices
    def next_row(self):
        r = self.row_i
        self.row_i += 1
        return r

    def get_selected_device_info(self, dev_num: int) -> InputDeviceInfo:
        name = self.audio_device_combo_1.get() if dev_num == 1 else self.audio_device_combo_2.get()
        api = self.audio_device_host_api_combo_1.get() if dev_num == 1 else self.audio_device_host_api_combo_2.get()
        ch = self.audio_device_channels_var_1.get() if dev_num == 1 else self.audio_device_channels_var_2.get()
        resample = self.audio_device_resample_var_1.get() if dev_num == 1 else self.audio_device_resample_var_2.get()
        latency = self.audio_device_latency_slider_var_1.get() if dev_num == 1 else self.audio_device_latency_slider_var_2.get()

        import copy
        caps = copy.copy(self.device_map[name][api])
        if ch == "2ch":
            caps.channels = 2
        if not resample:
            caps.samplerate = 16000.0

        return InputDeviceInfo(name, api, latency, caps)

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
            "nsp_threshold": threshold,
            "min_chunk_size": min_chunk_size,
            "log_level": "INFO",
            "vac": self.vac_var.get(),
            "vad": self.vad_var.get(),
        }

        self.whisper_client = WhisperClient(self.server_url_var.get().strip(), port, params, self.audio_queue, self.resuts_queue)
        self.whisper_client_thread = threading.Thread(target=self.whisper_client.run)
        self.whisper_client_thread.start()

    def run_captions_overlay(self):
        self.captions_overlay = CaptionsOverlay(self.root_wnd, self.resuts_queue, self.gui_queue)
        self.captions_overlay.start()

    def run_audio_listener(self):
        min_chunk_size, valid = str_to_float(self.min_chunk_size_var.get())
        if not valid:
            print("Error: Invalid minimum chunk size. Setting to default: 0.6")
            min_chunk_size = 0.6

        dev_info_1 = self.get_selected_device_info(1)
        use_second_dev = self.use_second_audio_dev_var.get()
        if use_second_dev:
            self.audio_temp_queue_1 = queue.Queue()
            self.audio_listener = AudioListener(min_chunk_size, dev_info_1, self.audio_temp_queue_1)

            dev_info_2 = self.get_selected_device_info(2)
            self.audio_temp_queue_2 = queue.Queue()
            self.audio_listener_2 = AudioListener(min_chunk_size, dev_info_2, self.audio_temp_queue_2)
            self.audio_mixer = AudioMixer(self.audio_temp_queue_1, self.audio_temp_queue_2, self.audio_queue)
        else:
            # We use one device, puts the results directly to the audio_queue.
            self.audio_listener = AudioListener(min_chunk_size, dev_info_1, self.audio_queue)

    def stop_captioner(self):
        if self.audio_listener:
            print("Stopping audio listener...", flush=True)
            self.audio_listener.stop()
            self.audio_listener = None

        if self.audio_listener_2:
            print("Stopping audio listener 2...", flush=True)
            self.audio_listener_2.stop()
            self.audio_listener_2 = None

            print("Stopping audio mixer thread...", flush=True)
            self.audio_mixer.stop()
            self.audio_mixer = None

        if self.whisper_client:
            print("Stopping whisper client...", flush=True)
            self.whisper_client.stop()
            self.whisper_client_thread.join()
            self.whisper_client = None
            self.whisper_client_thread = None

        if self.captions_overlay:
            print("Stopping captions overlay...", flush=True)
            self.captions_overlay.stop()
            self.captions_overlay = None

        self.audio_queue = None
        self.resuts_queue = None
        self.audio_temp_queue_1 = None
        self.audio_temp_queue_2 = None

    def run_gui(self):
        self.root_wnd.mainloop()


class AudioListener:
    def __init__(self, min_chunk_size: float, input_device_info: InputDeviceInfo, result_queue: queue.Queue):
        self.min_chunk_size = min_chunk_size
        self.input_device_info = input_device_info
        self.is_first = True

        self.WHISPER_SAMPLERATE = 16000

        self.device_rate = int(input_device_info.samplerate)
        self.device_channels = input_device_info.channels
        self.in_stream_latency = input_device_info.latency
        self.blocksize = int(self.device_rate * self.in_stream_latency)

        self.audio_queue = queue.Queue()
        self.result_queue = result_queue

        self.stop_event = threading.Event()
        self.audio_thread = threading.Thread(target=self.run)
        self.audio_thread.start()

    def _audio_callback_old(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio callback status: {status}")
        pcm16 = (indata * 32767).astype(np.int16)
        audio_float = pcm16.astype(np.float32) / 32767.0
        self.audio_queue.put(audio_float.copy())

    def downmix_mono(self, data: np.ndarray) -> np.ndarray:
        # Convert multi-channel audio to mono by averaging channels
        if data.shape[1] > 1:
            return data.mean(axis=1)
        return data[:, 0]

    def resample_to_whisper(self, data: np.ndarray) -> np.ndarray:
        # Resample audio to 16kHz if necessary
        if self.device_rate == self.WHISPER_SAMPLERATE:
            return data
        from scipy.signal import resample_poly
        return resample_poly(data, self.WHISPER_SAMPLERATE, self.device_rate)

    def audio_callback(self, indata, frames, time_info, status):
        if status and not self.is_first:
            logger.warning(f"Audio callback status: {status}")

        mono = self.downmix_mono(indata)
        mono16k = self.resample_to_whisper(mono)
        if mono16k.dtype != np.float32:
            mono16k = mono16k.astype(np.float32)
        self.audio_queue.put(mono16k)

    def receive_audio_chunk(self):
        out = []
        minlimit = int(self.min_chunk_size * self.WHISPER_SAMPLERATE)
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

    def run(self):
        print_info = (
            f"Listening to [{self.input_device_info.index}] {self.input_device_info.name}\n"
            f"  API: {self.input_device_info.api}\n"
            f"  Channels: {self.device_channels}\n"
            f"  Samplerate: {self.device_rate}\n"
            f"  Blocksize: {self.blocksize} frames (latency ~{self.in_stream_latency} s)"
        )
        print(print_info, flush=True)

        with sd.InputStream(
            samplerate=self.device_rate,
            channels=self.device_channels,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self.audio_callback,
            device=self.input_device_info.index
        ):
            while not self.stop_event.is_set():
                chunk = self.receive_audio_chunk()
                if chunk is None or len(chunk) == 0:
                    continue

                self.result_queue.put(chunk)

    def stop(self):
        self.stop_event.set()
        self.audio_thread.join()
        self.audio_thread = None


class AudioMixer:
    """
    Mixes two audio sources (mic + VB-CABLE) from their queues into a single
    16 kHz mono stream for Whisper.
    """
    def __init__(self, mic_queue: queue.Queue, cable_queue: queue.Queue, result_queue: queue.Queue):
        self.mic_queue = mic_queue
        self.cable_queue = cable_queue
        self.result_queue = result_queue
        self.stop_event = threading.Event()

        self.mixing_thread = threading.Thread(target=self.run)
        self.mixing_thread.start()

        # Minimum chunk size in samples (same as your AudioListener min_chunk_size)
        #self.min_chunk_size = 8000  # e.g., 0.5 s at 16 kHz

    def stop(self):
        self.stop_event.set()
        self.mic_queue.put(None)
        self.cable_queue.put(None)
        self.mixing_thread.join()
        self.mixing_thread = None

    def receive_chunk_from_queue(self, q: queue.Queue):
        """
        Pull one chunk from a queue, or return None if empty.
        """
        try:
            return q.get(timeout=0.1)
        except queue.Empty:
            return None

    def run(self):
        while not self.stop_event.is_set():
            mic_chunk = self.receive_chunk_from_queue(self.mic_queue)
            cable_chunk = self.receive_chunk_from_queue(self.cable_queue)

            # Skip if both are empty
            if mic_chunk is None and cable_chunk is None:
                continue

            # Replace None with zeros for mixing
            if mic_chunk is None:
                mic_chunk = np.zeros_like(cable_chunk)
            if cable_chunk is None:
                cable_chunk = np.zeros_like(mic_chunk)

            # Align lengths
            min_len = min(len(mic_chunk), len(cable_chunk))
            mixed = mic_chunk[:min_len] + cable_chunk[:min_len]

            # Optional: prevent clipping
            mixed = np.clip(mixed, -1.0, 1.0)

            self.result_queue.put(mixed)


class AudioMixer2:
    """
    Mixes two audio sources (mic + VB-CABLE) into a single
    16 kHz mono stream, outputting uniform chunks.
    """
    def __init__(self, mic_queue: queue.Queue, cable_queue: queue.Queue, result_queue: queue.Queue, min_chunk_size: int):
        self.mic_queue = mic_queue
        self.cable_queue = cable_queue
        self.result_queue = result_queue
        self.stop_event = threading.Event()

        # Minimum chunk size in samples (e.g., 8000 = 0.5s @ 16kHz)
        self.min_chunk_size = min_chunk_size

        # Buffers to hold leftover samples until enough are collected
        self.buffer = np.zeros(0, dtype=np.float32)

    def stop(self):
        self.stop_event.set()

    def receive_chunk_from_queue(self, q: queue.Queue):
        """Pull one chunk from a queue, or return None if empty."""
        try:
            return q.get(timeout=0.05)
        except queue.Empty:
            return None

    def run(self):
        while not self.stop_event.is_set():
            mic_chunk = self.receive_chunk_from_queue(self.mic_queue)
            cable_chunk = self.receive_chunk_from_queue(self.cable_queue)

            # If nothing arrived, loop again
            if mic_chunk is None and cable_chunk is None:
                continue

            # Replace missing chunks with zeros
            if mic_chunk is None and cable_chunk is not None:
                mic_chunk = np.zeros_like(cable_chunk)
            elif cable_chunk is None and mic_chunk is not None:
                cable_chunk = np.zeros_like(mic_chunk)

            # If still both None, skip
            if mic_chunk is None or cable_chunk is None:
                continue

            # Align lengths
            min_len = min(len(mic_chunk), len(cable_chunk))
            mixed = mic_chunk[:min_len] + cable_chunk[:min_len]

            # Prevent clipping
            mixed = np.clip(mixed, -1.0, 1.0).astype(np.float32)

            # Append to buffer
            self.buffer = np.concatenate([self.buffer, mixed])

            # While enough samples, push out uniform chunks
            while len(self.buffer) >= self.min_chunk_size:
                out_chunk = self.buffer[:self.min_chunk_size]
                self.buffer = self.buffer[self.min_chunk_size:]
                self.result_queue.put(out_chunk)


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
        self.results_thread = threading.Thread(target=self.listen_for_results, args=(sock,))
        self.results_thread.start()

        # Step 3: stream audio
        try:
            while True:
                chunk = self.audio_queue.get()
                if chunk is None:
                    # Tell the server we're done
                    ccmn.send_ndarray(sock, np.array([], dtype=np.float32))
                    sock.shutdown(socket.SHUT_WR)
                    self.results_thread.join()  # wait for results thread to finish, should receive shutdown confirmation
                    self.results_thread = None
                    break

                ccmn.send_ndarray(sock, chunk)

        except (BrokenPipeError, ConnectionResetError) as e:
            print(f"Server connection closed while streaming audio: {e}")
        finally:
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

    def __init__(self, root_wnd, source_queue, gui_queue):
        self.source_queue = source_queue
        self.gui_queue = gui_queue
        self.captions_thread = None
        self.is_running = False

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

    def start(self):
        if not self.is_running:
            self.captions_thread = threading.Thread(target=self.run)
            self.captions_thread.start()
            self.is_running = True

    def run(self):
        while True:
            text, complete = self.source_queue.get()
            if text is None:
                break
            elif not text.strip():
                continue

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
                if self.displ_text and self.overlay_wnd is not None:
                    try:
                        self.gui_queue.put(lambda t=self.displ_text: self.overlay_wnd.after(0, lambda t=t: self.update_label(text=t)))
                    except tk.TclError:
                        pass

    def stop(self):
        if self.is_running:
            self.source_queue.put((None, None))  # signal to stop
            self.captions_thread.join()
            self.captions_thread = None
            self.is_running = False
            self.overlay_wnd.destroy()
            self.overlay_wnd = None

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
        if self.overlay_wnd:
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
