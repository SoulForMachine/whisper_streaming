#!/usr/bin/env python3
from whisper_online import *
from transformers import MarianTokenizer, MarianMTModel
from urllib.parse import urlparse, parse_qs
import tkinter as tk

import argparse
import os
import logging
import numpy as np
import sounddevice as sd
import queue
import threading
import requests
import json

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

# options from whisper_online
add_shared_args(parser)
parser.add_argument("--warmup-file", type=str, dest="warmup_file",
        help="Path to wav file to warm up Whisper (optional).")
parser.add_argument("--device", type=int, default=None,
        help="Audio input device ID (default: system default)")
parser.add_argument("--zoom-url", type=str, default=None,
        help="Zoom meeting URL to send captions to. If not set, captions are only printed to the console.")

args = parser.parse_args()
set_logging(args, logger, other="")


######### Mic processor

class ASRProcessor:
    def __init__(self, min_chunk, result_queue):
        self.min_chunk = min_chunk
        self.last_end = None
        self.is_first = True

        self.audio_queue = queue.Queue()
        self.result_queue = result_queue

        self.stop_event = threading.Event()

        # setting whisper object by args 
        self.SAMPLING_RATE = 16000
        asr, self.online_asr_proc = asr_factory(args)

        # warm up the ASR so first chunk isn’t slow
        msg = "Whisper is not warmed up. The first chunk processing may take longer."
        if args.warmup_file:
            if os.path.isfile(args.warmup_file):
                a = load_audio_chunk(args.warmup_file,0,1)
                asr.transcribe(a)
                logger.info("Whisper is warmed up.")
            else:
                logger.critical("The warm up file is not available. "+msg)
        else:
            logger.warning(msg)


    def audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio callback status: {status}")
        pcm16 = (indata * 32767).astype(np.int16)
        audio_float = pcm16.astype(np.float32) / 32767.0
        self.audio_queue.put(audio_float.copy())

    def receive_audio_chunk(self):
        out = []
        minlimit = int(self.min_chunk * self.SAMPLING_RATE)
        while sum(len(x) for x in out) < minlimit:
            try:
                # Get a chunk from audio queue. Timeout is slightly longer than minimum chunk duration.
                chunk = self.audio_queue.get(timeout=self.min_chunk * 1.2)
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
        self.online_asr_proc.init()

        with sd.InputStream(
            samplerate=self.SAMPLING_RATE,
            channels=1,
            dtype="float32",
            blocksize=int(self.SAMPLING_RATE * 0.5),
            callback=self.audio_callback,
            device=args.device
        ):
            print("Listening from mic... Press Ctrl+C to stop.", flush=True)
            while not self.stop_event.is_set():
                chunk = self.receive_audio_chunk()
                if chunk is None:
                    continue

                self.online_asr_proc.insert_audio_chunk(chunk)
                result = self.online_asr_proc.process_iter()
                if result and result[2]:
                    self.result_queue.put(result[2])
                    print(f"[ASR] {result[2]}", flush=True)

    def stop(self):
        self.stop_event.set()

    def finish(self):
        result = self.online_asr_proc.finish()
        if result and result[2]:
            self.result_queue.put(result[2])
            print(f"[ASR] {result[2]}", flush=True)

########## Translator

class Translator:
    def __init__(self, model_name, src_lang, dst_lang, source_queue, result_queue):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.translator = MarianMTModel.from_pretrained(model_name)
        self.src_lang = src_lang
        self.dst_lang = dst_lang
        self.source_queue = source_queue
        self.result_queue = result_queue
        self.current_text = ""

    FLUSH_TIMEOUT = 6  # seconds
    LONG_BUFFERED_TEXT_LENGTH = 150  # characters
    TOO_LONG_BUFFERED_TEXT_LENGTH = 200  # characters

    def add_text(self, text: str):
        # Append new text to the current buffer.
        if text != "":
            self.current_text += text

    def get_text_to_flush(self) -> str:
        # Check for sentence-ending punctuation to flush.
        s = self.current_text
        for i, c in enumerate(s):
            if s[i] in '.!?':
                self.current_text = s[i + 1:]
                return s[:i + 1]
            
        # No sentence end found. Is the text too long? Try to split at other punctuation characters,
        # searching from the end to give as much context as possible.
        if len(self.current_text) > self.LONG_BUFFERED_TEXT_LENGTH:
            for i in range(len(s) - 1, -1, -1):
                if s[i] in ',:;-–':
                    print(f"[Translator] flushing partial text (longer than {self.LONG_BUFFERED_TEXT_LENGTH} chars).", flush=True)
                    self.current_text = s[i + 1:]
                    return s[:i + 1]

        # No punctuation and the current string is way too long? Flush everything.
        if len(self.current_text) > self.TOO_LONG_BUFFERED_TEXT_LENGTH:
            print(f"[Translator] flushing very long partial text (longer than {self.TOO_LONG_BUFFERED_TEXT_LENGTH} chars).", flush=True)
            self.current_text = ""
            return s

        # Indicate that nothing is ready to flush
        return ""

    def translate_text(self, text: str):
        text_to_translate = f">>srp_Cyrl<< {text.strip()}"
        inputs = self.tokenizer(text_to_translate, return_tensors="pt", truncation=True)
        translated = self.translator.generate(**inputs)
        transl_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)
        self.result_queue.put(transl_text)

    def run(self):
        while True:
            try:
                timeout = self.FLUSH_TIMEOUT if self.current_text != "" else None
                text = self.source_queue.get(timeout=timeout)
            except queue.Empty:
                if self.current_text != "":
                    print("[Translator] Timeout reached, flushing all current text.", flush=True)
                    self.translate_text(self.current_text + "...")
                    self.current_text = ""
                continue

            if text is None:
                break

            self.add_text(text)

            while to_traslate := self.get_text_to_flush():
                print(f"[Translator] To translate: {to_traslate}", flush=True)
                self.translate_text(to_traslate)


########## Zoom captioner

class ZoomCaptioner:
    def __init__(self, source_queue, zoom_url=None, zoom_language="en"):
        self.source_queue = source_queue
        self.zoom_url = zoom_url
        self.zoom_language = zoom_language
        
        self.meeting_id = self._extract_meeting_id(zoom_url) if zoom_url else None
        if self.meeting_id:
            self.seq_map = self._load_sequence_map()
            self.sequence = self.seq_map.get(self.meeting_id, 0)
        else:
            self.seq_map = {}
            self.sequence = 0

    def run(self):
        while True:
            text = self.source_queue.get()
            if text is None:
                break
            elif not text.strip():
                continue

            print(f"Zoom Caption: {text}", flush=True)

            if self.zoom_url:
                # Build URL with lang + sequence params
                url = f"{self.zoom_url}&lang={self.zoom_language}&seq={self.sequence}"

                headers = {
                    "Content-Type": "plain/text",
                    "Content-Length": str(len(text))
                }

                try:
                    result = requests.post(url, headers=headers, data=text)
                    if result.ok:
                        self.sequence += 1
                        self.seq_map[self.meeting_id] = self.sequence
                        self._save_seq_map(self.seq_map)
                    else:
                        print(f"[ZoomCaptioner] Error sending to Zoom: {result.status_code} {result.text}")
                except requests.RequestException as e:
                    print(f"[ZoomCaptioner] Error sending to Zoom: {e}")

    @classmethod
    def _extract_meeting_id(cls, zoom_url):
        parsed = urlparse(zoom_url)
        params = parse_qs(parsed.query)
        return params.get("id", ["unknown"])[0]

    SEQ_FILE = "zoom_seq_map.json"

    @classmethod
    def _load_sequence_map(cls):
        if os.path.exists(cls.SEQ_FILE):
            with open(cls.SEQ_FILE, "r") as f:
                return json.load(f)
        return {}

    @classmethod
    def _save_seq_map(cls, seq_map):
        with open(cls.SEQ_FILE, "w") as f:
            json.dump(seq_map, f)


######### Overlay captioner

class OverlayCaptioner:
    def __init__(self, source_queue):
        self.source_queue = source_queue
        # Tkinter window for captions
        self.root = tk.Tk()
        self.root.title("Live Captions")
        self.root.configure(bg="black")
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.8)
        self.root.overrideredirect(True)   # remove title bar and border

        self.label = tk.Label(self.root, text="", fg="white", bg="black", font=("Arial", 18), wraplength=800, justify="center")
        self.label.pack(padx=20, pady=20)

        self.current_text = ""

        self.root.bind("<Button-1>", self.start_move)
        self.root.bind("<B1-Motion>", self.do_move)
        self.start_x = 0
        self.start_y = 0

        btn = tk.Button(self.root, text="X", fg="white", bg="black", command=self.root.destroy)
        btn.place(relx=1.0, y=0, anchor="ne") # anchor to top-right corner

        # get window and screen sizes
        self.root.update_idletasks()
        win_w = self.root.winfo_width()
        win_h = self.root.winfo_height()
        scr_w = self.root.winfo_screenwidth()
        scr_h = self.root.winfo_screenheight()

        # distance from bottom (pixels)
        margin_bottom = 100  

        # calculate position
        x = (scr_w // 2) - (win_w // 2)    # center horizontally
        y = scr_h - win_h - margin_bottom  # margin from bottom

        self.root.geometry(f"+{x}+{y}")

    def start_move(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def do_move(self, event):
        x = self.root.winfo_x() + (event.x - self.start_x)
        y = self.root.winfo_y() + (event.y - self.start_y)
        self.root.geometry(f"+{x}+{y}")

    def run(self):
        while True:
            text = self.source_queue.get()
            if text is None:
                break
            elif not text.strip():
                continue

            print(f"Overlay Caption: {text}", flush=True)

            if self.current_text.endswith(('.', '!', '?')):
                self.current_text = text
            else:
                self.current_text += text

            self.root.after(0, lambda: self.update_label(text=self.current_text))

    def run_gui(self):
        self.root.mainloop()

    def update_label(self, text):
        # get current bottom center point
        x = self.root.winfo_x()
        y = self.root.winfo_y()
        win_w = self.root.winfo_width()
        win_h = self.root.winfo_height()
        bc_pt_x = x + win_w // 2
        bc_pt_y = y + win_h

        self.label.config(text=text)
        self.root.update_idletasks()

        # keep bottom center point fixed
        new_win_w = self.root.winfo_width()
        new_win_h = self.root.winfo_height()
        new_x = bc_pt_x - new_win_w // 2
        new_y = bc_pt_y - new_win_h
        self.root.geometry(f"+{new_x}+{new_y}")


######### Main

if __name__ == "__main__":
    asr_queue = queue.Queue()
    transl_queue = queue.Queue()

    print("Starting Translator thread...", flush=True)
    translator = Translator("Helsinki-NLP/opus-mt-tc-base-en-sh", "en", "sr", asr_queue, transl_queue)
    transl_thread = threading.Thread(target=translator.run, daemon=True)
    transl_thread.start()

    overlay = None
    asr_thread = None

    print("Starting Captioner thread...", flush=True)
    if args.zoom_url:
        # to warm up Zoom captioner
        transl_queue.put("...")

        zoomer = ZoomCaptioner(transl_queue, args.zoom_url, "sr")
        caption_thread = threading.Thread(target=zoomer.run, daemon=True)
        caption_thread.start()
    else:
        overlay = OverlayCaptioner(transl_queue)
        caption_thread = threading.Thread(target=overlay.run, daemon=True)
        caption_thread.start()

        print("Starting ASR thread...", flush=True)
        proc = ASRProcessor(args.min_chunk_size, asr_queue)
        asr_thread = threading.Thread(target=proc.run, daemon=True)
        asr_thread.start()

    try:
        if not overlay:
            # If Zoom captioner is used, run ASR in the main thread
            print("Starting ASR loop...", flush=True)
            proc = ASRProcessor(args.min_chunk_size, asr_queue)
            proc.run()
        else:
            # If overlay captioner is used, run Tkinter mainloop in the main thread
            overlay.run_gui()
    except KeyboardInterrupt:
        print("Interrupted by user, stopping...", flush=True)
    finally:
        # this will flush any remaining text
        proc.finish()
        
        print("Stopping all threads...", flush=True)
        # signal the end of processing
        asr_queue.put(None)
        transl_queue.put(None)

    if asr_thread:
        # signal ASR thread to stop
        proc.stop()
        print("ASR thread exiting...", flush=True)
        asr_thread.join()

    print("Translator thread exiting...", flush=True)
    transl_thread.join()

    print("Captioner thread exiting...", flush=True)
    caption_thread.join()
