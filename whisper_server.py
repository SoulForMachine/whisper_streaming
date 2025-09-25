import socket
import captioner_common as ccmn
from whisper_online import *
from transformers import MarianTokenizer, MarianMTModel
from urllib.parse import urlparse, parse_qs

import argparse
import os
import logging
import queue
import threading
import requests
import json
import re


######### Mic processor

class ASRProcessor:
    def __init__(self, args, audio_queue: queue.Queue, result_queue: queue.Queue):
        self.audio_queue = audio_queue
        self.result_queue = result_queue

        # Create whisper online processor object with args
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

    def run(self):
        self.online_asr_proc.init()

        while True:
            chunk = self.audio_queue.get()
            if chunk is None:
                print("--- quitting ASR thread.")
                break

            self.online_asr_proc.insert_audio_chunk(chunk)
            result = self.online_asr_proc.process_iter()
            if result and result[2]:
                self.result_queue.put(result[2])
                print(f"[ASR] {result[2]}", flush=True)

    def stop(self):
        self.audio_queue.put(None)

    def finish(self):
        result = self.online_asr_proc.finish()
        if result and result[2]:
            self.result_queue.put(result[2])
            print(f"[ASR] {result[2]}", flush=True)

########## Translator

class Translator:
    def __init__(self, model_name, src_lang, dst_lang, source_queue, result_queue, only_complete_sent: bool):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.translator = MarianMTModel.from_pretrained(model_name)
        self.src_lang = src_lang
        self.dst_lang = dst_lang
        self.source_queue = source_queue
        self.result_queue = result_queue
        self.only_complete_sent = only_complete_sent
        self.current_text = ""

    FLUSH_TIMEOUT = 6  # seconds

    def add_text(self, text: str):
        # Append new text to the current buffer.
        if text != "":
            if self.current_text != "" and self.current_text[-1] != ' ' and text[0] != ' ':
                self.current_text += ' '
            self.current_text += text

    def get_sentence(self) -> str:
        text = self.current_text
        """
        Extracts the first sentence from text, keeping consecutive punctuation.
        Returns the sentence or an empty string if no sentence-ending punctuation is found.
        """
        # Regex: sentence until first [.!?], include all consecutive [.!?]
        match = re.search(r'^(.*?[.!?]+)(\s*)(.*)$', text, flags=re.S)
        if not match:
            # no complete sentence, return the current partial text
            return (self.current_text, False)

        sentence = match.group(1).strip()
        self.current_text = match.group(3).lstrip()
        return (sentence, True)

    def translate_text(self, text: str):
        text_to_translate = f">>srp_Cyrl<< {text[0]}"
        inputs = self.tokenizer(text_to_translate, return_tensors="pt", truncation=True)
        translated = self.translator.generate(**inputs)
        transl_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)
        self.result_queue.put((transl_text, text[1]))

    def run(self):
        last_partial_len = 0

        while True:
            try:
                timeout = self.FLUSH_TIMEOUT if self.current_text != "" else None
                text = self.source_queue.get(timeout=timeout)
            except queue.Empty:
                if self.current_text != "":
                    print("[Translator] Timeout reached, flushing all current text.", flush=True)
                    self.translate_text((self.current_text + "...", True))
                    self.current_text = ""
                    last_partial_len = 0
                continue

            if text is None:
                break

            self.add_text(text)

            complete_sentence = True
            while complete_sentence:
                to_translate = self.get_sentence()
                text = to_translate[0]
                complete_sentence = to_translate[1]

                if text != "":
                    if complete_sentence:
                        self.translate_text(to_translate)
                        last_partial_len = 0
                    elif (not self.only_complete_sent) and (len(text) - last_partial_len > 50):
                        last_partial_len = len(text)
                        self.translate_text(to_translate)


########## Zoom caption sender

class ZoomCaptionSender:
    def __init__(self, source_queue, zoom_url, zoom_language="en"):
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
            text, complete = self.source_queue.get()
            if text is None:
                break
            elif not text.strip():
                continue

            #print(f"Sending Zoom caption: {text}", flush=True)

            if self.zoom_url and complete:
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


########## Caption sender: sends captions to the client

class CaptionSender:
    def __init__(self, source_queue, conn):
        self.source_queue = source_queue
        self.conn = conn

    def run(self):
        while True:
            text, complete = self.source_queue.get()
            if text is None:
                break
            elif not text.strip():
                continue

            ccmn.send_json(self.conn, {
                "type": "translation",
                "lang": "sr",
                "text": text,
                "complete": complete,
            })


########## WhisperPipeline

class WhisperPipeline:
    def __init__(self, client_params, conn):
        logger = logging.getLogger(__name__)
        parser = argparse.ArgumentParser()

        # options from whisper_online
        add_shared_args(parser)

        # additional options
        parser.add_argument("--warmup-file", type=str, dest="warmup_file",
                help="Path to wav file to warm up Whisper (optional).")
        parser.add_argument("--audio-input-device", type=int, default=None,
                help="Audio input device ID (default: system default)")
        parser.add_argument("--zoom-url", type=str, default=None,
                help="Zoom meeting URL to send captions to. If not set, captions are only printed to the console.")

        self.args = parser.parse_args()

        # Update args with client provided values
        for key, value in client_params.items():
            if hasattr(self.args, key):
                setattr(self.args, key, value)

        set_logging(self.args, logger, other="")

        self.audio_queue = queue.Queue()
        self.asr_queue = queue.Queue()
        self.transl_queue = queue.Queue()

        zoom_url = self.args.zoom_url.strip() if self.args.zoom_url and self.args.zoom_url.strip() else None

        print("Starting Translator thread...", flush=True)
        self.translator = Translator("Helsinki-NLP/opus-mt-tc-base-en-sh", "en", "sr", self.asr_queue, self.transl_queue, only_complete_sent=bool(zoom_url))
        self.transl_thread = threading.Thread(target=self.translator.run, daemon=True)
        self.transl_thread.start()

        print("Starting Caption sender thread...", flush=True)
        if zoom_url:
            # to warm up Zoom
            self.transl_queue.put(("...", True))

            self.caption_sender = ZoomCaptionSender(self.transl_queue, zoom_url, "sr")
            self.caption_thread = threading.Thread(target=self.caption_sender.run, daemon=True)
            self.caption_thread.start()
        else:
            self.caption_sender = CaptionSender(self.transl_queue, conn)
            self.caption_thread = threading.Thread(target=self.caption_sender.run, daemon=True)
            self.caption_thread.start()

        print("Starting ASR thread...", flush=True)
        self.asr_proc = ASRProcessor(self.args, self.audio_queue, self.asr_queue)
        self.asr_thread = threading.Thread(target=self.asr_proc.run, daemon=True)
        self.asr_thread.start()

    def process(self, arr):
        self.audio_queue.put(arr)

    def stop(self):
        # this will flush any remaining text
        self.asr_proc.finish()

        print("Stopping all threads...", flush=True)

        # signal ASR thread to stop
        self.asr_proc.stop()
        print("ASR thread exiting...", flush=True)
        self.asr_thread.join()

        self.asr_queue.put(None)
        print("Translator thread exiting...", flush=True)
        self.transl_thread.join()

        self.transl_queue.put((None, None))
        print("Caption sender thread exiting...", flush=True)
        self.caption_thread.join()
        print("--- all threads stopped.", flush=True)

        try:
            self.asr_proc = None
            self.translator = None
            self.caption_sender = None
        except Exception as e:
            print(f"--- Error releasing objects: {e}", flush=True)
        print("--- all objects released.", flush=True)


def whisper_server(host="0.0.0.0", port=5000):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind((host, port))
    srv.listen(1)

    try:
        while True:
            print(f"Server listening on {host}:{port}")
            conn, addr = srv.accept()
            print(f"Connected by {addr}")
            handle_client(conn)
    except KeyboardInterrupt:
        print("Interrupted by user, stopping...")
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        print("Server shutting down.")
        srv.close()


def handle_client(conn):
    try:
        # Step 1: receive params
        params = ccmn.recv_json(conn)
        if params is None:
            print("Failed to receive params.")
            conn.close()
            return
        print("Received params:", params)

        pipeline = WhisperPipeline(params, conn)

        # Step 2: confirm initialization
        ccmn.send_json(conn, {"type": "status", "value": "ready"})

        # Step 3: receive audio chunks
        while True:
            arr = ccmn.recv_ndarray(conn)
            if arr is None:
                print("Client disconnected unexpectedly.")
                break

            # Empty array signals shutdown
            if len(arr) == 0:
                print("Client disconnecting.")
                ccmn.send_json(conn, {
                    "type": "status",
                    "value": "shutdown"
                })
                break

            # Send it to the pipeline
            pipeline.process(arr)

    except (ConnectionResetError, OSError) as e:
        print(f"Connection lost: {e}.")
    finally:
        print("--- stopping pipeline and closing connection.")
        try:
            pipeline.stop()
        except Exception as e:
            print(f"--- Error closing connection: {e}")
        print("--- pipeline stopped.")
        conn.close()
        
        print("--- connection closed.")


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser()

    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host adress. Default: 0.0.0.0")
    parser.add_argument("--port", type=int, default=5000, help="Port number. Default: 5000")
    args = parser.parse_args(sys.argv[1:])

    whisper_server(args.host, args.port)
