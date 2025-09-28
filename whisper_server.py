import socket
import captioner_common as ccmn
from transformers import MarianTokenizer, MarianMTModel
from urllib.parse import urlparse, parse_qs
import multiprocessing as mp

import argparse
import os
import logging
import queue
import threading
import requests
import json
import re


######### WhisperOnline

class WhisperOnline:
    def __init__(self, client_params: dict):
        import whisper_online as wo

        logger = logging.getLogger(__name__)
        parser = argparse.ArgumentParser()

        # options from whisper_online
        wo.add_shared_args(parser)
        args = parser.parse_args()

        # Update args with client provided values
        for key, value in client_params.items():
            if hasattr(args, key):
                setattr(args, key, value)

        wo.set_logging(args, logger, other="")

        # Create whisper online processor object with args
        self.asr, self.asr_proc = wo.asr_factory(args)

        # warm up the ASR so first chunk isn’t slow
        warmup_file = "data/samples_jfk.wav"
        if os.path.isfile(warmup_file):
            a = wo.load_audio_chunk(warmup_file,0,1)
            self.asr.transcribe(a)
            logger.info("Whisper is warmed up.")
        else:
            logger.warning("The warm up file is not available. Whisper is not warmed up. The first chunk processing may take longer.")

    # Call before using the object for new audio stream
    # Not used for now, as we create a new object for each client
    def clear(self):
        self.asr_proc.init()

######### Mic processor

class ASRProcessor:
    def __init__(self, client_params: dict, audio_queue: mp.Queue, result_queue: mp.Queue):
        self.client_params = client_params
        self.audio_queue = audio_queue
        self.result_queue = result_queue

        self.asr_subproc = None
        self.is_running = False

    def start(self):
        if not self.is_running:
            self.asr_subproc = mp.Process(
                target=asr_subprocess_main,
                args=(self.client_params, self.audio_queue, self.result_queue),
                daemon=False
            )
            self.asr_subproc.start()
            self.is_running = True

    def stop(self):
        if self.is_running:
            self.audio_queue.put(None)
            self.asr_subproc.join()
            self.asr_subproc = None
            self.is_running = False


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

        self.transl_thread = None
        self.is_running = False

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

    def start(self):
        if not self.is_running:
            self.transl_thread = threading.Thread(target=self.run)
            self.transl_thread.start()
            self.is_running = True

    def stop(self):
        if self.is_running:
            self.source_queue.put(None)
            self.transl_thread.join()
            self.transl_thread = None
            self.is_running = False

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
        self.caption_thread = None
        self.is_running = False

        self.meeting_id = self._extract_meeting_id(zoom_url) if zoom_url else None
        if self.meeting_id:
            self.seq_map = self._load_sequence_map()
            self.sequence = self.seq_map.get(self.meeting_id, 0)
        else:
            self.seq_map = {}
            self.sequence = 0

    def start(self):
        if not self.is_running:
            self.caption_thread = threading.Thread(target=self.run)
            self.caption_thread.start()
            self.is_running = True

    def stop(self):
        if self.is_running:
            self.source_queue.put((None, None))
            self.caption_thread.join()
            self.caption_thread = None
            self.is_running = False

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
        self.caption_thread = None
        self.is_running = False

    def start(self):
        if not self.is_running:
            self.caption_thread = threading.Thread(target=self.run)
            self.caption_thread.start()
            self.is_running = True

    def stop(self):
        if self.is_running:
            self.source_queue.put((None, None))
            self.caption_thread.join()
            self.caption_thread = None
            self.is_running = False

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
        self.audio_queue = mp.Queue()
        self.asr_queue = mp.Queue()
        self.transl_queue = queue.Queue()

        zoom_url = client_params.get("zoom_url")

        print("Starting Translator thread...", flush=True)
        self.translator = Translator("Helsinki-NLP/opus-mt-tc-base-en-sh", "en", "sr", self.asr_queue, self.transl_queue, only_complete_sent=bool(zoom_url))
        self.translator.start()

        print("Starting Caption sender thread...", flush=True)
        if zoom_url:
            zoom_url = zoom_url.strip()
            self.transl_queue.put(("...", True))  # to warm up Zoom
            self.caption_sender = ZoomCaptionSender(self.transl_queue, zoom_url, "sr")
        else:
            self.caption_sender = CaptionSender(self.transl_queue, conn)
        self.caption_sender.start()

        print("Starting ASR thread...", flush=True)
        self.asr_proc = ASRProcessor(client_params, self.audio_queue, self.asr_queue)
        self.asr_proc.start()

    def process(self, arr):
        self.audio_queue.put(arr)

    def stop(self):
        print("Stopping all threads...", flush=True)

        print("ASR thread exiting...", flush=True)
        self.asr_proc.stop()

        print("Translator thread exiting...", flush=True)
        self.translator.stop()

        print("Caption sender thread exiting...", flush=True)
        self.caption_sender.stop()

        self.asr_proc = None
        self.translator = None
        self.caption_sender = None


class WhisperServer:
    def __init__(self, host="0.0.0.0", port=5000):
        self.conn = None
        self.server_thread = None
        self.is_running = False
        self.host = host
        self.port = port

    def start(self):
        if not self.is_running:
            self.server_thread = threading.Thread(target=self.run, daemon=True)
            self.server_thread.start()
            self.is_running = True

    def run(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind((self.host, self.port))
        srv.listen(1)

        try:
            while True:
                print(f"Server listening on {self.host}:{self.port}")
                conn, addr = srv.accept()
                print(f"Connected by {addr}")
                self.handle_client(conn)
        except KeyboardInterrupt:
            print("Interrupted by user, stopping...")
        except Exception as e:
            print(f"Server error: {e}")
        finally:
            print("Server shutting down.")
            srv.close()
            self.is_running = False

    def handle_client(self, conn: socket.socket):
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
                chunk = ccmn.recv_ndarray(conn)
                if chunk is None:
                    print("Client disconnected unexpectedly.")
                    break

                # Empty array signals that the client is diconnecting.
                if len(chunk) == 0:
                    print("Client gracefully disconnecting.")

                    # Stop the pipeline. This will flush any remaining text.
                    pipeline.stop()

                    # Confirm shutdown
                    ccmn.send_json(conn, {
                        "type": "status",
                        "value": "shutdown"
                    })

                    conn.shutdown(socket.SHUT_WR)
                    break

                # Send received chunk to the pipeline
                pipeline.process(chunk)

        except (ConnectionResetError, OSError) as e:
            print(f"Connection lost: {e}.")
        finally:
            conn.close()
            print("Connection closed.")


def asr_subprocess_main(client_params: dict, audio_queue: mp.Queue, asr_queue: mp.Queue):
    """
    Runs in the subprocess: construct WhisperOnline here, then
    read audio chunks from audio_queue and put results to asr_queue.
    """
    whisper_online = WhisperOnline(client_params)

    # loop until sentinel
    while True:
        chunk = audio_queue.get()
        if chunk is None:
            break

        whisper_online.asr_proc.insert_audio_chunk(chunk)
        result = whisper_online.asr_proc.process_iter()
        if result and result[2]:
            asr_queue.put(result[2])
            print(f"[ASR] {result[2]}", flush=True)

    # flush any remaining audio
    result = whisper_online.asr_proc.finish()
    if result and result[2]:
        asr_queue.put(result[2])


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    import sys

    parser = argparse.ArgumentParser()

    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host adress. Default: 0.0.0.0")
    parser.add_argument("--port", type=int, default=5000, help="Port number. Default: 5000")
    args = parser.parse_args(sys.argv[1:])

    whisper_server = WhisperServer(args.host, args.port)
    whisper_server.start()

    while True:
        try:
            threading.Event().wait(1)
            if whisper_server.is_running is False:
                break
        except KeyboardInterrupt:
            print("Stopping server...")
            break
