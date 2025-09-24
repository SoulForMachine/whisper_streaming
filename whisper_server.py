import socket
import captioner_common as ccmn

def process_with_whisper(arr, processor):
    """
    Fake whisper_streaming call for demo.
    Replace with actual processor.insert_audio_chunk(arr).
    """
    text = f"Decoded {arr.shape[0]} samples"
    return {"lang": "en", "text": text, "partial": False}

def audio_server(host="0.0.0.0", port=5000, processor=None):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind((host, port))
    srv.listen(1)
    print(f"Server listening on {host}:{port}")

    conn, addr = srv.accept()
    print(f"Connected by {addr}")

    # Step 1: receive params
    params = ccmn.recv_json(conn)
    if params is None:
        print("Failed to receive params.")
        conn.close()
        srv.close()
        return
    print("Received params:", params)

    zoom_url = params.get("zoom_url")

    # Step 2: confirm initialization
    ccmn.send_json(conn, {"type": "status", "value": "ready"})

    # Step 3: receive audio chunks
    while True:
        arr = ccmn.recv_ndarray(conn)
        if arr is None:
            break

        print(f"Received chunk with shape {arr.shape}")

        # Process with whisper
        result = process_with_whisper(arr, processor)

        if result:
            if zoom_url:
                # TODO: push to Zoom
                print("Would send to Zoom:", result)
            else:
                ccmn.send_json(conn, {
                    "type": "translation",
                    "lang": result["lang"],
                    "text": result["text"],
                    "partial": result["partial"],
                })

    conn.close()
    srv.close()

if __name__ == "__main__":
    audio_server(host="localhost")
