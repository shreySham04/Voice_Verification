import os

os.environ["SPEECHBRAIN_FETCH_STRATEGY"] = "COPY"

import PySimpleGUI as sg
import threading
import subprocess
import time

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.inference.ASR import EncoderDecoderASR
import sounddevice as sd

# --- 1. SETTINGS (from both your files) ---
# Enrollment settings
AUDIO_DIR = "sample_spunds"
OUTPUT_FILE = "final_live_voiceprint.npy"

# Verification settings
SAMPLE_RATE = 16000
DURATION = 2.5
VERIFICATION_THRESHOLD = 0.50
MIN_VOICE_SCORE = 0.40


# --- 2. ENROLLMENT LOGIC (from enroll.py) ---
# We run this in a thread to avoid freezing the UI
def do_enrollment(window):
    """
    Runs the complete enrollment process and sends updates
    to the UI window.
    """
    try:
        window.write_event_value('-ENROLL-UPDATE-', "Starting enrollment...")

        # --- COLLECT AUDIO FILES ---
        if not os.path.exists(AUDIO_DIR):
            window.write_event_value('-ENROLL-UPDATE-', f"⚠️ Folder '{AUDIO_DIR}' not found.")
            return

        wav_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
        if not wav_files:
            window.write_event_value('-ENROLL-UPDATE-', f"⚠️ No .wav files found in '{AUDIO_DIR}'.")
            return

        window.write_event_value('-ENROLL-UPDATE-', f"🧩 Found {len(wav_files)} audio files.")

        # --- LOAD ENROLLMENT MODEL ---
        # Note: This model is already loaded in the main thread,
        # but we load it here again for a clean, separate enrollment.
        # For a more optimized app, you'd pass the model in.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        enroll_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )
        enroll_model.eval()

        embeddings = []

        # --- PROCESS EACH FILE ---
        for i, wav_file in enumerate(wav_files):
            file_path = os.path.join(AUDIO_DIR, wav_file)
            signal, sr = torchaudio.load(file_path)

            # Convert to mono
            if signal.shape[0] > 1:
                signal = torch.mean(signal, dim=0, keepdim=True)

            # Resample if needed
            if sr != SAMPLE_RATE:
                signal = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(signal)

            with torch.no_grad():
                emb = enroll_model.encode_batch(signal.to(device))
                emb = F.normalize(emb, p=2, dim=-1)
                emb = emb.squeeze().cpu().numpy()
                embeddings.append(emb)
                window.write_event_value('-ENROLL-UPDATE-', f"({i + 1}/{len(wav_files)}) Processed {wav_file}")

        # --- AVERAGE ALL EMBEDDINGS ---
        embeddings = np.array(embeddings)
        final_embedding = np.mean(embeddings, axis=0)
        np.save(OUTPUT_FILE, final_embedding)

        window.write_event_value('-ENROLL-UPDATE-', "\n✅ Enrollment complete!")
        window.write_event_value('-ENROLL-UPDATE-', f"Saved master voiceprint to '{OUTPUT_FILE}'")
        window.write_event_value('-ENROLL-FINISHED-', None)
    except Exception as e:
        window.write_event_value('-ENROLL-UPDATE-', f"\n❌ ERROR: {e}")
        window.write_event_value('-ENROLL-FINISHED-', None)


# --- 3. VERIFICATION LOGIC (from speechy.py) ---
# These are helper functions that will be used by the verification thread
def verify_and_transcribe_signal(audio_signal, model, asr_model, target_voiceprint, device):
    """Compares audio to voiceprint AND transcribes the speech."""
    try:
        audio_rms = np.sqrt(np.mean(audio_signal ** 2))
        if audio_rms < 0.01:
            return 0.0, "SILENCE", ""

        audio_tensor = torch.tensor(audio_signal).float().unsqueeze(0).to(device)

        with torch.no_grad():
            # Speaker Verification
            test_embedding = model.encode_batch(audio_tensor)
            test_embedding = F.normalize(test_embedding, p=2, dim=-1)
            test_embedding = test_embedding.squeeze().cpu().numpy().reshape(1, -1)
            similarity = cosine_similarity(target_voiceprint, test_embedding)[0][0]

            # Speech-to-Text (ASR)
            relative_length = torch.tensor([1.0]).to(device)
            transcription = asr_model.transcribe_batch(audio_tensor, relative_length)
            transcribed_text = transcription[0][0]

        return similarity, "VOICE", transcribed_text.lower()

    except Exception as e:
        print(f"Error during verification: {e}")
        return None, "ERROR", ""


def trigger_unlock_action():
    """Triggers the desired action upon successful authentication."""
    # This will still open Notepad in the background
    try:
        subprocess.Popen("notepad.exe")
    except Exception as e:
        print(f"Failed to open app: {e}")


# This is the main function for the verification THREAD
def verification_loop(window, stop_event, model, asr_model, target_voiceprint, device):
    """
    The main listening loop that runs in a separate thread.
    """
    try:
        while not stop_event.is_set():
            window.write_event_value('-VERIFY-STATUS-', ("🎙️ Listening...", "cyan"))

            myrecording = sd.rec(
                int(DURATION * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32'
            )
            sd.wait()  # Wait for recording to finish

            # If 'Stop' was clicked while waiting, exit now
            if stop_event.is_set():
                break

            myrecording_1d = myrecording.flatten()
            score, status, text = verify_and_transcribe_signal(
                myrecording_1d, model, asr_model, target_voiceprint, device
            )

            # Send the results back to the main UI thread
            window.write_event_value('-VERIFY-RESULT-', (score, status, text))

    except Exception as e:
        window.write_event_value('-VERIFY-STATUS-', (f"❌ Error: {e}", "red"))

    # Tell the UI thread that we have stopped
    window.write_event_value('-VERIFY-STOPPED-', None)


# --- 4. THE GUI ---

def main_gui():
    sg.theme("DarkGrey2")

    # --- Layouts for each tab ---
    enroll_layout = [
        [sg.Text("Run enrollment to create/update your master voiceprint.")],
        [sg.Text(f"It will process .wav files from the '{AUDIO_DIR}' folder.")],
        [sg.Button("Run Enrollment Process", key="-ENROLL-", size=(25, 2))],
        [sg.Text("Output:")],
        [sg.Multiline(size=(80, 15), key="-ENROLL-OUTPUT-", disabled=True, autoscroll=True, reroute_stdout=True,
                      reroute_stderr=True)]
    ]

    verify_layout = [
        [sg.Text("Press 'Start' to begin voice verification.")],
        [sg.Button("Start Listening", key="-START-", size=(20, 2), button_color=("white", "green")),
         sg.Button("Stop Listening", key="-STOP-", size=(20, 2), button_color=("white", "red"), disabled=True)],
        [sg.HSeparator()],
        [sg.Text("STATUS:", font=("Helvetica", 14)),
         sg.Text("Idle", key="-STATUS-", size=(40, 1), font=("Helvetica", 14, "bold"), text_color="grey")],
        [sg.Text("SCORE:", font=("Helvetica", 14)),
         sg.Text("0.0000", key="-SCORE-", size=(20, 1), font=("Helvetica", 14, "bold"), text_color="white")],
        [sg.Text("HEARD:", font=("Helvetica", 14)),
         sg.Text("...", key="-HEARD-", size=(40, 1), font=("Helvetica", 14, "bold"), text_color="white")],
    ]

    # --- Main Window Layout ---
    layout = [
        [sg.TabGroup([
            [sg.Tab("Verification", verify_layout)],
            [sg.Tab("Enrollment", enroll_layout)]
        ])]
    ]

    window = sg.Window("Voice Guardian", layout, finalize=True)

    # --- Load Models (Do this once at the start) ---
    window["-STATUS-"].update("Loading models... (this may take a moment)", text_color="orange")
    window.refresh()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )
        model.eval()

        asr_model = EncoderDecoderASR.from_hparams(
            source="speechbrain/asr-crdnn-rnnlm-librispeech",
            savedir="pretrained_models/asr-crdnn-rnnlm-librispeech",
            run_opts={"device": device}
        )
        window["-STATUS-"].update("Models loaded. Loading voiceprint...", text_color="yellow")
        window.refresh()

        target_voiceprint = np.load(OUTPUT_FILE)
        target_voiceprint = target_voiceprint.reshape(1, -1)

        window["-STATUS-"].update("✅ Ready to Start.", text_color="green")

    except FileNotFoundError:
        window["-STATUS-"].update(f"❌ '{OUTPUT_FILE}' not found. Please run enrollment.", text_color="red")
        target_voiceprint = None
    except Exception as e:
        # Close the main loading window first
        window.close()

        # Now, create a new popup window that will stay on screen
        sg.popup_error(
            "CRITICAL MODEL ERROR",
            "The application could not load the AI models and must close.",
            f"\nError details: {e}",
            "\nPlease copy this full error message and report it."
        )
        return

    # --- GUI Event Loop ---
    stop_event = None
    verify_thread = None

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            if stop_event:
                stop_event.set()  # Tell thread to stop
            if verify_thread:
                verify_thread.join()  # Wait for thread to finish
            break

        # --- Enrollment Tab Events ---
        if event == "-ENROLL-":
            window["-ENROLL-"].update(disabled=True)
            window["-ENROLL-OUTPUT-"].update("")  # Clear output
            # Run enrollment in a separate thread to keep UI responsive
            threading.Thread(target=do_enrollment, args=(window,), daemon=True).start()

        elif event == "-ENROLL-UPDATE-":
            window["-ENROLL-OUTPUT-"].update(values[event] + "\n", append=True)

        elif event == "-ENROLL-FINISHED-":
            window["-ENROLL-"].update(disabled=False)
            # Reload the voiceprint in case it was just created/updated
            try:
                target_voiceprint = np.load(OUTPUT_FILE)
                target_voiceprint = target_voiceprint.reshape(1, -1)
                window["-STATUS-"].update("Voiceprint reloaded. Ready.", text_color="green")
            except Exception as e:
                window["-STATUS-"].update(f"Error loading new voiceprint: {e}", text_color="red")

        # --- Verification Tab Events ---
        if event == "-START-":
            if target_voiceprint is None:
                window["-STATUS-"].update("Please run enrollment first!", text_color="red")
            else:
                window["-START-"].update(disabled=True)
                window["-STOP-"].update(disabled=False)

                stop_event = threading.Event()
                verify_thread = threading.Thread(
                    target=verification_loop,
                    args=(window, stop_event, model, asr_model, target_voiceprint, device),
                    daemon=True
                )
                verify_thread.start()

        elif event == "-STOP-":
            if stop_event:
                stop_event.set()  # Signal the thread to stop
            window["-STOP-"].update(disabled=True)

        # --- Events from Verification Thread ---
        elif event == "-VERIFY-STATUS-":
            status_text, color = values[event]
            window["-STATUS-"].update(status_text, text_color=color)

        elif event == "-VERIFY-RESULT-":
            score, status, text = values[event]

            if status == "SILENCE":
                window["-STATUS-"].update("😴 Detected silence...", text_color="grey")
                window["-SCORE-"].update("0.0000")
                window["-HEARD-"].update("...")
                continue
            if status == "ERROR":
                window["-STATUS-"].update("Error processing audio.", text_color="red")
                continue

            # Update UI elements
            window["-SCORE-"].update(f"{score:.4f}")
            window["-HEARD-"].update(f'"{text}"')

            # --- The Decision ---
            if score > VERIFICATION_THRESHOLD:
                window["-STATUS-"].update(f"✅ ACCESS GRANTED!", text_color="lightgreen")
                trigger_unlock_action()

                # Optional: Stop listening after success
                # if stop_event:
                #     stop_event.set()
                # window["-STOP-"].update(disabled=True)

            elif score < MIN_VOICE_SCORE:
                window["-STATUS-"].update(f"❌ LOW CONFIDENCE", text_color="yellow")
            else:
                window["-STATUS-"].update(f"❌ ACCESS DENIED (Impostor)", text_color="red")

        elif event == "-VERIFY-STOPPED-":
            window["-STATUS-"].update("Idle", text_color="grey")
            window["-START-"].update(disabled=False)
            window["-STOP-"].update(disabled=True)
            verify_thread = None

    window.close()


# --- Run the App ---
if __name__ == "__main__":
    main_gui()