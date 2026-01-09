import os

os.environ["SPEECHBRAIN_FETCH_STRATEGY"] = "COPY"

import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from speechbrain.inference.classifiers import EncoderClassifier

# --- 1. LOAD THE MODEL ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading pre-trained model...")
model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)
model.eval()
print("Model loaded successfully.")

# --- 2. ENROLLMENT ---

# This is your list of 26 files (with the typo fixed for #24)
target_audio_files = [
    "C:/Users/shrey/PycharmProjects/sample_spunds/Recording.wav",
    "C:/Users/shrey/PycharmProjects/sample_spunds/Recording (2).wav",
    "C:/Users/shrey/PycharmProjects/sample_spunds/Recording (3).wav",
    "C:/Users/shrey/PycharmProjects/sample_spunds/Recording (4).wav",
    "C:/Users/shrey/PycharmProjects/sample_spunds/Recording (5).wav",
    "C:/Users/shrey/PycharmProjects/sample_spunds/Recording (6).wav",
    "C:/Users/shrey/PycharmProjects/sample_spunds/Recording (7).wav",
    "C:/Users/shrey/PycharmProjects/sample_spunds/Recording (8).wav",
    "C:/Users/shrey/PycharmProjects/sample_spunds/Recording (9).wav",
    "C:/Users/shrey/PycharmProjects/sample_spunds/Recording (10).wav",
    "C:/Users/shrey/PycharmProjects/sample_spunds/Recording (11).wav",
    "C:/Users/shrey/PycharmProjects/sample_spunds/Recording (12).wav",
    "C:/Users/shrey/PycharmProjects/sample_spunds/Recording (13).wav",
    "C:/Users/shrey/PycharmProjects/sample_spunds/Recording (14).wav",
    "C:/Users/shrey/PycharmProjects/sample_spunds/Recording (15).wav",
    "C:/Users/shrey/PycharmProjects/sample_spunds/Recording (16).wav",
    "C:/Users/shrey/PycharmProjects/sample_spunds/Recording (17).wav",
    "C:/Users/shrey/PycharmProjects/sample_spunds/Recording (18).wav",
    "C:/Users/shrey/PycharmProjects/sample_spunds/Recording (19).wav",
    "C:/Users/shrey/PycharmProjects/sample_spunds/Recording (20).wav",
    "C:/Users/shrey/PycharmProjects/sample_spunds/Recording (21).wav",
    "C:/Users/shrey/PycharmProjects/sample_spunds/Recording (22).wav",
    "C:/Users/shrey/PycharmProjects/sample_spunds/Recording (23).wav",
    "C:/Users/shrey/PycharmProjects/sample_spunds/Recording (24).wav",
    "C:/Users/shrey/PycharmProjects/sample_spunds/Recording (25).wav",
    "C:/Users/shrey/PycharmProjects/sample_spunds/Recording (26).wav"
]

print(f"\nEnrolling target speaker using {len(target_audio_files)} utterances...")

all_embeddings = []
valid_files_processed = 0

for audio_file in target_audio_files:
    try:
        signal, fs = torchaudio.load(audio_file)
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
            signal = resampler(signal)

        with torch.no_grad():
            embedding = model.encode_batch(signal.to(device))

        embedding = F.normalize(embedding, p=2, dim=-1)
        all_embeddings.append(embedding.squeeze().cpu().numpy())
        valid_files_processed += 1

    except Exception as e:
        print(f"  Warning: Error processing {audio_file}. Skipping. Error: {e}")

if valid_files_processed > 0:
    target_voiceprint = np.mean(all_embeddings, axis=0)
    np.save("temp_voiceprint_v192.npy", target_voiceprint)
    print(f"\nEnrollment complete! (dimensions: {target_voiceprint.shape})")
    print("New 'target_voiceprint.npy' (192-dim) has been saved.")
else:
    print("\nError: No files processed.")
