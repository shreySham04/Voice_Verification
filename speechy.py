import os
os.environ["SPEECHBRAIN_FETCH_STRATEGY"] = "COPY"

import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from speechbrain.inference.classifiers import EncoderClassifier

# ==========================================================
# STEP 1: SET DEVICE (GPU if available)
# ==========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ==========================================================
# STEP 2: LOAD PRETRAINED MODEL
# ==========================================================
try:
    print("\nLoading pre-trained ECAPA-TDNN model...")
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )
    model.eval()
    print("✅ Model loaded successfully.\n")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# ==========================================================
# STEP 3: ENROLLMENT (REGISTER YOUR VOICE)
# ==========================================================
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

print(f"Enrolling target speaker using {len(target_audio_files)} recordings...\n")

all_embeddings = []
valid_files_processed = 0

for audio_file in target_audio_files:
    try:
        signal, fs = torchaudio.load(audio_file)

        # Resample if not 16kHz
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
            signal = resampler(signal)

        with torch.no_grad():
            embedding = model.encode_batch(signal.to(device))
            embedding = F.normalize(embedding, p=2, dim=-1)
            all_embeddings.append(embedding.squeeze().cpu().numpy())
            valid_files_processed += 1

    except Exception as e:
        print(f"⚠️ Warning: Could not process {audio_file}. Skipping. Error: {e}")

if valid_files_processed == 0:
    print("\n❌ CRITICAL: No valid audio files processed.")
    print("Check the file paths or recordings.")
    exit()

target_voiceprint = np.mean(all_embeddings, axis=0)
np.save("target_voiceprint.npy", target_voiceprint)
print(f" Enrollment complete! ({valid_files_processed} files processed)")
print("Master voiceprint saved as 'target_voiceprint.npy'\n")

# ==========================================================
# STEP 4: VERIFICATION FUNCTION
# ==========================================================
def verify_speaker(test_audio_file, saved_voiceprint_path="target_voiceprint.npy"):
    """Compare a test audio file with the saved master voiceprint."""
    try:
        target_voiceprint = np.load(saved_voiceprint_path).reshape(1, -1)

        signal, fs = torchaudio.load(test_audio_file)
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
            signal = resampler(signal)

        with torch.no_grad():
            test_embedding = model.encode_batch(signal.to(device))
            test_embedding = F.normalize(test_embedding, p=2, dim=-1)
            test_embedding = test_embedding.squeeze().cpu().numpy().reshape(1, -1)

        similarity = cosine_similarity(target_voiceprint, test_embedding)[0][0]
        return similarity

    except FileNotFoundError:
        print(f" Error: Voiceprint not found at {saved_voiceprint_path}")
        return None
    except Exception as e:
        print(f" Verification error for {test_audio_file}: {e}")
        return None
if valid_files_processed > 0:
    print("\n--- STARTING VERIFICATION TEST ---")

    # Replace with your test files
    test_file_impostor = "C:/Users/shrey/PycharmProjects/sample_spunds/impo_sound.wav"
    test_file_target  = "C:/Users/shrey/PycharmProjects/sample_spunds/Record-002.wav"

    VERIFICATION_THRESHOLD = 0.65

    # Test 1: Target Speaker (You)
    print("\n[TEST] Your own voice sample:")
    score_target = verify_speaker(test_file_target)
    if score_target is not None:
        print(f"Similarity score: {score_target:.4f}")
        if score_target > VERIFICATION_THRESHOLD:
            print(f" ACCEPTED (Score > {VERIFICATION_THRESHOLD})")
        else:
            print(f" REJECTED (Score <= {VERIFICATION_THRESHOLD})")

    # Test 2: Impostor Speaker
    print("\n[TEST] Impostor sample:")
    score_impostor = verify_speaker(test_file_impostor)
    if score_impostor is not None:
        print(f"Similarity score: {score_impostor:.4f}")
        if score_impostor > VERIFICATION_THRESHOLD:
            print(f" FALSE ACCEPTANCE (Score > {VERIFICATION_THRESHOLD})")
        else:
            print(f" REJECTED (Score <= {VERIFICATION_THRESHOLD})")
