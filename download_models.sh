#!/bin/bash

# Define the target directory
MODEL_DIR="models/SenseVoiceSmall"
BASE_URL="https://huggingface.co/FunAudioLLM/SenseVoiceSmall/resolve/main"

# Create directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# List of files required for the engine
FILES=(
    "model.pt"
    "config.yaml"
    "am.mvn"
    "chn_jpn_yue_eng_ko_spectok.bpe.model"
    "configuration.json"
)

echo "--- ANABELLE Model Downloader ---"

for FILE in "${FILES[@]}"
do
    TARGET_PATH="$MODEL_DIR/$FILE"
    
    if [ -f "$TARGET_PATH" ]; then
        echo "[EXISTS] $FILE already present. Skipping..."
    else
        echo "[MISSING] $FILE not found. Starting download..."
        # -L follows redirects
        # -C - resumes partial downloads (very useful for the 1GB model.pt)
        # -o specifies the output path
        curl -L -C - "$BASE_URL/$FILE?download=true" -o "$TARGET_PATH"
        
        if [ $? -eq 0 ]; then
            echo "[SUCCESS] Finished downloading $FILE"
        else
            echo "[ERROR] Failed to download $FILE. Check your connection."
        fi
    fi
done

echo "--- Download Sync Complete ---"