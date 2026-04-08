#!/bin/bash
# Sync only the 6065 MRI scans used by experiments to a remote server
# Usage: bash sync_mri_to_server.sh user@server
set -e

REMOTE="${1:?Usage: bash sync_mri_to_server.sh user@server}"
REMOTE_MRI_DIR="/data/mri-alzheimer"

# CSV with scan paths
ALL_CSV="/home/tanguy/medical/alzheimer/experiments/multimodal_fusion/data/combined_trajectory/all.csv"

# 1. Build file list (resolve paths to /data/mri-alzheimer/)
FILELIST=$(mktemp)
cut -d',' -f2 "$ALL_CSV" | tail -n +2 | \
    sed 's|/home/tanguy/medical/|/data/mri-alzheimer/|' | \
    sort -u > "$FILELIST"

echo "Files to transfer: $(wc -l < "$FILELIST")"

# 2. Create remote directory structure
echo "Creating remote directories..."
cut -d',' -f2 "$ALL_CSV" | tail -n +2 | \
    sed 's|/home/tanguy/medical/|/data/mri-alzheimer/|' | \
    xargs -I{} dirname {} | sort -u | \
    ssh "$REMOTE" "sudo mkdir -p \$(cat -) && sudo chown -R \$USER:\$USER $REMOTE_MRI_DIR"

# 3. Rsync only the needed files
echo "Starting rsync (~39 Go)..."
rsync -avhP --files-from="$FILELIST" / "$REMOTE:/"

rm "$FILELIST"

echo ""
echo "Done! Synced $(wc -l < "$FILELIST" 2>/dev/null || echo 6065) scans to $REMOTE"
