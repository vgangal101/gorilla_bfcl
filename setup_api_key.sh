#!/bin/bash
# Setup script for GPT-4o API key
# Run this once to create the API key file

# ============================================================================
# Configuration
# ============================================================================

# Where to store the API key file (keep it private)
API_KEY_FILE="${HOME}/.ssh/gpt4o_api_key.sh"

# ============================================================================
# Main Script
# ============================================================================

echo "========================================================================"
echo "Setting up GPT-4o API Key"
echo "========================================================================"
echo ""
echo "This script will help you set up your OpenAI API key securely."
echo ""
echo "API key file will be stored at: $API_KEY_FILE"
echo ""

# Check if .ssh directory exists
if [ ! -d "${HOME}/.ssh" ]; then
    echo "Creating ~/.ssh directory..."
    mkdir -p "${HOME}/.ssh"
    chmod 700 "${HOME}/.ssh"
fi

# Check if API key file already exists
if [ -f "$API_KEY_FILE" ]; then
    echo "WARNING: API key file already exists at $API_KEY_FILE"
    read -p "Do you want to overwrite it? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping setup"
        exit 0
    fi
fi

# Prompt for API key
echo ""
read -sp "Enter your OpenAI API key (input will be hidden): " OPENAI_API_KEY
echo ""

# Validate API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: API key cannot be empty"
    exit 1
fi

# Create the API key file
cat > "$API_KEY_FILE" << EOF
#!/bin/bash
# OpenAI API Key Configuration
# Generated: $(date)

export OPENAI_API_KEY="$OPENAI_API_KEY"
EOF

# Secure file permissions (read/write for owner only)
chmod 600 "$API_KEY_FILE"

echo ""
echo "✓ API key file created successfully"
echo "  Location: $API_KEY_FILE"
echo "  Permissions: 600 (read/write for owner only)"
echo ""
echo "You can now run the SLURM script:"
echo "  sbatch evaluate_gpt4o_mini.slurm"
echo ""
