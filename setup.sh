#!/bin/bash

echo "🚀 Setting up Depth Surge 3D..."

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for Python 3
if ! command_exists python3; then
    echo "❌ Error: Python 3 is not installed. Please install Python 3.9 or later."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python $required_version or later is required. Found: $python_version"
    exit 1
fi

echo "✅ Python $python_version found"

# Check for uv package manager
if command_exists uv; then
    echo "✅ uv package manager found - using for fast setup"
    use_uv=true
else
    echo "📦 uv not found - using pip (consider installing uv for faster setup: curl -LsSf https://astral.sh/uv/install.sh | sh)"
    use_uv=false
fi

# Setup virtual environment and dependencies
if [ "$use_uv" = true ]; then
    echo "📦 Setting up environment with uv..."
    uv sync
else
    echo "📦 Setting up environment with pip..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# Download Video-Depth-Anything repository if not present
if [ ! -d "video_depth_anything_repo" ]; then
    echo "🔄 Downloading Video-Depth-Anything repository..."
    git clone https://github.com/DepthAnything/Video-Depth-Anything.git video_depth_anything_repo
    if [ $? -eq 0 ]; then
        echo "✅ Video-Depth-Anything repository downloaded successfully"
    else
        echo "❌ Failed to download Video-Depth-Anything repository"
        exit 1
    fi
else
    echo "✅ Video-Depth-Anything repository already exists"
fi

# Create models directory if not present
mkdir -p models

# Download Video-Depth-Anything-Large model if not present
model_path="models/Video-Depth-Anything-Large/video_depth_anything_vitl.pth"
if [ ! -f "$model_path" ]; then
    echo "🔄 Downloading Video-Depth-Anything-Large model (~1.3GB, this may take a while)..."
    mkdir -p "models/Video-Depth-Anything-Large"

    # Try using curl first, then wget as fallback
    if command_exists curl; then
        curl -L "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth" \
             -o "$model_path" --progress-bar
    elif command_exists wget; then
        wget "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth" \
             -O "$model_path" --progress=bar
    else
        echo "❌ Error: Neither curl nor wget found. Please install one of them to download the model."
        echo "   You can manually download the model from:"
        echo "   https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth"
        echo "   and place it at: $model_path"
        exit 1
    fi

    if [ -f "$model_path" ]; then
        echo "✅ Model downloaded successfully"
        echo "📊 Model size: $(du -h "$model_path" | cut -f1)"
    else
        echo "❌ Failed to download model"
        exit 1
    fi
else
    echo "✅ Video-Depth-Anything-Large model already exists"
    echo "📊 Model size: $(du -h "$model_path" | cut -f1)"
fi

# Check if FFmpeg is available
if command_exists ffmpeg; then
    echo "✅ FFmpeg found"
else
    echo "⚠️  FFmpeg not found - install it for video processing:"
    if command_exists apt-get; then
        echo "   sudo apt-get install ffmpeg"
    elif command_exists brew; then
        echo "   brew install ffmpeg"
    elif command_exists dnf; then
        echo "   sudo dnf install ffmpeg"
    else
        echo "   Please install FFmpeg for your system"
    fi
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📖 Quick start:"
echo "   # Start the web interface (recommended for most users):"
echo "   ./run_ui.sh"
echo ""
echo "   # Or use command line directly:"
echo "   python depth_surge_3d.py --help"
echo "   python depth_surge_3d.py input_video.mp4"
echo ""
echo "🔧 Available models:"
echo "   - Video-Depth-Anything-Large (335M params) - ✅ Downloaded"
echo ""
echo "📦 Additional models available (optional):"
echo "   ./download_models.sh small  # Video-Depth-Anything-Small (24.8M params, fastest)"
echo "   ./download_models.sh base   # Video-Depth-Anything-Base (97.5M params, balanced)"
echo ""
echo "💡 Tips:"
echo "   - Use --vr-resolution 16x9-720p for quick tests"
echo "   - Use --processing-mode batch for faster processing"
echo "   - Try custom resolutions: --vr-resolution custom:1920x1080"
echo ""
echo "📁 Output will be saved to ./output/ directory"