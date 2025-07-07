#!/bin/bash
# Quick start script for gender classification training

echo "=== Barkopedia Gender Classification Training ==="
echo

# Check if we're in the right directory
if [ ! -f "train_gender_ast.py" ]; then
    echo "Error: train_gender_ast.py not found. Please run this script from the TASK2 directory."
    exit 1
fi

# Default parameters
EPOCHS=10
BATCH_SIZE=16
LEARNING_RATE=1e-4
DEBUG_MODE=""
HF_TOKEN=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            DEBUG_MODE="--debug"
            EPOCHS=1
            BATCH_SIZE=4
            echo "Debug mode enabled (1 epoch, batch size 4)"
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --hf_token)
            HF_TOKEN="--hf_token $2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --debug              Run in debug mode (1 epoch, small batch)"
            echo "  --epochs EPOCHS      Number of training epochs (default: 10)"
            echo "  --batch_size SIZE    Batch size (default: 16)"
            echo "  --learning_rate LR   Learning rate (default: 1e-4)"
            echo "  --hf_token TOKEN     HuggingFace token for dataset access"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./results/gender_classification_${TIMESTAMP}"

echo "Training Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE" 
echo "  Learning Rate: $LEARNING_RATE"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Debug Mode: $([ -n "$DEBUG_MODE" ] && echo "Yes" || echo "No")"
echo

# Run training
python train_gender_ast.py \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --save_dir "$OUTPUT_DIR" \
    --apply_cleaning \
    --max_duration 5.0 \
    $DEBUG_MODE \
    $HF_TOKEN

echo
echo "Training completed! Results saved to: $OUTPUT_DIR"
