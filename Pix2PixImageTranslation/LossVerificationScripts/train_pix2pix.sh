#!/bin/bash

# Function to display help message
usage() {
    echo "Usage: $0 --dataroot DATASET_PATH --name EXPERIMENT_NAME --n_epochs NUM_EPOCHS --n_epochs_decay NUM_EPOCHS_DECAY --runs RUNS"
    exit 1
}

# Default values
dataroot=""
name=""
n_epochs=""
n_epochs_decay=""
runs=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataroot) dataroot="$2"; shift ;;
        --name) name="$2"; shift ;;
        --n_epochs) n_epochs="$2"; shift ;;
        --n_epochs_decay) n_epochs_decay="$2"; shift ;;
        --runs) runs="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Verify that all required arguments are provided
if [ -z "$dataroot" ] || [ -z "$name" ] || [ -z "$n_epochs" ] || [ -z "$n_epochs_decay" ] || [ -z "$runs" ]; then
    echo "Error: Missing required arguments."
    usage
fi

# Create or empty the final loss files
loss_data_dir="checkpoints/${name}/loss_data"
mkdir -p "$loss_data_dir"  # create directory if it doesn't exist
chmod 755 "$loss_data_dir" # set the appropriate permissions
echo "run,epoch,total_iters,G_GAN,G_L1,G,D_fake,D_real,D" > "${loss_data_dir}/all_training_losses.csv"
echo "run,epoch,total_iters,G_GAN,G_L1,G,D_fake,D_real,D" > "${loss_data_dir}/all_validation_losses.csv"

# Run the training script for the specified number of runs
for ((run=1; run<=$runs; run++))
do
    echo "Running training run $run..."
    python train.py --dataroot "$dataroot" --name "$name" --model pix2pix --direction AtoB --gan_mode lsgan --print_freq 1 --n_epochs "$n_epochs" --n_epochs_decay "$n_epochs_decay" --run_number "$run"

    train_loss_file="${loss_data_dir}/train_losses_run${run}.csv"
    test_loss_file="${loss_data_dir}/test_losses_run${run}.csv"

    if [[ ! -f "$train_loss_file" ]]; then
        echo "Error: $train_loss_file not found."
        continue
    fi

    if [[ ! -f "$test_loss_file" ]]; then
        echo "Error: $test_loss_file not found."
        continue
    fi

    # Append the losses to the final files
    tail -n +2 "$train_loss_file" >> "${loss_data_dir}/all_training_losses.csv"
    tail -n +2 "$test_loss_file" >> "${loss_data_dir}/all_validation_losses.csv"
done

echo "Training completed for $runs runs."

# Call the Python script to compute averages and generate plots
python compute_average_loss.py --training_losses "${loss_data_dir}/all_training_losses.csv" --validation_losses "${loss_data_dir}/all_validation_losses.csv" --output_dir "$loss_data_dir"

echo "Averages computed and saved to avg_training_losses.csv and avg_validation_losses.csv."
echo "Plots generated and saved to avg_training_losses.png and avg_validation_losses.png."
