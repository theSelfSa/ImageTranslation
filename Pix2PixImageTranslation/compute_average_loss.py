import pandas as pd
import argparse
import matplotlib.pyplot as plt

def compute_averages(input_file, output_file):
    df = pd.read_csv(input_file)
    # Drop the 'run' column if it exists
    if 'run' in df.columns:
        df = df.drop(columns=['run'])
    # Group by 'epoch' and 'total_iters' and compute the mean for each group
    df_avg = df.groupby(['epoch', 'total_iters']).mean().reset_index()
    df_avg.to_csv(output_file, index=False)
    return df_avg

def plot_losses(df_train, df_val, column, output_file, title):
    plt.figure()
    plt.plot(df_train['total_iters'], df_train[column], label=f'Train Loss', marker='o')
    plt.plot(df_val['total_iters'], df_val[column], label=f'Val Loss', marker='o')
    plt.xlabel('Iteration')
    plt.ylabel(f'Loss')
    plt.title(title)
    plt.legend()
    plt.savefig(output_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_losses', required=True, help='Path to all_training_losses.csv')
    parser.add_argument('--validation_losses', required=True, help='Path to all_validation_losses.csv')
    parser.add_argument('--output_dir', required=True, help='Directory to save the averaged CSV files and plots')
    args = parser.parse_args()

    avg_training_losses = compute_averages(args.training_losses, f"{args.output_dir}/avg_training_loss.csv")
    avg_validation_losses = compute_averages(args.validation_losses, f"{args.output_dir}/avg_validation_loss.csv")

    plot_losses(avg_training_losses, avg_validation_losses, 'G', f"{args.output_dir}/generator_loss.png", 'Generator Loss')
    plot_losses(avg_training_losses, avg_validation_losses, 'D', f"{args.output_dir}/discriminator_loss.png", 'Discriminator Loss')

if __name__ == "__main__":
    main()
