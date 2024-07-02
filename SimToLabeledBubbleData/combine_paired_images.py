import cv2
import os
import argparse

def combine_paired_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if '_real.png' in filename:
            # Construct the corresponding fake image filename
            base_name = filename.replace('_real.png', '')
            real_image_path = os.path.join(input_dir, f'{base_name}_real.png')
            fake_image_path = os.path.join(input_dir, f'{base_name}_fake.png')
            
            # Check if both real and fake images exist
            if os.path.exists(real_image_path) and os.path.exists(fake_image_path):
                real_image = cv2.imread(real_image_path)
                fake_image = cv2.imread(fake_image_path)
                
                combined_image = cv2.hconcat([real_image, fake_image])
                
                combined_image_name = f'{base_name}.png'
                save_path = os.path.join(output_dir, combined_image_name)
                cv2.imwrite(save_path, combined_image)
                print(f'Saved: {save_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine paired images side by side.')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input directory containing paired images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory to save combined images.')
    args = parser.parse_args()

    combine_paired_images(args.input_dir, args.output_dir)
