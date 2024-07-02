"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import csv
import sys
import os

if __name__ == '__main__':
    opt = TrainOptions().parse() # get training options
    
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    opt.phase = "test"
    opt.serial_batches = True
    test_dataset = create_dataset(opt)
    opt.phase = "train"
    opt.serial_batches = False
    test_dataset_size = len(test_dataset)
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    print('The number of Testing images = %d' % test_dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    
    # Initialize CSV file
    loss_data_dir = os.path.join(opt.checkpoints_dir, opt.name, 'loss_data')
    os.makedirs(loss_data_dir, exist_ok=True) 
    test_csv_file = os.path.join(loss_data_dir, f'test_losses_run{opt.run_number}.csv')
    with open(test_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['run' ,'epoch', 'total_iters', 'G_GAN', 'G_L1', 'G', 'D_fake', 'D_real', 'D'])
        
    train_csv_file = os.path.join(loss_data_dir, f'train_losses_run{opt.run_number}.csv')
    with open(train_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['run' ,'epoch', 'total_iters', 'G_GAN', 'G_L1', 'G', 'D_fake', 'D_real', 'D'])

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()  
        # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
                

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
                # print(f'losses at epoch {epoch} and iter {epoch_iter}: {losses}')
                with open(train_csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        opt.run_number, epoch, total_iters,
                        losses['G_GAN'], losses['G_L1'], losses['G'],
                        losses['D_fake'], losses['D_real'], losses['D']
                    ])
                    
                # Set the model to evaluation mode
                model.eval()
                test_losses = []
                # Calculate test loss after each epoch
                # List to store test losses
                for i, data in enumerate(test_dataset):
                    model.set_input(data)  # Set input for the model
                    test_loss = model.compute_test_losses()  # Compute test losses
                    test_losses.append(test_loss)

                # Optionally, average the test losses and print or log them
                avg_test_loss = {k: sum(d[k] for d in test_losses) / len(test_losses) for k in test_losses[0]}
                # print(f'Average test losses at epoch {epoch}: {avg_test_loss}')

                with open(test_csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        opt.run_number, epoch, total_iters,
                        avg_test_loss['G_GAN'], avg_test_loss['G_L1'], avg_test_loss['G'],
                        avg_test_loss['D_fake'], avg_test_loss['D_real'], avg_test_loss['D']
                    ])
                    
                # Revert the model back to training mode
                model.train()

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (run %d,epoch %d, total_iters %d)' % (opt.run_number, epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
                
            iter_data_time = time.time()


        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            save_suffix = 'run_%d_epoch_%d' % (opt.run_number,epoch)
            model.save_networks(save_suffix)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

