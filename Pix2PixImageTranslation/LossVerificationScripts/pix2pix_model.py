import torch
from .base_model import BaseModel
from . import networks
import pandas as pd
import os


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the original GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer.
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1.
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_GAN', 'G_L1', 'G','D_real', 'D_fake','D']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self, is_test=False):
        """Run forward pass; called by both functions <optimize_parameters> and <test>.
        
        Parameters:
            is_test (bool): Indicates whether it's a test pass or not.
        """
        with torch.set_grad_enabled(not is_test):
            self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        print('Hello D')
        print(pred_fake.shape)
        print(type(pred_fake))
        df = pd.DataFrame(pred_fake.squeeze().detach().cpu().numpy())
        image_path =self.get_image_paths()
        image_name = [os.path.splitext(os.path.basename(path))[0] for path in image_path][0]
        df.to_csv(f"Exp/train/pred_fake_{image_name}_{self.opt.run_number}.csv",index=False, header=False)

        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        print(self.loss_D_fake)
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        print(pred_real.shape)
        print(type(pred_real))
        
        df = pd.DataFrame(pred_real.squeeze().detach().cpu().numpy())
        df.to_csv(f"Exp/train/pred_real_{image_name}_{self.opt.run_number}.csv",index=False, header=False)

        self.loss_D_real = self.criterionGAN(pred_real, True)

        print(self.loss_D_real)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        # print(self.loss_D)
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        print('Hello G')
        print(self.real_B.shape)
        print(type(self.real_B))
        print(self.visual_names)
        image_path =self.get_image_paths()
        image_name = [os.path.splitext(os.path.basename(path))[0] for path in image_path][0]
        df = pd.DataFrame(self.real_B.squeeze().detach().cpu().numpy().reshape(3, -1).T)
        df.to_csv(f"Exp/train/real_B_{image_name}_{self.opt.run_number}.csv",index=False, header=False)

        print(self.fake_B.shape)
        print(type(self.fake_B))
        df = pd.DataFrame(self.fake_B.squeeze().detach().cpu().numpy().reshape(3, -1).T)
        df.to_csv(f"Exp/train/fake_B_{image_name}_{self.opt.run_number}.csv",index=False, header=False)

        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        print(self.criterionL1(self.fake_B, self.real_B))
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   
        self.set_requires_grad(self.netD, True)  
        self.optimizer_D.zero_grad()     
        self.backward_D()                
        self.optimizer_D.step()          
        self.set_requires_grad(self.netD, False)  
        self.optimizer_G.zero_grad()        
        self.backward_G()                   
        self.optimizer_G.step()             

    def compute_test_losses(self,total_iters):
        """Compute test losses without backpropagation"""
        with torch.no_grad():
            self.forward(is_test=True)
            
            # Calculate generator losses
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB)
            loss_G_GAN = self.criterionGAN(pred_fake, True)
            loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
            test_loss_G = loss_G_GAN + loss_G_L1
            
            # Calculate discriminator losses
            pred_fake_D = self.netD(fake_AB.detach())
            loss_D_fake = self.criterionGAN(pred_fake_D, False)
            
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            pred_real_D = self.netD(real_AB)
            loss_D_real = self.criterionGAN(pred_real_D, True)
            
            test_loss_D = (loss_D_fake + loss_D_real) * 0.5

            df = pd.DataFrame(pred_fake_D.squeeze().detach().cpu().numpy())
            image_path =self.get_image_paths()
            image_name = [os.path.splitext(os.path.basename(path))[0] for path in image_path][0]
            df.to_csv(f"Exp/val/pred_fake_{total_iters}_{self.opt.run_number}.csv",index=False, header=False)

            df = pd.DataFrame(pred_real_D.squeeze().detach().cpu().numpy())
            df.to_csv(f"Exp/val/pred_real_{total_iters}_{self.opt.run_number}.csv",index=False, header=False)

            df = pd.DataFrame(self.real_B.squeeze().detach().cpu().numpy().reshape(3, -1).T)
            df.to_csv(f"Exp/val/real_B_{total_iters}_{self.opt.run_number}.csv",index=False, header=False)

            df = pd.DataFrame(self.fake_B.squeeze().detach().cpu().numpy().reshape(3, -1).T)
            df.to_csv(f"Exp/val/fake_B_{total_iters}_{self.opt.run_number}.csv",index=False, header=False)
        
        return {
            'G_GAN': loss_G_GAN.item(), 
            'G_L1': loss_G_L1.item(), 
            'G': test_loss_G.item(),
            'D_fake': loss_D_fake.item(),
            'D_real': loss_D_real.item(),
            'D': test_loss_D.item()
        }


