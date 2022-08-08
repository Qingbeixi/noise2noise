import os
import json
import torch
import torch.nn as nn

from datetime import datetime
from torch.optim import Adam, lr_scheduler, SGD

from .others.noise2noise_model import UNet
from .others.dataset import load_dataset
from .others.help_function import progress_bar, time_elapsed_since, show_on_epoch_end, show_on_report, plot_per_epoch, psnr, AvgMeter

from pathlib import Path


class Model(object):
    """we implement the noise2noise model according to the thesis of Jaakko Lehtinen, the model is unet which is the same in that thesis."""

    def __init__(self):
        """Initializes model."""
        """Here is the data initialize, we can change the parameters here. Since paser args has conflicts with the test.py"""

        self.trainable = True
        self.train_dir = 'others/data/train_data.pkl'
        self.valid_dir = 'others/data/val_data.pkl'
        self.noise_type = 'origin'
        self.ckpt_overwrite = True
        self.report_interval = 100 
        self.learning_rate = 0.001
        self.adam = [0.9, 0.99, 1e-8] # the parameter of adam
        self.IF_SGD=False
        self.batch_size = 50
        self.nb_epochs = 10
        self.loss = 'l2'
        self.plot_stats = True
        self.mini_train = True
        self.pre_train = False
        self.targets = False
        self._compile()

    def _compile(self):
        """Compiles model (architecture, loss function, optimizers, etc.)."""

        self.model = UNet(in_channels=3)

        # Set optimizer and loss, if in training mode
        if self.trainable:
            self.optim = Adam(self.model.parameters(),
                              lr=self.learning_rate,
                              betas=self.adam[:2],
                              eps=self.adam[2])
            

            if self.IF_SGD == True:
                self.optim = SGD(self.model.parameters(),
                                lr=self.learning_rate,
                                momentum=0.98, weight_decay=0.01
                                )

            # Learning rate adjustment
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim,
                patience=self.nb_epochs/4, factor=0.5, verbose=True)

            # Loss function
            if self.loss == 'l2':
                self.loss = nn.MSELoss()
            else:
                self.loss = nn.L1Loss()

        # CUDA support
        self.use_cuda = torch.cuda.is_available() # and self.cuda
        if self.use_cuda:
            self.model = self.model.cuda()
            if self.trainable:
                self.loss = self.loss.cuda()


    def load_pretrained_model(self):

        """Loads model from checkpoint file."""

        ckpt_fname = 'bestmodel.pth'

        print('Loading checkpoint from: {}'.format(ckpt_fname))

        model_path = Path(__file__).parent / "bestmodel.pth"
        # model = torch.load(model_path)

        if self.use_cuda:
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))

    def _print_params(self):
        """Formats parameters to print when training."""

        print('Training parameters: ')
        self.cuda = self.use_cuda
        param_dict = vars(self.p)
        pretty = lambda x: x.replace('_', ' ').capitalize()
        print('\n'.join('  {} = {}'.format(pretty(k), str(v)) for k, v in param_dict.items()))
        print()


    def save_model(self, epoch, stats, first=False):
        """Saves model to files, and can be used for plotting"""

        # Create directory for model checkpoints, if nonexistent
        if first:
            if self.targets:
                ckpt_dir_name = f'{datetime.now():{self.noise_type}-clean-%H%M}'
            else:
                ckpt_dir_name = f'{datetime.now():{self.noise_type}-%H%M}'
            if self.ckpt_overwrite:
                if self.targets:
                    ckpt_dir_name = f'{self.noise_type}-clean'
                else:
                    ckpt_dir_name = 'model' #self.noise_type

            self.ckpt_dir = os.path.join(self.ckpt_save_path, ckpt_dir_name)
            if not os.path.isdir(self.ckpt_save_path):
                os.mkdir(self.ckpt_save_path)
            if not os.path.isdir(self.ckpt_dir):
                os.mkdir(self.ckpt_dir)

        # Save checkpoint dictionary
        if self.ckpt_overwrite:
            fname_unet = '{}/origin_train valnoise{}.pth'.format(self.ckpt_dir,self.noise_type)
        else:
            valid_loss = stats['valid_loss'][epoch]
            fname_unet = '{}/n2n-epoch{}-{:>1.5f}.pth'.format(self.ckpt_dir, epoch + 1, valid_loss)
        print('Saving checkpoint to: {}\n'.format(fname_unet))
        torch.save(self.model.state_dict(), fname_unet)

        # Save stats to JSON
        fname_dict = '{}/origin_train valnoise{}.json'.format(self.ckpt_dir,self.noise_type)
        with open(fname_dict, 'w') as fp:
            json.dump(stats, fp, indent=2)

    def _on_epoch_end(self, stats, train_loss, epoch, epoch_start, valid_loader):
        """Tracks and saves starts after each epoch."""

        # Evaluate model on validation set
        print('\rTesting model on validation set... ', end='')
        epoch_time = time_elapsed_since(epoch_start)[0]
        valid_loss, valid_time, valid_psnr = self.eval(valid_loader)
        show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr)

        # Decrease learning rate if plateau
        self.scheduler.step(valid_loss)

        # Save checkpoint
        stats['train_loss'].append(train_loss)
        stats['valid_loss'].append(valid_loss)
        stats['valid_psnr'].append(valid_psnr)
        self.save_model(epoch, stats, epoch == 0)

        # Plot stats
        if self.plot_stats:
            loss_str = f'{self.loss.upper()} loss'
            plot_per_epoch(self.ckpt_dir, 'Valid loss', stats['valid_loss'], loss_str)
            plot_per_epoch(self.ckpt_dir, 'Valid PSNR', stats['valid_psnr'], 'PSNR (dB)')
    
    def predict(self, test_tensor):
        """predict the results"""
        test_tensor = test_tensor.float()
        with torch.no_grad():
            self.model.train(False)
            # self.load_pretrained_model()
            h = test_tensor.size(2)
            w = test_tensor.size(3)
            c = test_tensor.size(1)
            test_datas = test_tensor
            test_datas = test_datas.reshape(-1,c,w,h)
            for id in range(len(test_datas)):
                test_data = test_datas[id]
                if self.use_cuda:
                    test_data = test_data.reshape(1,c,w,h)
                    test_data = test_data.cuda()
                else:
                    test_data = test_data.reshape(1,c,w,h)
                if id == 0:
                    prediction = self.model(test_data)
                    if self.use_cuda:
                        prediction.cuda()
                else:
                    predict_new = self.model(test_data)
                    if self.use_cuda:
                        predict_new.cuda()
                    prediction = torch.cat([prediction,predict_new],dim=0)
                    if self.use_cuda:
                        prediction.cuda()
        
        return torch.clamp(prediction,0,255)
        

    def eval(self, valid_loader):
        """Evaluates denoiser on validation set."""
        with torch.no_grad():
            self.model.train(False)

            valid_start = datetime.now()
            loss_meter = AvgMeter()
            psnr_meter = AvgMeter()

            for batch_idx, (source, target) in enumerate(valid_loader):
                if self.use_cuda:
                    source = source.cuda()
                    target = target.cuda()

                # Denoise
                source_denoised = self.model(source)

                # Update loss
                loss = self.loss(source_denoised, target)
                loss_meter.update(loss.item())

                # Compute PSRN
                for i in range(self.batch_size):
                    source_denoised = source_denoised.cpu()
                    target = target.cpu()
                    psnr_meter.update(psnr(source_denoised[i], target[i]).item())

            valid_loss = loss_meter.avg
            valid_time = time_elapsed_since(valid_start)[0]
            psnr_avg = psnr_meter.avg

        return valid_loss, valid_time, psnr_avg

           
    def train(self, train_input, train_target, num_epochs):
        """Trains model only on training set."""

        self.model.train(True)
        train_input = train_input.float()
        train_target = train_target.float()
        print('ok')

        # load pretrained model
        if self.pre_train == True:
            self.load_pretrained_model()

        # self._print_params()
        """Here is to monitor the training process and get psnr, while we ignore this part for test.py"""
        # num_batches = int(train_input.size(0)/self.batch_size)
        # assert num_batches % self.report_interval == 0, 'Report interval must divide total number of batches'

        # Dictionaries of tracked stats
        stats = {'noise_type': self.noise_type,
                 'train_loss': [],
                 'valid_loss': [],
                 'valid_psnr': []}

        # Main training loop
        train_start = datetime.now()
        self.nb_epochs = num_epochs
        for epoch in range(self.nb_epochs):
            # print('EPOCH {:d} / {:d}'.format(epoch + 1, self.nb_epochs))

            # Some stats trackers
            epoch_start = datetime.now()
            train_loss_meter = AvgMeter()
            loss_meter = AvgMeter()
            time_meter = AvgMeter()

            # Minibatch SGD
            for batch_idx,b in enumerate(range(0, train_input.size(0), self.batch_size)):
                source = train_input.narrow(0, b, self.batch_size)
                target = train_target.narrow(0, b, self.batch_size)
                batch_start = datetime.now()

                # progress_bar(batch_idx, num_batches, self.report_interval, loss_meter.val)

                if self.use_cuda:
                    source = source.cuda()
                    target = target.cuda()

                # Denoise image
                source_denoised = self.model(source)

                loss = self.loss(source_denoised, target)
                loss_meter.update(loss.item())

                # Zero gradients, perform a backward pass, and update the weights
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # Report/update statistics
                time_meter.update(time_elapsed_since(batch_start)[1])
                if (batch_idx + 1) % self.report_interval == 0 and batch_idx:
                    # show_on_report(batch_idx, num_batches, loss_meter.avg, time_meter.avg)
                    train_loss_meter.update(loss_meter.avg)
                    loss_meter.reset()
                    time_meter.reset()

            # Epoch end, save and reset tracker
            # self._on_epoch_end(stats, train_loss_meter.avg, epoch, epoch_start, valid_loader)
            train_loss_meter.reset()

        train_elapsed = time_elapsed_since(train_start)[0]
        # print('Training done! Total elapsed time: {}\n'.format(train_elapsed))






###########################################    Train    #########################################
# def parse_args():
#     """Command-line argument parser for training."""

#     # New parser
#     parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')

#     # Data parameters
#     # parser.add_argument('-t', '--train-dir', help='training set path', default='others/data/train_data.pkl')
#     # parser.add_argument('-v', '--valid-dir', help='test set path', default='others/data/val_data.pkl')
#     # parser.add_argument('-n', '--noise-type', help='noise_type', default='origin', type=str)
#     # # parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='bestmodel')
#     # parser.add_argument('--ckpt-overwrite', help='overwrite model checkpoint on save', action='store_true',default=True)
#     # parser.add_argument('--report-interval', help='batch report interval', default=100, type=int)
#     # # parser.add_argument('--loading-pth', help='the direction for loading model', default='bestmodel.pth', type=str)

#     # # Training hyperparameters
#     # parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.001, type=float)
#     # parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
#     # parser.add_argument('-s','--IF-SGD',help='whether to use sgd as a replace of adam', default=False)
#     parser.add_argument('-b', '--batch-size', help='minibatch-size', default=32, type=int)
#     # parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=10, type=int)
#     # parser.add_argument('-l', '--loss', help='loss function', choices=['l1', 'l2'], default='l2', type=str)
#     # parser.add_argument('--cuda', help='use cuda', action='store_true')
#     # parser.add_argument('--plot-stats', help='plot stats after every epoch', action='store_true')
#     parser.add_argument('-mini','--mini-train', help='load less train pictures', default=True)
#     # parser.add_argument('-pre','--pre-train', help='load pre-trained model', default=False)

#     # Corruption parameters
#     # parser.add_argument('--targets', help='use clean targets for training', action='store_true')

#     return parser.parse_args()


if __name__ == '__main__':

    # Parse training parameters
    # params = parse_args()

    """Since we need to training on test.py, we cannot use the load function on our own, which return a Dataloader"""
    # Train/valid datasets
    # train_loader = load_dataset(r'C:\Users\qingjun\Desktop\project\Miniproject_1\others\data\train_data.pkl',mini_train=True, data_type='train')
    # valid_loader = load_dataset(r'C:\Users\qingjun\Desktop\project\Miniproject_1\others\data\val_data.pkl', mini_train=True,data_type='val')
    # print(train_loader)
    # Initialize model and train

    source, target = torch.load(r'C:\Users\qingjun\Desktop\project\Miniproject_1\others\data\train_data.pkl')

    n2n = Model()

    # Train the model
    epochs = 5
    n2n.train(source, target,epochs)


    # Get the prediction
    # test_tensor, target = torch.load(params.valid_dir)
    # prediction = n2n.predict(test_tensor[:20])