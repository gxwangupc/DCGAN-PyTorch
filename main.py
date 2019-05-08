import torch            # This line is not necessary or line 5 is not necessary.
import argparse
import os
import random
import torch.utils.data  # This line is not necessary.
import torchvision.utils # 'import torchvision' is okay.
from Generator import Generator
from Discriminator import Discriminator


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', default='./data', help='path to the root of dataset')
parser.add_argument('--workers', type=int, default=2, help='number of worker threads for loading the data with Dataloader')
parser.add_argument('--batch_size', type=int, default=64, help='batch size used in training')
parser.add_argument('--img_size', type=int, default=64, help='the height / width of the input images used for training')
parser.add_argument('--nz', type=int, default=100, help='size of the latent vector z')
parser.add_argument('--ngf', type=int, default=64, help='size of feature maps in G')
parser.add_argument('--ndf', type=int, default=64, help='size of feature maps in D')
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to run')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate for training')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 hyperparameter for Adam optimizers')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs available')
parser.add_argument('--dev', type=int, default=1, help='which CUDA device to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--results', default='./results', help='folder to store images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
opt = parser.parse_args()
print(opt)

"""
Create a folder to store images and model checkpoints
"""
'''
try:
    os.makedirs(opt.results)
except OSError:
    pass
'''
if not os.path.exists(opt.results):
    os.mkdir(opt.results)

"""
Set random seed for reproducibility.
"""
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
#Sets the seed for generating random numbers. Returns a torch._C.Generator object.
torch.manual_seed(opt.manualSeed)

"""
Use GPUs if available.
"""
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
device = torch.device("cuda:0" if opt.cuda else "cpu")
torch.cuda.set_device(opt.dev)

"""
Create the dataset.
"""
'''
#torchvision.datasets.ImageFolder(root, transform=None, target_transform=None, \
 loader=<function default_loader>, is_valid_file=None)
 #root: Root directory path.
 #transform: transform (callable, optional): A function/transform that takes in \
   #an PIL image and returns a transformed version. 
'''
'''
torchvision.datasets: contains a lot of datasets
'''
'''
#torchvision.transforms.Compose(transforms): Composes several transforms together.
#torchvision.transforms.Resize(size, interpolation=2): Resize the input PIL Image to the given size.
 #interpolation: Desired interpolation. Default is PIL.Image.BILINEAR.
#torchvision.transforms.CenterCrop(size): Crops the given PIL Image at the center.\
#torchvision.transforms.ToTensor(): Convert a PIL Image or numpy.ndarray to tensor.
#torchvision.transforms.Normalize(mean, std, inplace=False):Normalize a tensor image with mean and standard deviation. \
 #Given mean: (M1,...,Mn) and std: (S1,..,Sn) for n channels, this transform will normalize each channel of the input.
 #mean: Sequence of means for each channel.
 #std : Sequence of standard deviations for each channel.
'''
if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = torchvision.datasets.ImageFolder(root=opt.dataroot,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.Resize(opt.img_size),
                                   torchvision.transforms.CenterCrop(opt.img_size),
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    nc=3
elif opt.dataset == 'lsun':
    dataset = torchvision.datasets.LSUN(root=opt.dataroot, classes=['bedroom_train'],
                        transform=torchvision.transforms.Compose([
                            torchvision.transforms.Resize(opt.img_size),
                            torchvision.transforms.CenterCrop(opt.img_size),
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    nc=3
elif opt.dataset == 'cifar10':
    dataset = torchvision.datasets.CIFAR10(root=opt.dataroot, download=True,
                           transform=torchvision.transforms.Compose([
                               torchvision.transforms.Resize(opt.img_size),
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    nc=3

elif opt.dataset == 'mnist':
    dataset = torchvision.datasets.MNIST(root=opt.dataroot, download=True,
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.Resize(opt.img_size),
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize((0.5,), (0.5,)),
                       ]))
    nc=1

elif opt.dataset == 'fake':
    dataset = torchvision.datasets.FakeData(image_size=(3, opt.img_size, opt.img_size),
                            transform=torchvision.transforms.ToTensor())
    nc=3

assert dataset
'''
Create the dataloader.
#torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0,\
 collate_fn=<function default_collate>, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)
 #batch_size: how many samples per batch to load.
 #shuffle: set to True to have the data reshuffled at every epoch.
 #num_workers: how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
'''
dataset = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=int(opt.workers))

"""
Custom weights initialization called on netG and netD.
All model weights shall be randomly initialized from a Normal distribution with mean=0, stdev=0.02. 
Note We set bias=False in both Conv2d and ConvTranspose2d.
"""
def weights_init(input):
    classname = input.__class__.__name__
    if classname.find('Conv') != -1:
        input.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        input.weight.data.normal_(1.0, 0.02)
        input.bias.data.fill_(0)

'''
Create a generator and apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
'''
netG = Generator(nz=opt.nz, ngf=opt.ngf, nc=nc, ngpu=opt.ngpu).to(device)
netG.apply(weights_init)
'''
Load the trained netG to continue training if it exists.
'''
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

'''
Create a discriminator and apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
'''
netD = Discriminator(nc=nc, ndf=opt.ndf, ngpu=opt.ngpu).to(device)
netD.apply(weights_init)
'''
Load the trained netD to continue training if it exists.
'''
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

"""
A Binary Cross Entropy loss is used and two Adam optimizers are responsible for updating 
 netG and netD, respectively. 
"""
'''
#torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
  #Creates a criterion that measures the Binary Cross Entropy between the target and the output.
'''
'''
#torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
'''
loss = torch.nn.BCELoss()
optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

#Create batch of latent vectors that we will use to visualize
 #the progression of the generator.
 #torch.randn(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
fixed_noise = torch.randn(opt.batch_size, opt.nz, 1, 1, device=device)
#Establish convention for real and fake labels during training.
real_label = 1
fake_label = 0

"""
Training.
"""
#This line may increase the training speed a bit.
torch.backends.cudnn.benchmark = True

print("Starting Training Loop...")
for epoch in range(opt.nepoch):
    for i, data in enumerate(dataset, 0):
        """
        (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        """
        '''
        Train with real batch.
        '''
        netD.zero_grad()
        #Format batch
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)
        #Forward propagate real batch through D.
        output = netD(real_cpu)
        #Calculate loss on real batch.
        errD_real = loss(output, label)
        #Calculate gradients for D in the backward propagation.
        errD_real.backward()
        D_x = output.mean().item()

        '''
        Train with fake batch.
        '''
        #Sample batch of latent vectors.
        noise = torch.randn(batch_size, opt.nz, 1, 1, device=device)
        #Generate fake image batch with G.
        fake = netG(noise)
        label.fill_(fake_label)

        #Classify all fake batch with D.
        #.detach() is a safer way for the exclusion of subgraphs from gradient computation.
        output = netD(fake.detach())
        #Calculate D's loss on fake batch.
        errD_fake = loss(output, label)
        #Calculate the gradients for fake batch.
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        #Get D's total loss by adding the gradients from the real and fake batches.
        errD = errD_real + errD_fake
        #Update D.
        optimizerD.step()

        """
        (2) Update G network: maximize log(D(G(z)))
        """
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for Generator cost
        #Since we just updated D, perform another forward propagation of fake batch through D
        output = netD(fake)
        #Calculate G's loss based on this output.
        errG = loss(output, label)
        #Calculate the gradients for G.
        errG.backward()
        D_G_z2 = output.mean().item()
        #Update G.
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.nepoch, i, len(dataset),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        if i % 100 == 0:
            '''
            #torchvision.utils.save_image(tensor, filename, nrow=8, padding=2, normalize=False,\
             range=None, scale_each=False, pad_value=0)
             #Save a given Tensor into an image file.
            '''
            torchvision.utils.save_image(real_cpu,
                    '%s/real_samples.png' % opt.results,
                    normalize=True)
            fake = netG(fixed_noise)
            torchvision.utils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (opt.results, epoch),
                    normalize=True)

    """
    Save the trained model.
    """ 
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.results, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.results, epoch))