import torch

"""
The discriminator network that not shown in the paper.
    nc  : number of color channels
    ndf : size of feature maps of D, 128 in the paper. Here is 64.
    ngpu: number of CUDA devices available
"""
class Discriminator(torch.nn.Module):

    def __init__(self, nc, ndf, ngpu):
        super(Discriminator, self).__init__()
        self.nc   = nc
        self.ndf  = ndf
        self.ngpu = ngpu
        '''
        torch.nn.Sequential(*args):
        #A sequential container. Modules will be added to it in the order they are passed in the constructor.\
         #Alternatively, an ordered dict of modules can also be passed in.
        '''
        '''
        torch.nn.Conv2d: Applies a 2D convolution operator over an input image.
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,\
         bias=True, padding_mode='zeros')
        '''
        '''
        torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
        #negative_slope: Controls the angle of the negative slope. Default: 1e-2
        #inplace: can optionally do the operation in-place.
        '''
        '''
        torch.nn.BatchNorm2d: Applies Batch Normalization over a 4D input 
          #(a mini-batch of 2D inputs with additional channel dimension).
        #torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #num_features: C from an expected input of size (N, C, H, W)
        '''
        '''
        activation function in the last layer:
        torch.nn.Sigmoid()
        '''
        self.net  = torch.nn.Sequential(
            # self.nc(1 or 3) * 64 * 64
            torch.nn.Conv2d(self.nc, self.ndf, 4, stride=2, padding=1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),

            # self.ndf(64) * 32 * 32
            torch.nn.Conv2d(self.ndf, self.ndf * 2, 4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(self.ndf * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),

            # self.ndf*2(128) * 16 * 16
            torch.nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(self.ndf * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),

            # self.ndf*4(256) * 8 * 8
            torch.nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(self.ndf * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),

            # self.ndf*8(512) * 4 * 4, stride = 1
            torch.nn.Conv2d(self.ndf * 8, 1, 4, stride=1, padding=0, bias=False),
            torch.nn.Sigmoid()
        )

    """
    Forward propogation of D.
    """
    '''
    #torch.nn.parallel.data_parallel(module, inputs, device_ids=None, \
     output_device=None, dim=0, module_kwargs=None)
     #module: the module to evaluate in parallel, self.net
     #input : inputs to the module
     #device_ids:GPU ids on which to replicate module
     #output_device:GPU location of the output Use -1 to indicate the CPU. (default: device_ids[0])
    '''
    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = torch.nn.parallel.data_parallel(self.net, input, range(self.ngpu))
        else:
            output = self.net(input)
        '''
        squeeze(a, axis=None): remove the dimensions that equal 1.
        '''
        return output.view(-1, 1).squeeze(1)
