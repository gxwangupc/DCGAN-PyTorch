import torch

"""
The generator network shown in Figure 1 in the paper.
    nz  : input latent vector
    ngf : size of feature maps of G, 128 in the paper. Here is 64.  
    nc  : number of color channels
    ngpu: number of CUDA devices available
"""
class Generator(torch.nn.Module):

    def __init__(self, nz, ngf, nc, ngpu):
        super(Generator, self).__init__()
        self.nz   = nz
        self.ngf  = ngf
        self.nc   = nc
        self.ngpu = ngpu
        '''
        torch.nn.Sequential(*args):
        #A sequential container. Modules will be added to it in the order they are passed in the constructor.\
         #Alternatively, an ordered dict of modules can also be passed in.
        '''
        '''
        torch.nn.ConvTranspose2d: Applies a 2D transposed convolution operator over an input image.
        #torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, \
         padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
        #bias=True: adds a learnable bias to the output.
        '''
        '''
        torch.nn.BatchNorm2d: Applies Batch Normalization over a 4D input 
          #(a mini-batch of 2D inputs with additional channel dimension).
        #torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #num_features: C from an expected input of size (N, C, H, W)
        '''
        '''
        torch.nn.ReLU: Applies the rectified linear unit function element-wise:
        #torch.nn.ReLU(inplace=False)
        #inplace=True: can optionally do the operation in-place.
        '''
        '''
        activation function in the last layer:
        torch.nn.Tanh()
        '''
        self.net  = torch.nn.Sequential(
            # self.nz(100), project and reshape, stride=1
            torch.nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, stride=1, padding=0,bias=False),
            torch.nn.BatchNorm2d(self.ngf * 8),
            torch.nn.ReLU(inplace=True),

            # self.ngf*8(512) * 4 * 4, stride = 2
            torch.nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(self.ngf * 4),
            torch.nn.ReLU(inplace=True),

            # self.ngf*4(256) * 8 * 8, stride = 2
            torch.nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(self.ngf * 2),
            torch.nn.ReLU(inplace=True),

            # self.ngf*2(128) * 16 * 16, stride = 2
            torch.nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(self.ngf),
            torch.nn.ReLU(inplace=True),

            # self.ngf(64) * 32 * 32, stride = 2
            torch.nn.ConvTranspose2d(self.ngf, self.nc, 4, stride=2, padding=1, bias=False),
            torch.nn.Tanh()
            # self.nc(1 or 3) * 64 * 64
        )

    """
    Forward propogation of G.
    """
    '''
    #torch.nn.parallel.data_parallel(module, inputs, device_ids=None, \
     #output_device=None, dim=0, module_kwargs=None)
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
        return output
