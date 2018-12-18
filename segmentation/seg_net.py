import torch
import torch.nn as nn  
from torch.nn import init


class Cconv(nn.Module):
    ''' perform convolution accross each dimention separately
        and then sum. reduces from c*k^3 to c*3*k params for each 
        conv filter
    '''
    def __init__(self, in_c, out_c, k, d, s, factored):
        super().__init__()
        self.factored = factored

        if factored:
            self.layer_x = nn.Conv3d(in_c, out_c, kernel_size=(k,1,1), 
                     padding=((k+d-1)//2,0,0), dilation=d, stride=s)

            nn.init.kaiming_normal_(self.layer_x.weight, mode='fan_out')



            self.layer_y = nn.Conv3d(in_c, out_c, kernel_size=(1,k,1), 
                     padding=(0,(k+d-1)//2,0), dilation=d, stride=s)

            nn.init.kaiming_normal_(self.layer_y.weight, mode='fan_out')



            self.layer_z = nn.Conv3d(in_c, out_c, kernel_size=(1,1,k), 
                     padding=(0,0,(k+d-1)//2), dilation=(1,1,d), stride=s)

            nn.init.kaiming_normal_(self.layer_z.weight, mode='fan_out')

        else:
            kwargs = {"kernel_size": k, "padding": (k+d-1)//2, "dilation":d, "stride":s}
            print(kwargs)
            self.layer = nn.Conv3d(in_c, out_c, **kwargs)

            # nn.init.kaiming_normal_(self.layer.weight, mode='fan_out')
            # only commented for consistancy, bring back


    def __call__(self, x):
        if self.factored:
#                         print(x.size(), self.layer_v(x).size(), self.layer_h(x).size())
            return self.layer_x(x)+self.layer_y(x)+self.layer_z(x)
#                         return self.layer_v(self.layer_h(x)) 
        else:
            return self.layer(x)


# In[ ]:

class UpCconv(nn.Module):
    ''' perform convolution accross each dimention separately
        and then sum. reduces from c*k^3 to c*3*k params for each 
        conv filter
    '''
    def __init__(self, in_c, out_c, k, d, s, factored):
        super().__init__()
        self.factored = factored

        if factored:
#             self.layer_x = nn.Conv3d(in_c, out_c, kernel_size=(k,1,1), 
#                      padding=((k+d-1)//2,0,0), dilation=d, stride=s)


            # TODO: fix because broken (s!=1 doesn't work)

            self.layer_x = nn.ConvTranspose3d(
                in_c, out_c, kernel_size=(k,1,1), 
                output_padding=(s//2,0,0), padding=((k+d-1)//2, d//2, d//2), 
                dilation=d, stride=s)
            
            nn.init.kaiming_normal_(self.layer_x.weight, mode='fan_out')



            self.layer_y = nn.ConvTranspose3d(
                in_c, out_c, kernel_size=(1,k,1), 
                output_padding=(0,s//2,0), padding=(d//2,(k+d-1)//2,d//2), 
                dilation=d, stride=s)

            nn.init.kaiming_normal_(self.layer_y.weight, mode='fan_out')



            self.layer_z = nn.ConvTranspose3d(
                in_c, out_c, kernel_size=(1,1,k), 
                output_padding=(0,0,s//2), padding=(d//2,d//2,(k+d-1)//2), 
                dilation=d, stride=s)

            nn.init.kaiming_normal_(self.layer_z.weight, mode='fan_out')

        else:
            self.layer = nn.ConvTranspose3d(
                in_c, out_c, kernel_size=k, 
                output_padding=s//2, padding=(k+d-1)//2, 
                dilation=d, stride=s)

    def __call__(self, x):
        if self.factored:
#             print(x.shape,"->","\n\t",self.layer_x(x).shape, "\n\t",self.layer_y(x).shape,"\n\t", self.layer_z(x).shape)
            return self.layer_x(x)+self.layer_y(x)+self.layer_z(x)
        else:
            return self.layer(x)



class Net(nn.Module):
    k = 7
    d = 1
    s = 2
    b1_c = 16
    b2_c = 64    
    b3_c = 64    
    
    
    def __init__(self):
        super(Net,self).__init__()
        k = self.k
        d = self.d
        s = self.s
        b1_c = self.b1_c
        b2_c = self.b2_c
        b3_c = self.b3_c
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        def conv3d(in_c, out_c, k=k, s=s, factored=True):
                
            layer = Cconv(in_c, out_c, k, d, s, factored)
              
            return layer
        
        def up_conv3d(in_c, out_c, s=s, factored=True):
#             layer =  nn.ConvTranspose3d(
#                 in_c, out_c, kernel_size=k, 
#                 output_padding=s//2, padding=(k+d-1)//2, 
#                 dilation=d, stride=s)
            
            # this is only commented out for consistancy with non-conda version
#           nn.init.kaiming_normal_(layer.weight, mode='fan_out')

            layer = UpCconv(in_c, out_c, k, d, s, factored)

            return layer
        
        # Returns 3D batch normalisation layer
        def bn(planes):
            layer = nn.BatchNorm3d(planes)
            # Use mean 0, standard deviation 1 init
            init.constant_(layer.weight, 1)
            init.constant_(layer.bias, 0)
            return layer

        # preproc
        self.conv_pre1 = conv3d(1, 3, factored=False, s=1)
        self.bn1_pre = bn(3)
        self.conv_pre2 = conv3d(3, 3, factored=False, s=1)
        self.bn2_pre = bn(3)
        self.conv_pre3 = conv3d(3, 3, factored=False, s=1)
        self.bn3_pre = bn(3)


        # down
        self.conv1 = conv3d(3, b1_c, factored=False)
        self.bn1 = bn(b1_c)

        self.conv2 = conv3d(b1_c, b2_c, factored=False)
        self.bn2 = bn(b2_c)
        
        self.conv3 = conv3d(b2_c, b3_c, s=1)
        self.bn3 = bn(b3_c)
        
        
        # low-res processing
        self.conv1_lrp = conv3d(b3_c, b3_c, s=1)
        self.bn1_lrp = bn(b3_c)

        self.conv2_lrp = conv3d(b3_c, b3_c, s=1)
        self.bn2_lrp = bn(b3_c)
        
        self.conv3_lrp = conv3d(b3_c, b3_c, s=1)
        self.bn3_lrp = bn(b3_c)
        
        self.conv4_lrp = conv3d(b3_c, b3_c, s=1)
        self.bn4_lrp = bn(b3_c)
        
        self.conv5_lrp = conv3d(b3_c, b3_c, s=1)
        self.bn5_lrp = bn(b3_c)
        
        self.conv6_lrp = conv3d(b3_c, b3_c, s=1)
        self.bn6_lrp = bn(b3_c)
        
        self.conv7_lrp = conv3d(b3_c, b3_c, s=1)
        self.bn7_lrp = bn(b3_c)
        
        
        # up
        self.conv3u = up_conv3d(2*b3_c, b2_c, s=1, factored=True)
        self.bn3u = bn(b2_c)

        self.conv2u = up_conv3d(2*b2_c, b1_c, factored=False)
        self.bn2u = bn(b1_c)
        
        self.conv1u = up_conv3d(2*b1_c, 3, factored=False)
        self.bn1u = bn(3)
        
        
        # post 
        self.conv_post1 = conv3d(12, 3, factored=False, s=1)
        self.bn1_post = bn(3)
        # self.conv_post2 = conv3d(12, 3, factored=False, s=1)
        # self.bn2_post = bn(3)
        self.conv_post3 = conv3d(12, 1, factored=False, s=1)
        # self.bn3_post = bn(3)

        
    def forward(self, x):
        k = self.k
        d = self.d
        s = self.s
        b1_c = self.b1_c
        b2_c = self.b2_c
        b3_c = self.b3_c
        
#         print(x.size())
        
        # pre
        # size = x.size()
        # size[1] = 3
        # x = xp0 = x.expand(size)

        # inp = x
        # print(x.size()) 
        x = xpre1 = self.relu(self.bn1_pre(self.conv_pre1(x))) # + x
        x = xpre2 = self.relu(self.bn2_pre(self.conv_pre2(x))) + x
        x = xpre3 = self.relu(self.bn3_pre(self.conv_pre3(x))) + x

        # down
        
        x = x1 = self.relu(self.bn1(self.conv1(x)))
        x = x2 = self.relu(self.bn2(self.conv2(x)))
        x = x3 = self.relu(self.bn3(self.conv3(x)))
        
        
        # low-res processing w/ residual connections
        x = self.relu(self.bn1_lrp(self.conv1_lrp(x))) + x
        x = self.relu(self.bn2_lrp(self.conv2_lrp(x))) + x
        x = self.relu(self.bn3_lrp(self.conv3_lrp(x))) + x
        x = self.relu(self.bn4_lrp(self.conv4_lrp(x))) + x
        x = self.relu(self.bn5_lrp(self.conv5_lrp(x))) + x
        x = self.relu(self.bn6_lrp(self.conv6_lrp(x))) + x
        x = self.relu(self.bn7_lrp(self.conv7_lrp(x))) + x
        
        # up
#         print(x.shape, x3.shape)
        x = torch.cat((x, x3), dim=1) 
        del x3
        x = self.relu(self.bn3u(self.conv3u(x)))
        
#         print("x shape:", x.shape)

        x = torch.cat((x, x2), dim=1)
        del x2
        x = self.relu(self.bn2u(self.conv2u(x)))
        
#         print("x shape:", x.shape)

        x = torch.cat((x, x1), dim=1)
        del x1
        x = self.relu(self.bn1u(self.conv1u(x)))


        #post
        inp = torch.cat((x, xpre1, xpre2, xpre3), dim=1)
        x = self.relu(self.bn1_post(self.conv_post1(inp))) + x

        inp = torch.cat((x, xpre1, xpre2, xpre3), dim=1)
        # x = self.relu(self.bn2_post(self.conv_post2(inp))) + x
        
        # inp = torch.cat((x, xpre1, xpre2, xpre3), dim=1)
        x = self.conv_post3(inp)


        return self.sigmoid(x)#, x

