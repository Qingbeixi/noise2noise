from torch.nn.functional import fold, unfold
from torch import empty 
import torch

""" Sequential """

class Sequential():
    """
    This function is used to assemble the model
    We've chosen to put the zero_grad function which call the zer_grad of all the 
    layers
    """

    def __init__(self, *layers):
        
        #Include also activation functions
        self.all_layers = layers
        
    def _call_(self, x):
        
        return self.forward(x)
    
    def zero_grad(self):
        #Put the gradients of the weight and bias to zero
        
        for layer in self.all_layers:
            layer.zero_grad()
            
    def param(self):
        
        self.tot_par = []
        for single_layer in self.all_layers:
            for param in single_layer.param():
                if len(param) != 0:
                    self.tot_par.insert(len(self.tot_par),param)
        return self.tot_par

    def forward(self, forward_input):
        
        for single_layer in self.all_layers:
            forward_input = single_layer.forward(forward_input)
        return forward_input

    def backward(self , grad):
        
        r_layers = self.all_layers[::-1]
        for single_layer in r_layers:
            grad = single_layer.backward(grad)
        return grad
    

class Upsampling():
    """
    After multiple tries, the results with Upsampling where not as good as TransposeConv2D,
    though we've let this function,
    because we're eager to know how to fix it, feedbacks are welcomed.
    The different steps here are upsampling with nearest neighbors thanks to the function
    repeat_interleave done on the two dimensions, then the same conv as previoulsy.
    The backward is first the same as in conv, and then the gradients are added thanks 
    the different reshape, sum and transposed.
    The results can go to a decent results (though less than TransposeConv2d )
    But the definition of the image at the end seems altered.
    """
    
    def __init__(self,in_channels, out_channels, kernel_size, size=(4,4),stride=1,
                 padding=0, dilation=1, bias = 1):
        
        
        if int == type(dilation):
            
            self.dilat=(dilation, dilation)
        else:
            self.dilat=dilation
            
        if int ==  type(stride):
            
            self.stri = (stride, stride)
        else:
            self.stri = stride
        
        if  int == type(kernel_size):
            
            self.kernel_size = (kernel_size, kernel_size)
        else: self.kernel_size = kernel_size
        
        if int == type(padding):
            
            self.pad = (padding, padding)
        else:
            self.pad = padding
        self.size = size
        self.in_ = in_channels
        self.out_ = out_channels
        self.k_0 = self.kernel_size[0]
        self.k_1 = self.kernel_size[1]
        self.bias_bool = bias
        

        if  1 == self.bias_bool:
            self.bias = empty(self.out_).uniform_(-1/((self.k_0*self.k_1*self.out_)**0.5),
                                                  1/((self.k_0*self.k_1*self.out_)**0.5))
            
            self.g_b = empty(self.bias.size()).zero_()
        else:
            self.bias = []
            self.g_b = []
            
        self.weight = empty(self.in_, self.out_, self.k_0, self.k_1)\
                        .uniform_(-1/((self.k_0*self.k_1*self.out_)**0.5), 
                                  1/((self.k_0*self.k_1*self.out_)**0.5))
        self.g_w = empty(self.weight.size()).zero_()
        
    def zero_grad(self):
        
        self.g_w.zero_()
        if self.bias_bool: self.g_b.zero_()
        
    def param(self):
        return [(self.weight, self.g_w), (self.bias, self.g_b)]   
        
    def forward(self, forward_input):
        #upsampling pytorch
        #print(input.size())
        self.x = forward_input.repeat_interleave( self.size[1], dim=3).repeat_interleave( self.size[0], dim=2)
        #print(self.x.size())
        stri_0, stri_1 = self.stri
        dilat_0, dilat_1 = self.dilat
        pad_0, pad_1 = self.pad
        
        # from the informations of the project and pytorch conv2D website
        
        x_0, x_1, x_2, x_3 = self.x.shape
        
        
        self.co = unfold(self.x, kernel_size=self.kernel_size, padding=self.pad,
                         stride=self.stri, dilation=self.dilat)
        
        w_r = self.weight.reshape(self.out_,-1)
        res = w_r @ self.co
        if self.bias_bool:
            res += self.bias.reshape(1, -1, 1)
        
        #calculating the output dimension cf pytorch conv2d documentation
        formula_1 = (x_2+2*pad_0-dilat_0*(self.k_0-1)-1)/stri_0+1
        formula_2 = (x_3+2*pad_1 -dilat_1*(self.k_1-1)-1)/stri_1+1
    
        #inting those formulas otherwise it bugs
        size_out_1 = int(formula_1)
        size_out_2 = int(formula_2)
        
        sorti = res.reshape(x_0, self.out_, size_out_1, size_out_2)
        return  sorti
        

    def backward(self, gradient):
        
        self.g_b.data = gradient.sum(dim=[0, 2, 3])
        
        
        x_0, x_1, x_2, x_3 = self.x.shape
        dim_inp = (x_2, x_3)
    
        
        #reshaping the gradients, making batch size last dimension
        r_grad = gradient.transpose(0, 3)
        r_grad = r_grad.transpose(0, 1)
        r_grad = r_grad.transpose(1, 2)
        r_grad = r_grad.reshape(self.out_, -1)
       
        
        # taking back the unfold from forward
        r_x =self.co.transpose(0, 1)
        r_x = r_x.transpose(0, 2)
        r_x = r_x.reshape(r_grad.shape[1], -1)
        
        #calculating the weights gradients
        self.g_w.data = r_grad @ r_x
        self.g_w.data = self.g_w.data.reshape(self.weight.shape)
        
        # multiplying the gradients by the weights
        w_reshape = self.weight.reshape(self.out_, -1)
        w_reshape = w_reshape.T
        dx_res = w_reshape @ r_grad
        
        
        size_of_reshape = self.co.transpose(0, 1)
        size_of_reshape = size_of_reshape.transpose(1, 2)
        size_of_reshape = size_of_reshape.shape
        
        #reshaping for the fold
        dx_res = dx_res.reshape(size_of_reshape)
        dx_res = dx_res.transpose(0, 1)
        dx_res = dx_res.transpose(0, 2)
        

        #finally folding
        resul = fold(dx_res, dim_inp, kernel_size=self.kernel_size,
                    stride=self.stri, dilation=self.dilat, padding=self.pad)
        
        n = self.size[0]
        shape_0 = int(resul.shape[0])
        shape_1 = int(resul.shape[1])
        
        #adding the grad together
        dim_1 = int(resul.shape[2] * resul.shape[3]/n)
        step_1 = resul.reshape(shape_0,shape_1,dim_1,n).sum(3)
        
        
        step_2 = step_1.reshape(shape_0,shape_1,int(resul.shape[2]),int(resul.shape[2]/n))
        
        dim_2 = int(resul.shape[2]*resul.shape[2]/n/n)
        dim_3 = int(resul.shape[2]/n)
        step_3 = step_2.transpose(2,3).reshape(shape_0,shape_1,dim_2,n).sum(3)\
                .reshape(shape_0,shape_1,dim_3,dim_3).transpose(2,3)
    
        return step_3

class TransposeConv2d():
    
    """
    This function is pretty similar to the conv one 
    The forward and backward are basically inverted
    
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding=(0,0), dilation=(1,1),bias = 1):

        if int == type(dilation):
            self.dilat=(dilation, dilation)
        else:
            self.dilat=dilation
            
        if int ==  type(stride):
            self.stri = (stride, stride)
        else:
            self.stri = stride
        
        if  int == type(kernel_size):
            self.kernel_size = (kernel_size, kernel_size)
        else: self.kernel_size = kernel_size
        
        if int == type(padding):
            self.pad = (padding, padding)
        else:
            self.pad = padding

        self.in_ = in_channels
        self.out_ = out_channels
        

        self.bias_bool = bias
        self.k_0, self.k_1 = self.kernel_size
        
        if self.bias_bool == 1:
            self.bias = empty(self.out_)
            self.bias = self.bias .uniform_(-1/((self.k_0*self.k_1*self.out_)**0.5),
                                                  1/((self.k_0*self.k_1*self.out_)**0.5))
            self.g_b = empty(self.bias.size())
            self.g_b =  self.g_b.zero_()
        else:
            self.bias = []
            self.g_b = []
            
        self.weight = empty(self.in_, self.out_, self.k_0, self.k_1)
        self.weight =  self.weight.uniform_(-1/((self.k_0*self.k_1*self.out_)**0.5), 
                                  1/((self.k_0*self.k_1*self.out_)**0.5))
                        
        self.g_w = empty(self.weight.size())
        self.g_w = self.g_w.zero_()
        
    def zero_grad(self):
        
        self.g_w.zero_()
        if self.bias_bool: self.g_b.zero_()   
    
    def param(self):
        
        return [(self.weight,self.g_w),(self.bias,self.g_b)]
    
    def __call__(self,x):
        
        return self.forward(x)
    
    def forward(self, forward_input):
        
        self.x = forward_input
        
        x_0, x_1, x_2, x_3 = self.x.shape
        stri_0, stri_1 = self.stri
        dilat_0, dilat_1 = self.dilat
        pad_0, pad_1 = self.pad
        
        self.reshape_x = self.x.transpose(0, 3)
        self.reshape_x =self.reshape_x.transpose(0, 1)
        self.reshape_x =self.reshape_x.transpose(1, 2)
        self.reshape_x =self.reshape_x.reshape(self.in_, -1)
        
        
        self.reshape_w = self.weight.reshape(self.in_, -1)
        self.reshape_w = self.reshape_w.T
        
        self.y_before = self.reshape_w @ self.reshape_x
        
        y_b_0 = self.y_before.shape[0]
        
        self.y_before.data = self.y_before.reshape(y_b_0, -1, x_0)
        self.y_before.data = self.y_before.transpose(0, 2)
        self.y_before.data = self.y_before.transpose(1, 2)
        
        formula_1 = dilat_0*(self.k_0-1) + 1 - 2*pad_0 + stri_0*(-1 + x_2) 
        formula_2 = dilat_1*(self.k_1-1) + 1 - 2*pad_1 + stri_1*(-1 + x_3)
        
        self.y = fold(self.y_before,(formula_1 ,formula_2), self.kernel_size, 
                      dilation=self.dilat, padding=self.pad, stride=self.stri)
        
        if not self.bias_bool == 0:
            self.y.data = self.y.data + self.bias.reshape(1, -1, 1, 1)
        
        return  self.y


    def backward(self, gradient):
        
        x_0, x_1, x_2, x_3 = self.x.shape
        
        self.x_g = gradient
        
        #summing them with respect to dim 1
        self.g_b.data = self.x_g.sum(dim=[0,2,3])
        
        self.un_x_g= unfold(self.x_g, kernel_size=self.kernel_size, stride=self.stri,
                            dilation=self.dilat, padding=self.pad)
        
        weight_r = self.weight.reshape(self.in_, -1)
        self.grad_input = weight_r @ self.un_x_g
        
        self.un_x_g_reshape = self.un_x_g.transpose(0, 2)
        self.un_x_g_reshape =self.un_x_g_reshape.transpose(1, 2)
        
        r_x_0 = self.reshape_x.shape[1]
        self.un_x_g_reshape =self.un_x_g_reshape.reshape(r_x_0, -1)
        
        g_w_applied = self.reshape_x @  self.un_x_g_reshape
        self.g_w.data = g_w_applied.reshape(self.weight.shape)
        
        
        size_grad = self.grad_input.shape[0]
        g_i_r = self.grad_input.reshape(size_grad, self.in_, x_2, x_3)
        return g_i_r
    
class Conv2d():
    
    """
    The forward part of the convutionnal layer is based from
    the data of the project.
    The backward is inspired by a github repo quoted in the report and code
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias = 1):
        
        #some conditions for having the good type
        if int == type(dilation):
            self.dilat=(dilation, dilation)
        else:
            self.dilat=dilation
            
        if int ==  type(stride):
            self.stri = (stride, stride)
        else:
            self.stri = stride
        
        if  int == type(kernel_size):
            self.kernel_size = (kernel_size, kernel_size)
            
        else: self.kernel_size = kernel_size
        
        if int == type(padding):
            self.pad = (padding, padding)
        else:
            self.pad = padding
         
        #initialing the rest
        
        self.bias_bool = bias
        self.in_ = in_channels
        self.out_ = out_channels
        self.k_0, self.k_1 = self.kernel_size
        
        # Uniform distribution for the bias and weights as in pytorch conv2D 
        if self.bias_bool == 1:
            self.bias = empty(self.out_)
            self.bias = self.bias .uniform_(-1/((self.k_0*self.k_1*self.out_)**0.5),
                                                  1/((self.k_0*self.k_1*self.out_)**0.5))
            self.g_b = empty(self.bias.size())
            self.g_b = self.g_b.zero_()
            
        self.weight = empty(self.in_, self.out_, self.k_0, self.k_1)
        self.weight =  self.weight.uniform_(-1/((self.k_0*self.k_1*self.out_)**0.5), 
                                  1/((self.k_0*self.k_1*self.out_)**0.5))
        self.g_w = empty(self.weight.size())
        self.g_w = self.g_w.zero_()
        
    def __call__(self,x):
        
        #function for testing it
        return self.forward(x)
    
    def zero_grad(self):
        
        self.g_w.zero_()
        if self.bias_bool: self.g_b.zero_()    
    
    def param(self):
        return [(self.weight, self.g_w), (self.bias, self.g_b)]

    def forward(self, forward_input):
        #making the code softer
        stri_0, stri_1 = self.stri
        dilat_0, dilat_1 = self.dilat
        pad_0, pad_1 = self.pad
        
        # from the informations of the project and pytorch conv2D website
        self.x = forward_input
        x_0, x_1, x_2, x_3 = self.x.shape
        
        
        self.co = unfold(self.x, kernel_size=self.kernel_size, padding=self.pad,
                         stride=self.stri, dilation=self.dilat)
        
        w_r = self.weight.reshape(self.out_,-1)
        res = w_r @ self.co
        if self.bias_bool:
            res += self.bias.reshape(1, -1, 1)
        
        #calculating the output dimension cf pytorch conv2d documentation
        formula_1 = (x_2+2*pad_0-dilat_0*(self.k_0-1)-1)/stri_0+1
        formula_2 = (x_3+2*pad_1 -dilat_1*(self.k_1-1)-1)/stri_1+1
    
        #inting those formulas otherwise it bugs
        size_out_1 = int(formula_1)
        size_out_2 = int(formula_2)
        
        sorti = res.reshape(x_0, self.out_, size_out_1, size_out_2)
        return  sorti

    def backward(self, gradient):
        
        #inspired by https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/layers.py
        #summing them with respect to dim 1
        self.g_b.data = gradient.sum(dim=[0, 2, 3])
        
        
        x_0, x_1, x_2, x_3 = self.x.shape
        dim_inp = (x_2, x_3)
    
        
        #reshaping the gradients, making batch size last dimension
        r_grad = gradient.transpose(0, 3)
        r_grad = r_grad.transpose(0, 1)
        r_grad = r_grad.transpose(1, 2)
        r_grad = r_grad.reshape(self.out_, -1)
       
        
        # taking back the unfold from forward
        r_x =self.co.transpose(0, 1)
        r_x = r_x.transpose(0, 2)
        r_x = r_x.reshape(r_grad.shape[1], -1)
        
        #calculating the weights gradients
        self.g_w.data = r_grad @ r_x
        self.g_w.data = self.g_w.data.reshape(self.weight.shape)
        
        # multiplying the gradients by the weights
        w_reshape = self.weight.reshape(self.out_, -1)
        w_reshape = w_reshape.T
        dx_res = w_reshape @ r_grad
        
        
        size_of_reshape = self.co.transpose(0, 1)
        size_of_reshape = size_of_reshape.transpose(1, 2)
        size_of_reshape = size_of_reshape.shape
        
        #reshaping for the fold
        dx_res = dx_res.reshape(size_of_reshape)
        dx_res = dx_res.transpose(0, 1)
        dx_res = dx_res.transpose(0, 2)
        

        #finally folding
        return fold(dx_res, dim_inp, kernel_size=self.kernel_size,
                    stride=self.stri, dilation=self.dilat, padding=self.pad)



class SGD():
    
    def __init__(self, params ,lr):
        self.params = params
        self.learning = lr 
        
    def __call__(self, x):
        return self.forward(x)        
    def param (self):
        return self.params
    
    def step(self):
    #applying the gradient with the learning rate
        for [param, g_params] in self.params:
            param.data = param.data -g_params* self.learning
    
    def forward(self, forward_input):
        return None
    
    def backward(self, gradient):
        return None



class ELU():

    def __init__(self, alpha=0.2):
        self.data = None
        self.alpha = alpha

    def forward(self, data_in):
        self.data = data_in
        data_neg = self.alpha * (torch.exp(self.data) - 1)
        data_pos = self.data
        return torch.max(data_pos, torch.tensor(0.)) + torch.min(data_neg, torch.tensor(0.))

    def backward(self, grad_dl_dout):
        return grad_dl_dout * ((self.data > 0) + self.alpha * (self.data < 0))

    def param(self):
        return []
    def zero_grad(self):
        pass


class SeLU():

    def __init__(self, alpha=1.6733, lambda_=1.0507):

        self.data = None
        self.alpha = alpha
        self.lambda_ = lambda_
    
    def __call__(self, x):
        return self.forward(x)  

    def forward(self, data_in):
        self.data = data_in
        data_neg = self.lambda_ * self.alpha * (torch.exp(self.data) - 1)
        data_pos = self.lambda_ * self.data
        return torch.max(data_pos, torch.tensor(0.)) + torch.min(data_neg, torch.tensor(0.))

    def backward(self, grad_dl_dout):
        return grad_dl_dout * self.lambda_ * ((self.data > 0) + self.alpha * (self.data < 0))

    def param(self):
        return []
    def zero_grad(self):
        pass

class ReLU() :
    def __init__(self):
        
        super().__init__()
        self.activation = 0
        
    def param(self):
        
        return []
    
    def zero_grad(self):
        
        pass  
    def __call__(self, x):
        
        return self.forward(x)  
    
    def forward(self, forward_input):
        
        self.activation = torch.max(forward_input, torch.tensor(0.))
        return self.activation

    def backward(self, gradient):
        
        mask_deri = self.activation > 0
        return mask_deri*gradient

    

class Sigmoid():
    
    def __init__(self):
        
        super().__init__()
        self.activation = 0
        
    def __call__(self, x):
        
        return self.forward(x)  
    
    def zero_grad(self):
        
        pass
    
    def param(self):
        
        return []
    
    def forward(self, forward_input):
        
        self.activation=(torch.exp(-forward_input)+1)**-1
        return self.activation

    def backward(self, gradient):
        
        deri = 1-self.activation
        deri = deri*self.activation
        return deri*gradient

    
    

class MSE():
    
    def __init__(self):
        
        pass
    
    def __call__(self,x,y):
        
        return self.forward(x,y)
    
    def param(self):
        
        pass
    
    def forward(self, guess, true_):
        
        self.guess = guess
        self.target = true_
        self.err = guess - true_
        
        square_error = (self.err)**2
        
        self.loss_av = square_error.mean()
        return self.loss_av

    def backward(self):
        
        g_0, g_1, g_2, g_3 = self.guess.shape
        tot_size = g_0*g_1*g_2*g_3
        
        top = 2 * (self.err)
        bottom = tot_size
        
        self.grad = top / bottom
        return self.grad

