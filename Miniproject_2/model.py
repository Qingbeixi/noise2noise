from .others.module import Sequential,Conv2d,ReLU,TransposeConv2d,Sigmoid,MSE,SGD,ELU,SeLU
from pathlib import Path
from torch import load as Load # only load the dataset
import math
import pickle
# import json
# from numpy import array as Array

class Model():
    def __init__(self):
        
        self.model = Sequential(
            Conv2d(3, 96, kernel_size = (2, 2), stride = (2,2),padding = (0,0),dilation = (1,1),bias = 1),
            ReLU(),
            Conv2d(96, 256, kernel_size = (4, 4), stride=(2,2),padding=(0,0),dilation=(1,1),bias = 1),
            ReLU(),
            TransposeConv2d(256, 96,kernel_size=(4, 4), stride=(2,2),dilation = (1,1),padding = (0,0), bias = 1),
            ReLU(),
            TransposeConv2d(96, 3, kernel_size = (2, 2), stride=(2,2),dilation=(1,1),padding=(0,0),bias=1),
            Sigmoid()
)


        self.optimizer = SGD(params = self.model.param() , lr = 4)
        self.criterion = MSE()

    def load_pretrained_model(self):

        model_path = Path(__file__).parent / "bestmodel.pth"
        with open(model_path, 'rb') as handle:
            load_list = pickle.load(handle)

        t = 0 # set the index
        for layer in self.model.all_layers:
            if len(layer.param()) != 0:
                # print(layer)
                layer.weight = load_list[t][0]
                # print(layer.weight.shape)
                layer.bias = load_list[t+1][0]
                # print(layer.bias.shape)
                t = t + 2


    def train(self, train_input, train_target,num_epochs):

        losses = []
        # test_losses = []

        train_input = train_input.float()/255
        train_target = train_target.float()/255
        
        # test_input = test_input.float()/255
        # test_target = test_target.float()/255

        mini_batch_size = 250
        for e in range(num_epochs):

            ps = 0
            mean_losses = 0
            
            for b in range(0, train_input.size(0), mini_batch_size):

                self.model.zero_grad()

                output = self.model.forward(train_input.narrow(0, b, mini_batch_size))
                
                loss = self.criterion.forward(output, train_target.narrow(0, b, mini_batch_size))
                mean_losses += loss

                
                loss_grad = self.criterion.backward()
                
                self.model.backward(loss_grad)
                
                
                self.optimizer.step()


                # print(loss)

                # ps += PSNR(output,train_target.narrow(0, b, mini_batch_size))
            
            
            mean_losses /= (train_input.size(0)/mini_batch_size)
                
            losses.append(mean_losses)

            # output = self.model.forward(test_input)
            # test_losses.append(self.criterion.forward(output, test_target))

            print("loss in epcoh {}, is {}".format(e,mean_losses))
            # print("test_loss in epcoh {}, is {}".format(e,test_losses[e]))
            
        return losses #, test_losses
    
    def predict(self,test_input):

        test_input = test_input.float()/255

        output = self.model.forward(test_input)*255  #TO threshold

        n,c,h,w = output.shape

        for i in range(n):
            for j in range(c):
                for k in range(h):
                    for l in range(w):
                        if output[i][j][k][l] < 0:
                            output[i][j][k][l] = 0
                        if output[i][j][k][l] > 255:
                            output[i][j][k][l] = 255

        return output

    def save_model(self):

        load_list = self.model.param()
        model_path = Path(__file__).parent / "bestmodel.pth"
        with open(model_path, 'wb') as handle:
            pickle.dump(load_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("the model is saved on bestmodel.pth")

def PSNR(original, compressed):
    mse = ((original - compressed) ** 2).mean()
    psnr = -10 * math.log10(mse + 10**-8)
    return psnr

if __name__ == '__main__':

    source, target = Load(r'C:\Users\qingjun\Desktop\Proj_341346\Miniproject_2\others\data\train_data.pkl')
    test_input, test_target = Load(r'C:\Users\qingjun\Desktop\Proj_341346\Miniproject_2\others\data\train_data.pkl')
    mode = Model()
    
    # mode.load_pretrained_model()
    mode.train(source[:50000], target[:50000],5)

    mode.save_model()

    # loss_dict = {
    #     'train_loss': Array(loss).tolist(),
    #     'test_loss' : Array(test_loss).tolist()
    # }

    # with open('Miniproject_2/plot/large_para_relu_4.json', 'w') as fp:
    #     json.dump(loss_dict, fp)

    # test_predict = mode.predict(test_input[:100])
    # print('The test PSNR is ',PSNR(test_predict/255,test_target[:100]/255))


