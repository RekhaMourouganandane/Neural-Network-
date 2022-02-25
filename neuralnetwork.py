import math
import random
class NeuralNetwork:
    def __init__(self, act_val, num_weights, neuron_position):
        self.act_val = act_val
        self.neuron_position = neuron_position
        self.wgt = []
        self.grd = 0
        self.del_value = []
        for i in range(num_weights):
            self.wgt.append(random.random())
            self.del_value.append(0)

    def weight_prod(self, previous_layer):
        y = 0
        for i in range(len(previous_layer)):
            y = y + float(previous_layer[i].act_val) * previous_layer[i].wgt[self.neuron_position]
        expo=math.exp(-y)
        self.act_val = 1 / (1 + expo)
#Feed Forward
def feedforward(inp):
    for i in range(len(inp_layer)-1):
        inp_layer[i].act_val = inp[i]
    inp_layer[-1].act_val = 1
    for j in range(len(hid_layer)-1):
        hid_layer[j].weight_prod(inp_layer)
    hid_layer[-1].act_val = 1
    for k in range(len(out_layer)):
        out_layer[k].weight_prod(hid_layer)

inp_layer = [NeuralNetwork(0, 2, 0), NeuralNetwork(0, 2, 1), NeuralNetwork(0, 2, 2)]
hid_layer = [NeuralNetwork(0, 2, 0), NeuralNetwork(0, 2, 1), NeuralNetwork(0, 2, 2)]
out_layer = [NeuralNetwork(0, 2, 0), NeuralNetwork(0, 2, 1)]

#Back propogation
def backppg(out):
    global out_error
    out_error = []
    for i in range(len(out)):
        err = out[i] - out_layer[i].act_val
        out_error.append(err)
    for i in range(len(out)):
        #lambda_value is 0.3
        out_layer[i].grd = 0.3 * out_layer[i].act_val * (1 - out_layer[i].act_val) * out_error[i]
    for i in range(len(hid_layer)):
        r = 0
        for i in range(len(hid_layer)):
            r = out_layer[i].grd * hid_layer[i].wgt[i]
         #lambda_value is 0.3
        hid_layer[i].grd = 0.3 * hid_layer[i].act_val * (1 - hid_layer[i].act_val) * r
    for i in range(len(hid_layer)):
        #learning_rate is 0.4 & Momentum is 0.2
        hid_layer[i].del_value[i] = 0.4 * out_layer[i].grd * hid_layer[i].act_val + 0.2 * hid_layer[i].del_value[i]

    for i in range(len(inp_layer)):
        delta_weight = []
        for index in range(len(inp_layer[i].wgt)):
            #learning_rate is 0.4 & Momentum is 0.2
            wgts = 0.4 * hid_layer[index].grd * inp_layer[i].act_val + 0.2 * hid_layer[i].del_value[i]
            delta_weight.append(wgts)
        for j in delta_weight:
            inp_layer[i].del_value[i] = j

    for i in range(len(hid_layer)):
        for index in range(len(hid_layer[i].wgt)):
            hid_layer[i].wgt[index] = hid_layer[i].wgt[index] + hid_layer[i].del_value[index]

    for i in range(len(inp_layer)):
        for index in range(len(inp_layer[i].wgt)):
            inp_layer[i].wgt[index] = inp_layer[i].wgt[index] + inp_layer[i].del_value[index]

    # print("error",out_error)
    # print("inp_del_value",inp_layer[0].del_value)
    # print("hidden_del_value",hid_layer[0].del_value)
    # print("op_del_value",out_layer[0].del_value)
    # print("wgt",inp_layer[0].wgt)

def training_set():
        file=open('train.csv')
        total = 0
        n=0
        for line in file:
            inp1=float(line.split(',')[0])
            inp2=float(line.split(',')[1])
            out1=float(line.split(',')[2])
            out2=float(line.split(',')[3])
            n=n+1
            
            global inp
            inp = [inp1, inp2]
            out = [out1,out2]
            global inp_layer
            inp_layer = []
            for i in range(len(inp)):
                inp_layer.append(NeuralNetwork(0, 2, i))
            global hid_layer
            hid_layer = []
            for i in range(2):
                hid_layer.append(NeuralNetwork(0, 2, i))
            global out_layer
            out_layer = []
            for i in range(2):
                out_layer.append(NeuralNetwork(0, 0, i))

            feedforward(inp)
            backppg(out)
            feedforward(inp)

            err_out1 = out1 - out_layer[0].act_val
            err_out2 = out2 - out_layer[1].act_val
            e1=err_out1 ** 2
            e2=err_out2 ** 2
            total= total+(e1+e2/ 2)
        rmse = math.sqrt(total / n)
        return rmse

def test_set():
        file=open('test.csv')
        total = 0
        n=0
        for line in file:
            inp1=float(line.split(',')[0])
            inp2=float(line.split(',')[1])
            out1=float(line.split(',')[2])
            out2=float(line.split(',')[3])
            n=n+1
            global testset_inputs
            testset_inputs = [inp1, inp2]
            feedforward(testset_inputs)
            global testset_outputs
            testset_outputs = []
            for i in range(len(out_layer)):
                testset_outputs.append(out_layer[i].act_val)
            err_out1 = out1 - testset_outputs[0]
            err_out2 = out2 - testset_outputs[1]
            total= total+((err_out1 ** 2) + (err_out2 ** 2) / 2)
        rmse = math.sqrt(total / n)
        return rmse

def epoch():
    epochs = 20
    for i in range(epochs):
        Rmse1 = training_set()
        Rmse2 = test_set()
        print('training_set : ',Rmse1,'test_set : ',Rmse2)

file=open('data.csv')
inputx=[]
inputy=[]
output_x=[]
output_y = []
for line in file:
    inputx.append(float(line.split(',')[0]))
    inputy.append(float(line.split(',')[1]))
    output_x.append(float(line.split(',')[2]))
    output_y.append(float(line.split(',')[3]))
maxvalue_1 = max(inputx)
maxvalue_2 = max(inputy)
maxvalue3 = max(output_x)
maxvalue4 = max(output_y)
minvalue_1 = min(inputx)
minvalue_2 = min(inputy)
minvalue_3 = min(output_x)
minvalue_4 = min(output_y)

file = open('weights.txt','w')

for i in range(len(inp_layer)):
    for w in inp_layer[i].wgt:
        file.write(str(w))
        file.write(',')
file.write('\n')

for i in range(len(hid_layer)):
    for w in hid_layer[i].wgt:
        file.write(str(w))
        file.write(',')
file.write('\n')
file.write(str(maxvalue_1))
file.write(',')
file.write(str(minvalue_1))
file.write(',')
file.write('\n')

file.write(str(maxvalue_2))
file.write(',')
file.write(str(minvalue_2))
file.write(',')
file.write('\n')

file.write(str(maxvalue3))
file.write(',')
file.write(str(minvalue_3))
file.write(',')
file.write('\n')

file.write(str(maxvalue4))
file.write(',')
file.write(str(minvalue_4))
file.write(',')
file.write('\n')

epoch()

