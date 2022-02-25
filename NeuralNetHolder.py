import neural as nn
class NeuralNetHolder:

    def __init__(self):
        self.inp_layer = [nn.NeuralNetwork(0, 2, 0), nn.NeuralNetwork(0, 2, 1), nn.NeuralNetwork(0, 2, 2)]
        self.hid_layer = [nn.NeuralNetwork(0, 2, 0), nn.NeuralNetwork(0, 2, 1), nn.NeuralNetwork(0, 2, 2)]
        self.out_layer = [nn.NeuralNetwork(0, 2, 0), nn.NeuralNetwork(0, 2, 1)]

    def feedforward(self,inp):
        for i in range(len(self.inp_layer)-1):
            self.inp_layer[i].act_val = inp[i]
        self.inp_layer[-1].act_val = 1

        for h in range(len(self.hid_layer)-1):
            self.hid_layer[h].weight_prod(self.inp_layer)
        self.hid_layer[-1].act_val = 1

        for o in range(len(self.out_layer)):
            self.out_layer[o].weight_prod(self.hid_layer)

    def predict(self, inprow):
        inprow = inprow.split(',')
        for i in range(len(inprow)):
            inprow[i] = float(inprow[i])


            file=open('weights.txt','r')
            inp_w = []
            out_w = []
            total_w = []
            for line in file:
                total_w.append(line.split('\n'))
            inp_w = total_w[0]
            inp_w.pop(-1)
            inp_w = inp_w[0].split(',')
            inp_w.pop(-1)
            for i in range(len(inp_w)):
                inp_w[i] = float(inp_w[i])

            out_w = total_w[1]
            out_w.pop(-1)
            out_w = out_w[0].split(',')
            out_w.pop(-1)
            for i in range(len(out_w)):
                out_w[i] = float(out_w[i])

            xaxis_dst = total_w[2]
            xaxis_dst.pop(-1)
            xaxis_dst = xaxis_dst[0].split(',')
            xaxis_dst.pop(-1)
            for i in range(len(xaxis_dst)):
                xaxis_dst[i] = float(xaxis_dst[i])

            yaxis_dst = total_w[3]
            yaxis_dst.pop(-1)
            yaxis_dst = yaxis_dst[0].split(',')
            yaxis_dst.pop(-1)
            for i in range(len(yaxis_dst)):
                yaxis_dst[i] = float(yaxis_dst[i])

            xaxis_velo = total_w[4]
            xaxis_velo.pop(-1)
            xaxis_velo = xaxis_velo[0].split(',')
            xaxis_velo.pop(-1)
            for i in range(len(xaxis_velo)):
                xaxis_velo[i] = float(xaxis_velo[i])

            yaxis_velo = total_w[5]
            yaxis_velo.pop(-1)
            yaxis_velo = yaxis_velo[0].split(',')
            yaxis_velo.pop(-1)
            for i in range(len(yaxis_velo)):
                yaxis_velo[i] = float(yaxis_velo[i])

        for i in range(len(self.inp_layer)):
            if i == 0:
                self.inp_layer[i].wgt.append(inp_w[0])
                self.inp_layer[i].wgt.append(inp_w[1])
            elif i == 1:
                self.inp_layer[i].wgt.append(inp_w[2])
                self.inp_layer[i].wgt.append(inp_w[3])
            elif i == 2:
                self.inp_layer[i].wgt.append(inp_w[4])
                self.inp_layer[i].wgt.append(inp_w[5])

        for i in range(len(self.hid_layer)):
            if i == 0:
                self.hid_layer[i].wgt.append(out_w[0])
                self.hid_layer[i].wgt.append(out_w[1])
            elif i == 1:
                self.hid_layer[i].wgt.append(out_w[2])
                self.hid_layer[i].wgt.append(out_w[3])
            elif i == 2:
                self.hid_layer[i].wgt.append(out_w[4])
                self.hid_layer[i].wgt.append(out_w[5])
        
        input_normalise = []
        input_normalise.append((xaxis_dst[0]-inprow[0])/(xaxis_dst[0]-xaxis_dst[1]))
        input_normalise.append((yaxis_dst[0] - inprow[1])/(yaxis_dst[0]-yaxis_dst[1]))
        
        print(input_normalise)
        self.feedforward(input_normalise)
        output_denormalise= []
        output_denormalise.append((xaxis_dst[0]-(self.out_layer[0].act_val*(xaxis_dst[0]-xaxis_dst[1]))))
        output_denormalise.append((yaxis_dst[0] - (self.out_layer[1].act_val * (yaxis_dst[0] - yaxis_dst[1]))))

        return output_denormalise
