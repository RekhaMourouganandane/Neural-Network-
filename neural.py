import math
class NeuralNetwork:
    def __init__(self, act_val, num_weights, neuron_position):
        self.act_val = act_val
        self.neuron_position = neuron_position
        self.wgt = []

    def weight_prod(self, prev_layer):
        y = 0
        for i in range(len(prev_layer)):
            y = y + float(prev_layer[i].act_val) * prev_layer[i].wgt[self.neuron_position]
        expo=math.exp(-y)
        self.act_val = 1 / (1 + expo)
