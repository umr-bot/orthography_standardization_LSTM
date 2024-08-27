import numpy as np

class MetricHandler:

    def __init__(self, data_dir, hdd, num_layer, model_num, model_name, val_or_test="val"):
        self.data_dir = data_dir
        self.hdd = hdd
        self.num_layer = num_layer
        self.model_num = model_num
        self.model_name = model_name
        self.val_or_test = val_or_test

    def get_mean_metrics(self, num_layer=None):
        model_num = str(self.model_num)
        if num_layer == None: num_layer = self.num_layer
        sensitivity = [[] for i in range(5)]
        specificity,corr_acc = [[] for i in range(5)], [[] for i in range(5)]
        for i in range(5):
            with open(self.data_dir+self.model_name+"_"+str(num_layer)+"/foldset_"+str(i+1)+"_hdd_"+str(self.hdd)+"/"+self.val_or_test+"_model_"+str(self.model_num)+"/metrics.txt") as f:
                lines = [line for line in f]
            metric = lines[-1].split(',')
            sensitivity[i] = float(metric[0].split()[1]); specificity[i] = float(metric[1].split()[1]);
            corr_acc[i] = float(metric[-1].split()[1]);
        mean_spec = np.average(specificity)
        mean_sens = np.average(sensitivity)
        mean_cacc = np.average(corr_acc)
        #print(f"{mean_sens:.4f} & {mean_spec:.4f} & {mean_cacc:.4f}")
        return mean_spec, mean_sens, mean_cacc
        
    def print_metrics(self):
        mean_spec, mean_sens, mean_cacc = self.get_mean_metrics()
        print(f"{mean_sens:.4f} & {mean_spec:.4f} & {mean_cacc:.4f} \\")
