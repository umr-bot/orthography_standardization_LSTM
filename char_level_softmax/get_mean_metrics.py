import numpy as np
def get_mean_metrics(data_dir, model_num, val_or_test="val"):
    model_num = str(model_num)
    sensitivity = [[] for i in range(5)]
    specificity,corr_acc = [[] for i in range(5)], [[] for i in range(5)]
    for i in range(5):
        with open(data_dir+"/foldset_"+str(i+1)+"/"+val_or_test+"_model_"+model_num+"/metrics.txt") as f:
            lines = [line for line in f]
        metric = lines[-1].split(',')
        sensitivity[i] = float(metric[0].split()[1]); specificity[i] = float(metric[1].split()[1]);
        corr_acc[i] = float(metric[-1].split()[1]);
    mean_spec = np.average(specificity)
    mean_sens = np.average(sensitivity)
    mean_cacc = np.average(corr_acc)
    print(f"{mean_sens:.4f} & {mean_spec:.4f} & {mean_cacc:.4f}")


