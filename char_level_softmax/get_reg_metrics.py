# coding: utf-8
import os
root_dir="checkpoints/eng_reg"
dirs = os.listdir(root_dir)
ll = []
metrics=[]
for d in dirs:
    if "l1" in d:
        toks = d.split("_")
        L1_L2 = (toks[2],toks[5],toks[8],toks[11])
        ll.append(L1_L2)
        with open(os.path.join(root_dir,d,"foldset_1","val_model_10","metrics.txt")) as f: lines = [line for line in f]
        temp=lines[-1].rstrip('\n').split(' ')
        metric=[]
        for i in range(1,len(temp),2): metric.append(temp[i].strip(','))
        metrics.append(L1_L2+tuple(metric))
        
