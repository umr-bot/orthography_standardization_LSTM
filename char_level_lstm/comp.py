import os,re
import argparse

def get_metrics(err,cln,tar):
    """Take in as input file names of error file, cleaned file and target file. """
    with open(err) as f: 
        X = [line.strip() for line in f]
        #X = " ".join(X).split()
    with open(cln) as f: 
        predict = [line.strip() for line in f]
        #predict = " ".join(predict).split()
    with open(tar) as f: 
        target = [line.strip() for line in f]
        #actual = " ".join(actual).split()

    tp,fp,fn,tn = 0,0,0,0
    cor,incor=0,0 # cnt varibles for predicted spelling vs target spelling
    for i in range(len(predict)):
        # predict             target
        if X[i]!=predict[i] and X[i] != target[i]: tp+=1
        if X[i]!=predict[i] and X[i] == target[i]: fp+=1
     
        if X[i]==predict[i] and X[i] == target[i]: tn+=1
        if X[i]==predict[i] and X[i] != target[i]: fn+=1
        
        if X[i]!=predict[i] and X[i] != target[i] and predict[i] != target[i]: incor+=1
        if X[i]!=predict[i] and X[i] != target[i]  and predict[i] == target[i]: cor+=1

    print(f"\n{tp} {fp}\n{fn} {tn}")
    #plot_2d_conf_mat((tp,fp,fn,tn))
    if tp+fn == 0 : sensitivity = recall = 0
    else: sensitivity = recall = tp/(tp+fn) # % correctly identifying positives
    if tn+fp == 0 : specificity = 0
    else: specificity = tn/(tn+fp) # % correctly identifying negatives
    if tp+fp == 0 : precision = 0
    else: precision = tp/(tp+fp) # 
    if cor+incor == 0: cor_accuracy=0
    else: cor_accuracy = cor/(cor+incor)
    #f1 = 2*(precision*recall)/(precision+recall)
    #print(f"sensitivity: {sensitivity} specificity: {specificity} precision: {precision} f1: {f1}")
    print(f"sensitivity {sensitivity:.4f}, specificity {specificity:.4f}, precision {precision:.4f}, cor_accuracy {cor_accuracy:.4f}")

    TPR = sensitivity
    FPR = 1-specificity #= fp/(fp+tn)
    #print(f"true positive rate: {TPR} false positive rate: {FPR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode anahash cleaned files..")
    parser.add_argument("--errfile", help="path to error filled file")
    parser.add_argument("--clnfile", help="path to cleaned file ")
    parser.add_argument("--tarfile", help="path to target file")
    parser.add_argument("--clnfolder",help="path to decoded and cleaned files")
    parser.add_argument("--outfolder",help="path to output folder")

    args = parser.parse_args()
    if args.clnfolder != None:
        dir_list = os.listdir(args.clnfolder)
        dir_list.sort(key=lambda f: int(re.sub('\D', '', f)))
        #print("file_name,sensitivity,specificity,precision,f1")
        for fn in dir_list:
            print(fn,end=",")
            get_metrics(err=args.errfile,
                    cln=args.clnfolder+'/'+fn,
                    tar=args.tarfile)
    else: get_metrics(err=args.errfile,cln=args.clnfile,tar=args.tarfile)

