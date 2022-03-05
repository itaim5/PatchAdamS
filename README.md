# PatchAdamS
Imperceptible Multi-Patch Adversarial Attacks via Adversarial Programs

Run instructions:

main.py -d \<dataset\> -g \<GPUs\> -p \<parallel\> -s \<samples\> -a \<attacks_per_sample\> -i \<num_instructions\> -c \<num_candidates\>

dataset - mnist / cifar10 / fashion-mnist / imagenet  
GPUs - indices of allowed gpus, separated by commas, for example: "-g 0,1,2,3"  
parallel - number of concurrent attacks for a single GPU  
samples - number of test set samples for attacks generation  
attacks_per_samples - number of different attacks (different target class) per sample  
num_instructions - number of maximal instructions (patches) in a program  
num_candidates - number of candidates per iteration  
  
Run example:  
main.py -d mnist -g 0,1,2,3 -p 4 -s 30 -a 9 -i 5 -c 7
