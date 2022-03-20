# J2PREc
## Introduction 

Joint Product-Participant Recommendation is a new framework for group-buying (GB) recommendation. J2PRec fully exploits the various GB relations. 

## Environment Requirement

+ Python == 3.6.5. 
+ tensorflow == 1.14.0
+ numpy == 1.15.4
+ scipy == 1.1.0
+ sklearn == 0.20.0



## Datasets

+ beibei & brightkite & Yelp
   + In the Ciao and Epinion datasets, we have user' ratings towards items. The data is saved in a txt file (rating.txt) and the format is as follows:
   
         userid itemid rating
   
   + trust.txt: it contains the trust relations between users. There are two columns and both of them are userid, denoting there is a social relation between two users. 
   
         userid userid
         
   + train.txt: it contains data for train. Each line is a user with a list of her interacted items. 
   + test.txt: it contains data for test. Each line is a user with a list of her test items.
   
## Running Command 

+ Data Preparation 

    Please run the data_preprocess.py in the Data folder to generate the required files. 

+ Model Training and Testing

    Please run the following example codes for model training and testing: 

         python3.6 Main.py --dataset brightkite --regs [1e-4,1e-4] --layer_size [64,64] --embed_size 64 --lr 0.001 --epoch 1000 --batch_size 1024 

   
    You need to specify serveral parameters for training and testing:
   
    + dataset: beibei / brightkite / Yelp
    + regs: regularization weight 
    + layer_size: the number of layers and embedding size 
    + lr: learning rate
    + batch_size : the size of batch for training
    + epoch : the epoch for training 
   
   
   
   
