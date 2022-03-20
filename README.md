# J2PREc
## Introduction 

Joint Product-Participant Recommendation (J2PRec) is a new framework for group-buying (GB) recommendation, which recommends both candidate products and participants for maximizing the success rate of a GB. J2PRec first encodes the various relations in GB for learning enhanced user and product embeddings, and then jointly learns the product and participant recommendation tasks under a probabilistic framework to maximize the GB likelihood.

## Environment Requirement

+ Python == 3.6.5. 
+ tensorflow == 1.14.0
+ numpy == 1.15.4
+ scipy == 1.1.0
+ sklearn == 0.20.0



## Datasets

+ beibei & brightkite & gowalla

   
   + social_relation.txt
         + Introduction: social relations among users. 
         + Format:
               (<user_id1>\t<user_id2>), denoting there is a social relation between the two users.
   + train_id.txt: it contains data for training. Each line is a user with a list of her interacted items. 
   + tune.txt: it contains data for validation. Each line is a user with her validation set. 
   + test.txt: it contains data for testing. Each line is a user with her test item.
   + test.negative.txt: 
   + We provide an illustration example on brightkite dataset. The beibei and gowalla datasets can be downloaded here: https://drive.google.com/drive/folders/1PUUP7mA2xR-suP108jreXyOT-EcjnXBd?usp=sharing. 
   
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
   
   
   
   
