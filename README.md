Repository for Neuroimaging Data Catalysis Augmentation Pipeline Described in Paper: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5240695

Creating the pipeline is split into two parts:
  Part 1: Training of the WCGAN on SUVR data

          1: Use scripts/wgan-suvr.py or scripts/wgan-suvr-longform.py to train the WCGAN (either on given SUVR data or data of your own) - 
             use wgan-suvr.py for shorter training protocols and wgan-suvr-longform.py for longer training protocols that you would like to checkpoint
          2: You can alter the architecture of the WCGAN inside scripts/lib/gan_architecture.py

  Part 2: Linking trained WCGAN to Machine Learning Classifiers

          1: Run scripts/cgan_link_training_stat_sig.py 
          2: Select which machine learning classifier you want to optimize parameters for/use for classification (XGBoost, SVM, RandomForest, DecisionTree, LogReg)
          3: Synthetic data generation and filtration process will augment and test synthetic data points for model classification performance enhancement
          4: You can change the amount of synthetic data points tested by changing the "training_iterations" conditional of the while loop for model training.
             WCGAN is set to produce synthetic samples in batches of 100 (100 training iterations means 100*100 total synthetic data points tested.
          5: Successful synthetic candidates that passed the filtration process will be saved inside the scripts folder, along with stats about 
             the training process
![Demo](media/github_readme_part2-ezgif.com-video-to-gif-converter.gif)
