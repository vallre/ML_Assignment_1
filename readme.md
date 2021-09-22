**Note:** This repository is the part of the Assignment 1 of the Machine Learning course.

### Structure
    .
    ├── data                    # Flight delay dataset
    ├── docs                    # Documentation files
    ├── figures                 # Graphs describing the data
    ├── src                     # Source files
    └── readme.md
    
This repository contains 4 foulders. **data** folder contains a link to an airline flight delay dataset used to train machine learning model. **docs** folder contains supporting material related to the task, as well as the report of the findings. **figures** folder contains all graphs generated by the code. **src** folder contains python scripts that are used to train and test the chosen machine learning approaches and to analyse the data preprocessing and model training stages.

### How to run:

First, download the dataset from the provided link and put it into **data** folder.
The main executable of this assignment is the **Assignment1.py**. Ohter executables are used to further analyze the data related to the dataset and the training/testing processes.

Running **Assignment1.py** will preprocesses the dataset, train and test the 4 different machine learning models, then print model error scores to stdout, and at the end it will show a table of results. The 4 models tested in the this project are Simple Linear Regression, Polynomial Linear Regression of 3 different degrees, Lasso Regression, and Ridge Regression.

Running **data_profile.py** will generate the dataset profile in **docs** folder using pandas_profiling module.

Running **graph_generator.py** will generate a set of images in "figures" folder demostrating different graphs related to the dataset. The graphs include: graphs of all predictors versus the target (Delay in minutes), and graphs of 4 different predictors that seem to show greater relation to the target after removing a few outliers from the "Flight Duration". The names of the generated graphs are **Raw Date vs Delay**, **Raw Place vs Delay**, **Raw Time vs Delay**, and **Graphs after outlier removal**. Also, the program will output the percentage of data lost due to initial outlier removal to the stdout.

Running **train_outlier_analysis.py** will explore the effect of the second outlier removal based on the z-score of "Delay" with different threshold. The model used in the analysis is Second Degree Polynomial Linear Regression (the best performing model). The program will print the test error values to the stdout with the value of z-score threshold used. Also, the program will generate image titled **Effect of Delay outlier removal** in the "figures" folder in oreder to demostrate effect of the outlier removal on the training data itself.
