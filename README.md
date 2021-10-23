Prerequisites
-------------
* Python 3.x
* Set up Kaggle authentication: on your Kaggle account, under API, select `Create New API Token`, and a `kaggle.json` file will be downloaded on your computer. Move this file to `~/.kaggle/kaggle.json` on MacOS/Linux. Remember to run `chmod 600 ~/.kaggle/kaggle.json` to ensure it is only readable for you.
* Mac OSX users: Run `brew install libomp` to install OpenMP runtime (for Xboost). This step requires that homebrew has been installed.

Get started
------------

     # install virtualenv
     pip3 install virtualenv
     
     # create a virtual environment
     virtualenv venv --python=<path-to-python-3.*>
     
     # activate environment
     source venv/bin/activate
     
     # install requirements
     pip3 install -r requirements.txt
     
     # set the global environment variables
     source global_env.sh
     
     # install local python packages
     python3 setup.py install
     pip3 install -e .


Download, clean and encode the data
-----------------------------------
Kaggle data set used in the example: [the stroke-prediction-dataset from fedesoriano](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)

    sh $CLEAN_THE_DATA
    
View output files (e.g., CSV and HTML files) in `/data/processed`.
    
Compare binary classification models
------------------------------------

Quick and simple comparison of common binary classificagtion algorithms.

    sh $COMPARE_BINARY_CLFS
    
View output files (e.g., figures) in `/data/processed`.
    
Repository overview
-------------------

    .
    ├── README.md                       # README file
    ├── data                            # Directory for storing data
    │   ├── processed                   # Directory for processed - and related files
    │   └── raw                         # Directory for raw data
    ├── executables                     # Executable scripts
    │   └── data_prep.sh                # Executable for data cleaning and encoding
    ├── global_env.sh                   # Global environment variables
    ├── notebooks                       # Directory for Jupyter Notebooks
    │└── stroke_classification.ipynb    # Compare classifiers on a stroke dataset
    ├── requirements.txt                # Environment dependencies
    ├── setup.py                        # Setup script to install local modules
    └── src                             # Source code
        ├── __init__.py                 # __init__ file to create package
        ├── classifiers.py              # Classification related classes and functions
        ├── clean_data.py               # Data cleaning and encoding
        ├── explore.py                  # Various classes and functions to explore data 
        ├── fetch_kaggle_data.py        # Fetch Kaggle data sets
        └── plotter.py                  # Classes and functions for plotting

