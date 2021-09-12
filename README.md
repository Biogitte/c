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
