# Challenge ML 2025

## Getting started

### Data

Data can be downloaded throught Kaggle : https://www.kaggle.com/competitions/beacon-biosignals-sleep-staging-2025/data

You can either download it from the browser or use the CLI tool. Once you have your kaggle token you can just run 
```
mkdir data
cd data
uvx kaggle competitions download -c beacon-biosignals-sleep-staging-2025
unzip beacon-biosignals-sleep-staging-2025.zip
rm -rf kaggle  beacon-biosignals-sleep-staging-2025
```

### Code

To run the code just do 
```uv run main.py``` Or ```python run main.py ```

It will generate a hydra output folder

### Submit
Use ./submit_kaggle.sh file.csv "submission name" to submit to kaggle
Make sure you have a token.

## Code structure

utils:
- constants.py easy access to data constants for use in jupyter
- data_io.py : Code for reading data, saving metrics and generating submission
- evaluate.py: Evaluation 

Features :
 - Contains code to extract features from raw data. FeatureEpoch extractor implemented that generates mean,std and ptp for each epoch

Models : 
- Contains models (must have a .fit and a .predict)
##