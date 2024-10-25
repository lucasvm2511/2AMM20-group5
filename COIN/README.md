# COIN model experiments

Original code of the Siren model: https://github.com/vsitzmann/siren

To reproduce the results mentioned in the paper first you have to create a virtual environment (preferably with `Python 3.10`) and install all the required dependencies mentioned in the `requirements.txt` file:

After that, there are 5 notebooks containing code that was used to run experiments with COIN model:
- `train_coin_eurosat_all_bands` -  training loop for the COIN model for the EuroSAT dataset (13 bands)
- `train_coin_eurosat_rgb` -  training loop for the COIN model for the EuroSAT dataset (RGB)
- `train_coin_all_bands` -  training loop for the COIN model for our dataset (13 bands)
- `train_coin_rgb` -  training loop for the COIN model for our dataset (13 bands)
- `comparison` - visualisations and interpretation of the results

Note: don't forget to change paths to the datasets in the notebooks (by default it is assumed the data lies in the `data` subdirectoey)

In the `src` subdirectory you can find the following files:

- `data.py` - file with all datasets used in the research
- `model.py` - file with the code of the COIN model used in the research
- `utils.py` - file with useful functions (e.g. metrics, visualisations, image preprocessing)


