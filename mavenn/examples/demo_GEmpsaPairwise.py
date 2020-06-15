import mavenn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from mavenn.src.utils import ge_plots_for_mavenn_demo

# load data
mpsa_df = pd.read_csv(mavenn.__path__[0]+'/examples/datafiles/mpsa/psi_9nt_mavenn.csv')
mpsa_df = mpsa_df.dropna()
mpsa_df = mpsa_df[mpsa_df['values'] > 0]  # No pseudocounts

# split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(mpsa_df['sequence'].values, np.log10(mpsa_df['values'].values))

# load mavenn's GE model
GER = mavenn.Model(regression_type='GE',
                   X=x_train,
                   y=y_train,
                   model_type='pairwise',
                   learning_rate=0.001,
                   alphabet_dict='rna')

# fit model to data
GER.fit(epochs=200, use_early_stopping=True, early_stopping_patience=20, verbose=1)

# make predictions on held out test set
loss_history = GER.model.return_loss()
predictions = GER.model.predict(x_test)


# plot results using helper function
ge_plots_for_mavenn_demo(loss_history, predictions, y_test, x_test, GER)


