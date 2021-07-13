"""
USING GOOGLE COLABORATORY IN THIS EXAMPLE
"""

#Importing packages
import sEMG.preprocessing as preEMG
from sklearn.model_selection import train_test_split
from google.colab import drive

drive.mount('/content/drive/') #Mounting Drive in Colab

# Declare path, database and exercise type
folder = '/content/drive/MyDrive/Bases de Dados/'  # Folder where the database it is saved
database = 'Hand-Arm/IEE/'  # The database name
subj = '1'  # Subject of the database
exerc = '1'  # Exercise of the database
assay = 'A'  # Assay of the database

data = preEMG.Data(folder)  # Defining data object
emg_data = data.load_iee(database, subj, exerc, assay)  # Loading EMG data
emg_raw = preEMG.Signal(emg_data, 2000, 12)  # Defining signal objetct
emg_nonull = emg_raw.remove_null()  # Filtering null values in EMG data
emg_filtered = emg_raw.butter_filter(emg_nonull)  # Filtering EMG data with bandpass Butterworth filter
emg_avt = emg_raw.avt_filter(emg_filtered, 0.8, 0.2, 300, 15)  # Filtering EMG data with AVT filter
emg_feat = emg_raw.feature_extraction(emg_avt, 300, 15)  # Extracting features from  EMG data
emg_norm = emg_raw.normalize_signal(emg_feat)  #  Normalizing EMG data

X_EMG = emg_norm.loc[:, emg_norm.columns != 'label'].values  # Selecting predictor variables
y_emg = emg_norm['label'].values  # Selecting response variables


# Spliting EMG data to train and test
X_EMG_train, X_EMG_test, y_emg_train, y_emg_test = train_test_split(X_EMG, y_emg, test_size=0.25, random_state=0)

folder = '/content/drive/MyDrive/Bases de Dados/Hand-Arm/'  # Folder where desire save the pre-processing EMG data
database = 'Pre-Processing/IEE'  # Pre-processing database folder


# Saving the EMG data in the selected folder
data.save_pickle(folder+database+'emg_s1a1.pkl', X_EMG_train, y_emg_train, X_EMG_test, y_emg_test)
