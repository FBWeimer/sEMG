import sEMG.preprocessing as preEMG
from sklearn.model_selection import train_test_split


folder = 'D:/Estudo/Engenharia Biomédica/Base de Dados/'
database = 'Mão-Braço IEE/'
subj = '1'
exerc = '1'
assay = 'A'

data = preEMG.Data(folder)
emg_data = data.load_iee(database, subj, exerc, assay)
emg_raw = preEMG.Signal(emg_data, 2000, 12)
emg_nonull = emg_raw.remove_null()
emg_filtered = emg_raw.butter_filter(emg_nonull)
emg_avt = emg_raw.avt_filter(emg_filtered, 0.8, 0.2, 300, 15)
emg_feat = emg_raw.feature_extraction(emg_avt, 300, 15)
emg_norm = emg_raw.normalize_signal(emg_feat)

X_EMG = emg_norm.loc[:, emg_norm.columns != 'label'].values
y_emg = emg_norm['label'].values

X_EMG_train, X_EMG_test, y_emg_train, y_emg_test = train_test_split(X_EMG, y_emg, test_size=0.25, random_state=0)

database = 'Pré-Processamento/IEE'

data.save_pickle(folder+database+'emg_s1a1.pkl', X_EMG_train, y_emg_train, X_EMG_test, y_emg_test)