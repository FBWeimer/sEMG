"""

DOI: 10.1109/TNSRE.2020.3024947

References to pre-processing, extracted and numerated as paper:
    
[17] V. H. Cene and A. Balbinot, “Using the sEMG signal representativity
improvement towards upper-limb movement classification reliability,”
Biomed. Signal Process. Control, vol. 46, pp. 182–191, Sep. 2018.

[23] T. R. Farrell and R. F. Weir, “The optimal controller delay for myoelectric prostheses,” IEEE Trans.
Neural Syst. Rehabil. Eng., vol. 15, no. 1,pp. 111–118, Mar. 2007.

[24] T. R. Farrell, “Determining delay created by multifunctional prosthesis
controllers,” The J. Rehabil. Res. Develop., vol. 48, no. 6, pp. 21–37,
2011.

[29]  M. Atzori et al., “Characterization of a benchmark database for myoelectric movement classification,”
IEEE Trans. Neural Syst. Rehabil. Eng.,
vol. 23, no. 1, pp. 73–83, Jan. 2015. 

[30] V. Cene, M. Tosin, J. Machado, and A. Balbinot, “Open database
for accurate upper-limb intent detection using electromyography and
reliable extreme learning machines,” Sensors, vol. 19, no. 8, p. 1864,
Apr. 2019.

[31]  I. Kuzborskij, A. Gijsberts, and B. Caputo, “On the challenge
of classifying 52 hand movements from surface electromyography,” in Proc. Annu. Int. Conf. Eng. Med. Biol. Soc., Aug. 2012,
pp. 4931–4937.
"""

import numpy as np
from numpy import savetxt

import pandas as pd

import scipy.io as sio
from scipy import stats
from scipy import signal

import pickle

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split



class Data:
    
    def __init__(self, folder):
        
        self.folder = folder
    
    def load_iee(self, database, subj, exerc, assay):
        """


        Parameters
        ----------
        folder : str
            FOLDER WHERE THE DATA IS SAVED.
        database : str
            NAME OF THE DATABASE.
        subj : int
            NUMBER OF THE SUBJECT IN DATABASE.
        exerc : int
            NUMBER OF THE EXERCISE IN DATABASE.
        assay : str
            TYPE OF ASSAY IN DATABASE (A, B, C or D)
    
        Returns
        -------
        emgRAWData : numpy.ndarray
            MATRIX WHERE COLUMNS ARE THE CHANNEL AND LINES ARE THE EMG RAW DATA IN TIME.
        labels : numpy.ndarray
            VECTOR CONTAINING THE LABEL OF EACH MOVEMENT.
        emgRAWData_df: pandas.core.frame.DataFrame
            DATAFRAME CONTAINING THE EMG RAW DATA AND LABELS
    
        """  
        
        path = (self.folder + database + 's' + str(subj) + '/S' + str(subj) + \
                '_'  + assay + str(exerc) + '.mat')
            
        allData = sio.loadmat(path)
        emg_data = np.double(allData['emg'])
        labels = np.double(allData['restimulus'])
        
        emg_data = pd.DataFrame(emg_data)
        emg_data['label'] = labels
        emg_data.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 'label']
        
        return emg_data
    
    
    def save_pickle(self, path, X_EMG_train, y_emg_train, X_EMG_test, y_emg_test):

      with open(self.path+'emg_s1a1.pkl', mode = 'wb') as f:
    
        pickle.dump([X_EMG_train, y_emg_train, X_EMG_test, y_emg_test], f)
        
        


class Signal:
    
    def __init__(self, emg_raw_data, sF, nCh):
        
        self.emg_raw_data = emg_raw_data
        self.sF = sF
        self.nCh = nCh
        
    
    def remove_null(self, status=False):
        """
        
        
        Parameters
        ----------
        emgData : pandas.core.frame.DataFrame
            DATAFRAME CONTAINING THE EMG RAW DATA AND LABELS.
    
        Returns
        -------
        emgData : pandas.core.frame.DataFrame
            DATAFRAME CONTAINING THE EMG DATA AND LABELS, WITHOUT NULLS VALUES.
    
        """

        if status == False:
            
            pass
        
        elif status == True:
            
            print(self.emg_raw_data.isnull().all())
            
        self.emg_raw_data.apply(lambda x: x.fillna(x.mean(), inplace=True))
        emg_data_nonull = self.emg_raw_data
            
            
        return emg_data_nonull
    
    
    def butter_filter(self, emg_data, low=5, high=500, band_type='bandpass', order=4):
        

        b, a = signal.butter(4, [low, high], btype=band_type, fs=self.sF)
        channels = list(emg_data.loc[:, emg_data.columns != 'label'])
        emg_filtered = pd.DataFrame()
        
        for ch in channels:
            
            emg_filter_ch = emg_data[ch].values
            emg_filtered_ch = signal.lfilter(b, a, emg_filter_ch)
            emg_filtered[ch] = emg_filtered_ch
            
        emg_filtered['label'] = emg_data['label'].values
        
        return emg_filtered
    
    
    def avt_filter(self, emg_data, ff1, ff2, twS, overlapS):
        """
        
    
        Parameters
        ----------
        emgData : pandas.core.frame.DataFrame
            DATAFRAME CONTAINING THE EMG DATA AND LABELS.
        
        ff1 : float
            FACTOR OF MOVING AVERAGE BEHAVIOR OF THE FILTER
            
        ff2 : float
            FACTOR OF DYNAMIC FREQUENCY BEHAVIOR OF THE FILTER
    
        Returns
        -------
        emg : pandas.core.frame.DataFrame
            DATAFRAME CONTAINING THE EMG FILTERED DATA AND LABELS.
    
        """
        
        
        """
        
        Anthonyan Verdan Transform (AVT) filter [17], [30]
        Statistical Filtering.
        
        """
    
    
        emg = emg_data[['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']].values
        label = emg_data['label'].values
        
        ff1 = 0.8 # factor of moving average behavior of the filter
        ff2 = 0.2 # factor dynamic behavior of the filter
                
        tS = len(emg) # number of samples of EMG data
               

        for i in range(0, tS, overlapS):
            
            if (i + twS) > tS:
                
                break
            
            for ch in range(self.nCh):
                
                
                MSA = (np.mean(emg[i:i+twS-overlapS, ch])*ff1 + np.mean(emg[i+twS-overlapS:i+twS, ch])*ff2)
                MSSD = (np.std(emg[i:i+twS-overlapS, ch])*ff1 + np.std(emg[i+twS-overlapS:i+twS, ch])*ff2)
                
                highLim = MSA + MSSD
                lowLim = MSA - MSSD
                
                filteredData = np.where(emg[i:i+twS, ch] < lowLim, MSA, emg[i:i+twS, ch])
                filteredData = np.where(filteredData > highLim, MSA, filteredData)
                
                emg[i:i+twS, ch] = filteredData
        
        emg_data_avt = pd.DataFrame(emg)
        emg_data_avt.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
        emg_data_avt['label'] = label
        
        return emg_data_avt
    
    def moving_average(self, emg_data):
        
        filtered_DES = np.zeros(np.shape(emg_data))
        data = np.absolute(emg_data)
    
        N=10
        
        for i in range(self.nCh):
    
            data_padded = np.pad(data[:,i], (N//2, N-1-N//2), mode='edge')
            filtered_DES[:,i] = np.convolve(data_padded, np.ones((N,))/N, mode='valid') 
            
        return filtered_DES
    
    def feature_extraction(self, emg_data, twS, overlapS): 
    
        """
        Statistical Filtering.
        """
        
        emg = emg_data.loc[:, emg_data.columns != 'label'].values
        tS = len(emg)
        lengthFeat = int(np.round(((tS-twS)/overlapS)))
        
        label = np.zeros((lengthFeat, 1))
        feat_rms = np.zeros((lengthFeat, self.nCh))
        feat_std = np.zeros((lengthFeat, self.nCh))
        feat_var = np.zeros((lengthFeat, self.nCh))
        feat_mav = np.zeros((lengthFeat, self.nCh))
        feat_DES = np.zeros((lengthFeat, self.nCh))
        
        idx = 0   
        
        for i in range(0, tS-twS, overlapS):
            
            for ch in range(self.nCh):
                
                if ch==0:
                    
                    label[idx,0] = emg_data['label'][i:i+twS].mode()[0]
                    
                else:    
                    
                    feat_rms[idx,ch-1] = np.sqrt(np.mean(np.square(emg_data[str(ch+1)][i:i+twS].values)))
                    feat_std[idx,ch-1] = emg_data[str(ch+1)][i:i+twS].std()
                    feat_var[idx,ch-1] = emg_data[str(ch+1)][i:i+twS].var()
                    feat_mav[idx,ch-1] = emg_data[str(ch+1)][i:i+twS].mean()
                    
            if idx==0:    
                
                a = feat_mav[idx,:] #np.mean(signal[i:i+twS, 1:nCh])
                b = np.ones((self.nCh))
                
            elif idx > 0 and idx < lengthFeat-2: 
                
                b = a
                a = feat_mav[idx,:]
                
            elif idx == lengthFeat-1:
                
                break
            
    
            c = np.concatenate(([a], [b]), axis=0)   
            pca = PCA(n_components=1)
            pca.fit(c)
            feat_DES[idx,:] = pca.components_
            
            idx += 1    
        
        filtered_DES = self.moving_average(feat_DES)
        
        columns_list = ['label']
        for ch in range(1, 12+1): columns_list.append('RMS_'+str(ch))
        for ch in range(1, 12+1): columns_list.append('STD_'+str(ch))
        for ch in range(1, 12+1): columns_list.append('VAR_'+str(ch))
        for ch in range(1, 12+1): columns_list.append('MAV_'+str(ch))
        for ch in range(1, 12+1): columns_list.append('DES_'+str(ch))
        
        df_label = pd.DataFrame(label)
        df_RMS = pd.DataFrame(feat_rms)
        df_STD = pd.DataFrame(feat_std)
        df_VAR = pd.DataFrame(feat_var)
        df_MAV = pd.DataFrame(feat_mav)
        df_DES = pd.DataFrame(filtered_DES)
        
        emg_data_feat = pd.concat([df_label, df_RMS, df_STD, df_VAR, df_MAV, df_DES], axis=1)
        
        emg_data_feat.columns = columns_list
        
        
        return emg_data_feat
    
    def normalize_signal(self, emg_data):
        
        columns_list = list(emg_data)
        emg_data_norm = pd.DataFrame(normalize(emg_data.loc[:, emg_data.columns != 'label']))
        emg_data_norm['label'] = emg_data['label']
        emg_data_norm.columns = columns_list 

        
        return emg_data_norm

    
    def plot(dfEMG, title='sEMG signal'):
    
        df = dfEMG.loc[:, dfEMG.columns != 'label']
        sns.set()
        fig = plt.figure(figsize=(20, 10))
        plt.plot(df)
        fig.suptitle(title, fontsize=25)
        fig.subplots_adjust(top=0.95)
        plt.xlabel('Samples [n]', fontsize=22)
        plt.ylabel('Amplitude (V)', fontsize=22)
        plt.plot()
