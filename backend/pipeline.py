# pipeline.py -> pre-processing, feature extraction and classification are all done in this file. 

#Import Statements
import os, sys
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import sys
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy import stats as st
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pickle
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, roc_curve, RocCurveDisplay, roc_auc_score


class data_pipeline:
    
    def __init__(self, data):
        #Initialize class
        self.data = data
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = None
        self.history = None
        
    def get_all_data(self, dir, append_class_label):
        # Function to collect all files of any type and put into a pandas dataframe. Depending on the starting label of the file, 
        # function can append a class label to data if append_class_label = True 
        
        #Lists to hold strings of directory paths
        walk_list = []
        jump_list = [] 
        file_list = [] #holds all csv's
        
        #create hdf5 file and subgroups for data
        hdf = h5py.File('datafile.hdf5', 'w')
        jgroup = hdf.create_group('/Jonah')
        lgroup = hdf.create_group('/Liam')
        sgroup = hdf.create_group('/Salvina')
      
        # Loop through all files in specified dir and attempt to add them to data if they're a csv
        for path in tqdm(os.listdir(dir)):
            csv_path = os.path.join(dir, path)
            if csv_path.endswith(".csv"):
                
                #Check if path starts with w or j and put into respective list
                if (path.startswith('w')):
                    walk_list.append(csv_path)
                elif (path.startswith('j')):
                    jump_list.append(csv_path)

                # inserting data for each team member in hdf5 file
                if ('liam' in csv_path):
                    lgroup.create_dataset(csv_path, data=pd.read_csv(csv_path))
                elif ('jonah' in csv_path):
                    jgroup.create_dataset(csv_path, data=pd.read_csv(csv_path))
                elif('salvina' in csv_path):
                    sgroup.create_dataset(csv_path, data=pd.read_csv(csv_path))

                #Put all file directories in file list
                file_list.append(csv_path)
            else:
                print("Skipping over file", csv_path, "because not type readable type") 
        hdf.close()      
        #Return df with or without w/r column added depending on function parameter
        if(append_class_label):
            #If both are dataframes are not empty
            if ((len(jump_list) != 0) and (len(walk_list) != 0)):
                #Turn into pandas dataframe
                walk_df = pd.concat(map(pd.read_csv, walk_list), ignore_index=True)
                jump_df  = pd.concat(map(pd.read_csv, jump_list), ignore_index=True)
                
                #Append labels
                walk_df['label'] = 0
                jump_df['label'] = 1

                #Conjoin the two dataframes and return
                self.data = pd.concat([walk_df, jump_df])
                return self.data
            
            #In the case that only walk or jump data was found
            elif (len(walk_list) != 0):
                
                self.data = pd.concat(map(pd.read_csv, walk_list), ignore_index=True)
                self.data['label'] = '0'
                return self.data
                
            elif (len(jump_list) != 0):
                
                self.data  = pd.concat(map(pd.read_csv, jump_list), ignore_index=True)
                self.data['label'] = '1'
                return self.data
            
            else:
                print ("Both lists are empty, either missing 'w' or 'j' labeled csv files or no data found")
        
        #If append_class_label is false, return all files without appended label using the file_list list of all csvs     
        else:
            self.data = pd.concat(map(pd.read_csv, file_list), ignore_index=True)
            return self.data
  

    def filterdata(self, window_size, flag):
        #remove nan values 
        if self.data.isnull().values.any():
            self.data.interpolate(method="linear", inplace=True)
        
        #using moving average filter to remove noise
        filteredData= pd.DataFrame()
        filteredData['x_filtered'] = self.data.iloc[:, 1].rolling(window_size).mean()
        filteredData['y_filtered'] = self.data.iloc[:, 2].rolling(window_size).mean()
        filteredData['z_filtered'] = self.data.iloc[:, 3].rolling(window_size).mean()
        if (flag == 1):
            filteredData['label'] = self.data.iloc[:, 5]
        filteredData = filteredData.iloc[(window_size-1):,:] #remove first 5 (n-1) 
        self.data = filteredData
        return self.data
    
    def findfeatures(self, coordinate_list, listname):
        for window in coordinate_list: 
            maxim = np.max(window)
            minim = np.min(window)
            range_win = maxim - minim
            mean = np.mean(window)
            median = np.median(window)
            std = np.std(window)
            skew_win = skew(window)
            kurt_win = kurtosis(window)
            variance = np.var(window)
            rms_win = np.sqrt(np.mean(window**2))
            feature = [maxim, minim, range_win, mean, median, std, skew_win, kurt_win, variance, rms_win]
            listname.append(feature)

    def extractfeatures(self, window_size, flag):
        # seperate dataframe into 4 seperate lists with window size
        x_values, y_values, z_values, labellist = [],[],[],[]
        for i in range(0, self.data.shape[0] - window_size, window_size):
            newdf = self.data[i:i+window_size]
            x_values.append(newdf.iloc[:, 0].values)
            y_values.append(newdf.iloc[:, 1].values)
            z_values.append(newdf.iloc[:, 2].values)
            
            if (flag == 1):
            #find mode of label in window to determine label value 
                labellist.append(st.mode(newdf.iloc[:, 3].values, keepdims=True)[0][0])

        #find features for each coordinate in each window
        x_features, y_features, z_features = [], [], []
        self.findfeatures(x_values, x_features)
        self.findfeatures(y_values, y_features)
        self.findfeatures(z_values, z_features)

        #find correleation betwween pairs of coordinate for each window
        corr_coefxz, corr_coefyz, corr_coefxy = [], [], []
        for x_list, y_list, z_list in zip(x_values, y_values, z_values):
            corr_coefxy.append(np.corrcoef(x_list, y_list)[0, 1])
            corr_coefyz.append(np.corrcoef(y_list, z_list)[0, 1])
            corr_coefxz.append(np.corrcoef(x_list, z_list)[0, 1])

            # initialize array that will hold data for each window - 10 features for each coordinate
        everything = np.empty((len(x_features), 30))

        # concatenate coordinate arrays for each window 
        for i in range(len(x_features)):
            everything[i] =  x_features[i] + y_features[i] + z_features[i]

        # create dataframe
        features = pd.DataFrame(everything, columns=['x_max', 'y_max', 'z_max','x_min', 'y_min', 'z_min','x_range', 'y_range', 'z_range','x_mean', 'y_mean', 'z_mean',
                                                    'x_median', 'y_median', 'z_median','x_std', 'y_std', 'z_std', 'x_skew', 'y_skew', 'z_skew','x_kurt', 'y_kurt', 'z_kurt',
                                                    'x_variance', 'y_variance', 'z_variance', 'x_rms','y_rms', 'z_rms' ])
        
        # add additional feature to dataframe (11 feature total) and labels
        features['coeffxy'] = corr_coefxy
        features['coeffyz'] = corr_coefyz
        features['coeffxz'] = corr_coefxz 
        if (flag == 1):
            features['labels'] = labellist
            features.sample(frac=1).reset_index(drop=True)
        # data
        
        self.data = features
        return self.data
    
    def train(self):
        features = self.data
        #CLASSIFICATION
        x = features.iloc[:, :-1]
        y = features.iloc[:, -1]
        #90/10 data split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.1)


        with h5py.File('datafile.hdf5', 'a') as hdf:
            traingroup = hdf.create_group("/dataset/Train")
            testgroup = hdf.create_group("/dataset/Test")
            traingroup.create_dataset('x_train', data=self.x_train)
            traingroup.create_dataset('y_train', data=self.y_train)
            testgroup.create_dataset('x_test', data=self.x_test)
            testgroup.create_dataset('y_test', data=self.y_test)
            
        hdf.close()
        ss = StandardScaler()
        l_reg = LogisticRegression(max_iter=10000)
        clf = make_pipeline(ss, l_reg)
        clf.fit(self.x_train, self.y_train)
        y_pred = clf.predict(self.x_test)
        y_clf_prob = clf.predict_proba(self.x_test)
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("\n -------------Classification Report-------------\n")
        print(classification_report(self.y_test, y_pred))
        cm = confusion_matrix(self.y_test, y_pred)

        # Define labels for the classes
        labels = ['Walking', 'Jumping']

        # Create heatmap plot of the confusion matrix
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels)

        # Add plot and axis labels
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()
        fpr, tpr, _ = roc_curve(self.y_test, y_clf_prob[:,1], pos_label=clf.classes_[1])
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        plt.show()
        auc = roc_auc_score(self.y_test, y_clf_prob[:, 1])
        print('The auc score is: ', auc)

         # Plot the learning curve
        train_sizes, train_scores, val_scores = learning_curve(clf, x, y, train_sizes=np.linspace(0.1, 1.0, 10),cv=5, scoring='accuracy')
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)
       
        plt.figure()
        plt.title('Learning Curve')
        plt.xlabel('Number of Training Examples')
        plt.ylabel('Accuracy')
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std,alpha=0.1, color='r')
        plt.fill_between(train_sizes, val_scores_mean - val_scores_std,val_scores_mean + val_scores_std,alpha=0.1, color='g')
        plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Score')
        plt.plot(train_sizes, val_scores_mean, 'o-', color='g', label='Validation Score')
        plt.legend(loc='best')
        plt.show()


        # Plot the training curve
        pickle.dump(clf, open('model.pkl','wb'))   
        
# Main function if running pipeline.py
def main():
    model = data_pipeline(data =[])
    model.__init__
    model.get_all_data(dir='data', append_class_label = True)
    model.filterdata(window_size = 5, flag=1)
    model.extractfeatures(window_size = 500, flag=1)
    model.train()
    

#Call to main
if __name__ == "__main__":
    main()
    
  