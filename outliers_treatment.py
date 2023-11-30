import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class OutliersTreatment:
    """
        Treat with numerical outliers for each column in the DataFrame.

        Using two type of methods to determine the ouliers (iqr, z_score) dependeing on if the data is normally distributed or not.

        normally distributed => z_score
        skewed distribution => iqr

        Args:
            data    : The whole dataframe.
            columns : Selected columns in the dataframe.

        Functions:
            iqr and z_socre : calculate the min and max cutoff thresholds.
            fit: perform the outliers indices in the dataframe.
            transorm: transform the old data to new one (after removing outliers).
            plot: plot box plot for old and new data, also plot the distribution of the data before removing the data.
    """
    def __init__(
        self, 
        data: pd.DataFrame,
        columns: list = [],
    ):
        # defining the input data
        self.data = data
        # determine only the numerical columns
        self._numerical_columns = data[columns if not pd.Index(columns).empty else data.columns].select_dtypes(include=np.number).columns
        
        # check for numerical columns
        assert not self._numerical_columns.empty, "There is no numerical data in the selected columns" 
        
        # p-value(normal test) for each column
        self._normal_tests = stats.normaltest(data[self._numerical_columns])[1]
        
    @staticmethod
    def iqr(col_data):
        q1, q3 = col_data.quantile([0.25, 0.75])
        iqr = q3 - q1
        return q1 - 1.5 * iqr, q3 + 1.5 * iqr, col_data # min_threshold, max_threshold, new_data_col
    
    @staticmethod
    def z_score(col_data):
        return -3, 3, pd.Series(stats.zscore(col_data)) # min_threshold, max_threshold
    
    def fit(self):
        self.outliers_indices_ = pd.Index([])
        
        for indx, col in enumerate(self._numerical_columns):
            # normal distribution test to choose which method iqr or zscore
            outlier_method = self.z_score if self._normal_tests[indx] >= .05 else self.iqr
            
            min_thd, max_thd, new_data_col =  outlier_method(self.data[col])
            
            indices = new_data_col[(new_data_col < min_thd) | (new_data_col > max_thd)].index
            
            self.outliers_indices_ = self.outliers_indices_.union(indices)
            
        return self.outliers_indices_
            
    def transform(self):
        assert hasattr(self, "outliers_indices_"), "You should perform the fit function first" # make sure that fit is implemented
        
        self.new_data = self.data.drop(index=self.outliers_indices_).reset_index(drop=True)
        return self.new_data
    
    def plot(self):
        assert hasattr(self, "new_data"), "You should perform the fit & transform functions first" #make sure that fit and transform are implemented
        
        fig = plt.figure(figsize=(25, 30))
        fig.suptitle("Outliers Statistics")
        fig.subplots_adjust(hspace=0.5)
        sns.set_theme(style="whitegrid", palette="pastel")
        
        subfigs = fig.subfigures(len(self._numerical_columns), 1)

        for indx, col in enumerate(self._numerical_columns):
            subfigs[indx].suptitle(f"Outliers for {col} column")
                
            (ax1, ax2, ax3) = subfigs[indx].subplots(1, 3)
            
            sns.boxplot(data=self.data, x=col, ax=ax1) # before removing outliers
            sns.boxplot(data=self.new_data, x=col, ax=ax2) # after removing outliers
            sns.histplot(self.data[col], kde=True, ax=ax3) # data distribution before removing outliers
            
            subfigs[indx].subplots_adjust(bottom=0.25, right=0.8, top=0.75)

            ax1.set_title("before removing outliers")
            ax2.set_title("after removing outliers")
            ax3.set_title(f"data distribution before removing outliers")
            ax3.set_xlabel(f"{col} ({'Noraml Distribution' if self._normal_tests[indx] >= .05 else 'Skewed Distribution'})")

