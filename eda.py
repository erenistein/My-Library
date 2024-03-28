
import numpy as np                                                                                                                      
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('C:\Users\90553\PyKasif\assistant')
from eda import ProfillingReport
...
class ProfillingReport:
    """
        Bir veri setinin kesifci veri analizi icin kullanılan sınıf
        
    """
...
...
def __init__(self, df: pd.core.frame.DataFrame = None, continuous_variables: list = None, categorical_variables: list = None, target_variables: list = None): 
        """  
        Parameters
        ----------
        df : pd.core.frame.DataFrame, optional
            Kullanılacak veri seti.
        continuous_variables : list, optional
            Sürekli değişkenler.
        categorical_variables : list, optional
            Kategorik değişkenler.
        target_variables : list, optional
            Hedef değişkenler.
        """
        self.df = df
        self.variables = df.columns.to_list() if df is not None else []
        self.__continuous_variables = self.set_continuous_variables(continuous_variables)
        self.__categorical_variables = self.set_categorical_variables(categorical_variables)
        self.__target_variables = self.set_target_variables(target_variables)

...  
def __check_it_includes(self, main_list: list = [], sub_list: list = []) -> bool:
    """
    Bir alt listede yer alan bütün elemanların, ana listede olup olmadığını kontrol eder.
    """
    result = True
    for sub_i in sub_list:
        if sub_i not in main_list:
            result = False
            break  # Döngüyü buraya almalısınız, yoksa sadece ilk elemanı kontrol eder ve hemen döngüyü sonlandırır
    return result

 
 
 
... 

...
def set_continuous_variables(self, continuous_variables):
    """
    Sürekli değişkenleri ayarlamada kullanılır
    Parameters 
    ---------
    continuous_variables: list
        Sürekli değişkenler
    Raises
    ------
    AssertionError
        Continuous_variables değişkenin değerleri, veri setinde tanımlanmamışsa.
    Returns
    -------
    list
        Sürekli olan değişkenlerin listesi
    """
    assert self.__check_it_includes(self.variables, continuous_variables), "continuous_variables değişkenin değerleri, veri setinde tanımlanmalıdır"
    
    return continuous_variables

...
...
def get_continuous_variables(self):
    """ Sürekli degiskenleri dönderir
    Returns
    ------
    list 
    Sürekli olan degiskenlerin listesi
    """
    return self.continuous_variables
def set_categorical_variables(self, categorical_variables):
    """ 
    Kategorik degiskenleri ayarlamada kullanır
    Parameters
    ------
    categorical_variables:list
        kategorik degiskenler
    Assertions
    -----
    AssertionError
        categorical_variables degiskenin degerleri , veri setinde tanımlanmamışsa.
        Returns
        ------
    list
    kategorik olan degiskenlerin listesi
    """
    assert(self.__check_it_includes(self.variables,categorical_variables))," categorical_variables degiskenin degerleri , veri setinde tanımlanmalıdır"
    return categorical_variables
...
def get_categorical_variables(self):
    """ Kategorik degiskenleri dönderir
    Returns
    ---------
    list
        kategorik olan degiskenlerin listesi
        """
    return self.__categorical_variables
...
def set_target_variables(self, target_variables):
    """ 
    Hedef degiskenleri ayarlamada kullanılır
    Parameters
    ------
    target_variables: list
        Hedef degiskenler
    Raises
    ------
    AssertionError
        target_variables degiskenin degerleri, veri setinde tanımlanmalıdır.
    Returns 
    --------
    list
        Hedeflenen degiskenlerin listesi
    """
    assert self.check_it_includes(self.variables, target_variables), "target_variables degiskenin degerleri, veri setinde tanımlanmalıdır"
    return target_variables
                            
...
def get_target_variables(self):
    """ Hedef degiskenleri dönderir
    Returns
    -----
    list
        hedef degiskenlerin listesi
        """
    return self.__target_variables
...
def __pie_plot_and_table(self,names:list=None , values:list=None):
    """ Dairesel grafik ve tablo olusturur
        Parameters
        --------
        names:list
            dairesel grafik ve tabloda yer alacak degerlerin isimleri
        values:list
            dairesel grafik ve tabloda yer alacak degerler """
    plt.clf()
    plt.pie(values,labels=names, autopct='%1.1f%%')
    table=plt.table(cellText=np.asarray(values).reshape(-1,1),
    rowLabels=names,fontsize=180,bbox=[2,0.8/len(names),0.5,0.15*len(names)])
    plt.show()
...
def data_types(self):
    """ Veri setindeki veri tiplerni gösteriyor
    """
    self.pie.plot_and_table(names=self.df.dtypes.value_counts().index.astype(str).tolist(), values=self.df.dtypes.value_counts().values.tolist())
...
def missing_cell_count(self):
    """
    Veri setindeki kayıp hücre miktarını gösterir
    """
    missing_cell_count=self.df.isna().sum().sum()
    filled_cell_count=self.df.size-self.df.isna().sum().sum()
    self.pie.plot_and_table(names=['Kayıp Hücreler' , 'Dolu hücreler'],values=[ missing_cell_count, filled_cell_count])
...
...
def duplicate_row_count(self):
    """
    Veri setinde tekrarlayan satir miktarini gösterir
    """
    duplicated_row_count=self.duplicated().sum()
    unduplicated_row_count=self.df.shape[0]-self.df.duplicated().sum()

    self.pie.plot_and_table(  names=['Tekrarlanan satirlar', 'Tekrarlanmayan satırlar'], values=[duplicated_row_count, unduplicated_row_count])
...
def visualize_distribution(self):
    """
    Veri setindeki degiskenlerinin dagılım grafigini oluşturur
    """
    for var in self.variables:
        fig=self.df.loc[:,var].hist()
        plt.title(var)
        plt.show()
...
...
def __create_dispersion_measures(self, feature: str = None):
    """
    Bir özelliğin dağılım ölçülerini oluşturur.
    
    Bunlar:
    - count
    - std
    - min
    - %25
    - %50
    - %75
    - max
    - Skewness
    - Kurtosis
    
    Parameters
    ----------
    feature : str
        Veri setindeki özellik
        
    Returns
    -------
    pandas.core.series.Series
        Dağılım ölçülerini içerir
    """
    df_d = self.df.loc[:, feature].describe().T
    df_d = df_d.drop('mean')  # 'mean' satırını silmek için drop kullanılmalı ve köşeli parantez içinde 'mean' belirtilmeli
    df_d["Skewness"] = self.df.loc[:, feature].skew(axis=0)
    df_d["Kurtosis"] = self.df.loc[:, feature].kurtosis(axis=0)  # 'kurtosis' doğru yazılmalı

    return df_d


...
def dispersion_measures_of_a_feature(self, feature: str = None):
    """
    Veri setindeki bir özelliğin dağılım ölçülerinin grafiğini oluşturur
    """
    assert self.__check_it_includes(self.__continuous_variables, [feature]), "feature değişkenin değeri, sürekli değişkenlerde tanımlanmalıdır"

    f, (ax_box, ax_hist) = plt.subplots(2, gridspec_kw={"height_ratios": (.30, .70)})
    sns.boxplot(self.df[feature], ax=ax_box)
    ax_box.set(xlabel='')

    sns.histplot(data=self.df, x=feature, ax=ax_hist, kde=True)
    
    table = self.__create_dispersion_measures(feature)
    plt.table(cellText=table.values.reshape(-1, 1), rowLabels=table.index, colWidths=[0, 4], fontsize=18, bbox=[1.2, 0.05, 0.5, 1.5])
    plt.show()

...

...
def central_tendency_measures_of_a_feature(self,feature:str=None):
    """
        veri setindeki bir özellgin merkezi dagılımını gösteren degerlerin grafigini oluşturur
        """
    assert(self.__check_it_includes(self.__continuous_variables,[feature])),"feature degiskenin degeri , sürekli degiskenlerde tanımlanılmalıdır"
    df_ct=pd.DataFrame()
    df_ct["mode"]=self.df.loc[:,[feature]].mode().iloc[0,:]
    df_ct["median"]=self.df.loc[:,[feature]].median()
    df_ct["mean"]=self.df.loc[:,[feature]].mean()

    df_ct.plot.bar()
    plt.xticks([])
    table=plt.table(cellText=df_ct.T.values,colLabels=df_ct.T.columns,rowLabels=df_ct.T.index,fontsize=180)
    plt.show()
...
def covariance_matrix(self):
    """
    Veri setindeki sürekli değişkenlerin kovaryans matrisini oluşturur
    """
    df_cov = self.df.loc[:, self.__continuous_variables].cov()
    mask_cov = np.triu(np.ones_like(df_cov, dtype=bool))
    
    plt.figure(figsize=(10, 8))  # Örnek bir boyut
    sns.heatmap(df_cov, mask=mask_cov, annot=True, cmap="inferno", cbar_kws={"shrink": 0.5})
    plt.title("Kovaryans Matrisi")
    plt.xlabel("Değişkenler")
    plt.ylabel("Değişkenler")
    plt.show()

...
def correlation_analysis(self):
    """
    Veri setindeki sürekli değişkenlerin korelasyon grafiğini oluşturur.
    """
    df_corr = self.df.loc[:, self.__continuous_variables].corr()
    mask_corr = np.triu(np.ones_like(df_corr, dtype=bool))
    
    plt.figure(figsize=(10, 8))  # Örnek bir boyut
    sns.heatmap(df_corr, mask=mask_corr, annot=True, fmt=".2%", cmap="Blues", cbar_kws={"shrink": 0.5})
    plt.title("Korelasyon Grafiği")
    plt.xlabel("Değişkenler")
    plt.ylabel("Değişkenler")
    plt.show()

...
def principal_component_analysis_2d(self, feature_color: np.array = None):
    """
    Veri setini iki boyutlu incelemeyi sağlar.
    """
    from sklearn.decomposition import PCA  # from ile sklearn modülü doğru şekilde import edildi

    pca = PCA(n_components=2)  # n_components parametresi kullanıldı
    data = self.df.drop(columns=self.__target_variables)
    projected = pca.fit_transform(data)
    
    plt.scatter(projected[:, 0], projected[:, 1], c=feature_color)  # feature_color'un doğru şekilde kullanılması sağlandı
    ratios = pca.explained_variance_ratio_
    ratio_for_2d = ratios[0] + ratios[1]
    plt.title(f"Açıklanan varyans oranı: {ratio_for_2d}")  # f-string kullanıldı
    plt.xlabel('Bileşen 1')
    plt.ylabel('Bileşen 2')
    plt.colorbar()
    plt.show()

    ...
def jointplot(self,x_axis:str,y_axis:str):
    """
    veri setinde bulunan iki sürekli degiskenin ilişkisini inceler
    """
    assert(x_axis in self.variables and y_axis in self.variables)," x_axis ve y_axis veri setinde tanımlanmalıdır"
    sns.joinplot(x=x_axis, y=y_axis, data=self.df,kind="reg")
    ...
    

