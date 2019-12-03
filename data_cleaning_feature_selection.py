# Descriptive Analysis and Data Cleaning

def valcheck(df):
    print('Shape: ', df.shape)
    print('Threshold (round up): ', round(len(df.index) * 0.6))
    print("Zero Values: ", '\n', df.select_dtypes(include=[np.number]).eq(0).sum())
    print("Negative Values: ", '\n', df.select_dtypes(include=[np.number]).lt(0).sum())
    print("NA Values: ", '\n', df.select_dtypes(include=[np.number]).isna().sum())
    
def to_num(df, bunch):
    for i in bunch:
        df.replace(r'--', np.nan, inplace=True)
        df[i] = pd.to_numeric(df[i])
        
def navert(df):
    rdf = df.select_dtypes(include=[np.number])
    for i in rdf:
        if df[i].isnull().values.any() == True:
            df[i].fillna(df[i].mean(), inplace=True)
        else:
            pass
            
def summarize(pos, num):
    
    def valgraph(pos):
        #posave = plt.gcf()
        posgroup = pos.groupby('Player', as_index = False).mean().head(10)
        pteam = sns.barplot(x='Player', y='TotalValue', data = posgroup)
        pteam.set_xticklabels(pteam.get_xticklabels(), rotation=40, ha="right")
        plt.xlabel('Player')
        plt.ylabel('Total Value')
        plt.title('Player Vs. Total Value')
        sns.set(font_scale = 1)
        plt.tight_layout()
        plt.show()
        #posave.savefig(top_player.png', dpi = 5000)
    
    def box(num):
        for i in num.columns:
            #numsave = plt.gcf()
            num.boxplot(column = i, grid = False, fontsize = 18)
            plt.figure(figsize=(10,10))
            #numsave.savefig({}bar.png'.format(i))
            plt.show()
        
    print(pos.describe())
    print('\n')
    print(valgraph(pos))
    print('\n')
    print(box(num))
    
# Feature Selection
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import metrics 

## Correlation Matrix
def corrmat(pos):
    corrmat = pos.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20,20))
    g=sns.heatmap(qb[top_corr_features].corr(),annot=True,cmap="RdYlGn")
 
##Recursive Feature Elimination
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from sklearn import preprocessing

def rfe(posX, posy):
    norm = preprocessing.scale(posX)
    svm = LinearSVC(max_iter=3000)
    # create the RFE model for the svm classifier 
    # and select attributes
    rfe = RFE(svm)
    rfe = rfe.fit(norm, posy)
    # print summaries for the selection of attributes
    print(rfe.support_)
    print(rfe.ranking_)
 
##Univariate Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from functools import partial
from sklearn.feature_selection import mutual_info_classif, SelectKBest

def unisel(feature,target, data):
    discrete_feat_idx = [1, 5] # an array with indices of discrete features
    score_func = partial(mutual_info_classif, discrete_features=discrete_feat_idx)
    bestfeatures = SelectKBest(score_func, k = 'all')
    fit = bestfeatures.fit(feature,target)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(feature.columns) #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Stat','Score']  #naming the dataframe columns
    print(featureScores.nlargest(5,'Score'))  #print 10 best features
    
    stat = featureScores.nlargest(5,'Score')
    stat.reset_index(inplace=True)
    stat.drop('index', axis = 1, inplace=True)
    
    weights = [pop / sum(stat['Score']) for pop in stat['Score']]
    
    def agg(TotalValue, Var0, Var1, Var2, Var3, Var4):
        return TotalValue / ((Var0 * weights[0]) + 
                             (Var1 * weights[1]) + 
                             (Var2 * weights[2]) +
                             (Var3 * weights[3]) + 
                             (Var4 * weights[4])
                            )
    
    data['Aggr'] = agg(data['TotalValue'], data[stat['Stat'][0]], data[stat['Stat'][1]], data[stat['Stat'][2]], 
                       data[stat['Stat'][3]], data[stat['Stat'][4]])
    
    print('\n)')
    return data.sort_values('Aggr').head(10)
    

##Feature Importance
from sklearn.ensemble import ExtraTreesClassifier

def featimp(feature, target):
    model = ExtraTreesClassifier(n_estimators=200) #set random_state to create same values for each iteration
    model.fit(feature,target)
    print(model.feature_importances_)
    feat_importances = pd.Series(model.feature_importances_, index=feature.columns)
    feat_importances.nlargest(5).plot(kind='barh')
    plt.show()

##Principal Component Analysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def pcafunc(feature, target):
    global X_train, X_test, y_train, y_test 
    X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size = 0.3, random_state = 0)
    
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test) 
    
    pca = PCA(0.95)
    pca.fit(X_train)
    
    global X_train_pca, X_test_pca
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
  
