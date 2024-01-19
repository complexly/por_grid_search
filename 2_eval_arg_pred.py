## evaluate prediction on sample data sid0 that should loop over 100 samples
import pandas as pd
import numpy as np
import csv
import sys
import os
import gc
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score,mean_squared_error,r2_score
_= np.seterr(divide = 'ignore') 
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


## k nearest neighbor in constructing density
def knn(x,n):
    idx = (-x).argsort()[n:]
    y = np.copy(x)
    y[idx] = 0
    return y


# In[5]:

def eval_perf(df):
    train = df[~df.testidx]
    test = df[df.testidx].copy()
    # levels on all
    reg = LinearRegression().fit(train[['regionsum','indsum','density']],train.lograw)
    test['predraw'] = reg.predict(test[['regionsum','indsum','density']])
    levelr2 = r2_score(test.lograwraw,test.predraw)
    # growth on all
    growth8 = LinearRegression().fit(train.loc[(train.raw>0)&train.existafter8,['lograw','regionsum','indsum','density']],train[(train.raw>0)&train.existafter8].growth8)
    test['predgr8'] = growth8.predict(test[['lograwraw','regionsum','indsum','density']])
    growth8r2 = r2_score(test[test.exist&test.existafter8].growth8,test[test.exist&test.existafter8].predgr8)
    # levels on tradable
    regtr = LinearRegression().fit(train[train.indtype=='traded'][['regionsum','indsum','density']],train[train.indtype=='traded'].lograw)
    test['predraw'] = regtr.predict(test[['regionsum','indsum','density']])
    levelr2tr = r2_score(test[test.indtype=='traded'].lograwraw,test[test.indtype=='traded'].predraw)
    # growth on tradable
    growth8tr = LinearRegression().fit(train.loc[(train.raw>0)&train.existafter8&(train.indtype=='traded'),['lograw','regionsum','indsum','density']],train[(train.raw>0)&train.existafter8&(train.indtype=='traded')].growth8)
    test['predgr8tr'] = growth8tr.predict(test[['lograwraw','regionsum','indsum','density']])
    growth8r2tr = r2_score(test[test.exist&test.existafter8&(test.indtype=='traded')].growth8,test[test.exist&test.existafter8&(test.indtype=='traded')].predgr8tr)
    # levels on nontradable-geo
    reggeo = LinearRegression().fit(train[train.indtype=='geo'][['regionsum','indsum','density']],train[train.indtype=='geo'].lograw)
    test['predraw'] = reggeo.predict(test[['regionsum','indsum','density']])
    levelr2geo = r2_score(test[test.indtype=='geo'].lograwraw,test[test.indtype=='geo'].predraw)
    # growth on nontradable-geo
    growth8geo = LinearRegression().fit(train.loc[(train.raw>0)&train.existafter8&(train.indtype=='geo'),['lograw','regionsum','indsum','density']],train[(train.raw>0)&train.existafter8&(train.indtype=='geo')].growth8)
    test['predgr8geo'] = growth8geo.predict(test[['lograwraw','regionsum','indsum','density']])
    growth8r2geo = r2_score(test[test.exist&test.existafter8&(test.indtype=='geo')].growth8,test[test.exist&test.existafter8&(test.indtype=='geo')].predgr8geo)
    # levels on nontradable-public
    regpublic = LinearRegression().fit(train[train.indtype=='public'][['regionsum','indsum','density']],train[train.indtype=='public'].lograw)
    test['predraw'] = regpublic.predict(test[['regionsum','indsum','density']])
    levelr2public = r2_score(test[test.indtype=='public'].lograwraw,test[test.indtype=='public'].predraw)
    # growth on nontradable-public
    growth8public = LinearRegression().fit(train.loc[(train.raw>0)&train.existafter8&(train.indtype=='public'),['lograw','regionsum','indsum','density']],train[(train.raw>0)&train.existafter8&(train.indtype=='public')].growth8)
    test['predgr8public'] = growth8public.predict(test[['lograwraw','regionsum','indsum','density']])
    growth8r2public = r2_score(test[test.exist&test.existafter8&(test.indtype=='public')].growth8,test[test.exist&test.existafter8&(test.indtype=='public')].predgr8public)
    # levels on nontradable-service
    regservice = LinearRegression().fit(train[train.indtype=='service'][['regionsum','indsum','density']],train[train.indtype=='service'].lograw)
    test['predraw'] = regservice.predict(test[['regionsum','indsum','density']])
    levelr2service = r2_score(test[test.indtype=='service'].lograwraw,test[test.indtype=='service'].predraw)
    # growth on nontradable-service
    growth8service = LinearRegression().fit(train.loc[(train.raw>0)&train.existafter8&(train.indtype=='service'),['lograw','regionsum','indsum','density']],train[(train.raw>0)&train.existafter8&(train.indtype=='service')].growth8)
    test['predgr8service'] = growth8service.predict(test[['lograwraw','regionsum','indsum','density']])
    growth8r2service = r2_score(test[test.exist&test.existafter8&(test.indtype=='service')].growth8,test[test.exist&test.existafter8&(test.indtype=='service')].predgr8service)
    return [levelr2,levelr2tr,levelr2geo,levelr2service,levelr2public,growth8r2,growth8r2tr,growth8r2geo,growth8r2service,growth8r2public,reg.coef_[2],regtr.coef_[2],reggeo.coef_[2],regservice.coef_[2],regpublic.coef_[2],growth8.coef_[0],growth8.coef_[3],growth8tr.coef_[0],growth8tr.coef_[3],growth8geo.coef_[0],growth8geo.coef_[3],growth8service.coef_[0],growth8service.coef_[3],growth8public.coef_[0],growth8public.coef_[3]]



## iterate all relatedness matrices
def loop_prox(proxtype,df,var4den,sid):
    for pid in range(116):
        if os.path.isfile(f'performance/result/{proxtype}-{var4den}-{sid}-{pid}.tsv') and os.path.getsize(f'performance/result/{proxtype}-{var4den}-{sid}-{pid}.tsv')>0:
            continue
        if proxtype == 'colocation':
            proxmat = np.load(f'proximity/colocation/sample{sid}/{pid}.npy')
        else:
            proxmat = np.load(f'proximity/{proxtype}/{pid}.npy')
        resfile = open(f'performance/result/{proxtype}-{var4den}-{sid}-{pid}.tsv','a')
        writer = csv.writer(resfile,delimiter='\t')
        
        for n_neigh in [50,100,200,300,415]:
            try:
                proxmatknn = np.apply_along_axis(knn,  0, proxmat,n_neigh)
                den = proxmatknn.sum(axis=0)
                den[den==0] = 1
                df['density'] = (df[var4den].values.reshape(nrow,415) @ (proxmatknn / den)).flatten('C')
                if var4den in ['raw','rca']:
                    df['density'] = np.where(df.density>0,np.log(df.density),0)
                perfs = eval_perf(df)
                writer.writerow([proxtype,var4den,sid,pid,n_neigh]+perfs)
            except:
                with open(f'performance/error.txt','a') as ef:
                    ef.write(f'{proxtype},{var4den},{sid},{pid},{n_neigh}\n')
        resfile.close()
        gc.collect()
    print(f'{sid}-{var4den}-{proxtype} finished!')


## iterate all train-test data samples
for sid in range(0, 100):
    df = pd.read_parquet(f'data/sample{sid}.parquet')
    nrow = int(df.shape[0]/415)
    for proxtype in ["coproduction", "du1st",'country',"colocation"]:
        for var4den in ["raw","lograw","rca","rca2","pmi","ppmi","feresid","resid","posresid","bin","bin_rca","bin_feresid","bin_resid","bin_posresid"]:
            _ = loop_prox(proxtype,df,var4den,sid)

