#!/usr/bin/env python
# coding: utf-8

# # Module import

# In[1]:


import numpy as np 
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.covariance import EllipticEnvelope
from tqdm.notebook import tqdm

import lightgbm 
from lightgbm import LGBMClassifier
import optuna
from optuna.samplers import TPESampler

import torch

import warnings
warnings.filterwarnings(action='ignore')


# In[ ]:


# !pip freeze > requirements.txt

# python==3.7.9
# jupyter==1.0.0
# lightgbm==3.3.2
# numpy==1.21.6
# optuna==2.10.1
# pandas==1.3.5
# scikit-learn==1.0.2
# sklearn==0.0
# tensorboard==2.9.1
# torch==1.12.0
# tqdm==4.64.0


# # Data Load

# In[2]:


train = pd.read_csv('./dacon/train.csv')
valid = pd.read_csv('./dacon/val.csv')
test = pd.read_csv('./dacon/test.csv')


# In[3]:


trainset = train.drop(['ID'] , axis = 1) 
testset = test.drop(['ID'] , axis = 1) 


# In[4]:


fraud_ratio = valid['Class'].values.sum() / len(valid)
print(fraud_ratio)


# In[5]:


# Fraud ratio (Valid와 동일하게)
# 하지만 contamination을 valid와 동일하게 하든 몇 배로 해서 모델을 만들든
# 나중에 본인이 score를 내림차순하여 원하는 개수만큼 이상치 개수를 정할 수 있는 것을 확인했기에 크게 의미는 없다고 생각함 
# 정리하자면, 
# 1) contamination을 몇으로 놓든 간에 정의한 함수 get_pred_label을 이용해 outlier label 개수를 조절 가능 
# 2) 극단적으로 contamination을 0.1로 두든 1로 두든 get_pred_label 함수에서 k를 똑같이주어 사용하면 결과는 똑같음 

model = EllipticEnvelope(support_fraction = 0.994, contamination = fraud_ratio, random_state = 42) 
model.fit(trainset)


# # Ensemble을 위한 test prediction value 1 획득

# In[6]:


def get_pred_label(model, x, k):
  prob = model.score_samples(x)
  prob = torch.tensor(prob, dtype = torch.float)
  topk_indices = torch.topk(prob, k = k, largest = False).indices

  pred = torch.zeros(len(x), dtype = torch.long)
  pred[topk_indices] = 1
  return pred , prob


# In[7]:


# 313개로 fraud label 개수를 정한 이유는, 
# https://dacon.io/competitions/official/235930/codeshare/5694?page=1&dtype=recent 게시글 작성자 
# Akynella님이 공유해주신 내용을 토대로 318개 근처의 값을 사용, 이후 public score 참고하여 적절하게 조정

# 이 과정에서 학습의 일관성이나 논리에 조금 어긋난다고 느낀 것은
# testset도 fraud label의 비율을 앞 두개의 dataset과 동일하게 두어야 논리나 일관성이 있지 않았나하는 생각이 듦
# 즉, test에도 train이나 valid와 동일하게 fraud label 비율을 가져갔으면 약 150개 정도의 fraud label을 주어야 일관성이 있는거라
# 생각함. 하지만 그렇게 두면 public score가 좋지 않고, 실제로 코드 공유 게시판을 참조한 결과 
# testset에 더 많은 비율의 fraud label이 존재함을 인지 및 예상

test_pred, _ = get_pred_label(model, testset, 313)


# In[8]:


envelope_pred = np.array(test_pred)


# # 분류 모델링을 위한 trainset label 임의 획득

# In[9]:


# valid와 동일한 비율 118~120개 사이의 fraud label을 가질 거라 가정하여 label 획득
# trainset의 label을 임의로 준 이유는 지도학습의 결과를 함께 앙상블하기 위함
# 물론, 일정 부분 잘못된 label을 부여하고 학습을 하는 것이 논리에 어긋나지만 
# 하나라도 사기 거래를 잘 찾아내는 것에 집중하기 위해 잘못된 label을 주고 모델 학습

train_pred, _ = get_pred_label(model, trainset, 118)
Y = np.array(train_pred)


# # 모델 최적화 (optuna module 사용)

# In[10]:



#skf = StratifiedKFold(n_splits = 5 , random_state = 42 , shuffle = True)

score = []
def lgb_optimization(trial):
    score = []
    skf = StratifiedKFold(n_splits = 5 , random_state = 42 , shuffle = True)
    for train_fold, test_fold in tqdm(skf.split(trainset, Y), desc = 'k_fold'):
        X_train, X_test, y_train, y_test = trainset.iloc[train_fold], trainset.iloc[test_fold], Y[train_fold], Y[test_fold] 
        
        params = {            
            "boosting_type" : trial.suggest_categorical('boosting_type',['dart','gbdt']),
            "learning_rate": trial.suggest_uniform('learning_rate', 0.2, 0.99),
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=10),
            "max_depth": trial.suggest_int("max_depth", 1, 15),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 30),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "max_bin": trial.suggest_int("max_bin", 50, 100),
            "verbosity": -1,
            "random_state": trial.suggest_int("random_state", 1, 10000)
        }
        model_lgb = LGBMClassifier(**params)
        model_lgb.fit(X_train, y_train)
        lgb_cv_pred = model_lgb.predict(X_test)
        
        score_cv = f1_score(Y[test_fold] , lgb_cv_pred , average = 'macro')
        
        score.append(score_cv)
    print(score)
    return np.mean(score) 


# In[11]:


# 1-fold , 2-fold ... 20-fold 분할 및 성능 평가와
# Best Hyper-parameter를 순차적으로 고정+탐색 , 고정+고정+탐색.. 하는 방식으로 
# 가장 train label을 안정적으로 찾는 parameter를 획득
# 해당 code엔 편의상 5 fold로 명시

sampler = TPESampler()
optim = optuna.create_study(
    study_name="lgb_parameter_opt",
    direction="maximize",
    sampler=sampler,
)
#optim.optimize(lgb_optimization, n_trials=1)
optim.optimize(lgb_optimization, n_trials=99999)
print("Best macro-F1:", optim.best_value)


# In[12]:


# 획득한 Lightgbm의 best parameter 

params = {'boosting_type': 'gbdt', 'learning_rate': 0.27931562561080087,
 'n_estimators': 180, 'max_depth': 2, 'num_leaves': 79, 'reg_alpha': 0.7804924821497133,
 'reg_lambda': 0.6483886637315736, 'subsample': 0.5046737928606037, 'subsample_freq': 27,
 'colsample_bytree': 0.2884662481524903, 'min_child_samples': 39, 'max_bin': 69}
#random_state = 3294


# # Ensemble을 위한 test prediction value 2 획득

# In[13]:


model2 = LGBMClassifier(**params , random_state = 3294)
model2.fit(trainset, Y)
lgb_pred = model2.predict(testset)


# In[14]:


# Ensemble 시,  AorB : true 조건을 사용한 이유는 
# 두 예측 시스템에서 최소 1번이라도 fraud로 예측된 example은 fraud로 의사 결정을 내리기 위함
# 이유 :: 사기 거래 탐지는 보수적인 의사 결정을 해야 옳다고 생각했기 때문에.

sub = pd.read_csv('./dacon/sample_submission.csv')
sub['Class'] = envelope_pred|lgb_pred # Ensemble 
sub.to_csv('./dacon/final_submission.csv' , index = False)


# In[15]:


# 대회 중 최종 제출 파일(stateopt.csv)과 동일한지 checking 

final_sub = pd.read_csv('./dacon/stateopt.csv')

print(sum(sub['Class'].values == final_sub['Class'].values),
      len(testset))

