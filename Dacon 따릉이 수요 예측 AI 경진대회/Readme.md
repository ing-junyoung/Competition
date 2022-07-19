# Public 5위 , Private 1위 / 이전끝 / Ensemble(LightGBM+CatBoost) <br>


## 데이터셋 구축 및 전처리

<br>
● 외부 데이터는 모두 기상개방포털을 활용(https://data.kma.go.kr/cmmn/main.do)


● **cl**  :  운량, 흐림의 정도를 표현할 수 있을 것이라는 판단에 추가 <br>
● **hi/pi**  :  열지수/체감온도, 온도가 주요한 피쳐 중 하나라고 생각했기에 관련 변수들을 추가 <br>
● **hr**  :  강수 지속 시간, 새벽이나 이른 오전, 늦은 밤에만 잠시 올 경우 수요에 큰 영향을 주지 않을 것 같다는 판단에 추가 <br>
● 엄밀하게 하면, 시간대 별 강수 현황을 가져와야 하지만 전처리의 문제로 인해 강수 시간으로 대체하여 사용 <br>
● 모든 결측값은 중앙값으로 대체하여 사용 

<br>
<br>

## 파생변수 생성 

● **weekend** : 주말에 따른 수요량 변화를 고려하기 위해 추가, 주말과 평일의 차이가 유의미하다고 <br>
생각들지 않기 때문에 따로 공휴일은 고려하지 않음 <br>
특히 평일 출/퇴근 시간에 따릉이 수요량이 많다고는 하지만 주말에 타는 사람이 그만큼 많기에 이러한 차이가 상쇄되었다고 판단 <br>
● **discomfort** : 불쾌지수, hi, pi와 마찬가지로 기온 관련 변수 추가 <br>
● date를 year, month, day, dayofyear로 분할 <br>
● *dayofyear와 day는 성능에 악영향을 끼쳐 최종적으로 제외* <br>

<br>
<br>

## Feature Selection & Drop 

● 논리적으로 필요가 없다고 생각한 변수를 제거하며 모델 입력 변수를 탐색 <br>
● 예로, 시간 변수가 있으므로 date 변수 제거 <br>
● 강수 지속 시간과 "강수량X강수지속시간"이 있기 때문에 precipitation(강수량) 변수 제거 <br>
● 강수 관련 변수들이 습도를 대신한다 생각하기 때문에 humidity 제거<br>
● wind_mean이나 wind_max는 체감 온도나 강수 시간(태풍의 경우), 최저 기온(추위)등으로 설명할 수 있다고 생각하기 때문에 제외<br>
● 사전 제거 이후, 이 변수 조합을 기반으로 Multiple Linear Regression에서의 Backward Elimination(후진제거법)과 같은 맥락으로 <br>
성능을 살펴보며 순차적으로 변수를 제거하며 최적의 성능을 낼 수 있는 변수 조합을 찾으려 함<br>

<br>
<br>


## Optimization 

● 전체 기간을 예측하기 위해서 따로 데이터를 Split하지 않음 : 특정 기간에 대한 데이터가 빠지면 해당 기간 예측력이 떨어지기 때문에 <br>
● 어쩔 수 없이 CV를 통해 과적합을 방지하면서 Loss를 줄여 나가는 방향으로 파라미터를 최적화 시도 <br>
CV를 통해 얻어진 파라미터를 활용하여 모델을 전체 학습 데이터로 학습하고, 예측값에는 연도별 상승분인 1.3을 곱해줌 <br>

          
**2018 -> 2019의 상승분은 약 1.9배** <br>
**2019 -> 2020의 상승분은 코로나임에도 불구하고 약 1.2배**<br>
● 코로나 관련 제한 사항들이 완화 됐기 때문에, 기존 상승률에 어느정도 더욱 상승했을 것이라고 추측하여 2020 -> 2021의 상승분은 1.3을 적용 <br>
<br>
<br>
## 최종 사용 모델 : LGBM + Catboost 

● 최적화 Libary optuna를 사용하여 optimization 진행 <br>
### Ensemble : Averaging <br>
● 각 모델의 결과값들을 평균내는 에버리징 앙상블 방법을 적용 <br>
● 이외에도 XGB, NN, ExtraTree, Randomforest 등 다양한 모델을 최적화 <br>
● 최종적으로 사용한 LGBM과 Catboost에 비해 CV 성능이 현저히 낮아 2가지 모델만 에버리징 <br>

### Post Processing <br>

● "학습에 사용하지 않은 변수"인 "강수량"을 활용하여 전체 Output에 "일괄적인 후처리"가 가능할 것이라고 판단<br>
● 비가 오면 확실히 대여량이 크게 감소함 <br>
● 연도에 따라 이용객이 증가했더라도, 비가 오는날은 모두가 동일하게 타지 않을거라고 생각한뒤 이를 확인 <br>
● *자세한 내용은 code내 explannation* <br>

<br>
<br>

## ETC(Other Trial) 
전부 성능이 나빠져서 최종적으로는 사용하지 않음

### EDA

● 상관계수를 고려하여 변수를 선택해 보았으나 상관관계가 높다고 해서 성능에 좋은 영향을 끼치진 않았음<br>
● 연도별로 대여량을 비교해보았을 때 봄/가을에 비해 여름/겨울의 대여량이 달라 이를 나누어 학습을 시도<br>
● 군집분석을 활용해 월별로 군집을 나누어 학습 시도<br>
● 강수 여부에 따라 데이터를 나누어 학습 시도

### Model 관점 
● Xgboost / Deep Neural Networks / Adaboost / RandomForest / ExtraTree 등 <br>
● Xgboost는 단일 model로 사용할 때는 성능이 나쁘진 않았으나 Ensemble 과정에서는 오히려 좋지 않았기에 제외 <br>
● 개인적인 판단이지만 Hyper parameter 조절 대상 개수가 적은 sklearn에서 제공하는 ML Model은 성능이 크게 좋지 않았음 <br>
● 워낙, 깔끔하게 정제된 정형 Dataset이기에 Model Selection 보다는 Optimization에 집중 <br>
