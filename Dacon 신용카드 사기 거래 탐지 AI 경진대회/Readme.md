## Training idea

<br>
<br>

### 1) Elliptic Envelope  
Outlier Detection을 위한 알고리즘 중 하나이며,<br>
이 방법은 Detection을 위해 정상 데이터의 Gaussian distribution을 가정하고 <br>
분포로부터 정상 데이터의 모양을 정의하는 것으로부터 시작된다. <br>
이와 공분산 추정을 통해 타원형의 정상 데이터 영역으로부터 외부에 위치하는 데이터를 Outlier로 판단한다.<br>
본 경진대회에서는 통계치 추정이나 후처리를 위한 validation dataset에 대해서만 label이 주어졌기 때문에 <br>
우선 validset에 대해 해당 알고리즘을 사용해보았고 약 92% 정도의 f1-score(macro)를 확인했다. <br>
이후 testset에 해당 알고리즘을 적용하여 label을 구했고 이를 ensemble을 위한 prediction value1로 얻었다. <br>
<br>
<br>

### 2) To get label(train) using Elliptic Envelope 
trainset의 label이 주어지진 않았지만 '만약 envelope의 방법이 어느 정도 준수한 성능을 보여준다면<br>
임의로 trainset에 label을 부여하고 지도학습을 해볼 수 있지 않을까?'하는 가벼운 생각으로 envelope 방법을 사용해 trainset에 label을 부여했다.<br> 
이후 데이터셋의 크기가 충분하기에<br> 
Lightgbm 알고리즘과 최적화 모듈 optuna를 사용해 하이퍼 파라미터 최적화와 함께 분류 모델링을 진행했다.<br> 
이후 획득한 파라미터를 토대로 만든 모델에 testset을 입력해 ensemble을 위한 prediction value2를 얻어냈다.<br>
<br>
<br>
### 3) AutoEncoer (본 대회의 최종 제출에선 사용하진 않았음)
AutoEncoder는 unsupervised learning Anmaly Detection에서 대표적으로 사용되는 방법이다.<br> 
해당 경진대회에서는 Encoder의 입력과 Decoder의 출력 간 코사인 유사도를 측정해 Outlier를 판별하는 방법을 택했다. <br>
(전제로, validset을 통해 매우 imbalance한 dataset임을 확인했기에 testset 또한 그런 양상을 띄울거라고 판단했다.) <br>
만약 대부분, 정상 데이터를 입력으로 받아 설계된 AutoEncoder라면 Decoder가 정상 데이터를 잘 복원해낼 수 있다는 기대로<br>
모델을 학습했다. 이를 통해, 만약 Encoder의 입력과 Decoder의 출력 간 <br>
코사인 유사도가 0.95 미만이라면 Outlier로 이상이라면 정상 데이터로 라벨을 부여했다. <br>
정리하면, trainset으로 pretrained된 model을 얻고 이 model에 testset을 입력으로 주어 ensemble을 위한 prediction value3을 얻어냈다. <br>
<br>
<br>
### 4) Ensemble 
해당 경진대회는 사기거래를 탐지해내는 것이 목적이기에 보다 보수적인 의사 결정이 필요하다고 판단했다.<br>
이에 3개의 prediction value 중 한번이라도 Outlier라고 예측된 데이터는 Outlier라고 판단하는 ensemble을 진행했다.<br>

### 5) 그 외 

### Fraud ratio (Valid와 동일하게)
contamination을 valid와 동일하게 하든 몇 배로 해서 모델을 만들어도 나중에 본인이 score를 내림차순하여<br>
원하는 개수만큼 이상치 개수를 정할 수 있는 것을 확인했기에 크게 의미는 없다고 생각함. <br> 
<br>
정리하자면, <br>
1) contamination을 몇으로 놓든 간에 정의한 함수 get_pred_label을 이용해 outlier label 개수를 조절 가능 <br>
2) 극단적으로 contamination을 0.1로 두든 1로 두든 get_pred_label 함수에서 k를 똑같이주어 사용하면 결과는 똑같음 <br>
<br>
<br>
