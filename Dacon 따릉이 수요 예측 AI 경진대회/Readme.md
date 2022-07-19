# Requirements

```
# 데이터셋 구축 및 전처리
외부 데이터는 모두 기상개방포털을 활용(https://data.kma.go.kr/cmmn/main.do)
일부 변수 계산법 및 일부 아이디어는 기존 따릉이 수요예측 대회 1등인 다람이도토리님의 코드를 참고(https://dacon.io/competitions/official/235837/codeshare/3724?page=1&dtype=recent)
cl : 운량, 흐림의 정도를 표현할 수 있을 것이라는 판단에 추가
hi/pi : 열지수/체감온도, 온도가 주요한 피쳐 중 하나라고 생각했기에 관련 변수들을 추가
hr : 강수 지속 시간, 새벽이나 이른 오전, 늦은 밤에만 잠시 올 경우 수요에 큰 영향을 주지 않을 것 같다는 판단에 추가
엄밀하게 시간대 별 강수 현황을 가져와야 하지만 전처리의 문제로 인해 강수 시간으로 대체하여 사용
결측값 처리는 Soobin님의 코드를 참고(https://dacon.io/competitions/open/235915/codeshare/5164?page=1&dtype=recent)
mean으로 하셨지만 median으로 변경한 경우 소폭 성능 향상이 있어 median을 선택
```
