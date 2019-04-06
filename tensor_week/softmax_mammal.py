import tensorflow as tf
import numpy as np
print(tf.__version__)
#[털, 날개]
x_data = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]]
)

#기타, 포유류, 조류
#원핫 인코딩
y_data = np.array([
   [1, 0, 0], # 기타
   [0, 1, 0], # 포유류
   [0, 0, 1], # 조류
   [1, 0, 0], # 기타
   [0, 1, 0], # 포유류
   [0, 0, 1] # 조류
])

#*****
#신경망 모델 구성
#*****
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

w=tf.Variable(tf.random_uniform([2, 3], -1, 1.))
#신경망 neural network 앞으로는 nn으로 표기
#nn 은 2차원으로 [입력층(특성), 출력층(레이블)] -> [2,3] 으로 정합니다

b= tf.Variable(tf.zeros([3]))
#b는 편향, 앞으로 편향은 b로 표기
#w는 가중치, 앞으로는 가중치는 w로 표기
#b는 각 레이어의 아웃풋 갯수로 설정함
#b는 최종결과값의 분류갯수인 3으로 설정함

L = tf.add(tf.matmul(X,w),b)

#가중치와 편향을 이용해 계산한 결과 값에
L = tf.nn.relu(L)
#TF 에서 기본적으로 제공하는 활성화함수인 ReLU 함수를 적용

model = tf.nn.softmax(L)
#softmax()을 사용해서 출력값을 사용하기 때문에 쉽게 안듦
#소프트맥스 함수는 다음처럼 결과값을 전체합이 1인 확률로 만들어주는 함수
# 예) [8.04, 2.76, -6.52] ->[0.53, 0.24, 0.23]

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(model), axis = 1))
optimizer = tf. train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

#비율 함수응 최소화시키면 -> 경사도를 0으로 만들어 가면 그 값이 최적화된 값일 것이가

#**********
#신경망 학습 모델
#**********
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(100):
    sess.run(train_op, {X: x_data, Y: y_data })
    if (step + 1) % 10 ==10:
        print( step +1, sess.run(cost, {X: x_data, Y: y_data}))

#결과확인
prediction = tf.argmax(model, 1)
target = tf.array(Y,1)
print("예측값", sess.run(prediction, {X: x_data}))
print("실제값", sess.run(target, {Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f'%sess.run(accuracy*100, {X: x_data, Y: y_data}))
