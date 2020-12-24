# MEMO
```
.constant() //상수

.Session() //tensor에 데이터를 넣어 흐르게 함.

.run() //실행

.add() //더하기

.placeholeder(), feed_dict={a:a_data} //변수, 값을 나중에 할당.

.Variable() //변수, 자동으로 업데이트.

.random_normal(Shapes) //랜덤 값 반환

.reduce_mean() //평균

.square() //제곱

.GradientDescentOptimizer() //미니 배치 확률적 경사하강법(SGD) 구현.

.minimize() //최소화

.global_variables_initializer() //.Variable()를 초기화.

.append() //append

.plot() //plot

.show() //show

.reduce_sum() //총합

.assign() //.Variable()의 값 변경.

.compute_gradients() //compute_gradients

.apply_gradients() //apply_gradients

.matmul() //matmul

.loadtext() //text 불러오기.

.set_random_seed() //랜덤 값 시드, 다른 환경에서도 같다.

.string_input_producer() //Queue, text 를 Filename Queue 에 쌓기.

.TextLineReader() //Queue, text 를 Reader 로 연결.

.read() //Queue, text 읽기.

.decode_csv() //Queue, text decode

.batch() //Queue, text batch

.Coordinator() //Queue, Coordinator 생성.

.start_queue_runners() //Queue, Queue 를 Thread 로 시작.

.request_stop() //Queue, 중지

.join() //Queue, 대기

.sigmoid() //S 자 곡선

.log() //로그

.cast() //새로운 자료형

.equal() //값이 같은지

.softmax() //softmax

.arg_max() //arg_max

.one_hot() //one_hot

.reshape() //reshape

.softmax_cross_entropy_with_logits() //softmax_cross_entropy_with_logits

.format() //format

.flatten() //flatten

.PrettyPrinter() //PrettyPrinter

.InteractiveSession() //InteractiveSession

.array() //Array

.pprint() //Pprint

.shape() //Shape

.eval() //Eval

.squeeze() //Array 정리

.expand_dims() //Array 정렬

.stack() //Array 쌓기

.ones_like() //One 으로 바꿈.

.zeros_like() //Zero 로 바꿈.

zip() //Zip

.nn.relu() //Relu

.random.randn() //지정 범위 내 랜덤 값 반환

.nn.dropout() //Dropout

.imshow() //Imshow

.nn.conv2d() //Conv2d

.swapaxes() //Swapaxes

.subplot() //Subplot

.nn.max_pool() //Maxpool

.contrib.rnn.BasicRNNCell() //Basic RNN Cell

.nn.dynamic_rnn() //Dynamic Rnn

.BasicLSTMCell() //Basic LSTM Cell

.zero_state() //Zero State

.contrib.layers.fully_connected() //Fully Connected

.ones() //Ones

.contrib.seq2seq.sequence_loss() //Sequence Loss

.AdamOptimizer() //AdamOptimizer

.get_variable() //Get Variable

.MultiRNNCell() //Multi RNN Cell

.min() //최소

.max() //최대

MinMaxScaler //MinMaxScaler

.xlabel() //X Label

.ylabel() //Y Label
```
