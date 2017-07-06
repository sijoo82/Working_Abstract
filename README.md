
# Linear Model for Regression
## <span style="font-family: 'Malgun Gothic'; color:#3333ff;';">3.1 Linear Basis Function Models</span>
<span style="font-family: 'Malgun Gothic';">
 Regression을 위한 가장 간단한 선형 모델은 입력 변수의 선형 조합을 포함한다.



<span style="color:#ff33ff;">$eq. 3.1$</span>
 $y(x,w) = w_0 + w_1 x + ... + w_d x_d$


 여기서 $x=(x_1,..., x_d)^T$이다. 이것은 종종 Linear Regression으로 알려져 있다. 이 모델의 중요한 속성은 파라미터 $w_0, ..., w_D$의 선형 함수라는 것이다. 하지만 이것은 입력 변수 x의 선형 함수이며, 이것은 이 모델의 중요한 한계로 작용한다. 따라서, 우리는 입력 변수의 고정된 비선형 함수의 선형 조합을 통해 모델의 클래스를 확장한다.

<span style="color:#ff33ff;">$eq. 3.2$</span>
$y(x,w)=w_0+\sum_{j=1}^{M-1}w_j\phi_j(x)$

 여기서 $\phi_j(x)$ 는 basis 함수로 알려져 있다. 이 모델의 전체 파라미터의 수는 M개가 될 것이다.

이 파라미터 $w_0$ 는 데이터의 고정된 오프셋을 나타내며, 때로 Bias 파라미터로 불린다. 종종 편리함을 위해, $\phi_0(x) = 1$ 로 basis function을 추가적으로 정의한다.

<span style="color:#ff33ff;">$eq. 3.3$</span>
$y(x,w)=\sum_{j=0}^{M-1}w_j\phi_j(x)=w^T\phi(x)$

여기서 $w=(w_0,...,w_M-1)^T$ 그리고, $\phi = (\phi_0, ..., \phi_M-1)^T$ 이다. 패턴 인식의 많은 실제 응용분야에서, 기본 데이터 변수를 특징 추출 또는 고정된 전처리의 형태로 적용된다.

만약 원본 변수가 벡터 x로 구성되면, 특징은 basis Function ${\phi_j(x)}$ 형태로 표현할수 있다.

비선형 basis function을 사용함으로써, $y(x,w)$ 은 입력 벡터 $x$에 대해
non linear function으로 모델링 할 수 있다. 그러나, 수식 3.2의 형태의 함수는 $w$에 대해 linear하기 때문에, linear Model이라 불린다. 파라미터의 선형성(linearity)는 모델의 클래스 분석을 매우 쉽게 하는 역할을 한다. 그러나, 어떤 경우에는 한계가 존재하며, 이것은 3.6절에서 다루도록 한다.

Chapter 1에서 다룬 ploynomial Regression의 예는 하나의 입력 변수 $x$에 대한 모델링하는 예이다. 그리고,  
$\phi_j(x)=x^j$ 인 x의 power 형태로 basis fucntion을 사용한 것이다.
Polynomial basis function의 제약은 입력 변수의 전역 함수라는 것이다. 따라서, 입력 공간의 한 영역의 변화는 모든 다른 영역에 영향을 준다. 이것은 __spline function(Hastie et al., 2001)__ 와 같이, 입력 공간을 여러개의 영역으로 분할하고, 각 영역을 다른 Polynomial 로 fit함으로써 해결할 수 있다.

basis function을 위한 많은 가능한 선택들이 있다. 예를 들면,

<span style="color:#ff33ff;">$eq. 3.4$</span>
$\phi_j(x) = exp \{-\frac{(x-\mu_j)^2}{2s^2} \}$


여기서 $\mu_j$ 는 입력공간에서 basis function의 위치를 제어한다. 그리고, 파라미터 $s$는 공간의 스케일을 제어한다.
이것은 일반적으로 __Gaussian basis function__ 이라 불린다.

basis function은 적응적인 파라미터 $w_j$에 의해 곱해지기 때문에, 확률적 해석을 요구하지 않으며, 정규화된 coefficient는 중요하지않다.

다른 가능한 함수는 아래와 같이, sigmoidal basis function이 있다.

<span style="color:#ff33ff;">$eq. 3.5$</span>
$\phi_j(x) = \sigma(\frac{x-\mu_j}{s})$

여기서 $\sigma$는 Logistic sigmoid function으로 정의된다.

<span style="color:#ff33ff;">$eq. 3.6$</span>
$\sigma(a) = \frac{1}{1+exp(-a)}$

동등하게, 우리는 tanh function을 사용할 수 있다. 왜냐하면 $tanh(a) = 2\sigma(a)-1$ 에 의해 Logistic sigmoid를 표현할 수 있기 때문이다. 그리고, 일반적으로 logisitic sigmoid function의 선형 조합은 'tanh' function의 선형 조합과 같다(equivalent). basis function의 다양한 선택은 그림 3.1에서 보여준다.



<img src="./image/fig3.1.bmp">


또 다른 가능한 basis function은 sinusoidal functions의 확장에 따른 Fourier basis가 있다. 각 basis function은 특정한 빈도(frequency)를 표현하고, 공간적으로 무한히 확장할 수 있다. <span style="color:#ff33ff;">"By contrast, basis functions that are localized to finite regions of input space necessarily comprise a spectrum of different spatial frequencies." </span>

많은 신호 처리 응용에서, 공간과 빈도(frequency) 모두 localize되는 basis function은 고려된다. 이러한 것은 __wavelets__ 으로 알려져 있다.

 <span style="color:#ff33ff;">생략</span>

### <span style="font-family: 'Malgun Gothic'; color:#55FFff;';">3.1.1 Maximum likelihood and least squares</span>

Chapter 1에서 우리는 Polynomial Functions을 데이터셋에 대해 sum-of-squares error를 최소화함으로써 fitting하였다. 또한 우리는 이러한 error function이 Gaussian noise model을 가정하고, Maximum likelihood로 유도될 수 있음을 보였다. 다시 이 문제로 돌아와서, least square 접근법을 고려해보자. 그리고, 이것이 Maximum likelihood와 관련된 것을 좀 더 세밀하게 살펴보자.

그 전에, 우리는 target 변수 $t$는 결정 함수 $y(x,w)$ 에 Gaussian noise가 추가되어 결정된다고 가정한다.

<span style="color:#ff33ff;">$eq. 3.7$</span>
$t = y(x,w)+\epsilon$

여기서 $\epsilon$은 precision이 $\beta$ (inverse variance) 인 zero mean Gaussian random variable이다.

<span style="color:#ff33ff;">$eq. 3.8$</span>
$p(t|x,w,\beta) = N(t|y(x,w),\beta^{-1})$

상기해라, 만약 우리가 squared loss function을 가정한다면, 새로운 입력 x에 대한 최적 예측은 target 변수의 conditional mean에 의해 주어질 것이다.

3.8 형태의 Gaussian conditional distribution 의 경우, conditional mean은 아래와 같다.

><span style="color:#ff33ff;">$eq. 3.9$</span>
$E[t|x] = \int tp(t|x)dt = y(x,w)$

Gaussian noise 의 가정은 x가 주어졌을 때, t의 조건부 분포(conditional distribution)가 unimodal임을 나타내며, 어떤 응용에서는 부적절할 수 있다.
conditional Gaussian distribution 혼합으로 확장은 multimodal conditional distribution을 허용한다. 이것은 14.5.1 절에서 다루도록 하겠다

이제 target values $t_1, ..., t_N$에 대해 $X=\{x_1,...,x_N\}$ 를 생각해보자. 우리는 multivariate taget의 single observation과 구분하기 위해, target variable $\{t_n\}$ 의 컬럼 벡터를 __t__ 로 정의한다. 이러한 데이터 포인트는 eq. 3.8의 분포로부터 독립적으로 선택된다고 가정하자. 그러면 우리는 아래와 같이 likelihood function으로 표현된 수식을 얻을 수 있다.

><span style="color:#ff33ff;">$eq. 3.10$</span>
$p( t | X,w,\beta) = \prod_{n=1}^{N} {N(t_n|w^T\phi(x_n), \beta)}$

여기서 우리는 eq. 3.3을 사용하였다. 저것이 Regression과 classification과 같은 supervised Learning 문제임을 주목하자. 우리는 입력 변수의 분포는 모델링을 위해 찾지 않는다. 따라서, x는 conditional variables의 셋에서 항상 나타날 것이다. 그리고, 표기법을 유지하기위해 $p(t|x,w,\beta)$ 의 표현에서 x를 제거할 것이다. likelihood function에 logarithm을 적용하고, eq. 1.46의 형태로 만들면,

><span style="color:#ff33ff;">$eq. 3.11$</span>
$\ln p( t | w,\beta) = \sum_{n=1}^N{\ln N(t_n|w^T\phi(x_n), \beta^{-1})}$
$= \frac{N}{2}\ln\beta - \frac{N}{2}\ln(2\pi) - \beta E_D(w)$


여기서 sum-of-squares error function은 아래와 같이 정의된다.

><span style="color:#ff33ff;">$eq. 3.12$</span>
$E_D(w)= \frac{1}{2}\sum_{n=1}^{N}\{t_n - w^T\phi(x_n)\}^2$







---
---
### 1.5.5 Loss function for Regression
<span style="font-family: 'Malgun Gothic';">

지금까지 우리는 분류 문제에 대해 토론하였다. 우리는 이제 앞에서 언급했던, curve fitting의 예제와 같은 Regression 문제의 경우를 볼 것이다. 결정 단계는 각각의 입력 $x$에 대해 $t$의 값인 특정 $y(x)$ 를 선택하는 것으로 구성된다. 이 것을 가정하면, 우리는 loss $L(t, y(x))$ 를 만들 수 있다. 평균 혹은 기대값 loss는 아래와 같다.

$E[L] = \int\int{L(t,y(x))p(x,t)dxdt}$

Regression 문제에서 일반적인 loss function은 squared loss이다 $L(t,y(x)) = {y(x)-t}^2$
이러한 경우, loss의 기대값은 아래와 같다.

$E[L] = \int\int{\{y(x)-t\}^2p(x,t)dxdt}$

우리의 목적은 $E[L]$을 최소화하는 $y(x)$를 선택하는 것이다. 만약 flexible function y(x)라고 가정한다면, 우리는 아래와 같이 미분 할 수 있다.

$\frac{\delta E[L]}{\delta y(x)} = 2\int{\{y(x)-t\}p(x,t)dt} = 0$

$y(x)$에 대해 풀면, 확룰의 덧셈과 곱셈 법칙을 사용하여 정리하면,
</span>
$\frac{\delta E[L]}{\delta y(x)} = 2\int{y(x)p(x,t)-tp(x,t)dt} = 2 y(x)p(x) -2 \int{tp(x,t)dt} = 0$


$2y(x)p(x) = 2\int{tp(x,t)dt}$


$y(x) = \frac{\int{tp(x,t)dt}}{p(x)}=\int{tp(t|x)dt} = E_t[t|x]$  

---




### 3.2 The Bias-Variance Decomposition

지금까지 우리는 regression을 위한 linear models의 토론에서, 기저 함수(basis function)의 수와 형태를 고정한다고 가정하였다. 1장에서 보았듯이, maximum likelihood를 사용하거나, least square를 사용하는 것은 제한된 크기의 데이터 셋을 사용하여 복잡한 모델을 학습한다면 오버피팅을 야기시킬 수 있다. 그러나, 오버피팅을 피하기 위해, 기저 함수의 수를 제한하는 것은 데이터의 중요한 경향을 모델링하는데 유연함을 잃어 버릴 수도 있다.

물론 앞에서, 많은 파라메터에 대해 모델링을 수행할 때, 오버피팅을 조절하기 위해 Regularization의 설명에도 불구하고, 여전히 Regularization Coefficient lamda 에 대한 안정적인 값을 어떻게 결정할 것인지에 대한 의문이 남아있다. Regularization Coefficient와 weight vector에 대한 regularized error를 최소화하는 솔루션을 찾는 것은 올바른 접근 방법이 아니다, 왜냐하면, 이것은 람다=0인 경우인 unregularized solution을 따르기 때문이다.

이전 장에서 보았듯이, 오버피팅은 maximum likelihood의 안좋은 속성이다. 그리고, 이것은 베이지안 설정에서 파라미터에 대해 marginalize 할때 발생하지 않는다. 이번장에서는 우리는 모델의 복잡성에 대한 베이지안 관점을 고려할 것이다. 그 전에, 모델 복잡성과 관련된 이슈(bais-variance trade-off)를 frequentist viewpoint에서 고려할 것이다. 비록 여기서는 간단한 예제를 통해 아이디어를 쉽게 설명하기 위해 linear basis function 개념을 사용하더라도 일반적인 적용성을 갖는다

1.5.5절에서, regression 문제의 결정 이론에 대해 다룰때, 우리는 최적의 예측 모델을 찾기 위한 방법으로 조건부 분포 p(t|x)로 이용하여 다양한 Loss 함수를 고려하였다.

자주 사용되는 squared loss function을 선택하고, 조건부 기대값에 의해 최적 예측은 결정된다. 수식은 아래와 같다.

><span style="color:#ff33ff;">$eq. 3.36$</span>
$h(x)=E[t|x]=\int tp(t|x)dt$


이러한 관점에서, 모델 파라미터의 maximum likelihood 에서 나온 sum-of-squares error function과 결정 이론에서 나온 squared loss function 사이를 구분할 가치가 있다. 우리는 조건부 분포 p(t|x)를 결정하기 위해, 예를 들어, 정칙화(Regularization) 또는 fully Bayesian approach와 같은, least squares 보다 정교한 방법을 사용할 수도 있다. 이것들은 예측을 만들기위한 목적을 위해, squared loss function으로 조합할 수 있다.

1.5.5절에서 보여줬듯이,  expected squared loss는 아래와 같은 형태로 쓸 수 있다.

><span style="color:#ff33ff;">$eq. 3.37$</span>
$E[L]=\int \{y(x)-h(x)\}^2p(x)dx + \int \int \{h(x)-t\}^2p(x,t)dxdt$

두번째 항을 살펴보자. y(x)에 독립적이다. 데이터의 내부 노이즈로부터 발생하며, 이것은 loss의 기댓값의 최소값을 표현한다. 첫 항은 우리가 선택한 함수 y(x)에 의존적이다. 그리고 우리는 이 항을 최소화하기 위한 y(x)를 찾을 것이다. 이 항은 음수가 아니기 때문에, 이 항의 가장 작은 값은 0일 것이다. 만약 충분한 데이터가 제공된다면, 우리는 정확하고 아주 이상적인 차수의 regression 함수 h(x)를 찾을 수 있고, 이것은 y(x)에 대해 최적의 선택을 나타내는 것이다. 그러나, 실제로는 데이터 집합 D에는 유한한 데이터의 수 N개의 샘플만을 가지고 있기 때문에, regression 함수 h(x)를 정확하게 알 수 없다.


만약 우리가 파라메터 벡터 w에 의한 y(x,w) 함수를 사용하여 h(x)를 모델링한다면, 베이지안 관점에서 우리의 모델의 불확실성은 w에 의한 사후 분포(posterior distribut)를 통해 표현된다. 반면 빈도 확률론자(frequentist)들은 데이터 셋 D에 기반하여 w의 추정을 포함하고, 다음과 같은 사고 실험을 통해 이 추정의 불확실성을 해석하는 것 대신에 시도한다.

우리는 p(t,x) 분포로부터 독립적으로 N크기의 데이터 셋을 가지고 있다고 가정하자. 주어진 어떤 데이터 셋 D에 대해 우리는 학습 알고리즘을 적용할 수 있으며, 예측 함수 y(x;D)를 얻을 수 있다. 다른 데이터 셋은 다른 함수를 찾을 것이고, 따라서 다른 squared loss 값을 나타낼 것이다. 특정 학습 알고리즘의 성능은 여러 데이터 셋에 걸쳐 평균을 취함으로써 평가할 수 있다.

수식 3.37의 첫 항에서 적분을 특정 데이터 셋에 대해 고려해보자

><span style="color:#ff33ff;">$eq. 3.38$</span>
$\{y(x; D)-h(x)\}^2$

위의 값은 특정 데이터 셋의 의존적이기 때문에, 우리는 다양한 데이터 셋에서 평균을 취할 것이다.
만약 우리는 위의 수식에서 $E_D[y(x; D)]$ 를 더하고 차감하면 다음과 같은 수식을 얻을 수 있다.

><span style="color:#ff33ff;">$eq. 3.39$</span>
$\{y(x; D)-E_D[y(x; D)]+E_D[y(x; D)]-h(x)\}^2$
$= \{y(x; D)-E_D[y(x; D)]\}^2+\{E_D[y(x; D)]-h(x)\}^2$
$+2\{y(x;D)-E_D[y(x; D)]\}\{E_D[y(x; D)]-h(x)\}$
</span>
