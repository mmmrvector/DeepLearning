## CS229 ##

### Lecture 1  Introduction and Basic Concepts ###

### Lecture 2  Supervised Learning. Linear Regression ###

+ #### Part 1 Linear Regression ####

  + **LMS algorithm**

    为了得到预测方程
    $$
    h(x) = \sum_{i=0}^{d}θ_ix_i = θ^Tx
    $$
    需要一个成本函数(cost function)
    $$
    J(θ)=\frac{1}{2}\sum_{i=1}^{n}(h_\theta(x^{(i)})-y^{(i)})^2
    $$
    要求得最小的$J(\theta)$，采用梯度下降算法(gradient descent)，给定一个初始值θ，然后重复一下步骤更新θ：
    $$
    \theta_j:=\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta)
    $$
    （此更新在j = 0, 1, ... d上同时发生），$\alpha$为学习率，通过对$J(θ)$求偏导可得
$$
    \theta_j:=\theta_j+\alpha(y^{(i)}-h_\theta(x^{(i)}))x_j^{(i)}
$$
这个更新规则被称为最小均方差更新规则(LMS, least mean square)，也被称为Widrow-Hoff学习规则

目前只有一个训练数据$i$。有两种方法更改此更新规则，使其用于多个训练数据。

+ 法一 **批梯度下降法**

  重复以下计算直至收敛
  $$
  \theta_j:=\theta_j+\alpha\sum_{i=1}^n(y^{(i)}-h_\theta(x^{(i)}))x_j^{(i)},(for\ every\ j)
  $$
  此方法每一步都会用到整个训练集中的每一个数据，因此被称为批梯度下降法(Batch gradiant descent)

+ 法二 **随机梯度下降法**

  Loop {

  for $i = 1$ to $n$, {

  $θ_j:=θ_j+\alpha(y^{(i)} - h_θ(x^{(i)}))x_j^{(i)}$

  }

  }

  + **The normal equations**
  
    此方法我们将通过明确求出$θ_j$导数的方式求$J$的最小值。首先介绍一些数学符号
  
    + Matrix derivatives
      $$
      \nabla_Af(A)=\begin{bmatrix}\frac{\partial f}{\partial A_{11}}&...&\frac{\partial f}{\partial A_{1d}}\\\vdots&\ddots&\vdots\\\frac{\partial f}{\partial A_{n1}}&...&\frac{\partial f}{\partial A_{nd}}\end{bmatrix}
      $$
    
    + Least squares revisited
    
      重新审视最小二乘法
    
      定义设计矩阵$X$ 为训练集的输入数据
      $$
      X = \begin{bmatrix}(x^{(1)})^T\\\vdots\\(x^{(n)})^T\end{bmatrix}
      $$
      令$\vec y$表示训练集中的目标值
      $$
      \vec y = \begin{bmatrix} y^{(1)}\\ \vdots \\ y^{(n)} \end{bmatrix}
      $$
      由于$h_θ(x^{(i)})= (x^{(i)})^Tθ$，所以
      $$
      X\theta - \vec y = \begin{bmatrix} h_\theta(x^{(1)}) - y^{(1)} \\ \vdots \\ h_\theta(x^{(n)}) - y^{(n)} \end{bmatrix}
      $$
      又由于$z^Tz = \sum_iz_i^2$
    
      因此
      $$
      \begin{aligned}\frac{1}{2}(X\theta - \vec y)^T(X\theta - \vec y)&=\frac{1}{2}\sum_{i=1}^n(h_\theta(x^{(i)})-y^{(i)})^2\\&=J(\theta)
      \end{aligned}
      $$
      为了最小化$J$，需要求出$θ$的导数
      $$
      \begin{aligned}
      \nabla _\theta J(\theta) &= \nabla _\theta \frac{1}{2}(X\theta - \vec y)^T(X\theta - \vec y) \\&= \nabla _\theta \frac{1}{2}((X\theta)^T(X\theta) - (X\theta)^T\vec y - \vec y^T(X\theta) + \vec y^T \vec y)\\&=\nabla _\theta \frac{1}{2}(\theta^T(X^TX)\theta - \vec y^T(X\theta)-\vec y^T (X\theta))\\&= \nabla _\theta \frac{1}{2}(\theta^T(X^TX)\theta - 2 (X^T \vec y)^T\theta)\\ & = \frac {1}{2}(2X^TX\theta - 2(X^T \vec y))\\&=X^TX\theta-X^T \vec y 
      \end{aligned}
      $$
  
  + **Problistic interpretation**
  
    采用最大似然估计的方法解释了为什么最小二乘法是一个合理的选择
  
  + **Locally weighted linear regression（局部加权线性回归）**
  
    LWR算法假设有足够多的训练数据，使得选择特征变得不再那么重要。
  
    LWR算法：
  
    1. 寻找使得 $\sum_iw^{(i)}(y^{(i)}-θ^Tx^{(i)})^2$ 最小的 $\theta$
    2. 输出$θ^Tx$
  
    其中 $w^{(i)}$ 表示非负的权重值。一个相对标准的权重选择方法如下:
    $$
    w^{(i)} = exp(-\frac{(x^{(i)} - x)^2}{2\tau^2})
    $$
    权重的选择与点 $x$ 有关。如果 $|x^{(i)} -x|$ 越小，则 $w^{(i)}$ 越接近于1， 如果 $|x^{(i)} - x|$  越大，则 $w^{(i)}$ 越小。
  
    因此 $θ$ 将给予那些距离查询点 $x$ 近的点更高的权重。参数 $\tau$ 控制了一个训练样本权重下降的速度，被称为带宽参数。是一个超参数。
  
    LWR是一个非参数算法，而线性回归则是一个参数学习算法。所谓参数学习算法，有固定明确的参数，一旦确定就不会改变了，而参数学习算法则需要保留训练样本，每进行一次预测就要重新学习一遍。
  
+ #### Part 2 Classification and logistic regression

  + **Logistic regression**

    预测函数 $h_θ(x)$ 为以下的形式
    $$
    h_\theta(x)=g(\theta^Tx) = \frac{1}{1 + e^{-\theta^Tx}}
    $$
    其中
    $$
    g(z)=\frac{1}{1 + e^{-z}}
    $$
    被称为**logistic function** 或者 **sigmoid function**

    sigmoid函数的导数如下：
    $$
    \begin{aligned}
    g'(z)=&\frac{d}{dz}\frac{1}{1+e^{-z}}\\
    =&\frac{1}{(1+e^{-z})^2}(e^{-z})\\=&\frac{1}{(1+e^{-z})}·(1-\frac{1}{(1+e^{-z})})\\
    =&g(z)(1-g(z))
    \end{aligned}
    $$
    在采用逻辑回归模型的情况下，如何求出 $θ$ 。

    假设
    $$
    \begin{aligned}
    &P(y=1|x;\theta)\ =\ h_\theta(x)\\
    &P(y=0|x;\theta)\ =\ 1-h_\theta(x)
    \end{aligned}
    $$
    可以写成下面这种形式
    $$
    P(y|x;\theta)=(h_\theta(x))^y(1-h_\theta(x))^{1-y}
    $$
    假设有n个独立产生的训练数据，则参数的似然函数可以写成
    $$
    \begin{aligned}
    L(\theta)\ =& \ p(\vec y|X;\theta)\\
    =&\ \prod_{i=1}^np(y^{(i)}|x^{(i)};\theta)\\
    =&\ \prod_{i=1}^n(h_\theta(x^{(i)}))^{y^{(i)}}(1-h_\theta(x^{(i)}))^{1-y^{(i)}}
    \end{aligned}
    $$
    对似然函数取对数
    $$
    \begin{aligned}
    \ell(\theta)\ =& \ {\rm log}L(\theta)\\
    =& \ \sum_{i=1}^ny^{(i)}{\rm log}h(x^{(i)})+(1-y^{(i)}){\rm log}(1-h(x^{(i)}))
    \end{aligned}
    $$
    采用梯度下降的方法求最大似然函数，用向量的方法写
    $$
    \theta:=\theta+\alpha\nabla_\theta\ell(\theta)
    $$
    假如只有一个训练数据
    $$
    \begin{aligned}
    \frac{\partial}{\partial\theta_j}\ell(\theta)=&\ (y\frac{1}{g(\theta^Tx)}-(1-y)\frac{1}{1-g(\theta^Tx)})\frac{\partial}{\partial\theta_j}g(\theta^Tx)\\
    =&\ (y\frac{1}{g(\theta^Tx)}-(1-y)\frac{1}{1-g(\theta^Tx)})g(\theta^Tx)(1-g(\theta^Tx))\frac{\partial}{\partial\theta_j}\theta^Tx\\
    =&\ (y(1-g(\theta^Tx))-(1-y)g(\theta^Tx))x_j\\
    =&\ (y-h_\theta(x))x_j
    \end{aligned}
    $$
    在上面的计算过程中，使用了 $g'(z)=g(z)(1-g(z))$ 的结果。

    因此随机梯度上升的方法如下：
    $$
    \theta_j:=\theta_j+\alpha(y^{(i)}-h_\theta(x^{(i)})x_j^{(i)}
    $$
  
+ **Digression: The perceptron learning algorithm（感知机算法）**
  
+ **Another algorithm for maximizing $\ell(θ)$**
  
  首先介绍一下牛顿迭代法。它的迭代规则如下：
  $$
    \theta:=\theta-\frac{f(\theta)}{f'(\theta)}
  $$
    设 $θ_r$ 是 $f(\theta)=0$ 的根，选取 $θ_0$ 作为 $\theta_r$ 的初始值，过点 $(θ_0,f(θ_0))$ 作曲线的切线 $L$ 。$L:y=f(θ_0)+f'(θ_0)(θ-θ_0)$ ，则 $L$ 与横轴交点为 $θ_1=θ_0-\frac{f(θ_0)}{f'(θ_0)}$ ，则称 $θ_1$ 为 $θ_r$  的第一次近似值。重复以上步骤可得到 $θ_r$ 的近似值。
  
  可以使用牛顿迭代法来求解 $f(\theta)=0$ 。因此如果我们希望求得 $\ell(\theta) $ 的最大值，可以通过求解一阶导数的零点。令 $f(θ)=\ell'(\theta)$ ，则迭代规则为：
  $$
    \theta:=\theta-\frac{\ell'(\theta)}{\ell''(\theta)}
  $$
    由于 $θ$ 是一个向量，因此迭代规则更新为：
  $$
    \theta:=\theta-H^{-1}\nabla_\theta\ell(\theta)
  $$
    其中 $H$ 为**Hessian**矩阵:
  $$
    H_{ij}=\frac{\partial^2\ell(\theta)}{\partial\theta_i\partial\theta_j}
  $$
  
  
   牛顿迭代法收敛的速度一般比梯度下降法要快，但是牛顿法一次迭代的开销要比梯度下降法多。只要**Hessian**矩阵的维数不高，通常牛顿法要更快些。当牛顿法被用于求解逻辑回归问题的对数似然函数最大值时，这个方法又被称为**Fisher Scoring**
  
+ #### Part 3 Generalized Linear Models(广义线性模型) ####

  + #### The exponential family ####

    定义一类分布属于指数族，当它可以写成以下形式：
    $$
    p(y;\eta)=b(y){\rm exp}(\eta^TT(y)-a(\eta))\ \ \ \ \ (3)
    $$
    此处 $\eta$ 被称为分布的自然参数（也被称为规范参数）；$T(y)$ 是充分统计量（对于我们锁考虑的分布，通常情况下有 $T(\eta)=y$）; $a(\eta)$ 被称为对数划分函数。这一项 $e^{-a(\eta)}$ 的本质是起到了正则化常数的作用，确保了分布 $p(y;\eta)$ 的总和或是积分在 $y$ 到1上。

    以Bernouli分布和Gauss分布为例，来说明他们属于指数族分布。

    Bernouli分布可以写成如下的形式
    $$
    \begin{aligned}
    p(y;\phi)=&\ \phi^y(1-\phi)^{1-y}\\
    =&\ {\rm exp}(y{\rm log}\phi+(1-y){\rm log}(1-\phi))\\
    =&\ {\rm exp}(({\rm log}\frac{\phi}{1-\phi})y+{\rm log}(1-\phi))
    \end{aligned}
    $$
    因此自然参数由 $\eta={\rm log}\frac{\phi}{1-\phi}$ 给出。有趣的是，如果把 $\eta$ 的这个定义转化为用 $\eta$ 来求解 $\phi$ ，可以得到 $\phi=\frac{1}{1+e^{-\eta}}$ ，这就是sigmoid函数。

    考虑高斯分布，当导出线性回归时， $\sigma^2$ 的值对于最终选择 $θ$ 和 $h_θ(x)$ 无影响，因此 $\sigma^2$ 可以选择任意值。为了简化推导，选择$\sigma^2 = 1$。则有:
    $$
    \begin{aligned}
    p(y;\mu)=&\ \frac{1}{\sqrt{2\pi}}{\rm exp}(-\frac{1}{2}(y-\mu)^2)\\
    =&\ \frac{1}{\sqrt{2\pi}}{\rm exp}(-\frac{1}{2}y^2)·{\rm exp}(\mu y-\frac{1}{2}\mu^2)
    \end{aligned}
    $$

  + #### Constructing GLMs ####

    考虑一个分类或回归问题，我们希望得到随机变量y关于x的函数。为了获得这个问题的广义线性模型，我们有几以下三个假设：

    1. $y|x;θ \sim{\rm ExpotionalFamily}(\eta) $ 
    2. 给定 $x$ ，目标是得到给定 $x$ 的 $T(y)$ 的期望值。在大多数例子中，有 $T(y)=y$ ，这意味着我们希望学习假设 $h(x)$ 满足 $h(x)=E[y|x]$  
    3. 自然参数 $\eta$ 和输入 $x$ 呈线性相关： $\eta=θ^Tx$ 

    第三个假设似乎是最不合理的，在我们设计GLMs配方中，就其本身而言，它可能被认为是一种设计选择而不是一种假设。这三个假设/设计选择使我们能够派生出非常优雅的一类学习算法，即GLMs。

    + **Ordinary Least Squares(普通最小二乘法)**

      为了证明普通最小二乘法是GLM族中的一种模型，考虑目标变量 $y$ 连续，且给定 $x$ ， $y$ 服从高斯分布 ${\mathcal N}(\mu,\sigma^2)$ 。如之前所说，在指数分布族中的高斯分布模型形式下，有 $\mu=\eta$ ，所以：
      $$
      \begin{aligned}
      h_\theta(x)=&\ E[y|x;\theta]\\
      =&\ \mu\\
      =&\ \eta\\
      =&\ \theta^Tx
      \end{aligned}
      $$
      其中第一个等式由假设2而来。第二个等式由条件 $y|x;\theta \sim {\mathcal N}(\mu,\sigma^2)$ 而来，$y$ 的期望为 $\mu$ 。第三个等式由之前的证明中得到，高斯分布作为一种指数分布，其 $\mu =\eta$  。第四个等式由假设3而来 。

    + **Logistic Regression**

      考虑逻辑回归。由于 $y$ 假设为二值0或1，因此选择伯努利分布族来对 $y$ 在给定 $x$ 情况下的分布进行建模。在之前的证明中，我们知道，伯努利分布中的 $\phi=\frac{1}{1+e^{-\eta}}$ 。此外，如果 $y|x;θ \sim {\rm Bernoulli}(\phi)$ ，那么 ${\rm E}[y|x;θ]=\phi$ 。所以我们将得到:
      $$
      \begin
      {aligned}h_θ(x)=&\ E[y|x;θ]\\=&\ \phi\\=&\ \frac{1}{1+e^{-\eta}}\\=&\ \frac{1}{1+e^{-\theta^Tx}}
      \end{aligned}
      $$

    + **Softmax Regression**

      考虑一种分类问题，在这种问题中，响应变量有 $k$ 种值可以选择，即 $y\in\{1,2,...,k\}$ 。

      为了参数化这k种输出，令 $\phi_i=p(y=i;\phi)$ ，令 $p(y=k;\phi)=1-\sum_{i=1}^{k-1}\phi_i$ 因为 $\phi_k$ 完全由 $\phi_1,...,\phi_{k-1}$ 决定。 所以它不是参数。为了书写方便，写为 $\phi_k$。

      定义 $T(y)\in{\mathbb R}^{k-1}$ 如下形式: 
      $$
      T(1)=\begin{bmatrix}1\\0\\0\\ \vdots \\0 \end{bmatrix},T(2)=\begin{bmatrix}0\\1\\0\\\vdots\\0\end{bmatrix},...,T(k-1)=\begin{bmatrix}0\\0\\0\\\vdots\\1\end{bmatrix},T(k)=\begin{bmatrix}0\\0\\0\\\vdots\\0\end{bmatrix}
      $$
      与之前不同的是，此处没有 $T(y)=y$ ，并且 $T(y)$ 是一个 $k-1$ 维的向量，而不是实数。我们将用$ (T(y))_i$ 来表示 $T(y)$ 的第 $i$ 个元素。

      另外介绍一种符号，指示函数 $1\{·\}$ 表示如果括号内值为真则返回1，值为假则返回0。因此我们可以将 $T(y)$ 和 $y$ 之间的关系写为 $(T(y))_i=1\{y=i\}$  。此外，还有 ${\rm E}[(T(y))_i]=P(y=i)=\phi_i$ 

      下面证明此种情况亦属于指数分布族：
      $$
      \begin{aligned}
      p(y;\theta)=&\ \phi_1^{1\{y=1\}}\phi_2^{1\{y=2\}}...\phi_k^{1\{y=k\}}\\
      =&\ \phi_1^{1\{y=1\}}\phi_2^{1\{y=2\}}...\phi_k^{1-\sum_{i=1}^{k-1}1\{y=i\}}\\
      =&\ \phi_1^{(T(y))_1}\phi_2^{(T(y))_2}...\phi_k^{1-\sum_{i=1}^{k-1}(T(y))_i}\\
      =&\ {\rm exp}((T(y))_1{\rm log}(\phi_1)+(T(y))_2{\rm log(\phi_2)+...+(1-\sum_{i=1}^{k-1}(T(y))_i){\rm log}(\phi_k)}\\
      =&\ {\rm exp}((T(y)_1){\rm log}(\phi_1/\phi_k)+(T(y))_2{\rm log}(\phi_2/\phi_k)+...+(T(y))_{k-1}{\rm log}(\phi_{k-1}/\phi_k)+{\rm log}(\phi_k)\\
      =&\ b(y){\rm exp}(\eta^TT(y)-a(\eta))
      \end{aligned}
      $$
      其中
      $$
      \begin{aligned}
      \eta=&\ \begin{bmatrix}
      {\rm log}(\phi_1/\phi_k)\\
      {\rm log}(\phi_2/\phi_k)\\
      \vdots\\
      {\rm log}(\phi_{k-1}/\phi_k)
      \end{bmatrix},\\
      \alpha(\eta)=&\  -{\rm log}(\phi_k)\\
      b(y)=&\ 1
      \end{aligned}
      $$
      我们有连接函数：
      $$
      \eta_i={\rm log}\frac{\phi_i}{\phi_k}
      $$
      为了反转连接函数，派生出响应函数，有：
      $$
      \begin{aligned}
      e^{\eta_i}=&\ \frac{\phi_i}{\phi_k}\\
      \phi_ke^{\eta_i}=&\ \phi_i\ \ \ \ \ \ (4)\\
      \phi_k\sum_{i=1}^ke^{\eta_i}=&\ \sum_{i=1}^k\phi_i
      =1\end{aligned}
      $$
      这表明 $\phi_k=\frac{1}{\sum_{i=1}^ke^{\eta_i}}$ ，代入等式（4）中，得到响应函数
      $$
      \phi_i=\frac{e^{\eta_i}}{\sum_{j=1}^ke^{\eta_j}}
      $$
      这个将 $\eta$ 映射到 $\phi$ 的函数被称为**softmax**函数。

      根据假设3，又有 $\eta_i=θ_i^Tx$ ，其中 $θ_1,...,θ_{k-1}\in{\mathbb R}^{d+1}$ 是我们模型的参数。为了书写方便 ，定义 $θ_k = 0$ ，因此 $ \eta_k=θ_k^Tx=0$ 。模型可写成：
      $$
      \begin{aligned}
      p(y=i|x;\theta)=&\ \phi_i\\
      =&\ \frac{e^{\eta_i}}{\sum_{j=1}^ke^{\eta_j}}\\
      =&\ \frac{e^{\theta_i^Tx}}{\sum_{j=1}^ke^{\theta_j^Tx}}
      \ \ \ \ \ \ \ \ (5)
      \end{aligned}
      $$
      这个用于多重分类的模型被称为 **softmax regression** ，是逻辑回归的一种推广形式。

      预测方程将变为:
      $$
      \begin{aligned}
      h_\theta(x)=&\ {\rm E}[T(y)|x;\theta]\\
      =&\ {\rm E}
      \begin{bmatrix}
      \begin{matrix}
      1\{y=1\}\\
      2\{y=2\}\\
      \vdots\\
      1\{y=k-1\}
      \end{matrix}
      \ 
      |x;\theta
      \end{bmatrix}\\
      =&\ 
      \begin{bmatrix}
      \phi_1\\
      \phi_2\\
      \vdots\\
      \phi_{k-1}
      \end{bmatrix}\\
      =&\ 
      \begin{bmatrix}
      \frac{{\rm exp}(\theta_1^Tx)}{\sum_{j=1}^k{\rm exp}(\theta_j^Tx)}\\
      \frac{{\rm exp}(\theta_2^Tx)}{\sum_{j=1}^k{\rm exp}(\theta_j^Tx)}\\
      \vdots\\
      \frac{{\rm exp}(\theta_{k-1}^Tx)}{\sum_{j=1}^k{\rm exp}(\theta_j^Tx)}\\
      \end{bmatrix}
      \end{aligned}
      $$
      假如我们有已个训练数据集包含n个样本 $\{(x^{(i)},y^{(i)});i=1,...,n\}$ ，想要学习这个模型的参数 $θ$ 。首先写下似然函数：
      $$
      \begin{aligned}
      \ell(\theta)=&\ \sum_{i=1}^n{\rm log}(y^{(i)}|x^{(i)};\theta)\\
      =&\ \sum_{i=1}^n{\rm log}\prod_{l=1}^k(\frac{e^{\theta_l^Tx^{(i)}}}{\sum_{j=1}^ke^{\theta_j^Tx^{(i)}}})
      \end{aligned}
      $$
      其中第二行等式根据 等式（5）中 $p(y|x;θ)$ 的定义而来。接下来可以通过梯度上升或牛顿迭代法来求 $\ell(θ)$ 的最大值，从而获得参数的最大似然估计值



