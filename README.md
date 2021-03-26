# Maching Learning Practices

# Practices 1 

**I have learned Linear Regression and Iterative Search for Optimal through several samples**
* Part 1
  * Iterative Search for Optimal with these functions
    
    <img src="http://github.com/RubenGiC/Practices-about-Maching-learning/blob/main/P1/Images/Tex2Img_1616618057.jpg?raw=true" alt="f(x,y)">
    </br>
    <img src="http://github.com/RubenGiC/Practices-about-Maching-learning/blob/main/P1/Images/Tex2Img_1616618271.jpg?raw=true" alt="Tex2Img_1616618271.jpg">

## Gradient Descent Implementation 

<p>The Gadient Descent algorithm is a general iterative optimization technique, where reach a local optimum.</p>

<img src="http://github.com/RubenGiC/Practices-about-Maching-learning/blob/main/P1/Images/descarga.gif?raw=true" alt="Gradient Descent">

<p>For this I need a small constant that will be my learning rate, which is denoted by η. We must be careful with this constant, since if it is too large it may not reach the optimal local and if it is too small, it reaches the optimal local, but it entails a great computational cost. </p>
<p>Therefore, the optimal learning rate will depend on the topoligy of our loss landscape, which in turn depends on the data set.</p>

<img src="http://github.com/RubenGiC/Practices-about-Maching-learning/blob/main/P1/Images/learning%20rate.png?raw=true" alt="learning rate.png">

<p>Before explaining the algorithm of mathematically, you need some values, apart from the learning rate, an initial value W0, which will be our stating point and the function that we want minimize.</p>

<p>The general equation is:</p>

<img src="http://github.com/RubenGiC/Practices-about-Maching-learning/blob/main/P1/Images/gd.png?raw=true" alt="gd.png">

<p>where Ein/Wj is the gradient of the differemtiable function.</p>

<p>this action is carried out simultaneously for all values of <b>j in N</b>, where <b>N</b> is a set of integer values.</p>

<p>And it's repeated applying the same calculation, until the cost of function can't be minimized any more or when it exceeds an X number of iterations, since it could be the case that it doesn't find the minimum cost function.</p>

## Implementation Pseudo code

```c++ 
//where w is the initial values, eta the learning rate, f the function and gradF is the gradient of the function
GradientDescend(w, eta, maxIter, f, gradF){
  it = 0;// number of iterations run
  while(it < maxIter and f(w) change){
    w = w - (eta * gradF(w));
    ++it;
  }
  return w, it;
}
```

# Bibliography
* https://www.jeremyjordan.me/nn-learning-rate/
* https://sigmoidal.ai/metodo-de-ensemble-vantagens-da-combinacao-de-diferentes-estimadores/

