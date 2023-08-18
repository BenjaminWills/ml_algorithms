# Classification trees

Classification trees are a subset of decision trees that have a discrete output. They aren't necessarily machine learning algorithms in the strictest sense, they simply take a set of data and break it down into many regions through a series of boolean statements to filter the data down.

## How do they work?

We begin with a whole dataset. This dataset will have one classification column and an arbritrary amount of non classification columns. The aim of the tree is to be able to take this data, and split it into two. 

### Information entropy

What makes a good split? Suppose we have a binary classification column, in an ideal world we would want to split on some inequality: $x_n \leq \lambda$, all the data that suits this inequality will go to the left child of the root, and all the rest will go to the right. In this ideal world the left child contains one class and the right child contains the other class. Unfortunately this is normally not the case. So we need a measure of how much information each split of the data contains, suppose $p$ s the proportion of class one relative to class two, then we define the **information entropy** as follows:

$$
E(\bold{p}) = - \sum_i p_i \log{p_i}
$$

Since we only care about $0 \leq p_i \leq 1$, this function is very meaningful! In our case $\bold{p} = (p,1-p)$, such that when we substitute this into our entropy equation we get:

$$
E(\bold{p}) = - (1-p)\log{(1-p)} - p\log{p}
$$

Thus, when p = 1, i.e when our dataset entirely contains one class

$$
E(0,1) = - (1)\log{(1)} - 0\log{0} = 0
$$

The information entropy is 0, hooray! Further the function reaches its max disorder when $p = 0.5$.

$$
E(0.5,0.5) = -\log{\frac{1}{2}} = \log{2}
$$

This is the maximum value of entropy! Intuitively the data is "messy" when the proportions are equally 0.5, as there is no information present. Thus we conclude that enropy is a measure of **disorder**, the higher the entropy, the more the disorder!

### Splitting the data

Now that we have a measure of disorder, we wish to find a tree such that the disorder is minimised, such that every **leaf node**, a node with no children, has an entropy as close to zero as possible.

Suppose that we are splitting one datapoint, that datapoint will contain some subset of the data, we want to find the exact value of that data for which splitting it on that value yields the largest **information gain**. 

#### Information gain

Intuitively information gain in this context would mean organising the classes, splitting the data in such a way that the majority of one class moves to one side and visa versa. The maths works out the same way. To maximise information gain is to minimise entropy gain, an analogoous statement is to say: to maximise order is to minimse disorder. 

When we split our data on some arbritrary value $\lambda$, we naturally have 2 datasets, suppose these datasets make up $\omega_1, \omega_2$% of the dataset each, so $\omega_1 + \omega_2 = 1$. Then we can measure information gain as follows:

$$
\text{information gain} = E(parent) - \sum_i \omega_iE(\text{child}_i)
$$

If we revisit our intuition, to maximise information gain here, means that we must minimise the only non constant term, $\sum_i \omega_iE(\text{child}_i)$, aka the combined entropy of the split!

So now we know the rules. We aim to split the data arbritrarily many times on a value that splits the datasets to maximise information gain. This process quickly becomes a tree, the outcome is then a fully divided dataset that minimises entrop.

Thus, we now know why we can't really call this a ML algorithm! But it's pretty cool nonetheless. It shows that we can accurately split up any dataset using some relatively simple rules! We also generate these rules from the dataset it's self! It works like our brains would and I find that facinating.