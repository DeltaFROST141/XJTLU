# The basic understanding of code
## Channel splitting
### Get the time domain
Input parameter: x - time series data, w - window size

Output parameter: y - multi dimensional matrix

The **first layer and second layer loops** are the following:

The first layer choose time series to be compared, and the second layer use it to compare with another series.
For example, there are 10 series in total.

When the outer loop process the first series, the middle loop will process from second series to tenth series. There are nine comparison.

The same, the outer loop process the second series, the middle loop will process from third series to tenth series. There are eight comparison.

According to the above statement, the number of comparisons of n time series data is calculated using the combination number `C(n,2)`. When n is ten, the number of comparisons will be $45$.

When the loops are processed, the `x1` and `x2` will do Multiple reassignments according to the idea of the following(comparison between different series). 

```matlab
t1 = [min(x1((k-1)*w+1:k*w)), max(x1((k-1)*w+1:k*w))];
t2 = [min(x2((k-1)*w+1:k*w)), max(x2((k-1)*w+1:k*w))];
```
`t1` and `t2` are defined in the inner loop and loop over subset(w is window size and k is current window index)

And then use `meshgrid()` to generate the 2*2 matrix for the `T1` and `T2` respectively.
For example:

```makefile
T1 = 1 3
     1 3
each rows are the copy of t1

T2 = 2 2 
     4 4
each columns are the copy of t2

The combination will be:
(1,2),(3,2),(1,4),(3,4)
```

The total pairs of `T1` and `T2` are $\frac{t(t-1)}{2} \times g$.
In any moment, there is only one pair and it will be replaced soon.

So through the `meshgrid()`, I can get the combination of each windows.

And then `T1(:)` and `T2(:)` will become column vector and is assign to the third dimension of the y variable.

Let me explain the `y`: The first dimension tracks which two rows are being compared, the second dimension stores the combination of data in those two rows, and the third dimension tracks which window is being considered(slice operation).

In a nutshell, the first dimension is the number of row combination(10 - 45); the second is 2(always `T1` and `T2`);the third dimension is two times the number of windows(if windows number g is 10, the result will be 20)


