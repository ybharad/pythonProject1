# This file is to revise python
# write a function to reverse a string
import random


def rev_str(str):
    rs = str[::-1]
    return(rs)

# find the first non repeated character
#     count all charaters in a string and then get the first non repeated character

def fnrc(str):
    d = {}
    for i in range(0:len(str)):
        k = str[i]

# looping through strings
s = 'yashas bharadwaj'

for i in range(0,len(s)):
    print[i]

print(range[0:4])


d = {}
for a in 'yashas':
    d[a] = d.get(a,0) +1


# create a list of numbers using range
ln = list(range(10))

# check for palindrome
def palin(str):
    if str == str[::-1]:
        print('palindrome')
    else:
        print('not a palindrome')

palin('bobcat')

# remove duplicates from a list
def dedup(lst):
    dlist = []
    for i in lst:
        if i not in dlist:
            dlist.append(i)
        else:
            pass
    return(dlist)

dedup([1,2,3,3,4,5,5,6])

# count words in a sentence
def cnt_wrd(str):
    d = {}
    s = str.split()
    for i in s:
        if i in d.keys():
            d[i] += 1
        else:
            d[i] = 1
    return(d)

cnt_wrd('mary had a little lamb had a a had mary lamb a')

kl = []
vl = []
for k,v in dt.items():
    kl.append(k)
    vl.append(v)


tp = ()
for i in zip(kl,vl):
    tp.add(i)







# fibonacci sequence upto  certain number 'n'
# using while loops:
def fib(int):
    fib = [0,1]
    while (fib[-1] + fib[-2]) < int:
        fib.append(fib[-1]+fib[-2])
    return(fib)

# fibonacci sequence containing n numbers
def fibn(n):
    fib = [0,1]
    while len(fib) < n:
        fib.append(fib[-1]+ fib[-2])
    return(fib)


# factorial calculation: using loops
def fact(n):
    if n == 0:
        return(1)
    else:
        k = 1
        for i in range(1,n+1):
            k = k*i
    return(k)

# factorial calculation calling the same function:
def fact1(n):
    if n == 0:
        return(1)
    else:
        return(n * fact1(n-1))

# list comprehension
lst2 = [i*i for i in range(1,8) if i%2 == 0]


# # write a function to get the below:
# You have a list of tuples that describes a group of balls in a box. Each tuple has the colour and the number of balls in that color.

# Input: balls = [('red', 3), ('blue', 5), ('green', 1)], N = 5
#Output: ['red', 'red', 'blue', 'blue', 'green']

balls = [('red', 3), ('blue', 5), ('green', 1)]

import random
pb = []
clr = []
for i in balls:
    pb.append(i[1])
    clr.append(i[0])
print(pb)
print(clr)

res = []
for i in range(5):
    res.append(random.choices(clr,pb))
print(res)

## =================== ## ===================## ===================## ===================## ===================

# important functions for testing regression metrics like MAE (mean absolute error)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
def score_dataset(X_train,X_valid,y_train,y_valid):
    model = RandomForestRegressor(n_estimators = 100, random_state= 1)
    model.fit(X_train,y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid,preds)

# practice model building
import pandas as pd

train = pd.read_csv('/Users/ybharadwaj2/Downloads/kaggle_data/train.csv')
train.head()


train.shape
train.dtypes

# describe data
train.describe()

# check distribution of sale price
train.SalePrice.describe()
import matplotlib.pyplot as plt

# function to generate histogram plots
def hist1(colname,df):
    plt.hist(df[colname], bins = 10, color = 'skyblue')
    plt.xlabel(colname)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {colname}')
    plt.grid(True)
    plt.show()

# plot using the function
hist1('SalePrice',train)

# count nulls in the columns of train
train.head()

nulcols = [c for c in train.columns if train[c].isnull().any()]

null_train = train.isnull().sum()
null_train

# get categorical columns and numerical columns:

calg_col = [x for x in train.columns if train[x].dtype == "object"]


#

def max_profit(prices):
    b = prices.index(min(prices))
    print(b)
    p1 = prices[b:len(prices)]
    print(p1)
    s = p1.index(max(p1))
    print(s)
    profit = (p1[s] - prices[b])

    return(profit)


# correlation between 2 variables
import math

def mean(x):
    return sum(x) / len(x)


def sd(x):
    m = mean(x)
    ss = sum((i - m) ** 2 for i in x)
    return math.sqrt(ss / len(x))


def corr(x, y):
    if len(x) != len(y):
        return print('vectors of different lengths')

    else:
        n = len(x)
        xm = mean(x)
        ym = mean(y)
        xxm = sum([i - xm for i in x])
        yym = sum([i - ym for i in y])
        cv = xxm * yym
        sds = (sd(x) * sd(y))
    return cv / sds

# word frequency counter - using dictionary:
def wfreq(str):
    str = str.split()
    d = {}
    for i in str:
        if i in d.keys():
            d[i] = d[i]+1
        else:
            d[i] = 1
    return d

# character frequency counter
def cfreq(cstr):
    crs = [i for i in cstr if i != " "]
    d = {}
    for i in crs:
        if i in d.keys():
            d[i] = d[i] + 1
        else:
            d[i] = 1
    return d


# get the key of a dict when a value is given
def get_key(dict, value):
    key = [key for key, val in dict.items() if val == value]
    return key


# Anagram checker: Anagrams are 2 strings having the same characters in the same frequency
def anagrm(str1, str2):
    d1 = cfreq(str1)
    d2 = cfreq(str2)
    if d1 == d2:

        print('words are Anagram')
    else:
        print('words are not Anagram')


### use random library to generate random choices
import random
ua = list(ua)
weights = [0.3,0.3,0.4]
s = ['a','b','c']
random.choices(s,weights)


### - Implement a Library book tracking using python classes

class book:
    def __init__(self, bname, cost, qty):
        self.bname = bname
        self.cost = cost
        self.qty = qty

    def lends(self, bname, qty):
        self.bname = bname
        self.qty = self.qty - qty


class library():
    def __init__(self):
        self.books = []

    def add_book(self,book):
        self.books.append(book)
        print('book added:',book.bname)





b1 = book('got1', 20, 10)
b2 = book('got2', 25, 15)
b3 = book('got3', 20, 12)
b4 = book('got4', 20, 15)
b5 = book('hp1', 10, 15)

b1.lends('got1',2)

lib = library()
lib.add_book(b1)



k = 1
for e in l:
    k = k*e
    print(k)


def trailing_zeroes(n):
    ans = 1
    for i in range(1,n+1):
        ans = i * ans
    print(ans)
    ans1 = str(ans)
    print(ans1[::-1])
    tz = 0
    for k in ans1[::-1]:
        if k == '0':
            tz = tz + 1
            print(tz)
        else:
            return
    return tz



def two_sum(input: list[int], target: int) -> list[int]:
    t = []
    ix = []
    for i in range(len(input)):
        for k in range(i + 1, len(input)):
            l = (input[i], input[k])
            t.append(l)
    print(t)
    for h in t:
        if target == sum(h):
            ix = [input.index(h[0]),input.index(h[1])]
            print(ix)
            return(ix)
    return [-1, -1]

two_sum([1,4,6,10],2)


# -- get the smallest multiple - the number must be smallest and must be fully divisible by all
# numbers upto the target

def factorial(n):
    f = 1
    for i in range(2,n+1,1):
        f = f*i
    return f

def smallest_multiple(target):
    l = []
    f = factorial(target)
    for i in range(target + 1 ,f ,1):
        pd = True
        for k in range(1,target+1):
            if (i % k != 0):
                pd = False
                break
        if (pd == True):
            print(i)

smallest_multiple(5)


# enumerate gives index and the iterable object
fruits = ["apple", "banana", "cherry", "date"]
for i , f in enumerate(fruits):
    print(i,f)

fts = ('MA', 'CA', 'TX', 'NC')
for i,t in enumerate(fts):
    print(i,t)


# -- three sum code:
# given an array calculate the triplets that equal to the target sum




def threesum(arr, tsum):

    arr.sort()
    res = []

    for i,x in enumerate(arr):
        if i > 0 and x == arr[i-1]:
            continue
        l,r = i+1, len(arr) -1
        # print(i, x, l,r)
        while l < r:
            ts = arr[i] + arr[l] + arr[r]
            if ts < tsum and l < r:
                l += 1
            elif ts > tsum:
                r -= 1
            else:
                res.append([arr[i],arr[l],arr[r]])
                l +=1
                while arr[l] == arr[l-1] and l < r:
                    l += 1
    print(res)
    return res

# Example usage
nums = [-1, 0, 1, 2, -1, -4]
target = 0
print(threesum(nums, target))
























