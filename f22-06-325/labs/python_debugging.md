---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# <font color='red'>Python functions, Error handling, Exceptions, Debugging</font>

# Functions
Imagine that you have to open a file, read the contents of the file and close it. Pretty trivial, right? Now imagine that you have to read ten files, print their output or perform some computation on the contents and then close them. Now you don't want to sit there and type file i/o operations for every file. What if there are over 500 files?

This is where the functions come in. A function is a block of organized and reusable code in a program that performs a specific task which can be incorporated into a larger program or reused by passing different sets of parameters.

The advantages of using functions are:
- Allowing code reuse.
- Reducing code duplication.
- Improving readability while reducing the complexity of the code.

There are two basic types of functions: 
- Built-in functions 
- User defined functions. 

We have been using [built-in](https://docs.python.org/3/library/functions.html) functions for quite some time without actually understanding how a function works. This is the beauty of Python. Now let’s see how you can create your own functions and call them in your code.

## Defining Functions

A function is defined using the `def` keyword followed by the name of the function. The parameters or the arguments should be placed within the parentheses followed by the function name. The code block within every function starts with a colon and should be indented.

```{code-cell} ipython3
def mul(a, b):
    return('{} * {} = {}'.format(a, b, a*b))
    
print(mul(4, 5))
```

## Function Arguments

A function can be called by using following types of formal arguments:

- Required arguments
- Keyword arguments
- Default arguments
- Variable-length arguments

### Required Arguments:

Required arguments are passed to a function in correct positional order. The number of arguments being passed should be equal to the number or arguments expected by the function that is defined. Let's take a look at the example:

```{code-cell} ipython3
def info(name, sem):
    print('My name is: ',name)
    print('This is semester',int(sem))
```

```{code-cell} ipython3
info('Abilash', 2)
print()
print()
# What if we change the order in which we are passing the arguments?
info(2, 'Abilash')
```

We can see how swapping the order of function arguements can break it. There are cases where you will be able to make the function works independently of the order of arguement inputs.

+++

### Keyword Arguements

```{code-cell} ipython3
info(sem=2, name='Abilash')
```

Here by specifiying which input we are giving first we can change the order of our arguement inputs.

```{code-cell} ipython3
info('Abilash')
```

```{code-cell} ipython3
def info2(name, sem=2):
    print('My name is: ',name)
    print('This is semester',int(sem))

info2('Abilash')
```

You can see above that in `info` when the arguement sem was not given it threw a `TypeError` unlike in the case of `info2`. This is because we defined a default value for sem in info2 which will be used if no value is given during the function call.

+++

### Variable Length Arguements

At some point, you may need to process the function for more than the arguments that you specified when you defined the function. `print()` is on such examples. These arguments can be of variable length and are not named in the function definition, unlike required and default arguments. So how do you handle this?

```{code-cell} ipython3
def names(course, *names):
    print('Name of course: ',course)
    print('Name of students in the course:')
    for name in names:
        print(name)

names('Python', 'Jim', 'Jack', 'Mat')
```

## Type setting functions

In Python, you can indicate the function argument and return value type. This makes it easier to read your code and for others to work on it.

```{code-cell} ipython3
def numberremover(text: str, delim: int) -> str:
    '''
    We are trying to remove a specific number from a string. We take in a string text and integer int
    to return the editted string.
    '''
    newtext = text.split(str(delim))

    return ''.join(newtext)

numberremover('Lets3test4this3and333see if this69works',3)
```

However, it is essential to remember that if we give an input of the wrong type, python will ***NOT*** throw and error. We need to have other ways to check for this if we desire it. We will look at it at the end of the next section.

+++

# Exception Handling


An exception is a python object that represents an error. It is an event, which occurs during the execution of a program that disrupts the normal flow of the program's instructions. When such a situation occurs and if python is not able to cope with it, it raises an exception. We have been seeing errors like TypeError and NameError or IndentationError throughout our tutorial which caused our application or that code to stop the execution. To prevent this from happening, we have to handle such exceptions.

Following is a hierarchy for some built-in exceptions in python:

+++

``````
BaseException
 +-- KeyboardInterrupt
 +-- Exception
      +-- StandardError
      |    +-- ArithmeticError
      |    |    +-- FloatingPointError
      |    |    +-- OverflowError
      |    |    +-- ZeroDivisionError
      |    +-- AssertionError
      |    +-- ImportError
      |    +-- LookupError
      |    |    +-- IndexError
      |    |    +-- KeyError
      |    +-- NameError
      |    +-- RuntimeError
      |    |    +-- NotImplementedError
      |    +-- SyntaxError
      |    +-- SystemError
      |    +-- TypeError
      |    +-- ValueError

      +-- Warning
           +-- DeprecationWarning
           +-- PendingDeprecationWarning
           +-- RuntimeWarning
           +-- SyntaxWarning
           +-- UserWarning
           +-- FutureWarning
``````

+++

Let's take a look at an example

```{code-cell} ipython3
print(x)
```

Here we try to access a variable that has not been defined yet. Python raises a NameError and the execution halts. There are basically two ways to handle this error.

Make sure the variable is defined first.

Use try-catch block. Place the code to be executed inside the try block and place the exception to be handled in the except block.

```{code-cell} ipython3
try:
    print(x)
except NameError:
    print("You have not defined this variable yet!!!")
```

As observed from the above example, our execution continued even after we tried to print the non-exsistent variable x.

+++

# Argument of an Exception

An exception can have an argument, which is a value that gives additional information about the problem that caused the exception. The contents of argument vary by exception.

Here we see the case of a divide by zero error.

```{code-cell} ipython3
print(1/0)
```

Lets try to use a try-catch block with an error message.

```{code-cell} ipython3
for i in range(3, -3, -1):
    try:
        print(round(1 / i,1))
    except ZeroDivisionError as err:
        print('i =',i,'    Zero Division Error: ', str(err.args[0]))
```

## Finally clause

+++

`finally` keyword is a clause which contains the block of code that will always be executed regardless of whether there was any exception in the code or not. This is generally used to cleanup some resources in a program, especially when using file I/O operations.

```{code-cell} ipython3
fhandler = None
try:
    # Open file in read-only mode. Try renaming file to test1.txt
    fhandler = open('./sample_datasets/test.txt', 'r')
    # Read all lines
    print(fhandler.readlines())
except IOError:
    print('Error Opening File')
except ZeroDivisionError:
    print('You have a zero division error')
finally:
    # If the file was opened
    if fhandler:
        # Close the file
        fhandler.close
```

In the above example we can observe that we are trying to open a file and read its contents. If the file doesn't exist, it will raise an IOError exception. If that happens, our try-catch block will handle it. However once the file has been read, we need to close the file so that other processes or other functions in our code can access it. (Remember: when accessing/ modifying a file, the file is locked to that process which is performing the I/O operation on it. Unless the lock is released, no other process will be able to modify it.. ) 

To make sure we release the resources, in the `finally` block we are checking if fhandler is not null and closing it.

+++

## Assertions

We can check our code at anypoint during run time to see if things are running as expected. This can be done using `assert`.

```{code-cell} ipython3
def raiser(x: int, n:int) -> int:
    '''
    We want to raise number x  to the nth power(n>=1)
    x and n must be integers and positive.
    '''
    num = x
    for i in range(n-1):
        num *= x
        if i == 3:
            num *=-1
    return num

raiser(2,0),raiser(2,1),raiser(2,2)
```

We can see that the above function works flawlessly. As we know when we raise a positive number to the (n-1)th power, we expect a positive value. However, due to some flaws in the code it might not happen so. For example:

```{code-cell} ipython3
raiser(2,8)
```

We will then want to check if our code makes an error at any stage. This can be either due to inputs, the code flow or the output. We can check this by asserting at each stage if our output meets our expectation. `Assert  <check>, <error message>` The assert check must output a boolean result for it to work.

```{code-cell} ipython3
def raiser(x: int, n:int) -> int:
    '''
    We want to raise number x  to the nth power(n>=1)
    x and n must be integers and positive.
    '''
    assert type(x) == int
    assert x > 0
    assert type(n) == int
    assert n > 0
    
    num = x
    for i in range(n-1):
        num *= x
        if i == 3:
            num *=-1
        assert num > 0
        
    assert type(num) == int
    return num

raiser(2,8)
```

We see an AssertionError but have dont have an understanding of where in the loop it is happening. We can thus add a message.

```{code-cell} ipython3
def raiser(x: int, n:int) -> int:
    '''
    We want to raise number x  to the nth power(n>=1)
    x and n must be integers and positive.
    '''
    assert type(x) == int
    assert x > 0
    assert type(n) == int
    assert n > 0
    
    num = x
    for i in range(n-1):
        num *= x
        if i == 3:
            num *=-1
        assert num > 0, 'Error occuring when i = '+str(i)
        
    assert type(num) == int
    return num

raiser(2,8)
```

Note that asserting excessively can slow down code, so dont be overboard with them or comment them out after debugging
