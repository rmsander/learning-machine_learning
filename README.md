# Learning Machine Learning

![Example machine learning task: GPR](readme_images/gpr_ts_sine.png)

Welcome!  This repository is built for anyone interested in learning more about 
different facets of **machine learning**, a subfield of artificial intelligence.  
Whether you want to learn more about native Python, C++, or the advanced tutorials 
or algorithms driving state-of-the-art in machine learning and artificial intelligence today, 
my goal is for this repository to be a resource for your learning in these fields.

I'm also working on an open-source guide for machine learning and AI.  You can 
find it [here](https://docs.google.com/document/d/1XS8DIW_nVHKe8Gfp6hK24GMVcaM4TKbZCMEVxJfdG7Y/edit?usp=sharing) as a set of google docs, as well as [here](https://rmsander.github.io/documentation/index.html) as part of my personal website.  Both of 
these are continuously evolving and will have more content added to them in time.

## Installation
For full use of all Jupyter notebooks and libraries, please install requirements 
using either `conda` or `pip`:

- `conda`: `conda env create -f requirements.txt -n learning_ml python=3.7`
- `pip`: `pip install -r requirements.txt`

## Concepts
For learning about concepts in machine learning, computer vision, operating systems, and programming, please visit the `concepts/` directory to view the different pdf files for these concepts.  Concepts currently covered include:

1. Computer Vision and Deep Learning
2. Introductory Slides on Python

## Python Programming
Python is a powerful programming language for machine learning, probability, 
statistics, and scientific computing.  Many state-of-the-art frameworks for 
machine learning, such as **TensorFlow**, **PyTorch**, **NumPy**, **Scikit-Learn**, and **Keras** all have open-source implementations in Python.  You can find exercises for Python under the `python` directory.  See contents below:

* Under `python/intro_to_python`, you can find Python files covering fundamentals in Python, such as:
1. Python data structures (lists, tuples, dictionaries, strings, ints/floats)
2. Loops and iteration (for and while)
3. Conditional logic
4. Functions
5. I/O (input/output)
6. A brief intro to `numpy`

You can also find some introductory concepts in a more "notebook"-like fashion under `python/introductory_notebooks`.

* Under `python/machine_learning_in_python`, I have illustrated some of the applications of Python both inside and outside of AI, such as fingerprint detection, stock price prediction, and Principal Component Analysis.

* Under `python/machine_learning_fundamentals`, I have included some files for some fundamental concepts in machine learning, such as regression and K-Means Clustering.

* Under `python/python_package_tutorials`, I have included `jupyter` notebooks (with extension `.ipynb`).  Note that all of these tutorials are implemented with `.ipynb` files and can be run with [Jupyter Notebook](https://jupyter.org/) or [Google Colab](https://colab.research.google.com/).  Tutorials include:

1. `NumPy`
2. `OpenCV`
3. `TensorFlow`
4. `Keras`
5. `PyTorch`
6. `pandas` 
7. `gpytorch`

* Under `python/machine_learning_in_python/reinforcement_learning`, you can find some code for deep reinforcement learning from pixels.

* Under `python/machine_learning_in_python/adversarial_search`, you can find an application of Python to solving the game of 2048 using adversarial search techniques.

## C++ Programming
Exercises and examples for programming in C++ can be found in `./c++/`.  These examples are mainly derived from exercises with programming drones for autonomous racing in MIT's 16.485: Visual Navigation for Autonomous Vehicles course.  Some examples of code include:

1. Creating a `RandomVector` class that samples a random vector of arbitrary dimension from a uniform distribution.
2. `roscpp` exercises.

## Stata Programming
Stata, a statistical programming language, is also useful for data science, machine learning, and econometrics.  You can find content and examples for stata in   `./stata/`.

## Missing Something?  More Concepts You Would Like to See?
If you have questions or would like content to be added for a specific topic, 
please feel free to let me know via [this google form](https://forms.gle/yH4NxYYQsqjNexuQ8).  I'm always looking for ways to improve this website and my documentation.
If you're interested in learning more about machine learning through online articles, 
please check out my Medium blog [here](https://rmsander.medium.com).
