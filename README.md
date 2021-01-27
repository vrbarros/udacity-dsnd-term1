# Intro to Machine Learning with Pytorch Nanodegree (nd229)

This is a repository with 3 projects related to Intro to Machine Learning with Pytorch Nanodegree from Udacity.

## Getting Started

1. Install [Anaconda](https://www.anaconda.com/products/individual#Downloads) and dependecies
2. Clone this repository to your machine
3. Create a [virtual environment](https://docs.python.org/3/library/venv.html) to install dependencies
4. Activate the new virtual environment
5. Install packages dependencies `pip install -r requirements.txt`
6. Start jupyter-lab using the command `jupyter notebook`

## Project 1: Finding Donors for CharityML

![Project1](https://miro.medium.com/max/810/0*181Nbv4t1bPcXXXe.jpg)

CharityML is a fictitious charity organization located in the heart of Silicon Valley that was established to provide financial support for people eager to learn machine learning. After nearly 32,000 letters were sent to people in the community, CharityML determined that every donation they received came from someone that was making more than $50,000 annually. To expand their potential donor base, CharityML has decided to send letters to residents of California, but to only those most likely to donate to the charity. With nearly 15 million working Californians, CharityML has brought you on board to help build an algorithm to best identify potential donors and reduce overhead cost of sending mail. Your goal will be evaluate and optimize several different supervised learners to determine which algorithm will provide the highest donation yield while also reducing the total number of letters being sent.


## Project 2: Create Your Own Image Classifier

![Project2](https://cdn-media-1.freecodecamp.org/images/1*NRIFYyKmm8XJvQSZNZbaxQ.png)

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application.

Important topics:
* Pretrained Network
* Feedforward Classifier
* Training the network
* Validation Loss and Accuracy
* Testing Accuracy
* Saving the model
* Loading checkpoints
* Sanity Checking
* Model architecture
* Model hyperparameters
* Training with GPU
* Top K classes

## Project 3: Creating Customer Segments with Arvato

![Project3](https://cdn-images-1.medium.com/max/1024/1*YRcl1W-20x_p0usdYbR87A.jpeg)

In this project, our Bertelsmann partners AZ Direct and Arvato Financial Solutions have provided two datasets one with demographic information about the people of Germany, and one with that same information for customers of a mail-order sales company. You’ll look at relationships between demographics features, organize the population into clusters, and see how prevalent customers are in each of the segments obtained.

In this project, you will work with real-life data provided to us by our Bertelsmann partners AZ Direct and Arvato Finance Solution. The data here concerns a company that performs mail-order sales in Germany. Their main question of interest is to identify facets of the population that are most likely to be purchasers of their products for a mailout campaign. Your job as a data scientist will be to use unsupervised learning techniques to organize the general population into clusters, then use those clusters to see which of them comprise the main user base for the company. Prior to applying the machine learning methods, you will also need to assess and clean the data in order to convert the data into a usable form.

This projects uses:
* NumPy
* pandas
* Sklearn / scikit-learn
* Matplotlib (for data visualization)
* Seaborn (for data visualization)

**Important:**  The original dataset used for this project is not available in this repository, once that has private intelectual property.

## References

- [deep-learning-v2-pytorch](https://github.com/udacity/deep-learning-v2-pytorch) from Udacity
- [AIPND](https://github.com/udacity/AIPND) from Udacity

### Author
Victor Barros - [@vrbarros](https://github.com/vrbarros)