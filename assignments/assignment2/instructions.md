
# Deep Learning for Visual Computing - Assignment 2

The second assignment covers iterative optimization and parametric (deep) models for image classification.

## Part 1

This part is about experimenting with different flavors of gradient descent and different optimizers.

Download the data from [here](https://github.com/theitzin/dlvc2020/tree/master/assignments/assignment2). Your task is to implement `optimizer_2d.py`. See the code comments for instructions. The `fn/` folder contains sampled 2D functions for use with that script. You can add more functions if you want, [here](https://www.sfu.ca/~ssurjano/optimization.html) is a list of interesting candidates.

The goal of this part is for you to understand the optimizers provided by PyTorch better by playing around with them. Try different types (SGD, AdamW etc.), parameters, starting points, and functions. How long do different optimizers take to terminate. Is the global minimum reached? This nicely highlights the function and limitations of gradient descent, which we've already covered in the lecture.
