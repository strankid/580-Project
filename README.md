#Adaptive Learning with Dynamic Batch Creation

Using Near-Neighbors
Deep neural networks (DNNs) are the most
effective method for many classification tasks.
The standard way to train DNNs is through
backpropagation using stochastic gradient
descent (SGD) based on mini-batches of fixed
size. This training step is the computational
bottleneck of DNNs, and it can take several days
to train a large dataset on high-performance
GPUs. In order to speed up the training process,
we propose an adaptive batch selection method
that adaptively chooses which datapoints to
include in a mini-batch. In SGD, it is common
to fix the batch size and iteratively apply the
algorithm to all datapoints. Instead, our approach
involves using near-neighbors of datapoints that
led to the most learning (largest gradients). We
sample datapoints for the next batch from the
non-uniform probability distribution that is based
on the loss/gradients previously computed for
each datapoint. This allows SGD to focus on the
most relevant training datapoints and progress
faster.
