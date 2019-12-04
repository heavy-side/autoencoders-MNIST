# Neural Networks with MNIST 

Modified National Institute of Standards and Technology (MNIST) is a great dataset for machine learning. It is a dataset of grayscale handwritten digits that have been normalized and centered on a 28x28 pixel field. The gray levels are a result of the anti-aliasing technique used by the normalization algorithm. The dataset contains a labelled training set of 60,000 examples, and a labelled test set of 10,000 examples. 

<br />

## Autoencoders

An autoencoder is an unsupervised learning algorithm that aims to create an output that is equal to the input using backpropagation. An autoencoder consists of two parts: an encoder and a decoder. The encoder maps the input to a hidden layer of neurons and the decoder reconstructs the output using the information from that hidden layer. This basic neural network structure is equivalent to a multilayer perceptron.

The simplest autoencoder is a feedforward neural network with the same number of input and output neurons. The weights are initialized randomly and are trained through backpropagation with respect to some cost function. In this case, we want the output to match the input. Therefore, a good cost function will minimize the difference between the input and the output. 

<br />

<p align="center">
  <img width="550" height="295" src="/images/ae.png">
  <br />
  <em>Figure 1 - Autoencoder (Reyhane Askari)</em>
</p>

<br />
<br />

After training, the autoencoder should have learned to map the input to the output. Although it seems trivial to learn this identity mapping, placing constraints on the network can reveal the underlying structure of the unlabelled training dataset. 

<br />

### Dimensionality Reduction

An obvious constraint is to limit the number of hidden neurons. As long as the number of hidden neurons is less than the number of input and output neurons, the encoder will learn to compress the input and the decoder will learn to reconstruct the output from a compressed representation. The autoencoder will be able to capture the underlying structure in the data through the latent representation given by the neurons in the hidden layer. These low-dimensional representation are similar to results from PCA.

A SVD is performed on the MNIST training set, and the singular values are shown in Figure 2. Note that this is show on a log scale because the singular values become increasingly smaller. A useful rule of thumb is to retain enough singular values for a percentage of the total energy in Σ, defined as the sum of the squared singular values. Different autoencoders are constructed with a varying number of neurons in the hidden layer. 

<p align="center">
  <img width="550" height="295" src="/images/svd.png">
  <br />
  <em>Figure 2 - Singular values for MNIST training set in decreasing order</em>
</p>

<br />
<br />

### Sparse Representations (SAE)

### Denoising (DAE)

### Variational Autoencoders (VAE)


<br />
<br />
<br />
<br />


#### References
 - [Multilayer Perceptrons](https://pdfs.semanticscholar.org/7b79/cccc8de41d76a2ca20eacc3d39f7b45bff5f.pdf)
 - [Andrew Ng's Notes on Autoencoders](https://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf)
 - [Reyhane Askari's Notes on Autoencoders](https://reyhaneaskari.github.io/AE.htm)
