# Neural Networks with MNIST 

Modified National Institute of Standards and Technology (MNIST) is a dataset of grayscale handwritten digits that have been normalized and centered on a 28x28 pixel field. The gray levels are a result of the anti-aliasing technique used by the normalization algorithm. The dataset contains a labelled training set of 60000 examples, and a labelled test set of 10000 examples. 

## Autoencoders

An autoencoder is an unsupervised learning algorithm that aims to create an output that is equivalent to the input. An autoencoder consists of two parts: an encoder and a decoder. The encoder maps the input to a hidden layer of neurons and the decoder reconstructs the output using the information from that hidden layer. This basic neural network structure is equivalent to a multilayer perceptron. The simplest autoencoder is a feedforward neural network with the same number of input and output neurons. The weights are initialized randomly and are trained through backpropagation with respect to some cost function. In this case, we want the output to match the input. Therefore, a good cost function will minimize the difference between the input and the output. After training, the autoencoder should have learned to map the input to the output. Although it seems trivial to learn this identity mapping, placing constraints on the network can reveal the underlying structure of the unlabelled training dataset. 


<p align="center">
  <img width="550" height="415" src="/images/ae.png">
  <br />
  <em>Figure 1 - Simple feedforward neural network as an autoencoder with <br> a single hidden layer. </em>
  <br />
</p>

### Dimensionality Reduction

An obvious constraint is to limit the number of hidden neurons. As long as the number of hidden neurons is less than the number of input and output neurons, the encoder will learn to compress the input and the decoder will learn to reconstruct the output from a compressed representation. The autoencoder will be able to capture the underlying structure in the data through the latent representation given by the neurons in the hidden layer. These low-dimensional representation are similar to results from PCA.

<p align="center">
  <br />
  <img width="360" height="193" src="/images/svd.png">
  <br />
  <em>Figure 2 - Singular values for MNIST training set in decreasing <br> order. Certain values of varying energy have been chosen to <br> study the effect of different number of hidden neurons. Energy <br> is defined here as the sum of the square of all singular values.</em>
  <br />
  <br />
  <br />
  <img width="600" height="211" src="/images/svd_study.png">
  <br />
  <br />
  <em>Figure 3 - (Right) Ground truths from dataset. (Left) Autoencoder outputs with <br> varying number of hidden neurons. Columns show examples after 1, 10, 20, <br> and 100 epochs of training with a minibatch size of 200. Rows show examples <br> with 9, 46, 85, and 224 hidden neurons respectively. </em>
  <br />
  <br />
</p>

An SVD is performed on the MNIST training set, and the singular values are shown in **Figure 2**. Note that this is shown on a log scale because the singular values become increasingly small. A useful rule of thumb is to retain enough singular values for a percentage of the total energy in Σ, defined as the sum of the squared singular values. Different autoencoders are constructed with a varying number of neurons in the hidden layer and some visual results are shown in **Figure 3**. Increasing the number of neurons in the hidden layer causes the network to learn more features, allowing better reconstructions. However, if the number of neurons is greater than or equal to the number of input and output neurons, the network can learn to simply copy the input and pass the information through. Regularization can be employed to combat this problem and prevent overfitting to the training dataset.

### Sparse Learning (Sparse Autoencoder)

An alternative method to prevent the network from learning to copy with input without constraining the number of neurons is to reularize the activations of the hidden neurons. This encourages the network to learn encodings and decodings that rely only small activation from the hidden neurons. When the number of hidden neurons is large, the network can learn to selectively activate different regions depending on the input. This forces the network to relate a small number of neurons to underlying structures within the data. This constraint can be enforced by adding an activation penalty to the cost function. 

<p align="center">
  <br />
  <img width="350" height="197" src="/images/sae_study.png">
  <br />
  <br />
  <em>Figure 4 - (Row 1) Regular autoencoder weights without sparsity constraint. <br> (Row 2) Sparse autoencoder with L1 constraint. (Row 3) Sparse autoencoder <br> with KL Divergence constraint. </em>
  <br />
</p>

Sparsity can be enforced by constraining the average activations of the hidden neurons over the training set. Specifically, the activations can be averaged during training and added to the cost function in the form of a norm or KL divergence. **Figure 4** visually compares the weights of an autoencoder with and without sparsity constraints. By learning a sparse representation, hidden neuron weights show underlying structures in the data that are obscured in a normal autoencoder. Adding a sparsity constraint will regularize the cost function and prevent overfitting.

### Denoising (Denoising Autoencoder)

As opposed to changing the cost function, the training set can be augmented to increase the robustness of the network. By providing the network with noisy data and teaching the network to reconstruct the original data without noise, the network is encouraged to learn the underlying structures within the data. As expected, the underlying features learned in denoising are similar to the features learned by the sparse autoencoder shown in **Figure 5**. Because the denoising and sparse training result in similar learned weights, the sparse autoencoder also provides robustness against noise as seen in **Figure 6**.

<p align="center">
  <img width="350" height="93" src="/images/dae_study.png">
  <br />
  <em>Figure 5 - Denoising autoencoder weights </em>
  <br />
  <br />
  <br />
  <img width="350" height="270" src="/images/dae_study2.png">
  <br />
  <em>Figure 6 - (Row 1) Augmented images with noise. (Row 2) Sparse <br> autoencoder reconstruction. (Row 3) Denoising autoencoder reconstruction. <br> (Row 4) Original ground truth images without noise.</em>
  <br />
  <br />
</p>

## Classifiers

Classification is a supervised learning problem that aims to identify the correct class a given observation belongs to. To accomplish this goal ground truth labels must be provided for the training dataset. During training, the network learns to associate certain inputs to specific classes. A popular convention is to have one output node corresponding to each class. These output nodes are trained to output the probability that an input belongs to a given class. In order to ensure the all the outputs form a probability distribution, the final layer is passed through a softmax function. The ground truth labels are usually one-hot-encoded to be the same shape as the output layer of the network. Labels provided with the MNIST dataset allow supervised training for classification problems.

### Convolution Layers




#### References
[Autoencoder Inspiration (Dana Ballard)](/papers/Ballard_1978.pdf)
<br>
[Sparse Coding (Bruno Olshausen)](/papers/Olshausen_1996.pdf)
<br>
[Sparse Autoencoder (Andrew Ng)](/papers/Ng_2011.pdf)
<br>
[Denoising Autoencoder (Pascal Vincent)](/papers/Vincent_2008.pdf)

