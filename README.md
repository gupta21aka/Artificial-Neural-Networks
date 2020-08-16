# Artificial-Neural-Networks

An artificial neural network, usually simply called neural network, is an interconnected group of nodes, inspired by a simplification of neurons in a brain. Neural networks consist of input and output layers, as well as (in most cases) hidden layer(s) consisting of units that transform the input into something that the output layer can use. 

Although neural networks (also known as "perceptrons") have been around since the 1940s, they have only become a significant part of artificial intelligence in the last few decades. This is due to the arrival of a technique called “backpropagation”, which allows networks to adjust their hidden layers of neurons in situations where the outcome doesn’t match what the creator is hoping for — like a network designed to recognize dogs, which misidentifies a cat, for example.  Another important move forward was the introduction of deep learning neural networks, in which various layers of a multilayer network extract different features before it can identify what it is searching for.

There are multiple types of neural network, each of which come with their own specific use cases and levels of complexity. In this project, we study three networks: Variational Autoencoders, Deep Convolutional Generative Adversarial Networks and Capsule Generative Adversarial Networks.

## Project Paper and Results

https://github.com/gupta21aka/Artificial-Neural-Networks/blob/master/FAI_project_report.pdf

# Comparison of DCGAN, CapsuleGAN and Variational Autoencoder for Image Generation
 
This project compares the results of Variational Autoencoders (VAE), Deep Convolutional Generative Adversarial Networks (DCGAN) and Capsule Generative Adversarial Networks (CapsuleGAN) in generating images of handwritten digits from the MNIST dataset. Their performance is evaluated via loss computation and the generated images are compared after a certain fixed number of epochs.

# Dataset
The MNIST dataset has been used to train all the models and compare results. This dataset contains 60000 black-white, 28x28 pixel, labeled handwritten digit images. All numbers between 0-9 are represented and hence they have 10 distinct classes.

# Variational Autoencoders (VAE)
VAEs consist of two distinct networks, an encoder and a decoder. In the training phase, both networks are trained. The encoder is tasked with forming a distribution given the image input and the decoder is tasked with reconstructing an image after an input for it has been sampled from the previous distribution. Once trained, only the discriminator is required for image generation. Since the distribution that the encoder creates is a simple gaussian normal distribution with a mean and standard deviation, any inputs sampled from this will yield a valid image from the decoder.
	Since the encoder converts the image to a latent distribution, the input layer has a size of 784. Each part of the input represents a pixel of an image in the MNIST dataset. The encoder then passes the information through its network. The hidden layers are constructed to accomplish this. The output of the encoder consists of two layers, both the same size. These layers represent the mean and standard deviation of latent distribution.
	The decoder must take a value from this latent distribution. So, its input layer’s size is the same as the latent distribution layer’s size. The hidden layers of the decoder then build up such that the final output layer represents the reconstructed image, so it has a size of 784.
	The VAE is formed by connecting the output of the encoder to the input of the decoder through a sampling layer. The distribution that is formed represents an expectation of z given an image x. So, the decoder requires a sample from this z in order to reconstruct an image x’. Instead of directly sampling from the distribution, which is not possible, the reparameterization trick is used. Now instead of sampling from z directly, samples are taken from a unit normal distribution called epsilon(ϵ). The sample to the decoder is then computed as

	z = µ + σ * ϵ

µ is the mean of the distribution
σ is the standard deviation
ϵ is the sample taken from the unit normal distribution

In order to train this VAE, the loss must be calculated. The loss function can be divided into 2 parts. The reconstruction loss, which is how close the generated reconstructed image is to the original image and the KL divergence, which is how close the learnt distribution is to a unit normal distribution. This ensures that the encoder is optimized to create a distribution that is close to a unit normal one and the decoder is trained to create images that are close to the original training images.
	Reconstruction loss is the binary cross entropy loss and the KL divergence is 

	KL Loss = -0.5 * ∑ (1 + σ - µ2 - e σ)

The total loss is the sum of reconstruction loss and KL divergence.

# Deep Convolutional Generative Adversarial Networks (DCGAN)
DCGAN comprises of two networks, generator and discriminator. Generator has been trained on a 100-dimensional normal distribution with one dense layer of 12544 units and three transposed convolutional layers of 128, 64 and 1 units respectively. Striding has been used in each convolutional layer and additionally, padding has also been used, in order to keep the dimensions of each convolutional layer’s output image same as the dimensions of the layer’s input image. Each layer uses batch normalization and leaky rectified linear activation (LeakyRelu), except for the last layer, which uses hyperbolic tangent activation and no normalization.
The discriminator has been trained simultaneously using the real images from the dataset and the fake images generated by the generator. The discriminator network has a similar structure as the generator but reversed. It replaces the number of units in the dense layer with a single unit. Each layer in the discriminator uses leaky rectified linear activation and dropout. The architecture is modeled in Fig 3.
	The model was trained by sampling one minibatch of real images through the discriminator and generator and then computing binary cross entropy losses for both networks for each sample and then backpropagating. Given datapoints x from the target distribution generated by the generator G, and z from the noise distribution, the loss function for the discriminator network D, is given by,

	LD(x, z) = log(D(x))+log(1−D(G(z)))

and the loss function for generator is given by,

	LG (z) = log(1−D(G(z)))

In each training step, noise is fed as input to the generator which outputs an image. Discriminator takes the generated image and real image dataset to emit a probability how real the generated image is. Losses are calculated thereafter for both the models which are used to update gradients for generator as well as discriminator. The model learns with each training step and eventually begins to converge.

# CapsuleGAN
Capsule Networks
Originally introduced in a paper by Hinton et al., capsules are group of neurons that capture the presence of a feature entity and also represents the properties of those feature entities in vector outputs. The length of the vector shows the probability of feature detection and, its spatial representation is denoted as the direction the vector is pointed to. Therefore, a nonlinear squashing function is used for discriminator learning. This function ensures that shorter vectors get reduced to almost zero length and longer ones to just below one.
	When a feature is detected and its position with respect to an image changes, the length of the vector would remain the same and the direction would only change. Hence, as mentioned before, the equivariance of features makes it possible to transform the detected objects to one another easily. Each capsule would be trained to identify certain objects in the image. Hence while building the layers for neural networks, each layer can comprise of multiple capsules forming a capsule-layer and giving a vector representation for the output.
The higher layers receive inputs from the lower capsule layers. If ui is the vector received from the lower levels (previous capsule) and Wij (transformation matrix) is the corresponding weight matrices, then multiplying them would result in encoding important spatial and other relationships between lower and higher-level features.
 
## Incorporating Capsule Networks in GANs for Discriminator Structure
A basic 28x28 pixel image is given as input to the model. First step is a simple convolutional layer with 256 filters, 9-pixel kernel size and a stride value of 1. This layer is followed by LeakyRelU activation and Batch Normalization to ensure better performance. The non-linearity of the activation is added is here in order to ensure that there is no vanishing gradient. This layer, before the CapNet architecture begins, converts the pixel data, captures all the local features and then passes it on to the PrimaryCaps (the first layer in the capsule network). This part is a combination of multiple layers of Convolution, Reshape and Squashing functions. At this point the capsule architecture shows that the convolution is an 8D image vector of 32 feature maps, representing all the 256 filter of the previous layer and reshape would split all the 32 neurons into 8D vectors.
	The second part of the Capsule network comprises of DigitCaps Layers which incorporates the Routing-by-agreement algorithm. This layer receives its input first from the flattened output of the PrimaryCaps. The first step is that this output is passed through a Keras Dense layer which acts as the prediction vector, u ̂. The output vector from the Dense layer outputs 160 neurons and is passed through 3 routing iterations. This implies that the information is passed between capsules of successive layers. Hence, for the information in capsule at layer l, 〖h_i〗^l and each capsule in the layer above, 〖h_j〗^(l+1), a coupling coefficient c_ij, which is a softmax over the bias weights of the previous layer and u ̂, is computed iteratively. Each iteration is followed by a LeakyRelU activation. The final layer is a Keras Dense layer with a sigmoid activation and a single neuron. This will output a score that determines how close the generated output is to the actual image.
 
## Generator Structure
In this implementation of Generator, multiple deconvolutional layers have been used similar to DCGAN architecture. We take in a random noise to build the first layer which would be a fully connected layer. The second layer after reshaping sits on tops of this input noise and this is fed as input to a series of batch normalizations, UpSampling2D, which simply doubles the dimensions of the input, convolutions and activation functions. The main aim then is to resize and generate a 28x28 size image that can be then used to compare.

# Experiments
## Variational Autoencoder
To create the VAE, the encoder and decoder networks need to be configured according to the specifications mentioned previously. The images below have been obtained with the following training parameters. The encoder and decoder have a single hidden layer each and both are of size 512 and using Relu activation function. The latent distribution has a size of 2. The network is trained with a batch size of 64 and it uses Adam optimizer. The training was done for 5000 epochs.
	The above images clearly show that the network converges quickly in a few numbers of epochs. The images that are generated are distinguishable as handwritten digits. The quality of the images that are generated however is not good, they are blurry. A similar result is obtained when modifying the parameters of the network.	
	Tweaking various parts of the network can help improve the performance of the VAE. For example, modifying the number of latent variables in the network can help the generator converge faster. The code has also been modified to include additional hidden layers in the encoder and decoder.

## DCGAN
The model was trained on the MNIST dataset images for 150 epochs using the Adam optimizer with a learning rate of 0.0002 and a batch size of 128. All the minibatches were trained at each epoch sequentially. Fig 10,11,12 show the images generated after 1, 60 and 150 epochs. The images show a significant improvement from the first to 60th epoch. However, the difference between the images generated after 60 and 150 epochs is not as noteworthy. This is because the digits have started to show up in a good shape after 60 epochs and since the learning rate (0.0002) is constant, the image refines at a slower pace. Similar observation can be made by looking at the loss vs epoch graphs in Fig 13 and 14. The generator and discriminator losses change rapidly in the first few epochs but the change in loss subsides gradually thereafter.
 
## CapsuleGAN
The training of the generator was done using a combined model by stacking generator and discriminator. Initially a noise was fed to the generator to generate images. The discriminator would give the valid outputs for those generated images. A model was then constructed by stacking the generator and discriminator that takes noise as input, generates images and determines validity. This model was trained by taking the batch size as 32 by using the Keras training method- model.train_on_batch(). For training the discriminator, a random half of the batch was selected to create a noise. Keeping the generator static, the noise was used to generate fake images. Then the discriminator was trained on these fake images and real ones by feeding them into the training function one by one. The dataset was trained using GPU on Google Colab and training was done for 40000 epochs. The training loss is given by the abovementioned function- model.train_on_batch(). It runs a single gradient update on a single batch of data. Binary cross entropy loss was calculated, and Adam Optimizer has been used to train the data.
	At each epoch, the progress is shown in the terms of the image generated, Discriminator Loss & Accuracy and the Generator Loss. It can be observed that the generator and discriminator loss fluctuate a lot at lower epochs although it decreases over time and becomes constant over larger number of epochs and flatten at about 2.5. This looks like the model has found an optimum performance limit and cannot improve more. The discriminator began with a higher loss and then decreased over time, flattening at 0.85. The accuracy was observed to be around 40-60% throughout the training. The MNIST dataset showed good and recognizable images from the 5000th epoch itself.

# Conclusion
This paper compares 3 different image generation models namely VAEs, DCGANS and CapsuleGANs. After performing the various experiments listed above and comparing the results of the different models, we can conclude that VAEs tend to get trained faster than the other 2 models and are able to generate images quicker. However, the images that are generated are blurry and not sharp. GANs on the other hand can generate much sharper images but the time taken to train these networks is much higher. We also conclude that the images generated with Capsule GANs are sharper and converge faster as compared to DCGANs.

# References
Ayush Jaiswal, Wael AbdAlmageed, Yue Wu, Premkumar Natarajan. CapsuleGAN: Generative Adversarial Capsule Network. USC Information Sciences Institute. (2018)

Bang, D., & Shim, H. (2018). Improved training of generative adversarial networks using representative features. arXiv preprint arXiv:1801.09195.

Denton, E. L., Chintala, S., & Fergus, R. (2015). Deep generative image models using a laplacian pyramid of adversarial networks. In Advances in neural information processing systems (pp. 1486-1494).

Doersch, C. (2016). Tutorial on variational autoencoders. arXiv preprint arXiv:1606.05908.

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative Adversarial Nets. In: Advances in neural information processing systems. pp. 2672–2680 (2014)

Hernandez, E., Liang, D., Pandori, S., Periyasamy, V., & Singhal, S. Generative Models for Handwritten Digits

Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.

Kingma, D. P., & Welling, M. (2019). An introduction to variational autoencoders. Foundations and Trends in Machine Learning, 12(4), 307-392.

Marusaki, K., & Watanabe, H. (2020). Capsule GAN Using Capsule Network for Generator Architecture. arXiv preprint arXiv:2003.08047.

Radford, A., Metz, L., Chintala, S.: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. In: International Conference on Learning Representations (2016)

Sabour, S., Frosst, N., Hinton, G.E.: Dynamic routing between capsules. In: Advances in Neural Information Processing Systems. pp. 3859–3869 (2017)
