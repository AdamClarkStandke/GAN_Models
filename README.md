# Theoretical underpinnings of DCGAN
- - - - - - - - - - - - - -

As detailed in the book [Advanced Deep Learning with Python: Design and implement advanced next-generation AI solutions using TensorFlow and PyTorch](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X) by Ivan Vasilev. The DCGAN stems from the landmark paper introduced in 2014 titled [Generative Adversarial Nets](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf) The implementation of the paper's algorithm comes from a tensorflow tutorial titled [dcgan](https://www.tensorflow.org/tutorials/generative/dcgan)

### The generator: 
> To learn the generator’s distribution p<sub>g</sub> over data **x**, we define a prior on input noise variable p<sub>z</sub>(**z**), then represent a mapping to data space as G(**z**;θ<sub>g</sub>), where G is a differential function represented by a multilayer perceptron with parameters θ<sub>g</sub> [^1]. 

This is represented by the following code:
```python
# function that builds the generator
def build_generator(latent_input, weight_initialization, channel):
  model = Sequential(name='generator')
  # first fully connected layer to take in 1D latent vector/tensor z
  # and output a 1D tensor of size 12,544
  model.add(keras.layers.Dense(7*7*256, input_shape=(latent_input,)))
  # applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1
  # stabilizes training after the conv layer and before the activation function
  model.add(keras.layers.BatchNormalization())
  # activation function
  model.add(keras.layers.ReLU())
  # reshape previous layer into a 3D tensor 
  model.add(keras.layers.Reshape((7, 7, 256)))
  # first layer of upsampeling(i.e. deconvolution) of the 3D tensor to output a 7x7 feature map as determined by the stride  
  model.add(keras.layers.Conv2DTranspose(filters=128, kernel_size=(5,5), strides=(1,1), padding='same', kernel_initializer=weight_initialization))
  model.add(keras.layers.BatchNormalization()) 
  model.add(keras.layers.ReLU()) 
  # second layer of upsampeling(i.e. deconvolution) in which the volume depth is reduced to 64 
  # and outputs a feature map of size 14x14 as determined by the stride
  model.add(keras.layers.Conv2DTranspose(filters=64, kernel_size=(5,5), strides=(2,2), padding='same', kernel_initializer=weight_initialization)) 
  model.add(keras.layers.BatchNormalization())
  model.add(keras.layers.ReLU()) 
  # third layer upsampeling(i.e. deconvolution) in which the volume depth is reduced to 1 and the image is output as 28x28x1
  model.add(keras.layers.Conv2DTranspose(filters=channel, kernel_size=(5,5), strides=(2,2), padding='same', activation='tanh'))
  return model
```
Which is represented by the below image as found in the 2016 paper titled [UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS](https://arxiv.org/pdf/1511.06434.pdf) 

![alt text](https://github.com/aCStandke/GAN_Models/blob/main/Screenshot%202022-05-21%201.22.59%20AM.png "DCGAN generator")[^2].


### The discriminator:
> We also define a second multilayer perceptron D(**x**;θ<sub>d</sub>) that outputs a single scalar. D(**x**) represents the probability
that **x** came from the data rather than p<sub>g</sub>.[^1].

This is represented by the following code:
```python
def build_discriminator(width, height, depth, alpha=0.2):
  model = Sequential(name='discriminator')
  input_shape = (height, width, depth)
  # first layer of discriminator network that downsamples image to 14x14 as determined by stride and 
  # increases depth by 64
  model.add(keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(2,2), padding='same', input_shape = input_shape))
  model.add(keras.layers.BatchNormalization())
  model.add(keras.layers.LeakyReLU(alpha=alpha))
  # second layer of discriminator network that downsamples image to 7x7 and increaes depth to 128
  model.add(keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(2,2), padding='same'))
  model.add(keras.layers.BatchNormalization())
  model.add(keras.layers.LeakyReLU(alpha=alpha))
  # flatten 3D tensor to 1D tensor of size 7*7*128 = 6727
  model.add(keras.layers.Flatten()) 
  # apply dropout of 30% before feeding it to the dense layer
  model.add(keras.layers.Dropout(0.3))
  model.add(keras.layers.Dense(1, activation='sigmoid')) 
  return model
```
### Value function and Training:
> We train D to maximize the probability of assigning the correct label to both training examples and samples from G. We simultaneously train G to minimize log(1 − D(G(**z**))). In other words, D and G play the following two-player minimax game with
value function V(G, D):

![alt text](https://github.com/aCStandke/GAN_Models/blob/main/Screenshot%202022-05-21%202.20.44%20AM.png "value function")[^1].

However as the authors of the paper note this objective function does not perform in practice, since it may not provide sufficient gradients for the generator to acutally learn, especially during the early stages of learning when the discriminator is very accurate (i.e. outputing 0 rather than 1 so the gradient will be 0 and the weights of the generator will not move). So rather than training the generator to minimize log(1-D(G(**z**))), training is done to maximize log D(G(**z**)).[^1].  

- - - - - - - - - - - - - -
### Deep Convolutional GAN using MINST Fashion Dataset:


Using a deep convolution GAN to create fashion clothes from a Gaussian distribution trained using the MINST fashion data set. **_Click below on the picture of [Daphne](https://www.youtube.com/channel/UCpIqTXVAk0a14YdkWX-hn9Q) to show the video of the transformation from random noise into actual fashion clothes that I think Daphne would include in her wardrobe! 👗 (espically if it is a [White Party](https://en.wikipedia.org/wiki/White_Party)_**)

[![CLICK HERE](https://github.com/aCStandke/GAN_Models/blob/main/mqdefault.jpg)](https://youtu.be/4xKBck4LJjA)

- - - - - - - - - - - - - -
### Deep Convolutional GAN using the Celeb-A Faces Dataset:

Using a Using a deep convolution GAN to create new faces from a Gaussian distribution trained using the [Celeb-A Faces
dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). After just training for five epochs these were fake faces that were generated(note that some of them look realistic, especially the women with the blond hair, lol):

![alt text](https://github.com/aCStandke/GAN_Models/blob/main/epoch1.png "Epoch 1")
![alt text](https://github.com/aCStandke/GAN_Models/blob/main/epoch2.png "Epoch 2")
![alt text](https://github.com/aCStandke/GAN_Models/blob/main/epoch3.png "Epoch 3")
![alt text](https://github.com/aCStandke/GAN_Models/blob/main/epoch4.png "Epoch 4")
![alt text](https://github.com/aCStandke/GAN_Models/blob/main/epoch5.png "Epoch 5")

- - - - - - - - - - - -
# Theoretical underpinnings of Pix2Pix
- - - - - - - - - - - - - - - - - - - -
As detailed in the book [Advanced Deep Learning with Python: Design and implement advanced next-generation AI solutions using TensorFlow and PyTorch](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X) by Ivan Vasilev. The Pix2Pix paper [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf) develops upon ideas from the paper titled [Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf) introduced in 2014. The implementation of the Pix2Pix algorithm comes from a tensorflow tutorial titled [pix2pix: Image-to-image translation with a conditional GAN](https://www.tensorflow.org/tutorials/generative/pix2pix).

### Conditional GAN:
> Generative adversarial nets can be extended to a conditional model if both the generator and discriminator are conditioned on some extra information **y**. **y** could be any kind of auxiliary information, such as class labels or data from other modalities. We can perform the conditioning by feeding **y** into the both the discriminator and generator as additional input layer. In the generator the *prior distribution of input noise* p<sub>z</sub>(**z**), and **y** are combined in *joint hidden representation/distribution*, and the adversarial training framework allows for considerable flexibility in how this *joint hidden representation/distribution* is composed. [^4].

![alt text](https://github.com/aCStandke/GAN_Models/blob/main/Screenshot%202022-05-26%208.54.57%20PM.png "Conditional GAN")[^3]

### Value function:
D and G play the following two-player minimax game with the following value function V(G, D):

![alt text](https://github.com/aCStandke/GAN_Models/blob/main/conditional_value_func.png "conditional value function")[^4].

###

[^1]: [Generative Adversarial Nets](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)
[^2]: [UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS](https://arxiv.org/pdf/1511.06434.pdf)
[^3]: [Advanced Deep Learning with Python: Design and implement advanced next-generation AI solutions using TensorFlow and PyTorch](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X)
[^4]: [Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf)

