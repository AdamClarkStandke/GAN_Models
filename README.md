# Deep Convolutional GAN
- - - - - - - - - - - - 

Using a deep convolution GAN to create fashion clothes from a Gaussian distribution trained using the MINST fashion data set. **_Click below on the picture of [Daphne](https://www.youtube.com/channel/UCpIqTXVAk0a14YdkWX-hn9Q) to show the video of the transformation from random noise into actual fashion clothes that I think Daphne would include in her wardrobe! 👗 (espically if it is a [White Party](https://en.wikipedia.org/wiki/White_Party)_**)

[![CLICK HERE](https://github.com/aCStandke/GAN_Models/blob/main/mqdefault.jpg)](https://youtu.be/4xKBck4LJjA)


---------------------------

# Theoretical underpinnings of DCGAN
- - - - - - - - - - - - - -

As detailed in the book [Advanced Deep Learning with Python: Design and implement advanced next-generation AI solutions using TensorFlow and PyTorch](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X) by Ivan Vasilev. The DCGAN stems from the landmark paper introduced in 2014 titled [Generative Adversarial Nets](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)

## The generator: 
> To learn the generator’s distribution p_g over data **x**, we define a prior on input noise variable p_z(z), then represent a mapping to data space as G(**z**;theta_g), where G is a differential function represented by a multilayer perceptron with parameters theta_g [^1]. 

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

![alt text](https://github.com/aCStandke/GAN_Models/blob/main/Screenshot%202022-05-21%201.22.59%20AM.png "DCGAN generator")






## The discriminator:
> 





[^1]: [Generative Adversarial Nets](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)
