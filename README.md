# Theoretical underpinnings of GAN and DCGAN
As detailed in the book [Advanced Deep Learning with Python: Design and implement advanced next-generation AI solutions using TensorFlow and PyTorch](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X) by Ivan Vasilev. The DCGAN stems from the landmark paper introduced in 2014 titled [Generative Adversarial Nets](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf) The implementation of the paper's algorithm comes from a tensorflow tutorial titled [dcgan](https://www.tensorflow.org/tutorials/generative/dcgan)
## The generator: 
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
## The discriminator:
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
## GAN Loss function:
> We train D to maximize the probability of assigning the correct label to both training examples and samples from G. We simultaneously train G to minimize log(1 − D(G(**z**))). In other words, D and G play the following two-player minimax game with
value function V(G, D):
![alt text](https://github.com/aCStandke/GAN_Models/blob/main/Screenshot%202022-05-21%202.20.44%20AM.png "value function")[^1].
However as the authors of the paper note this objective function does not perform in practice, since it may not provide sufficient gradients for the generator to acutally learn, especially during the early stages of learning when the discriminator is very accurate (i.e. outputing 0 rather than 1 so the gradient will be 0 and the weights of the generator will not move). So rather than training the generator to minimize log(1-D(G(**z**))), training is done to maximize log D(G(**z**)).[^1].  
### Example 1: Deep Convolutional GAN using MINST Fashion Dataset
Using a deep convolution GAN to create fashion clothes from a Gaussian distribution trained using the MINST fashion data set. **_Click below on the picture of [Daphne](https://www.youtube.com/channel/UCpIqTXVAk0a14YdkWX-hn9Q) to show the video of the transformation from random noise into actual fashion clothes that I think Daphne would include in her wardrobe! 👗 (espically if it is a [White Party](https://en.wikipedia.org/wiki/White_Party)_**)

[![CLICK HERE](https://github.com/aCStandke/GAN_Models/blob/main/mqdefault.jpg)](https://youtu.be/4xKBck4LJjA)

The sorce code for this example can be found here: [DCGAN-MINST Fashion](https://github.com/aCStandke/GAN_Models/blob/main/DCGAN_Fashion.ipynb)

### Example 2: Deep Convolutional GAN using the Celeb-A Faces Dataset:

Using a Using a deep convolution GAN to create new faces from a Gaussian distribution trained using the [Celeb-A Faces
dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). After just training for five epochs these were fake faces that were generated(note that some of them look realistic, especially the women with the blond hair, lol):

![alt text](https://github.com/aCStandke/GAN_Models/blob/main/epoch1.png "Epoch 1")
![alt text](https://github.com/aCStandke/GAN_Models/blob/main/epoch2.png "Epoch 2")
![alt text](https://github.com/aCStandke/GAN_Models/blob/main/epoch3.png "Epoch 3")
![alt text](https://github.com/aCStandke/GAN_Models/blob/main/epoch4.png "Epoch 4")
![alt text](https://github.com/aCStandke/GAN_Models/blob/main/epoch5.png "Epoch 5")

The sorce code for this example can be found here: [DCGAN-Celeb-A Faces](https://github.com/aCStandke/GAN_Models/blob/main/DCGAN_Faces.ipynb)

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Theoretical underpinnings of Conditional GAN and Supervised Pix2Pix 
As detailed in the book [Advanced Deep Learning with Python: Design and implement advanced next-generation AI solutions using TensorFlow and PyTorch](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X) by Ivan Vasilev. The Pix2Pix paper [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf) develops upon ideas from the paper titled [Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf) introduced in 2014. The implementation of the Pix2Pix algorithm comes from a tensorflow tutorial titled [pix2pix: Image-to-image translation with a conditional GAN](https://www.tensorflow.org/tutorials/generative/pix2pix).

## Conditional GAN:
> Generative adversarial nets can be extended to a conditional model if both the generator and discriminator are conditioned on some extra information **y**. **y** could be any kind of auxiliary information, such as class labels or data from other modalities. We can perform the conditioning by feeding **y** into the both the discriminator and generator as additional input layer. In the generator the *prior distribution of input noise* p<sub>z</sub>(**z**), and **y** are combined in *joint hidden representation/distribution*, and the adversarial training framework allows for considerable flexibility in how this *joint hidden representation/distribution* is composed. [^4].
> 
> ![alt text](https://github.com/aCStandke/GAN_Models/blob/main/Screenshot%202022-05-26%208.54.57%20PM.png "Conditional GAN")
> [^3].

## Conditional GAN Loss function:
D and G play the following two-player minimax game with the following value function V(G, D):

![alt text](https://github.com/aCStandke/GAN_Models/blob/main/conditional_value_func.png "conditional value function")[^4].

## Paired Pix2Pix:
Paired Pix2Pix is a conditional GAN with an additional loss constraining the generator, which the paper outlines in section 3.1 is a L1 loss rather than the traidtional L2 loss. This helps with blurring.[^5].

## Paired Pix2Pix Loss function:

![alt text](https://github.com/aCStandke/GAN_Models/blob/main/Screenshot%202022-05-26%2011.50.28%20PM.png "generator l1 loss")[^5].

![alt text](https://github.com/aCStandke/GAN_Models/blob/main/Screenshot%202022-05-26%2011.50.51%20PM.png "total loss")[^5].

The architecture for the generator is described by the paper as the following:
> To give the generator a means to circumvent the bottleneck for information like this, we add skip connections, following the general shape of a “U-Net”... Specifically, we add **skip connections between each layer i and layer n − i, where n is the total number of layers. Each skip connection simply concatenates all channels at layer i with those at layer n − i**.[^5].
The architecture for the discriminator is described by the paper as the following:
> [To motivate the GAN discriminator to only model high-frequency structures in image that are generated] it is sufficient to restrict our attention to
the structure in local image patches. Therefore, we design
a discriminator architecture – which we term a **PatchGAN
– that only penalizes structure at the scale of patches. This
discriminator tries to classify if each N ×N patch in an image as real or fake**. We run this discriminator convolutionally across the image, averaging all responses to provide the ultimate output of D.[^5].

### Example 3: Paired Pix2Pix for Image Segmentation on the Cityscapes Dataset
The objective of this task is to transform a set of real-world images from the Cityscapes dataset[^7] into semantic segmentations. The dataset contains 5,000 finely annotated images split into training, and validation sets (i.e. 2975/500 split). The dense annotation contains 30 common classes of road, person, car, etc. as detailed by the following figure [^8]: 

![alt text](https://github.com/aCStandke/GAN_Models/blob/main/labels%20of%20colors.png "classes") 

**Default Paired Pix2Pix Model**:
After training the semantic segementation generator for 40,000 steps it was tested on the test set of the [Cityscape Dataset](https://www.cityscapes-dataset.com/) The following five test results were outputted to compare the actual semantic segmentation i.e. ground truth to the semantic segmentaion generator i.e. predicted image:

![alt text](https://github.com/aCStandke/GAN_Models/blob/main/download.png)
![alt text](https://github.com/aCStandke/GAN_Models/blob/main/test_four.png)
![alt text](https://github.com/aCStandke/GAN_Models/blob/main/test_one.png)
![alt text](https://github.com/aCStandke/GAN_Models/blob/main/test_three.png)
![alt text](https://github.com/aCStandke/GAN_Models/blob/main/test_two.png)

The default semantic segmentation generator model weights can be found here: [Default Semantic Segmentation Generator Model](https://github.com/aCStandke/GAN_Models/blob/main/saved_model.pb)

**Custom Paired Pix2Pix Model**:
This time the Pix2Pix generator was trained for 25,000 steps and used a lambada value of 1000 for the l1 loss function. Since the L1 loss  regularizes the generator model to output predicted images that are plausible translations of the source image, I decided to weight it 1 order of magnitude higher than [^5] especially when it came to segmenting riders(seemed to help). The following five test results were outputted detailing the some of the preditions of the semantic segmentaion generator i.e. predicted image [^9].

![alt text](https://github.com/aCStandke/GAN_Models/blob/main/pix2pixGen1000_l1loss_25k_1.png "pix2pix segmentation")  
![alt text](https://github.com/aCStandke/GAN_Models/blob/main/pix2pixGen1000_l1loss_25k_2.png "pix2pix segmentation")       
![alt text](https://github.com/aCStandke/GAN_Models/blob/main/pix2pixGen1000_l1loss_25k_3.png "pix2pix segmentation")
![alt text](https://github.com/aCStandke/GAN_Models/blob/main/pix2pixGen1000_l1loss_25k_4.png "pix2pix segmentation")

The custom semantic segmentation generator model weights can be found here: [Custom Semantic Segmentation Generator Model](https://github.com/aCStandke/GAN_Models/blob/main/pix2pixGen1000_l1loss_25k.pb) 

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Theoretical underpinnings of Cycle-Consistent Adverserial Networks (CycleGAN)
Unlike Pix2Pix in which paired training was required i.e. need both input and target pairs, CycleGan works on unpaired data i.e. no information is provided as to which input matches to which target.[^6]

## Unpaired CycleGAN Loss function:
> Our objective contains two types of terms: adversarial losses for matching the distribution of generated images to the data distribution in the target domain; and cycle consistency losses to prevent the learned mappings G and F from contradicting each other 
> 
> ![alt text](https://github.com/aCStandke/GAN_Models/blob/main/CycleGanLoss.png "total loss")
>
> We apply an adversarial losses to both mapping functions. For the mapping function G : X → Y and its discriminator D<sub>Y</sub> , we express the objective as:
>
> ![alt text](https://github.com/aCStandke/GAN_Models/blob/main/adverserialLoss.png "adverserial loss")
>
> where G tries to generate images G(*x*) that look similar to images from domain Y , while D<sub>Y</sub> aims to distinguish between translated samples G(*x*) and real samples *y*. G aims to minimize this objective against an adversary D that tries to maximize it, i.e., min<sub>G</sub>max<sub>D<sub>Y</sub></sub> *L*<sub>GAN</sub>(G, D<sub>Y</sub> , X, Y). We introduce a similar adversarial loss for the mapping function F : Y → X and its discriminator D<sub>X</sub> as well: i.e., min<sub>F</sub>max<sub>D<sub>X</sub></sub> *L*<sub>GAN</sub>(F, D<sub>X</sub>, Y, X).
> 
>Adversarial training can, in theory, learn mappings G
and F that produce outputs identically distributed as target
domains Y and X respectively. However,
with large enough capacity, a network can map the same
set of input images to any random permutation of images in
the target domain, where any of the learned mappings can
induce an output distribution that matches the target distribution. Thus, adversarial losses alone cannot guarantee
that the learned function can map an individual input *x*<sub>i</sub> to a desired output *y*<sub>i</sub> Thus to reduce the space of possible mapping functions, a constraint is introduced in which the mapping functions should be cycle-consistent in the forward direction i.e. x → G(x) → F(G(x)) ≈ x and in the backward direction y → F(y) → G(F(y)) ≈ y 
>
> ![alt text](https://github.com/aCStandke/GAN_Models/blob/main/cycleloss.png "cycle consistency loss")
> 
> Furthermore, for mapping paintings to photos (and thus also, photos to paintings), we find that it is helpful to introduce an additional loss to encourage the mapping to preserve color composition between the input and output. In particular, we adopt the technique of Taigman et al. and regularize the generator to be near an identity mapping when real samples of the target domain are provided as the input to the generator: 
> 
> ![alt text](https://github.com/aCStandke/GAN_Models/blob/main/identloss.png "identity loss")
> 
> Without Lidentity, the generator G and F are free to change the tint of input images when there is no need to. For example, when learning the mapping between Monet’s paintings and Flickr photographs, the generator often maps paintings of daytime to photographs taken during sunset, because such a mapping may be equally valid under the adversarial loss and cycle consistency loss
>
>[^6]


### Example 4: Unpaired CyleGAN for Image Segmentation on the Cityscape Images
The objective of this task is to transform a set of real-world images from the Cityscapes dataset[^7] into semantic segmentations. The dataset contains 5,000 finely annotated images split into training, and validation sets (i.e. 2975/500 split). The dense annotation contains 30 common classes of road, person, car, etc. as detailed by the following figure [^8]: 

![alt text](https://github.com/aCStandke/GAN_Models/blob/main/labels%20of%20colors.png "classes") 

Each of the models were trained for 10 epochs, even though they should be trained at least for 100 epochs(and the paper trained for 200 epochs). However, due to CycleGANs being very compute intensive even when using a GPU(or TPU), to do 10 epochs, which amounts to 29,750 steps it took 11 hours! So for 200 epochs, it would take 2 days of training the model, lol I am not that devoted! 

**CycleGAN w/ Pix2Pix U-Net Backbone w/ instance normalization:**

1. **LOSS FUNCTIONS:**
   - *Binary Focal Cross-Entropy*: [Focal Loss](https://medium.com/swlh/focal-loss-what-why-and-how-df6735f26616). Code can can be found here: [CycleGAN-FocalLoss](https://github.com/aCStandke/GAN_Models/blob/main/CycleGanUnet_BinaryFocalEntropy_Gamma2.ipynb)
   - *Binary Cross-Entropy*: 
2. **IMAGE POOL SIZE:**
   - *Pool Size 3 w/ Focal Cross-Entropy*: 
   - *Pool Size 3 w/ Binary Cross-Entropy*:
 
**CycleGAN w/ ResNet Backbone:**


1. **IMAGE POOL SIZE:**
   - *Pool Size 3*:
   - *Pool Size 5*:
   - *Pool Size 10*:

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
[^1]: [Generative Adversarial Nets](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)
[^2]: [UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS](https://arxiv.org/pdf/1511.06434.pdf)
[^3]: [Advanced Deep Learning with Python: Design and implement advanced next-generation AI solutions using TensorFlow and PyTorch](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X)
[^4]: [Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf)
[^5]: [pix2pix: Image-to-image translation with a conditional GAN](https://www.tensorflow.org/tutorials/generative/pix2pix)
[^6]: [Cycle-Consistent Adverserial Networks (CycleGAN)](https://arxiv.org/pdf/1703.10593.pdf)
[^7]: Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R.,
Franke, U., Roth, S., Schiele, B.: The cityscapes dataset for semantic urban scene
understanding. In: CVPR. (2016)
[^8]: [ICNet for Real-Time Semantic Segmentation](https://hszhao.github.io/papers/eccv18_icnet.pdf)
[^9]: Pretty good results, shows that increasing the l1 loss term provides significant improvements, especally when it comes to identifying pedestrians
