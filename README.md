# Machine Learning

There are two machine learning parts in this project. The first one is to create an image classification for skin type classification, determining whether it's oily, dry, or normal. The second one is to make skincare recommendations based on the classified skin type. 

## Skin Type Classification

The skin type is classified by analyzing the picture using a Convolutional Neural Network (CNN). The CNN classifies the image into three classes, which is oily, dry, and normal type. To enhance the model's accuracy, we used transfer learning from [TensorFlow Hub](https://www.tensorflow.org/hub) EfficientNetV2 B0. The training accuracy achieved is >97%, while the validation accuracy is >91%. However, one major challenge we encountered is the limited availability of high-quality face images.

## Recommender System

-

## Dataset

For the skin type classification, we created the dataset ourselves by crawling pictures for each skin type from various search engines. As for the skincare product data, we scraped it from [Female Daily](https://femaledaily.com/).

## References

Here are some recources that we uses as oru reference:

https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2
https://medium.com/mlearning-ai/understanding-efficientnet-the-most-powerful-cnn-architecture-eaeb40386fad
https://www.researchgate.net/publication/350578362_EfficientNetV2_Smaller_Models_and_Faster_Training
https://github.com/Randon-Myntra-HackerRamp-21/Skyn/tree/main/ML/Skin_metrics/Skin_type


## EfficientNetV2 B0 Transfer Learning

EfficientNet is considered one of the most powerful CNN architectures. In transfer learning scenarios, we often use the feature extractor model. Instead of training a model from scratch, we can use a pre-trained EfficientNet feature extractor to extract image features. These features can then be used to create a custom model for tasks like object detection, segmentation, or fine-grained classification. This allows us to benefit from the pre-trained model's ability to capture detailed image representations while tailoring the final layers to our specific needs.

When deciding which version of EfficientNet to use, we should consider the size of our dataset. If the dataset is small, using a larger and more complex EfficientNet model could lead to overfitting. Overfitting means that the model becomes too specialized in the training data and doesn't generalize well to new, unseen data. So, it's important to choose a smaller version of EfficientNet for small datasets to avoid overfitting and ensure better performance on new data.

In this apllication, we decided to use EfficientNetV2 B0.

![fficientNet B0](https://www.researchgate.net/publication/348470984/figure/fig2/AS:979961129209859@1610652348348/The-EffecientNet-B0-general-architecture.png)

recource: [General EfficientNet B0 Architecture](http://dx.doi.org/10.1109/ACCESS.2021.3051085)

EfficientNetV2 is an improved version of the EfficientNet model, designed to be smaller and faster while maintaining high accuracy. It introduces a combination of MBConv and Fused-MBConv blocks, which help make better use of server/mobile accelerators during training. The model was optimized through a neural architecture search, considering factors like accuracy, training efficiency, and parameter efficiency.

EfficientNetV2 achieves similar accuracy to previous models but trains much faster. It outperforms other state-of-the-art models, including Vision Transformers, in terms of efficiency. This demonstrates that well structured convolutional neural networks (CNNs) can still achieve excellent results in computer vision tasks, even in the face of transformer models.

resource: [EfficientNetV2: Faster, Smaller, and Higher Accuracy than Vision Transformers](https://towardsdatascience.com/efficientnetv2-faster-smaller-and-higher-accuracy-than-vision-transformers-98e23587bf04)






