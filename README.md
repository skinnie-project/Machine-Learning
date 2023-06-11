# Machine Learning

There are two machine learning parts in this project. The first one is to create an image classification for skin type classification, determining whether it's oily, dry, or normal. The second one is to make skincare recommendations based on the classified skin type. 

## Skin Type Classification

The skin type is classified by analyzing the picture using a Convolutional Neural Network (CNN). The CNN classifies the image into three classes, which is oily, dry, and normal type. To enhance the model's accuracy, we used transfer learning from [TensorFlow Hub](https://www.tensorflow.org/hub) EfficientNetV2 B0. The training accuracy achieved is >97%, while the validation accuracy is >91%. However, one major challenge we encountered is the limited availability of high-quality face images.

## Recommender System

-

## Dataset

For the skin type classification, we created the dataset ourselves by crawling pictures for each skin type from various search engines. As for the skincare product data, we scraped it from [Female Daily](https://femaledaily.com/).

## References

-

## Architecture

-





