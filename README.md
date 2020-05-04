# Improving Semantic Segmentation Through Clustering
In the field of Deep Neural Network and Deep Learning, it is commonly observed that it is very difficult for deep convolution neural network (DCNN) based semantic segmentation to estimate correct object boundary. We think that the reason lies in assigning labels for pixels on the object boundaries because the cascaded feature maps generated by DCNNs blur them. In order to segment foreground objects from background object the DCNN algorithm should classify the boundary pixel precisely.
This problem poses major hindrance to the accuracy in the semantic segmentation and it also leads to very high losses in ventures involving satellite imaging. Due to the involvement of very fine edges in 'Satellite images' missing them would lead to unbearable expenses and losses. Hence this becomes a major problem to be dealt with.
We solve this anomaly through the use of clustering methods. Clustering helps in Fine Edge Detection of the images and together with DeepLabV3 proves to be an indispensable tool for the solution.
After clustering the images, the segmentation of the Neural Network (here DeepLabV3) is enhanced by the use of our merging algorithm known as Disparting.
