
# Parallel-Computing

This repository contains implementation of several image and data processing models which have been implemented in such a way to minimize the running time by maximizing the CPU and GPU resource utilization. Different kinds of CPU parallelism such as instruction level parallelism, Multicore parallelism and vectorization were used.

Language: C++, CUDA

1. Correlated Pairs: Implemented a program that is used to find pairwise correlations between pixel rows in PNG images. <br> V1 (CPU baseline): ~9secs <br> V2a (instruction-level parallelism): ~3.6 secs <br> V2b (multicore parallelism): ~1.2 secs <br> V2c (vectorization): ~3.4 secs <br> V3a (all resources of CPU): ~1.9 secs <br> V3b (all resources of CPU): ~0.7 secs <br> V4 (GPU baseline): ~0.14 secs <br> V5 (fast GPU baseline): ~0.76 secs 

2. Image Segmentation: Implemented a program that segmented image in PNG files. Found the best way to partition a given figure in two parts - a monochromatic rectangle and a monochromatic background. The objective was to minimize the sum of squared errors. <br> V1 (fast CPU solution: ~4.6 secs <br> V2a (fast CPU solution for 1-bit images): ~0.76 secs.

3. Median Filter: Implemented a program for doing 2-dimensional median filtering with a rectangular window. <br> V1 (CPU baseline): ~4secs <br>V2 (multicore parallelism): ~0.7 secs.

4. Neural Network: Developed a Convolutional Neural Network to classify images in C++. <br> V1a (fast CPU solution): ~4.1secs <br> V1b (fast GPU solution): ~0.8secs.

5. Sorting: Implemented a parallel sorting algorithm which used the basic approach used in Merge-Sort and Quick-Sort. <br> V1 (Parallel Merge Sort): ~2.3 secs <br> V2 (Parallel Quick Sort): ~2.0 secs
