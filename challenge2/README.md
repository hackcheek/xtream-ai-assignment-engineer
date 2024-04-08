# Challenge 2

### Kubernetes and Kubeflow
To create this pipeline I used Kubeflow. This is a nice tool to orchestate processes and make them scalable

For this challenge I'm running kubeflow locally in my Mac M1. So, I started a k3s cluster using k3d where kubeflow gonna be running.


### New Data
I found full diamonds dataset in this github repository: [link](https://github.com/Pratik94229/Diamond-Price-Prediction-End-to-End-Project/blob/main/artifacts/raw.csv) \
Complete dataset has +192k records! For this challenge, I will use only 2k examples.


### Training Aspects
I'll train svm-rbf model with old data + new data for each training iteration. Because, this model doesn't consume a lot of resources.
