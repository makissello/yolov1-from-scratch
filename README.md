# yolov1-from-scratch
**YOLOv1 inspired face detection CNN built entirely from scratch just using Numpy, no TensorFlow etc.**

- 1000 lines of code, only using numpy for math
- Forward propagation and backward propagation built from scratch
- Used my own Architecture consisting of CONV, MAXPOOL and FC layers with ReLU activation
- All layers are also built from scratch
- implemented Non-max suppression and grid cell architecture from YOLOv1 paper
- uses Adam optimizer for parameter updates and batch gradient descent
- uses original loss function from YOLOv1 paper to account for coordinate loss, objectness loss and classification loss
- binary classification of object: Face
- currently runs really slow as it isn't optimized for vectorization ~1h for 1 Epoch of mini_batch_size=4 and (448, 448, 3) images on M2 Apple Silicon
