import numpy as np

class YOLO_Face_Detector:
    def __init__(self, input_shape=(448, 448, 3), learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, grid_size=7, boxes_per_cell=2, classes=1):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.S = grid_size
        self.B = boxes_per_cell
        self.C = classes

        # Initialize parameters with random values
        self.parameters = self.initialize_parameters()

    def initialize_parameters(self):
        """
        Initializes parameters for all layers.

        Arguments:
        None

        Returns:
        parameters -- Dictionary containing all initialized parameters
        """

        np.random.seed(16)
        parameters = {}

        # Conv layer 1
        parameters["W1"] = np.random.randn(7, 7, 3, 64) * 0.01
        parameters["b1"] = np.zeros((1, 1, 1, 64))

        # Conv layer 2
        parameters["W2"] = np.random.randn(3, 3, 64, 128) * 0.01
        parameters["b2"] = np.zeros((1, 1, 1, 128))

        # Conv layer 3
        parameters["W3"] = np.random.randn(3, 3, 128, 256) * 0.01
        parameters["b3"] = np.zeros((1, 1, 1, 256))

        # Conv layer 4
        parameters["W4"] = np.random.randn(3, 3, 256, 512) * 0.01
        parameters["b4"] = np.zeros((1, 1, 1, 512))

        # Conv layer 5
        parameters["W5"] = np.random.randn(3, 3, 512, 1024) * 0.01
        parameters["b5"] = np.zeros((1, 1, 1, 1024))

        # FC layer 1
        parameters["W6"] = np.random.randn(1024 * 7 * 7, 4096) * 0.01
        parameters["b6"] = np.zeros((1, 4096))

        S = self.S
        B = self.B
        C = self.C

        # FC layer 2 (output layer for binary classification)
        # 5 output classes: 1 class prob, 4 for bounding box (bx, by, bh, bw)
        parameters["W7"] = np.random.randn(4096, S * S * (B * 5 + C)) * 0.01
        parameters["b7"] = np.zeros((1, S * S * (B * 5 + C)))

        return parameters

    def forward_prop(self, X):
        """
        Executes the forward prop for this YOLO inspired model.

        Arguments:
        X -- Input data of shape (m, height, width, channels)

        Returns:
        Y_pred -- Predictions of this CNN
        caches -- List of caches for back prop
        """

        caches = []
        A = X

        print("Starting forward prop layer 1: ")

        # Conv layer 1 + ReLU + MaxPool
        Z1, cache = self.conv_forward(A, self.parameters["W1"], self.parameters["b1"], stride=2, pad=3)
        A1, cache_relu = self.relu_forward(Z1)
        A1, cache_pool = self.maxpool_forward(A1, pool_size=2, stride=2)
        caches.append((cache, cache_relu, cache_pool))

        print("Starting forward prop layer 2: ")

        # Conv layer 2 + ReLU + MaxPool
        Z2, cache = self.conv_forward(A1, self.parameters["W2"], self.parameters["b2"], stride=1, pad=1)
        A2, cache_relu = self.relu_forward(Z2)
        A2, cache_pool = self.maxpool_forward(A2, pool_size=2, stride=2)
        caches.append((cache, cache_relu, cache_pool))

        print("Starting forward prop layer 3: ")

        # Conv layer 3 + ReLU + MaxPool
        Z3, cache = self.conv_forward(A2, self.parameters["W3"], self.parameters["b3"], stride=1, pad=1)
        A3, cache_relu = self.relu_forward(Z3)
        A3, cache_pool = self.maxpool_forward(A3, pool_size=2, stride=2)
        caches.append((cache, cache_relu, cache_pool))

        print("Starting forward prop layer 4: ")

        # Conv layer 4 + ReLU + MaxPool
        Z4, cache = self.conv_forward(A3, self.parameters["W4"], self.parameters["b4"], stride=1, pad=1)
        A4, cache_relu = self.relu_forward(Z4)
        A4, cache_pool = self.maxpool_forward(A4, pool_size=2, stride=2)
        caches.append((cache, cache_relu, cache_pool))

        print("Starting forward prop layer 5: ")

        # Conv layer 5 + ReLU + MaxPool
        Z5, cache = self.conv_forward(A4, self.parameters["W5"], self.parameters["b5"], stride=1, pad=1)
        A5, cache_relu5 = self.relu_forward(Z5)
        A5, cache_pool5 = self.maxpool_forward(A5, pool_size=2, stride=2)
        caches.append((cache, cache_relu5, cache_pool5))

        print("Starting forward prop FC layer 1: ")

        # FC layer 1 + ReLU
        Z6, cache = self.fc_forward(A5, self.parameters["W6"], self.parameters["b6"])
        A6, cache_relu = self.relu_forward(Z6)
        caches.append((cache, cache_relu))

        print("Starting forward prop FC layer 2: ")

        # FC layer 2
        Z7, cache = self.fc_forward(A6, self.parameters["W7"], self.parameters["b7"])
        caches.append((cache,))

        Y_pred = Z7.reshape((-1, self.S, self.S, (self.B * 5 + self.C)))

        return Y_pred, caches

    def compute_loss(self, Y_pred, Y):
        """
        Compute the YOLO loss function with grid cells.

        Arguments:
        Y_pred -- Output from forward propagation (m, S, S, (B*5+C))
        Y -- Ground truth labels (m, S, S, (B*5+C))

        Returns:
        loss -- Loss value
        """
        m = Y_pred.shape[0]
        S = Y_pred.shape[1]  # Grid size
        B = 2  # Number of boxes per grid cell

        # Hyperparameters
        lambda_coord = 5.0  # Weight for bounding box coordinates
        lambda_noobj = 0.5  # Weight for no-object confidence

        loss = 0

        for i in range(m):
            for row in range(S):
                for col in range(S):
                    # Extract objectness scores for each box
                    pred_box1_conf = Y_pred[i, row, col, 4]
                    pred_box2_conf = Y_pred[i, row, col, 9]

                    # Extract ground truth objectness
                    true_obj = Y[i, row, col, 4]  # Using first box's objectness as indicator

                    # If there is an object in that cell then add some losses
                    if true_obj > 0:
                        # Find which predicted box has higher IoU with ground truth
                        pred_box1 = Y_pred[i, row, col, 0:4] # These 3 get the (x, y, h, w) values
                        pred_box2 = Y_pred[i, row, col, 5:9]
                        true_box = Y[i, row, col, 0:4]

                        iou1 = self.compute_iou(pred_box1, true_box)
                        iou2 = self.compute_iou(pred_box2, true_box)

                        # Box with higher IoU is responsible for this prediction
                        if iou1 > iou2:
                            responsible_box = pred_box1
                            responsible_conf = pred_box1_conf
                            box_idx = 0
                        else:
                            responsible_box = pred_box2
                            responsible_conf = pred_box2_conf
                            box_idx = 5

                        epsilon_sqrt = 1e-10

                        # Ensure box dimensions are positive before sqrt
                        responsible_box[2] = max(responsible_box[2], epsilon_sqrt)
                        responsible_box[3] = max(responsible_box[3], epsilon_sqrt)
                        true_box[2] = max(true_box[2], epsilon_sqrt)
                        true_box[3] = max(true_box[3], epsilon_sqrt)

                        # Coordinate loss (x, y, w, h)
                        # loss += lambda_coord * ( (x - x_hat)^2 - (y - y_hat)^2 )
                        loss += lambda_coord * ((responsible_box[0] - true_box[0])**2 +
                                                (responsible_box[1] - true_box[1])**2)

                        # loss += lambda_coord * ( (sqrt(w) - sqrt(w_hat))^2 - (sqrt(h) - sqrt(h_hat))^2 )
                        loss += lambda_coord * ((np.sqrt(responsible_box[2]) - np.sqrt(true_box[2]))**2 +
                                                (np.sqrt(responsible_box[3]) - np.sqrt(true_box[3]))**2)

                        # Confidence loss for responsible box
                        # loss += (C - C_hat)^2 where C_hat=1
                        loss += (responsible_conf - 1)**2

                        # Class prediction loss
                        # loss += ( p(c) - p_hat(c) )^2
                        pred_class = Y_pred[i, row, col, 10]
                        true_class = Y[i, row, col, 10]
                        loss += (pred_class - true_class)**2

                    # If there's no object in this cell
                    else:
                        # Confidence loss for both boxes
                        # loss += lamda_noobj * (C - C_hat)^2 where C_hat=0
                        loss += lambda_noobj * (pred_box1_conf**2 + pred_box2_conf**2)

        # Normalize by batch size
        loss /= m

        return loss

    def compute_iou(self, box1, box2):
        """
        Compute IoU between two bounding boxes.

        Arguments:
        box1, box2 -- Bounding boxes [x, y, w, h] where x,y is midpoint

        Returns:
        iou -- Intersection over Union
        """
        # Convert from center coordinates to corner coordinates
        box1_x1 = box1[0] - box1[2]/2
        box1_y1 = box1[1] - box1[3]/2
        box1_x2 = box1[0] + box1[2]/2
        box1_y2 = box1[1] + box1[3]/2

        box2_x1 = box2[0] - box2[2]/2
        box2_y1 = box2[1] - box2[3]/2
        box2_x2 = box2[0] + box2[2]/2
        box2_y2 = box2[1] + box2[3]/2

        # Calculate intersection area
        x1 = max(box1_x1, box2_x1)
        y1 = max(box1_y1, box2_y1)
        x2 = min(box1_x2, box2_x2)
        y2 = min(box1_y2, box2_y2)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate union area
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union = box1_area + box2_area - intersection

        # Calculate IoU
        iou = intersection / union

        return iou

    def predict(self, X):
        """
        Make predictions using the trained model with non-max suppression.

        Arguments:
        X -- Input data of shape (m, height, width, channels)

        Returns:
        final_boxes -- List of final bounding boxes after non-max suppression
        final_scores -- List of confidence scores for final boxes
        final_classes -- List of class predictions for final boxes
        """
        # Forward propagation
        Y_pred, _ = self.forward_prop(X)

        m = Y_pred.shape[0]
        S = Y_pred.shape[1]  # Grid size
        B = 2  # Number of boxes per grid cell

        all_boxes = []
        all_scores = []
        all_classes = []

        # Process each image in the batch
        for i in range(m):
            boxes = []
            scores = []
            classes = []

            # Process each grid cell
            for row in range(S):
                for col in range(S):
                    # Process each bounding box in the cell
                    for b in range(B):
                        # Box coordinates are relative to grid cell
                        # Convert to absolute coordinates
                        box_idx = b * 5

                        # Extract box data
                        x = Y_pred[i, row, col, box_idx + 0]
                        y = Y_pred[i, row, col, box_idx + 1]
                        w = Y_pred[i, row, col, box_idx + 2]
                        h = Y_pred[i, row, col, box_idx + 3]
                        confidence = Y_pred[i, row, col, box_idx + 4]

                        # Apply sigmoid to confidence
                        confidence = 1 / (1 + np.exp(-confidence))

                        # Convert from grid cell coordinates to image coordinates
                        x = (col + x) / S
                        y = (row + y) / S

                        # Class prediction (apply sigmoid)
                        class_idx = B * 5
                        class_pred = 1 / (1 + np.exp(-Y_pred[i, row, col, class_idx]))

                        # Only keep boxes with confidence above threshold
                        if confidence > 0.5:  # Confidence threshold
                            boxes.append([x, y, w, h])
                            scores.append(confidence)
                            classes.append(class_pred > 0.5)  # Binary classification

            # Apply non-max suppression
            final_boxes, final_scores, final_classes = self.non_max_suppression(
                np.array(boxes), np.array(scores), np.array(classes), iou_threshold=0.5
            )

            all_boxes.append(final_boxes)
            all_scores.append(final_scores)
            all_classes.append(final_classes)

        return all_boxes, all_scores, all_classes

    def non_max_suppression(self, boxes, scores, classes, iou_threshold=0.5):
        """
        Apply non-max suppression to remove redundant bounding boxes.

        Arguments:
        boxes -- Array of bounding boxes [x, y, w, h]
        scores -- Array of confidence scores
        classes -- Array of class predictions
        iou_threshold -- IoU threshold for suppression

        Returns:
        selected_boxes -- Array of selected boxes
        selected_scores -- Array of selected scores
        selected_classes -- Array of selected classes
        """
        # If no boxes, return empty arrays
        if len(boxes) == 0:
            return [], [], []

        # Sort boxes by confidence score (highest first)
        indices = np.argsort(scores)[::-1]
        boxes = boxes[indices]
        scores = scores[indices]
        classes = classes[indices]

        selected_indices = []

        # Non-max suppression
        while len(boxes) > 0:
            # Select box with highest score
            selected_indices.append(0)

            # If only one box left, break
            if len(boxes) == 1:
                break

            # Calculate IoU of the selected box with all remaining boxes
            ious = np.array([self.compute_iou(boxes[0], box) for box in boxes[1:]])

            # Keep boxes with IoU less than threshold
            mask = ious < iou_threshold
            boxes = np.concatenate(([boxes[0]], boxes[1:][mask]))
            scores = np.concatenate(([scores[0]], scores[1:][mask]))
            classes = np.concatenate(([classes[0]], classes[1:][mask]))

        # Get the selected boxes, scores, and classes
        selected_boxes = boxes
        selected_scores = scores
        selected_classes = classes

        return selected_boxes, selected_scores, selected_classes

    def zero_pad(self, X, pad):
        """
        Pads the input X with zeros.

        Arguments:
        X -- Input of shape (m, n_H, n_W, n_C)
        pad -- integer, amount of padding

        Returns:
        X_pad -- Padded input X
        """

        X_pad = np.pad(X, ((0,0), (pad, pad), (pad, pad), (0,0)), mode='constant', constant_values = (0,0))
        return X_pad

    def conv_single_step(self, a_slice_prev, W, b):
        """
        Implements a single convolution step. Applies a filter defined by W on a single slice of the previous layer output.

        Arguments:
        a_slice_prev -- Slice of previous activation layer of shape (f, f, n_C_prev)
        W -- Weight parameters of filter of shape (f, f, n_C_prev)
        b -- Bias parameter of filter of shape (1, 1, 1)

        Returns:
        Z -- Scalar, result of convolving the filter on a slice of the last activation
        """

        s = a_slice_prev * W
        Z = np.sum(s)
        Z = np.float64(Z + b)

        return Z

    def conv_forward(self, A_prev, W, b, stride, pad):
        """
        Implements the forward prop for a Convolutional layer.

        Arguments:
        A_prev -- Activations from previous layer of shape (m, n_H_prev, n_W_prev, n_C_prev)
        W -- Weights for filter of shape (f, f, n_C_prev, n_C)
        b -- Bias for filter of shape (1, 1, 1, n_C)
        stride -- Stride value
        padding -- Padding of A_prev

        Returns:
        Z -- Result of convolution operation on A_prev
        cache -- Cache for later back prop with (A_prev, W, b, stride, pad)
        """

        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        (f, f, n_C_prev, n_C) = W.shape

        n_H = int((n_H_prev - f + 2*pad) / stride) + 1
        n_W = int((n_W_prev - f + 2*pad) / stride) + 1

        Z = np.zeros((m, n_H, n_W, n_C))

        A_prev_pad = self.zero_pad(A_prev, pad)

        print(Z.shape)

        for i in range(m):
            print("i: " + str(i))
            a_prev_pad = A_prev_pad[i]

            for h in range(n_H):
                vert_start = h * stride
                vert_end = vert_start + f

                for w in range(n_W):
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    for c in range(n_C):
                        a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        weights = W[:, :, :, c]
                        biases = b[:, :, :, c]
                        Z[i, h, w, c] = self.conv_single_step(a_slice_prev, weights, biases)

        cache = (A_prev, W, b, stride, pad)

        return Z, cache


    def maxpool_forward(self, A_prev, pool_size, stride):
        """
        Implements the forward prop for a MaxPool layer.

        Arguments:
        A_prev -- Activations from previous layer of shape (m, n_H_prev, n_W_prev, n_C_prev)
        pool_size -- Size f of filter (f, f)
        stride -- Stride value

        Returns:
        A -- Output of the pool layer of shape (m, n_H, n_W, n_C)
        cache -- Cache used in the backprop containing A_prev
        """

        # Retrieve dimensions from previous activations
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        # Calculate output dimensions after applying max pooling
        n_H = int(1 + (n_H_prev - pool_size) / stride)
        n_W = int(1 + (n_W_prev - pool_size) / stride)
        n_C = n_C_prev

        # Initialize output matrix A
        A = np.zeros((m, n_H, n_W, n_C))

        for i in range(m):
            for h in range(n_H):
                vert_start = h * stride
                vert_end = vert_start + pool_size

                for w in range(n_W):
                    horiz_start = w * stride
                    horiz_end = horiz_start + pool_size

                    for c in range(n_C):

                        # Compute pooling operation on a slice of the input
                        a_prev_slide = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                        A[i, h, w, c] = np.max(a_prev_slide)

        cache = (A_prev, pool_size, stride)
        return A, cache

    def fc_forward(self, A_prev, W, b):
        """
        Implements the forward prop for a Fully Connected (FC) layer.

        Arguments:
        A_prev -- Activations from previous layer of shape (m, n_H_prev, n_W_prev, n_C_prev)
        W -- Weights for previous layer of shape (n_H_prev * n_W_prev * n_C_prev, n_H * n_W * n_C)
        b -- Bias for filter of shape (1, 1, 1, n_H * n_W * n_C)

        Returns:
        Z -- Output of the FC layer of shape (m, n_H * n_W * n_C)
        cache -- Cache containing A_prev, W, b for backpropagation
        """

        m = A_prev.shape[0]

        A_prev_flatten = A_prev.reshape(m, -1)
        Z = np.dot(A_prev_flatten, W) + b

        cache = (A_prev, W, b)
        return Z, cache

    def relu_forward(self, Z):
        """
        Implements forward prop for ReLU activation function.

        Arguments:
        Z -- Input to the ReLU activation of shape (m, n_H, n_W, n_C)

        Returns:
        A -- Output of ReLU activation of shape (m, n_H, n_W, n_C)
        cache -- Cache containing Z
        """

        A = np.maximum(Z, 0)

        cache = Z
        return A, cache

    def backward_prop(self, Y_pred, Y, caches):
        """
        Implement backward prop for the YOLO-inspired model with grid cells.

        Arguments:
        Y_pred -- Output from forward propagation (m, S, S, (B*5+C))
        Y -- Ground truth labels (m, S, S, (B*5+C))
        caches -- List of caches from forward propagation

        Returns:
        gradients -- Dictionary containing gradients of parameters
        """
        m = Y_pred.shape[0]
        S = Y_pred.shape[1]  # Grid size (e.g., 7)
        B = 2  # Number of bounding boxes per grid cell
        C = 1  # Number of classes (binary: face or not)

        # Initialize gradients dictionary
        gradients = {}

        # Initialize gradient of the output layer
        dY_pred = np.zeros_like(Y_pred)

        # Hyperparameters for loss function
        lambda_coord = 5.0  # Weight for bounding box coordinates
        lambda_noobj = 0.5  # Weight for no-object confidence

        # Compute gradients for each image in the batch
        for i in range(m):
            for row in range(S):
                for col in range(S):
                    # Check if there's an object in this cell
                    true_obj = Y[i, row, col, 4]  # Using first box's objectness as indicator

                    if true_obj > 0:  # If there's an object in this cell
                        # Find which predicted box has higher IoU with ground truth
                        pred_box1 = Y_pred[i, row, col, 0:4]
                        pred_box2 = Y_pred[i, row, col, 5:9]
                        true_box = Y[i, row, col, 0:4]

                        iou1 = self.compute_iou(pred_box1, true_box)
                        iou2 = self.compute_iou(pred_box2, true_box)

                        # Box with higher IoU is responsible for this prediction
                        if iou1 > iou2:
                            responsible_idx = 0  # First box is responsible
                        else:
                            responsible_idx = 5  # Second box is responsible

                        # Gradient for box coordinates (x, y)
                        dY_pred[i, row, col, responsible_idx] = lambda_coord * 2 * (Y_pred[i, row, col, responsible_idx] - Y[i, row, col, 0])
                        dY_pred[i, row, col, responsible_idx+1] = lambda_coord * 2 * (Y_pred[i, row, col, responsible_idx+1] - Y[i, row, col, 1])

                        # Gradient for box dimensions (w, h) - using square root in the loss
                        # d/dx(sqrt(x) - sqrt(y))^2 = 2(sqrt(x) - sqrt(y)) * 0.5/sqrt(x)
                        pred_w = Y_pred[i, row, col, responsible_idx+2]
                        pred_h = Y_pred[i, row, col, responsible_idx+3]
                        true_w = Y[i, row, col, 2]
                        true_h = Y[i, row, col, 3]

                        epsilon_sqrt = 1e-10

                        # Ensure box dimensions are positive before sqrt
                        pred_w = max(pred_w, epsilon_sqrt)
                        pred_h = max(pred_h, epsilon_sqrt)
                        true_w = max(true_w, epsilon_sqrt)
                        true_h = max(true_h, epsilon_sqrt)

                        dY_pred[i, row, col, responsible_idx+2] = lambda_coord * (np.sqrt(pred_w) - np.sqrt(true_w)) / np.sqrt(pred_w)
                        dY_pred[i, row, col, responsible_idx+3] = lambda_coord * (np.sqrt(pred_h) - np.sqrt(true_h)) / np.sqrt(pred_h)

                        # Gradient for objectness score of responsible box
                        # Using sigmoid: σ(x) = 1/(1+e^(-x))
                        # d/dx(σ(x) - 1)^2 = 2(σ(x) - 1) * σ(x)(1-σ(x))
                        pred_conf = 1 / (1 + np.exp(-Y_pred[i, row, col, responsible_idx+4]))
                        dY_pred[i, row, col, responsible_idx+4] = 2 * (pred_conf - 1) * pred_conf * (1 - pred_conf)

                        # Gradient for objectness score of non-responsible box (should be zero)
                        non_responsible_idx = 9 if responsible_idx == 0 else 4
                        pred_conf = 1 / (1 + np.exp(-Y_pred[i, row, col, non_responsible_idx]))
                        dY_pred[i, row, col, non_responsible_idx] = 2 * pred_conf * pred_conf * (1 - pred_conf)

                        # Gradient for class prediction
                        # Using sigmoid: σ(x) = 1/(1+e^(-x))
                        # d/dx(σ(x) - y)^2 = 2(σ(x) - y) * σ(x)(1-σ(x))
                        pred_class = 1 / (1 + np.exp(-Y_pred[i, row, col, 10]))
                        true_class = Y[i, row, col, 10]
                        dY_pred[i, row, col, 10] = 2 * (pred_class - true_class) * pred_class * (1 - pred_class)

                    else:  # If there's no object in this cell
                        # Gradient for objectness scores (should be zero)
                        # Using sigmoid: σ(x) = 1/(1+e^(-x))
                        # d/dx(σ(x))^2 = 2σ(x) * σ(x)(1-σ(x))
                        for b in range(B):
                            conf_idx = b * 5 + 4
                            pred_conf = 1 / (1 + np.exp(-Y_pred[i, row, col, conf_idx]))
                            dY_pred[i, row, col, conf_idx] = lambda_noobj * 2 * pred_conf * pred_conf * (1 - pred_conf)

        # Reshape dY_pred to match the original output shape from the network
        dY_pred_reshaped = dY_pred.reshape(m, -1)

        print("Starting backprop for FC layer 2...")

        # Backpropagate through the network
        # FC layer 2 (output layer)
        cache = caches[-1][0]
        dA6, gradients["dW7"], gradients["db7"] = self.fc_backward(dY_pred_reshaped, cache)

        print("Starting backprop for FC layer 1...")

        # FC layer 1 + ReLU
        cache, cache_relu6 = caches[-2]
        dZ6 = self.relu_backward(dA6, cache_relu6)
        dA5, gradients["dW6"], gradients["db6"] = self.fc_backward(dZ6, cache)

        print("Starting backprop for Conv layer 5...")

        # Conv layer 5 + ReLU + MaxPool
        cache, cache_relu5, cache_pool5 = caches[-3]
        dA5 = self.maxpool_backward(dA5, cache_pool5)
        dZ5 = self.relu_backward(dA5, cache_relu5)
        dA4, gradients["dW5"], gradients["db5"] = self.conv_backward(dZ5, cache)

        print("Starting backprop for Conv layer 4...")

        # Conv layer 4 + ReLU + MaxPool
        cache, cache_relu4, cache_pool4 = caches[-4]
        dA4 = self.maxpool_backward(dA4, cache_pool4)
        dZ4 = self.relu_backward(dA4, cache_relu4)
        dA3, gradients["dW4"], gradients["db4"] = self.conv_backward(dZ4, cache)

        print("Starting backprop for Conv layer 3...")

        # Conv layer 3 + ReLU + MaxPool
        cache, cache_relu3, cache_pool3 = caches[-5]
        dA3 = self.maxpool_backward(dA3, cache_pool3)
        dZ3 = self.relu_backward(dA3, cache_relu3)
        dA2, gradients["dW3"], gradients["db3"] = self.conv_backward(dZ3, cache)

        print("Starting backprop for Conv layer 2...")

        # Conv layer 2 + ReLU + MaxPool
        cache, cache_relu2, cache_pool2 = caches[-6]
        dA2 = self.maxpool_backward(dA2, cache_pool2)
        dZ2 = self.relu_backward(dA2, cache_relu2)
        dA1, gradients["dW2"], gradients["db2"] = self.conv_backward(dZ2, cache)

        print("Starting backprop for Conv layer 1...")

        # Conv layer 1 + ReLU + MaxPool
        cache, cache_relu1, cache_pool1 = caches[-7]
        dA1 = self.maxpool_backward(dA1, cache_pool1)
        dZ1 = self.relu_backward(dA1, cache_relu1)
        _, gradients["dW1"], gradients["db1"] = self.conv_backward(dZ1, cache)

        return gradients

    def relu_backward(self, dA, cache):
        """
        Implement backward pop for ReLU activation function.

        Arguments:
        dA -- Gradient of the cost with respect to the output of the ReLU of shape same like Z
        cache -- Z, stored in cache during forward prop

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """

        Z = cache
        dZ = np.array(dA, copy=True)

        dZ[Z <= 0] = 0
        return dZ

    def conv_backward(self, dZ, cache):
        """
        Implements the backward prop for a Convolution layer.

        Arguments:
        dZ -- Gradient of the cost of Conv layer of shape (m, n_H, n_W, n_C)
        cache -- Cache of output from conv_forward()

        Returns:
        dA_prev -- Gradient of the cost with respect to the input of the Conv layer of shape same like A_prev
        dW -- Gradient of the cost with respect to the weights of the Conv layer of shape same like W
        db -- Gradient of the cost with respect to the bias of the Conv layer of shape same like b
        """

        (A_prev, W, b, stride, pad) = cache

        (f, f, n_C_prev, n_C) = W.shape
        (m, n_H, n_W, n_C) = dZ.shape

        dA_prev = np.zeros(A_prev.shape)
        dW = np.zeros(W.shape)
        db = np.zeros(b.shape)

        A_prev_pad = self.zero_pad(A_prev, pad)
        dA_prev_pad = self.zero_pad(dA_prev, pad)

        for i in range(m):
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]

            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):

                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f

                        a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                        dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                        db[:,:,:,c] += dZ[i, h, w, c]

            dA_prev[i] = da_prev_pad[pad:-pad, pad:-pad]

        return dA_prev, dW, db

    def create_mask_from_window(self, x):
        """
        Creates a mask from an input x identifying the max entry of x.

        Arguments:
        x -- Input of shape (f, f)

        Returns:
        mask -- Output same shape as window, containing True at the position of max entry of x
        """

        mask = (x == np.max(x))
        return mask

    def maxpool_backward(self, dA, cache):
        """
        Implements backward prop for a MaxPool layer.

        Arguments:
        dA -- Gradient of the cost with respect to the output of the pooling layer of same shape like A
        cache -- (A_prev, f, stride) stored in cache during forward prop

        Returns:
        dA_prev -- Gradient of the cost with respect to the input of the pooling layer of same shape like A_prev
        """

        A_prev, f, stride = cache

        (m, n_H, n_W, n_C) = dA.shape

        dA_prev = np.zeros(A_prev.shape)

        for i in range(m):
            a_prev = A_prev[i]

            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):

                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f

                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]

                        mask = self.create_mask_from_window(a_prev_slice)

                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA[i, h, w, c]

        return dA_prev

    def fc_backward(self, dZ, cache):
        """
        Implements backward prop for a FC layer.

        Arguments:
        dZ -- Gradient of cost with respect to the output of the FC layer of shape (m, n_H * n_W * n_C)
        cache -- (A_prev, W, b) stored in cache during forward prop

        Returns:
        dA_prev -- Gradient of cost with respect to the input of the FC layer of shape (m, n_H, n_W, n_C)
        dW -- Gradient of cost with respect to the weights of the FC layer of shape (n_H_prev * n_W_prev * n_C_prev, n_H * n_W * n_C)
        db -- Gradient of cost with respect to the bias of the FC layer of shape (1, n_H * n_W * n_C)
        """

        A_prev, W, b = cache
        m = A_prev.shape[0]

        orig_shape = A_prev.shape

        A_prev_flatten = A_prev.reshape(m, -1)

        dA_prev_flatten = np.dot(dZ, W.T)
        dW = np.dot(A_prev_flatten.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)

        dA_prev = dA_prev_flatten.reshape(orig_shape)
        return dA_prev, dW, db

    def initialize_adam(self):
        """
        Initialize Adam optimizer parameters.

        Returns:
        v -- Dictionary containing the exponentially weighted average of the gradient
        s -- Dictionary containing the exponentially weighted average of the squared gradient
        """
        v = {}
        s = {}

        # Initialize v, s for all parameters
        for l in range(1, 8):
            v["dW" + str(l)] = np.zeros_like(self.parameters["W" + str(l)])
            v["db" + str(l)] = np.zeros_like(self.parameters["b" + str(l)])
            s["dW" + str(l)] = np.zeros_like(self.parameters["W" + str(l)])
            s["db" + str(l)] = np.zeros_like(self.parameters["b" + str(l)])

        return v, s

    def update_parameters_with_adam(self, gradients, v, s, t):
        """
        Update parameters using Adam optimizer.

        Arguments:
        gradients -- Dictionary containing gradients of parameters
        v -- Dictionary containing the exponentially weighted average of the gradient (first moment)
        s -- Dictionary containing the exponentially weighted average of the squared gradient (second moment)
        t -- Iteration number

        Returns:
        v -- Updated exponentially weighted average of the gradient
        s -- Updated exponentially weighted average of the squared gradient
        """
        v_corrected = {}
        s_corrected = {}

        # Update parameters for each layer
        for l in range(1, 8):

            # Moving average of the gradients
            v["dW" + str(l)] = self.beta1 * v["dW" + str(l)] + (1 - self.beta1) * gradients["dW" + str(l)]
            v["db" + str(l)] = self.beta1 * v["db" + str(l)] + (1 - self.beta1) * gradients["db" + str(l)]

            # Compute bias-corrected first moment estimate
            v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - np.power(self.beta1, t))
            v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - np.power(self.beta1, t))

            # Moving average of the squared gradients
            s["dW" + str(l)] = self.beta2 * s["dW" + str(l)] + (1 - self.beta2) * np.power(gradients["dW" + str(l)], 2)
            s["db" + str(l)] = self.beta2 * s["db" + str(l)] + (1 - self.beta2) * np.power(gradients["db" + str(l)], 2)

            # Compute bias-corrected second raw moment estimate
            s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - np.power(self.beta2, t))
            s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - np.power(self.beta2, t))

            # Update parameters
            self.parameters["W" + str(l)] -= self.learning_rate * v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + self.epsilon)
            self.parameters["b" + str(l)] -= self.learning_rate * v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + self.epsilon)

        return v, s

    def train(self, X, Y, num_epochs, batch_size=32, print_cost=True):
        """
        Train the YOLO-inspired model using batch gradient descent with Adam optimizer.

        Arguments:
        X -- Training data of shape (m, height, width, channels)
        Y -- Ground truth labels of shape (m, S, S, (B*5+C)) where:
             - S is the grid size (e.g., 7)
             - B is the number of boxes per grid cell (e.g., 2)
             - C is the number of classes (e.g., 1 for binary)
        num_epochs -- Number of epochs for training
        batch_size -- Size of mini-batches
        print_cost -- Whether to print cost during training

        Returns:
        costs -- List of costs during training
        """
        costs = []
        m = X.shape[0]
        num_batches = int(m / batch_size)

        # Initialize Adam parameters
        v, s = self.initialize_adam()
        t = 0  # Adam iteration counter

        # Training loop
        for epoch in range(num_epochs):
            epoch_cost = 0

            # Create mini-batches
            permutation = list(np.random.permutation(m))
            shuffled_X = X[permutation, :, :, :]
            shuffled_Y = Y[permutation, :, :, :]

            for batch in range(num_batches):
                # Get current mini-batch
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, m)
                X_batch = shuffled_X[start_idx:end_idx, :, :, :]
                Y_batch = shuffled_Y[start_idx:end_idx, :, :, :]

                # Forward propagation
                Y_pred, caches = self.forward_prop(X_batch)

                # Compute cost
                batch_cost = self.compute_loss(Y_pred, Y_batch)
                epoch_cost += batch_cost / num_batches

                # Backward propagation
                gradients = self.backward_prop(Y_pred, Y_batch, caches)

                # Update parameters
                t += 1
                v, s = self.update_parameters_with_adam(gradients, v, s, t)

            # Print cost
            if print_cost and (epoch % 5 == 0 or epoch == num_epochs - 1):
                print(f"Cost after epoch {epoch}: {epoch_cost}")

            costs.append(epoch_cost)

        return costs

# Dummy dataset for testing
X_train = np.random.randn(20, 448, 448, 3)  # 20 images, each 448x448x3
Y_train = np.random.randn(20, 7, 7, 11)  # 20 ground truth outputs for YOLO (7x7 grid, 2 boxes, 5 for each box, and 1 class)

# Initialize and train the YOLO model
yolo = YOLO_Face_Detector()
yolo.train(X_train, Y_train, batch_size=4, num_epochs=10)
