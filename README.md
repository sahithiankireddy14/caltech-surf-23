# Softmax & Mnist

1. [0. Instructions](#instructions)
2. [A. Safe Softmax](#parta)
3. [B. Mnist training & evaluation](#partb)

## 0. Instructions  <a name="instructions"></a>

1. There are two parts to this problem set. For Part A, provide an iPython notebook with the solution and answers to the question, titled `answers_a.ipynb`.
2. For Part B, provide a markdown report, titled `REPORT.md`, with answers to questions and a Colab/iPython notebook, titled `answers_b.ipynb` with the code that reproduces your experiments.
3. For the report, include 
  * thoughtful discussion (try to give explanations, not just answers!)
  * provide sources you might have used for your work (other github repos, papers, wikipedia etc.)
4. For the notebooks,
  * code should be clean, well commented and executable (I should be able to run it without any issues)
  * practice good coding practices (name variables relating to their use etc.)
  * GPU is optional but you can get access to a free one with Colab. In the menu, go to Runtime → Change runtime type → Set Hardware accelerator to GPU (default is None).
5. **You cannot collaborate with others**
6. You may look up anything on the internet but you can absolutely not plagiarize code from other people/sources. If you used a source, please reference it in your report.  
7. If you have questions, contact me!

## A. Safe Softmax <a name="parta"></a>

The problem is found in [SafeSoftmax](SafeSoftmax.md). Answer all questions and provide the implementation as well as the unit tests in an iPython notebook titled `answers_a.ipynb`.

## B. Mnist <a name="partb"></a>

1. Setup the Mnist dataset. 
  * [This](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html) might be of help.
  * Split the dataset into train/val/test (note that pytorch might provide an official train and test split, in which case, split the official train into train and val). Implement data loaders for each split. Discuss the process you went through to split the data. Why do we need these splits and what are good practices for creating data splits?
2. Setup model `Net`. Follow this template below
   ```python
   class Net(nn.Module):
    """
    Build the best MNIST classifier.
    """
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def train(model, device, train_loader, optimizer, epoch, log_interval):
        """
        This is your training function. When you call this function, the model is trained for 1 epoch.
        """
        raise NotImplementedError

    def test(model, device, test_loader):
        raise NotImplementedError
   ```
3. Implement the training loop. For example, you can use the following
    ```python
    # Feel free to change these pre-sets and experiment with different values.
    # Set random seed.
    seed = 33
    # Set batch size.
    batch_size = 64
    # Set learning rate
    lr = 1.0
    # Set total number of epochs
    epochs = 14
    # Set other hyperparameters of your choice.

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Pytorch has default MNIST dataloader which loads data at each iteration
    data_tansform = transforms.Compose([
        transforms.ToTensor(),  # Add data augmentation here
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('../data', train=True, download=True, transform=data_transform)

    # TODO: Add dataset split code here.

    # TODO: Add any relevant set up code.

    # Training loop
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval)
        test(model, device, val_loader)

        # You may optionally save your model at each epoch here
   ```
   * The model `Net` should be built using the following layers: Linear, Conv2d, MaxPool2d, AvgPool2d, ReLU, Softmax, BatchNorm2d, Dropout, Flatten. I leave it up to you to design the architecture. However, you might want to read some sources for effective network architectures to get inspired from.
   * Plot relevant metrics for training. Describe the metrics you use, why you chose them and what they capture. 
   * It's common practice in computer vision models to use data augmentations during training. Implement data augmentations and train your model. What do you observe. [This](https://pytorch.org/vision/stable/transforms.html) might be of help.
   * Discuss how you set up your architecture. Did you observe anything being particularly effective or ineffective?
   * Where the data splits useful? How did you use them?
4. Evaluate the model on the official test set. For example, 
    ```python
    # Add eval code.
    # The official test set is at:
    test_dataset = datasets.MNIST('../data', train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    ```
    * After you are done tweaking the network, evaluate the model on the official Mnist dataset. Write the evaluation code for model loss and accuracy. 
5. Performance analysis. Analyze what your network has learnd. You can use [scikit-learn](https://scikit-learn.org/stable/) for all parts of this question.
    * Plot per class precision-recall curves. Present them in your report. What do you see?
    * Present examples for each class where the classifier made a mistake. Can you distinguish any error modes?
    * Generate a confusion matrix for the test set. Present it in your report. What do you see?
    * Use your network to convert each image on the test set into a feature vector (the penultimate layer of the network). Visualize this high-dimensional embedding in 2D using tSNE (each class with its own color). What do you observe?






