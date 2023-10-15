=======
Support For Learning Rate Schedulers in Skorch
=======
Skorch, a powerful library for training PyTorch models, offers seamless integration with learning rate schedulers. Learning rate schedulers allow you to adapt the learning rate during training, leading to faster convergence and improved model performance. In this section, we'll explore how to use learning rate schedulers with Skorch to fine-tune your neural network training process.

What is a Learning Rate Scheduler?
----------

A learning rate scheduler dynamically adjusts the learning rate during training. It can be a crucial component of your training pipeline, enabling you to control the step size for updating the model's weights as the training progresses.

Using Learning Rate Schedulers in Skorch
Skorch allows you to integrate PyTorch learning rate schedulers seamlessly into your training process. Here's a step-by-step guide on how to use them:

1. Create Your Neural Network Model
Before you can use a learning rate scheduler, you need to define your neural network model using PyTorch. For example:

.. code:: python
    import torch
    import torch.nn as nn

    class YourModel(nn.Module):
        def __init__(self):
            super(YourModel, self).__init__()
            # Define your layers here


2. Create Your Skorch NeuralNet
Now, create a Skorch NeuralNet that wraps your PyTorch model. Make sure to specify the optimizer and learning rate scheduler in the NeuralNet constructor. Below is an example using the StepLR learning rate scheduler:

.. code:: python
    from skorch import NeuralNet
    from skorch.callbacks import LRScheduler

    from torch.optim import SGD
    from torch.optim.lr_scheduler import StepLR

    net = NeuralNet(
        YourModel,
        criterion=nn.CrossEntropyLoss,
        optimizer=SGD,
        optimizer__lr=0.01,
        optimizer__momentum=0.9,
        iterator_train__shuffle=True,
        callbacks=[
            ('scheduler', LRScheduler(StepLR, step_size=10, gamma=0.5)),
        ],
    )


In the example above, we set the optimizer to Stochastic Gradient Descent (SGD) and attach a StepLR learning rate scheduler with a step size of 10 and a decay factor of 0.5. You can customize the scheduler parameters to suit your needs.

3. Train Your Model
With your Skorch NeuralNet defined and the learning rate scheduler attached, you can start training your model as you normally would with scikit-learn:

.. code:: python
    net.fit(X_train, y_train)

The learning rate scheduler will automatically adjust the learning rate during training based on the specified schedule.

4. Monitor Training Progress
During training, Skorch will automatically keep you informed about the learning rate changes, allowing you to monitor the effect of the learning rate scheduler on your model's performance.

Conclusion
Learning rate schedulers are a valuable tool for fine-tuning neural network training, and Skorch simplifies their integration into your training pipeline. Experiment with different schedulers and monitor your model's progress to find the best strategy for your specific task. With Skorch, you have the flexibility to choose the scheduler that suits your needs, and you can easily adjust its parameters for optimal results.