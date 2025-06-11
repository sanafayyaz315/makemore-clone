makemore-clone: A Learning-by-Building Journey Through Language Models

I found a [gem](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) of a resource for anyone who wants to go beyond just using Large Language Models and instead dive deep into how they actually work.

This repo is my personal deep dive into building language models from scratch — inspired by [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy)’s excellent [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) series. This project replicates the core ideas of the MakeMore model: a simple character-level language model trained to generate new names — or in other words, to make more names.

I know I’m not the first (and definitely not the last) to share a makemore clone, but this one is mine. Every notebook here is a checkpoint in my own learning path, where I’ve explored concepts by building, breaking, and rebuilding.

Each notebook includes code, inline notes, and explanations — all written as I learned. No fancy abstractions, just raw, hands-on experimentation with ideas like bigrams, MLPs, transformers, and more.

# What's Included 


**1. Bigram by Counts:**
- A character-level language model that uses raw bigram frequencies to generate- new names.

- Calculates the probability of the next character based purely on the current one

- No learning involved. Just counting character transitions and normalizing

- Great for understanding statistical modeling before diving into neural networks

**2. Bigram Neural Network:**
- A single-layer neural network replaces the frequency table with trainable weights.

- Each character is embedded into a vector space using one-hot encoding

- Learns to predict the next character using gradient descent and cross-entropy loss

- Introduces core deep learning concepts like backpropagation, weight updates, and training loops

**3. MLP Language Model:**
- A more expressive architecture that uses a Multi-Layer Perceptron (MLP) with a context window of three characters.

- Context length (```block_size```) is set to 3, meaning the model uses the previous 3 characters to predict the next one (a trigram)

- Each character is passed through an embedding layer to convert discrete tokens into dense vectors

- These embeddings are concatenated and fed into a feedforward neural network with one or more hidden layers and non-linear activation functions (e.g., tanh)

- Learns richer patterns over longer sequences than bigram models and produces more coherent name generation

- Serves as a stepping stone to deeper architectures like transformers

**4. Weight Initialization, Vanishing Gradients & BatchNorm:**
- A focused exploration of training instability issues and how to overcome them.

- High Initial Loss from Improper Weight Initialization

    - Weights initialized with a standard normal distribution led to large logits

    - This caused unstable softmax outputs and high cross-entropy loss early in training

    - Resolved by scaling down initial weights (e.g., multiplying by a small factor) to bring logits closer to zero

- Vanishing Gradients due to tanh Saturation

    - Hidden activations were often pushed into the saturated regions of tanh, where gradients become very small. This led to slow or stalled learning

    - Mitigated first with naive weight scaling, then properly addressed using Kaiming initialization, which maintains variance of activations through layers

- Batch Normalization to Tackle Internal Covariate Shift

    - As training progresses, the distribution of hidden activations keeps shifting, known as **internal covariate shift**

    - This disrupts learning because each layer must constantly adapt to changing input distributions

    - BatchNorm normalizes the activations within a batch, reducing internal covariate shift, accelerating training, and improving stability

    - Result: smoother gradients, faster convergence, and better generalization


# How to Run
- Requirements
    - Python 3.8+
    - Jupyter Notebook
    - PyTorch, NumPy, Matplotlib

```
git clone https://github.com/sanafayyaz315/makemore-clone.git
cd makemore-clone
pip install -r requirements.txt
```
Open notebooks in order to follow along.

# Credits
Inspired by Andrej Karpathy’s original [tutorials](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ).

Built with ❤️, curiosity, and a desire to move from using LLMs to understanding them.