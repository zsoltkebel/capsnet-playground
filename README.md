# CapsNet Playground

Welcome to the CapsNet Playground! This project is an an interactive web visualisation of a capsule network and the dynamic routing algorithm between capsules, based on the paper [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829) by Sabour, Frosst, and Hinton.

The web visualisation is available at https://capsnet-playground.vercel.app.

Link to the GitHub repo: https://github.com/zsoltkebel/capsnet-playground.git

## Features

- Train a capsule network on the MNIST digits dataset.
- Load a pre-trained network.
- Make predicitons on input images.
- Observe and interact with the dynamic routing and see how coupling coefficients are refined.
- View capsule contributions to decision.
- View image reconstruction based on capsule values.

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/zsoltkebel/capsnet-playground.git
    cd capsnet-playground
    ```

2. Install dependencies:
    ```
    npm i
    ```

3. Start the development server:
    ```
    npm start
    ```

## References

- [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829) by Sabour, Frosst, and Hinton.

Happy experimenting!