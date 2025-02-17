![Logo](https://github.com/protekti/A.D.A.M/blob/ai/Group%2080.png?raw=true)

A.D.A.M. (Autonomous Driving and Assistance Model) is an intelligent self-driving system designed to deliver safe, efficient, and adaptive autonomous driving experiences. Leveraging real-time data processing and machine learning, A.D.A.M. continuously learns from its environment, adjusts to road and traffic conditions, and prioritizes passenger comfort and safety. With a focus on dynamic decision-making and precision, A.D.A.M. aims to redefine the standards of autonomous mobility by providing reliable, human-centered assistance on every journey.

***THIS PROJECT IS CURRENTLY IN ITS SIMLATOR STAGES AND SHOULD NOT BE USED FOR REAL SCENARIOS.***

## Understanding Model Names  

Each model has a specific naming convention that provides key details about its training process. Here's how to interpret them:  

### Example: `adam_v0.1a_1e`  
- **adam** → Base name of the model  
- **v0.1a** → Version of the training code  
- **1e** → Number of epochs (An epoch refers to one complete pass through the entire dataset. The dataset is divided into smaller batches that are fed into the algorithm during training. The number of epochs can range from 1 to infinity.)  

From this example, we can determine that the model is named "adam," was trained using training program version **0.1a**, and completed **1 epoch** of training.  

### Epoch Categories:  
- **Testing Models** (`< 100e`)  
  - Used for initial testing to verify that the AI model produces readable and displayable data.  
- **Development Models** (`≥ 100e`)  
  - Used to refine the training program and analyze model performance at lower epoch counts.  
- **Ready Models** (`≥ 500e`)  
  - Considered sufficiently trained, with a low percentage of inaccurate lane detections.

 Keep in mind that the larger the epoch number doesnt mean the best results. As of current testing model adam_v0.3a_350e.keras is the current best preforming model.

This structured naming system helps track model progress and categorize models based on their level of training and reliability.

## Types
As of now there are 2 different types of A.D.A.M.. There is the OPENCV branch which utilises AI for car detection as well as distance aproximation but the AI branch is 100% AI. The AI branch is still in heavy development and is still undergoing major changes.

## Installation

As of now there is no official instalation instruction as the current code is in PRE-ALPHA stages meaning that the code isnt stable enough for proper use.

## Support

For support, email protekti08@gmail.com or DM me on discord @protekti.
