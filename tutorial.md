# 🚀 CLEO Workflow Tutorial

Welcome to this tutorial for **Combinatorial Libraries to Explore and Optimize (CLEO)**! This guide will walk you through the process of designing protein libraries and using experimental data to guide exploration and optimization. The framework focuses on enhancing an already functional protein construct using experimental feedback.

## 🌐 Key Concepts

The two main themes of this workflow are:

1. [**Library Design**](#-library-design)  
   - [Library constraints](#-library-constraints)  
   - [Defining reward functions](#-defining-reward-functions)  
   - [Aligning proteinMPNN to rewards](#aligning-proteinmpnn-to-rewards)  
   - [Sampling and filtering sequence fragments](#sampling-and-filtering-sequence-fragments)  
   - [Reverse translating sequences to prepare for experimental assembly](#reverse-translating-sequences-to-prepare-for-experimental-assembly)  

2. [**Multi-Round Experimental Optimization**](#-multi-round-experimental-optimization)  
   - [Data collection philosophy and assay design](#data-collection-philosophy-and-assay-design)  
   - [Strategies for early rounds of testing](#strategies-for-early-rounds-of-testing)  
   - [Training sequence-to-function models](#training-sequence-to-function-models)  
   - [Proposing batch of sequences to test next](#proposing-batch-of-sequences-to-test-next)  
   - [Looping it all together](#-looping-it-all-together)

---

# 💻 Library Design

Library design is the foundation of creating effective protein constructs. While this tutorial uses PETase (a plastic-degrading enzyme) as an example, the workflow can apply widely to other proteins.

🔑 **Note**: The libraries we design focus on a single fold space. Within the library, there may be variations in conformations predicted by structure oracles.

## 📐 Library constraints

## 🛠 Defining reward functions

First, define the **reward metrics** predictive of protein function. If no experimental data exists yet:
- Use structural prediction confidence, atomic distances, and hamming distances to the parent sequence.

If experimental data is already available (e.g., from a first round of testing):
- Train a custom predictor on the experimental results and incorporate its scores into the reward.

## 🔧 Aligning ProteinMPNN to rewards:
1. Fix **2 amino acids** at the start and end of each fragment (e.g., for Golden Gate Assembly compatibility 🧬).
2. Use a **structure prediction oracle** to assess sequences and calculate rewards.  
   - Example reward metrics: catalytic distances, confidence scores, mutational distances.

3. Normalize metrics to [0, 1] and compute the **overall scalar reward** as a weighted sum. Starting with equal weights is a good default.  

   Example code snippets for fine-tuning and reward functions are available in the <add link to config file>.  

### Launching Runs
Once your run setup is configured:
```bash
# Placeholder for training command
executetraining config.yaml
```

📝 **Tip**: Use the <path-to-notebook> to visualize metrics during training and track convergence.

## 🎲 Sampling and Filtering Sequence Fragments

After the model converges, you'll want to **sample sequences** for testing:
- Example: Sample 5,000 sequences if aiming to order 384 to allow for filtering.  
- The **goal** is to analyze fragment compatibility in silico and eliminate incompatible ones.

### Filtering and Constraints:
1. Rank fragments based on aggregated metrics.  
2. Apply constraints:
   - Diversity in mutation space (avoid over-optimization for a single point).  
   - Ensuring uniform distance from the parent sequence.  

A detailed example notebook demonstrating sequence filtering is available at <link-to-notebook>.

## 🧬 Reverse Translation and Order Preparation

Reverse translating protein sequences into DNA sequences is the final step before ordering. Key considerations:
- Use orthogonal codon pairs to ensure compatibility for assembly methods (e.g., ⛓️ NEB Golden Gate Assembly).  
- Validate assembly methods using tools like [Benchling](https://www.benchling.com/) or NEB's assembly wizard.

Positive controls: Split the original, active sequence into the same fragments to benchmark your experimental data.

---

# 🔬 Multi-Round Experimental Optimization

## 🧪 Data Collection Philosophy and Assay Design
Establish a robust assay that demonstrates clear, reproducible signal for your **positive control** over background noise. For PETase:
- Use Golden Gate Assembly to create DNA constructs.
- Express protein in a cell-free system and measure activity (e.g., using a fluorescent product signal).
- Normalize measurements (e.g., divide activity rates by protein concentration).

### Data filtering
- Use **thresholds**: e.g., protein expression ≥ 0.1 µM and CoV < 1 for replicates.  
- Provide metadata and plate maps in your dataset for improved reproducibility.  
📓 Refer to <link-to-data-processing-folder> for scripts and an example notebook.

---

## 🦠 Early Rounds of Testing

Decide how to screen your library. Two possible scenarios:
1. **Random Sampling**: Works best if ~10% of the variants show measurable activity.  
2. **Independent Fragment Testing**: Test fragments individually with parent fragments, analyze, and recombine the best-performing ones.  

Example figure from PETase **independent testing** (to be added).

---

## 🤖 Training Sequence-to-Function Models

Once experimental data is collected:
- Train a **predictive model** to map sequences to fitness.  
- Use a simple MLP (Multi-Layer Perceptron) with dropout layers and ensemble training.  

Normalization is key:
1. Scale the activity data (e.g., z-score).  
2. Split into training/validation sets (10%-20% validation).  

👨‍💻 Config and example notebook for training available: <link-to-notebook>.

---

## 🧠 Proposing Batch of Sequences to Test Next

With a trained model, propose new sequences for experimentation:
- Small libraries (<1 billion): Predict activity for all sequences.  
- Large libraries: Use a batched acquisition function that combines **mean activity prediction** and diversity optimization.  

💡 Additional Filtering Tips:
- Consider sequences with a **range of activities** to improve calibration.  
- Filter by predicted activity threshold for high-confidence candidates.

Notebook guide for batch optimization here: <link-to-batch-optimization-notebook>.

---

## 🔄 Looping it all together

Repeat the process (test → train → propose) until the activity metric plateaus. Satisfied with the results? 🎉 Congratulations, you’ve optimized your protein construct!

For advanced users:
- Use predictors trained on all collected data to generate a new library: <add link to config>.  

---

Thanks for following this tutorial! We hope this helps streamline your protein fitness optimization endeavors. Good luck! 💪🧬