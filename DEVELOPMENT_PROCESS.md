# Development Process and AI Usage

This document describes the development process behind the project, including the AI tools used, selected prompts, key technical decisions, unsuccessful approaches, and subsequent iterations.

It aims to provide transparency into how AI-assisted development was used and how the resulting suggestions were evaluated and refined.

## AI Tools

| Tool | Purpose |
|---|---|
| Gemini Pro | Experiment design, technical consultations, and iterative code refinement |
| ChatGPT | Review of experimental results and comparison of interpretations |

AI-generated suggestions were treated as proposals rather than unquestioned conclusions. Technical decisions were refined based on the project's requirements, observed results, and practical limitations.

## Prompts

The following examples document selected prompts used during development. They illustrate how AI tools were used to design experiments and interpret the resulting explanations.

### 1. Experiment Design and Code Development

**Prompt: Initial explainability analysis**

```text
You are an expert in artificial intelligence, Graph Neural Networks (GNNs),
explainability, neuroscience, and EEG signal analysis for epileptic seizure
detection.

You have access to a trained ChebConv-based model with the following
architecture:

[model]

The model was trained on the CHB-MIT dataset using the following
patient-level split: patients 1–20 for training, patients 21–22 for
validation, and patients 23–24 for testing.

The model takes graphs constructed using Phase Locking Value (PLV)
[graph construction script]. The graph nodes correspond to EEG channels
and contain handcrafted features commonly used in epileptic seizure
detection [feature extraction script].

The features are computed over 6-second EEG segments. Each graph is
assigned a binary label: 0 for non-ictal segments and 1 for ictal
segments.

Design a comprehensive and technically sound explainability analysis
for this model that can be presented in a Jupyter Notebook.

Consider the following project-specific materials:

- [Mini-project description]
- [AGENTS.md]
- [PROCESS.md]

Explain the rationale behind the proposed methods and their limitations.
Prioritize analyses that provide meaningful insight into the model's
decision-making rather than producing visualizations without a clear
interpretation.
```

**Context:** This prompt was used at the beginning of the project to establish an initial analysis plan. The proposed experiments were subsequently refined through iterative development.

**Prompt: Choosing data for explainability analysis**

```text
Which data should be used for model explainability analysis: training
data or test data?

Consider the goal of evaluating whether the model's explanations
generalize beyond the samples used to fit the model.
```

**Context:** The initial suggestion favored training data. After this question was raised, the model revised its recommendation and emphasized the use of held-out test data for evaluation.

For an unbiased assessment, explanations should primarily be evaluated on held-out test samples. Training samples can still be useful for debugging and exploratory analysis, provided that these uses are distinguished from the final evaluation.

### 2. Result Interpretation

**Prompt: Reviewing experimental results**

```text
What do you think about these results?

They were generated using the following code:

[code]

The task is epileptic seizure detection using EEG signals, with 6-second
epochs, a ChebConv-based Graph Neural Network, handcrafted features,
and explainability analysis using GNNExplainer.

Analyze the results critically. Distinguish between observations directly
supported by the visualizations and hypotheses that would require
additional experiments to validate.

What conclusions can reasonably be drawn, and what are the main
limitations of this analysis?
```

**Context:** The purpose was to compare the AI-generated interpretation with the conclusions that initially emerged from inspecting the results.

## Key Technical Decisions

### 1. Replacing SHAP with GNNExplainer

The initial approach considered using conventional SHAP-based feature attribution. However, explaining a graph neural network introduces additional challenges because the model operates on graph structures as well as node features.

GraphSHAP was also considered, but GNNExplainer was ultimately selected as a method specifically designed to identify relevant subgraphs and feature information for graph neural network predictions.

This change made it possible to investigate both node-feature importance and the relevance of graph connections.

### 2. Prioritizing Feature Importance Analysis

The analysis was increasingly focused on handcrafted feature importance because the observed feature-importance patterns were more stable than the edge-importance visualizations in the initial experiments.

This allowed for a more consistent comparison of model behavior across different prediction categories, including true positives, true negatives, false positives, and false negatives.

Edge-importance analysis was retained as a complementary component, with its limitations taken into account.

## Challenges and Approaches That Did Not Work as Expected

### 1. Limitations of the Initial SHAP-Based Approach

In the first iteration, the LLM proposed experiments based on conventional SHAP, following the initial project requirements.

However, the proposed analysis did not provide the desired level of interpretability for this graph-based architecture, and some of the suggested experiments lacked a sufficiently clear connection to clinically meaningful questions.

The approach was therefore reconsidered, and GNNExplainer was selected to better accommodate the graph structure of the model.

### 2. Instability of Edge-Importance Explanations

During the initial experiments, the edge-importance visualizations generated by GNNExplainer sometimes differed across repeated runs on the same sample when using 200 optimization epochs.

Increasing the number of epochs to 1,000 substantially improved the observed stability, but also increased the computational cost of generating explanations.

As a result, greater emphasis was placed on handcrafted feature importance, which appeared more stable under the tested configurations and was more practical for comparing model predictions across multiple samples.

These observations motivated a stronger focus on feature-based analysis, while retaining edge importance as a complementary avenue for investigation.

## Development Iterations

1. **v1 — Initial SHAP-based approach:** An initial attempt to apply SHAP-based explanations to the trained model.
2. **v2 — Transition to GNNExplainer:** Adoption of a graph-oriented explanation method, followed by experiments investigating the usefulness and limitations of its outputs.
3. **Subsequent iterations — Feature importance and error analysis:** More detailed investigation of feature importance across different prediction outcomes, together with further analysis of edge-importance stability.

## Reflections

The development process highlighted several practical considerations for explainable graph neural networks:

- The choice of explanation method should reflect the structure and behavior of the model being analyzed.
- Explanation stability matters: visually plausible results are not necessarily reproducible.
- Increasing the optimization budget can improve stability but introduces additional computational costs.
- Feature importance and edge importance provide complementary perspectives and should not be treated as interchangeable.
- Model explanations can generate hypotheses about decision-making, but they do not independently establish causal relationships or clinical validity.

The final analysis should therefore be interpreted as an investigation of model behavior rather than a clinical validation of an EEG-based seizure detection system.