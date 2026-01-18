graph TD
    %% ================================
    %% ROOT
    %% ================================
    A["🤖 Artificial Intelligence (AI)<br/>Systems that exhibit intelligent behavior"]
    
    %% ================================
    %% MAJOR PARADIGMS
    %% ================================
    A --> B["⚙️ Symbolic / Declarative AI<br/>Logic, rules, planning, search"]
    A --> C["📊 Probabilistic AI<br/>Reasoning under uncertainty"]
    A --> D["🧩 Machine Learning (ML)<br/>Learn patterns from data"]
    
    %% ================================
    %% SYMBOLIC AI DETAILS
    %% ================================
    B --> B1["Logic (Propositional, FOL)"]
    B --> B2["Ontologies / Knowledge Graphs"]
    B --> B3["Planning (STRIPS, PDDL)"]
    B --> B4["Search (A*, CSP, SAT)"]
    B --> B5["Rule-based / Expert Systems"]
    
    %% ================================
    %% PROBABILISTIC AI DETAILS
    %% ================================
    C --> C1["Bayesian Networks"]
    C --> C2["Hidden Markov Models (HMMs)"]
    C --> C3["Markov Decision Processes (MDPs)"]
    C --> C4["Probabilistic Graphical Models (PGMs)"]
    
    %% ================================
    %% MACHINE LEARNING BRANCHES
    %% ================================
    D --> D1["Supervised Learning"]
    D --> D2["Unsupervised Learning"]
    D --> D3["Self-Supervised Learning"]
    D --> D4["Reinforcement Learning (RL)"]
    D --> D5["Semi-Supervised Learning"]
    D --> D6["Meta-Learning / Few-Shot Learning"]
    
    %% ================================
    %% SUPERVISED LEARNING MODELS
    %% ================================
    D1 --> D1a["Linear / Logistic Regression"]
    D1 --> D1b["Support Vector Machines (SVM)"]
    D1 --> D1c["Decision Trees"]
    D1 --> D1d["Random Forests / Gradient Boosting"]
    D1 --> D1e["k-Nearest Neighbors (k-NN)"]
    D1 --> D1f["Naive Bayes"]
    
    %% ================================
    %% UNSUPERVISED LEARNING MODELS
    %% ================================
    D2 --> D2a["Clustering (k-Means, DBSCAN, Hierarchical)"]
    D2 --> D2b["PCA / Dimensionality Reduction"]
    D2 --> D2c["Autoencoders"]
    D2 --> D2d["Anomaly Detection"]
    
    %% ================================
    %% NEURAL NETWORKS / DEEP LEARNING
    %% ================================
    D --> E["Neural Networks (NN)"]
    E --> E1["Shallow Neural Networks<br/>(Perceptrons, MLPs)"]
    E --> E2["Deep Neural Networks (DNNs)<br/>Deep Learning"]
    
    E2 --> F1["Convolutional Neural Networks (CNNs)"]
    E2 --> F2["Recurrent Neural Networks<br/>(RNN / LSTM / GRU)"]
    E2 --> F3["Transformers"]
    E2 --> F4["Graph Neural Networks (GNNs)"]
    E2 --> F5["Generative Models<br/>(GANs, VAEs, Diffusion, Flows)"]
    
    %% ================================
    %% APPLICATION DOMAINS
    %% ================================
    F1 --> APP1["Computer Vision<br/>(Classification, Detection, Segmentation)"]
    F2 --> APP2["Sequential Modeling<br/>(Time Series, Speech Recognition)"]
    F3 --> APP3["Language & Multimodal<br/>(LLMs, VLMs, Code Gen, ViT)"]
    F4 --> APP4["Graph-Based Tasks<br/>(Molecular Design, Social Networks)"]
    F5 --> APP5["Generative AI<br/>(Image/Video/Audio Synthesis)"]
    
    %% ================================
    %% REINFORCEMENT LEARNING DETAIL
    %% ================================
    D4 --> H1["Model-Free RL"]
    D4 --> H2["Model-Based RL"]
    D4 --> H3["Multi-Agent RL"]
    
    H1 --> H1a["Value-Based (Q-Learning, DQN)"]
    H1 --> H1b["Policy-Based (REINFORCE, PPO)"]
    H1 --> H1c["Actor-Critic (A3C, SAC)"]
    
    H2 --> H2a["Planning with Models"]
    H2 --> H2b["World Models"]
    
    %% ================================
    %% CONCEPTUAL FOUNDATION LAYER
    %% ================================
    C4 -.->|"Conceptual foundation"| C1
    C4 -.->|"Conceptual foundation"| C2
    C4 -.->|"Framework for"| C3
    
    %% ================================
    %% PROBABILISTIC → ML CONNECTIONS
    %% ================================
    C1 -.->|"Probabilistic inference"| D1a
    C1 -.->|"Graphical structure influences"| F4
    C2 -.->|"Historical precursor to"| F2
    C3 -.->|"Formal framework for"| D4
    C3 -.->|"State-action dynamics"| H1
    C3 -.->|"Planning foundation"| H2
    
    %% ================================
    %% SYMBOLIC → ML CONNECTIONS
    %% ================================
    B2 -.->|"Structure for"| F4
    B2 -.->|"Knowledge retrieval for"| APP3
    B3 -.->|"Reward shaping in"| D4
    B3 -.->|"Goal-oriented"| H2
    B4 -.->|"Combined with"| H1a
    B4 -.->|"MCTS + RL"| H2
    B1 -.->|"Reasoning structure for"| APP3
    
    %% ================================
    %% NEURAL NETWORK CROSS-CONNECTIONS
    %% ================================
    E1 -.->|"Applied to"| D1
    E1 -.->|"Applied to"| D2
    E2 -.->|"Applied to"| D1
    E2 -.->|"Applied to"| D2
    E2 -.->|"Enables"| D3
    E2 -.->|"Deep RL uses"| D4
    
    %% ================================
    %% SELF-SUPERVISED LEARNING FLOW
    %% ================================
    D3 -.->|"Pretraining for"| F3
    D3 -.->|"Contrastive methods for"| F1
    D3 -.->|"Masked modeling for"| APP3
    D3 -.->|"Representation learning for"| E2
    
    %% ================================
    %% DEEP RL EXPLICIT CONNECTIONS
    %% ================================
    H1 -.->|"Uses DNNs as approximators"| E2
    H2 -.->|"Uses DNNs for dynamics"| E2
    H2 -.->|"Grounded in"| C3
    H2b -.->|"Learned via"| F5
    
    %% ================================
    %% GENERATIVE MODEL CONNECTIONS
    %% ================================
    F5 -.->|"Pretraining method for"| APP3
    F5 -.->|"Extends autoencoders"| D2c
    
    %% ================================
    %% DIFFUSION-TRANSFORMER CONVERGENCE
    %% ================================
    F3 -.->|"Diffusion Transformers (DiT)"| F5
    
    %% ================================
    %% GNN HYBRID CONNECTIONS
    %% ================================
    F4 -.->|"Operates on"| B2
    
    %% ================================
    %% CLASSICAL ML FOUNDATIONS
    %% ================================
    D1f -.->|"Probabilistic classifier"| C1
    D2c -.->|"Representation learning"| E2
    
    %% ================================
    %% META-LEARNING CONNECTIONS
    %% ================================
    D6 -.->|"Applied to"| APP3
    D6 -.->|"Uses"| E2
    D6 -.->|"Adaptation in"| D4
    
    %% ================================
    %% ALIGNMENT & OPTIMIZATION LAYER
    %% ================================
    I1["🎯 RLHF / DPO / Constitutional AI<br/>(Alignment)"]
    I2["⚡ Optimization & Scaling<br/>(SGD, Adam, Backprop, Quantization, MoE)"]
    I3["🔗 Neuro-Symbolic AI<br/>(Hybrid reasoning)"]
    
    APP3 -.->|"Aligned via"| I1
    I1 -.->|"Uses policy optimization"| H1b
    I1 -.->|"Direct preference"| H1c
    
    I2 -.->|"Enables all gradient-based"| D
    I2 -.->|"Training foundation"| E
    I2 -.->|"Scales"| E2
    I2 -.->|"Efficient inference"| F3
    
    I3 -.->|"Integrates"| B
    I3 -.->|"Integrates"| E2
    F4 -.->|"Example of"| I3
    APP3 -.->|"Emerging in"| I3
    
    %% ================================
    %% MATHEMATICAL FLOW LAYER
    %% ================================
    FLOW1["📐 Representation<br/>(Symbols, Features, Embeddings)"]
    FLOW2["🎲 Uncertainty<br/>(Probability, Belief States)"]
    FLOW3["🔄 Optimization<br/>(Loss/Reward/Posterior)"]
    FLOW4["🎯 Action/Decision<br/>(Output, Policy, Inference)"]
    
    FLOW1 --> FLOW2
    FLOW2 --> FLOW3
    FLOW3 --> FLOW4
    
    B -.->|"Contributes"| FLOW1
    C -.->|"Contributes"| FLOW2
    D -.->|"Contributes"| FLOW3
    E -.->|"Contributes"| FLOW3
    D4 -.->|"Contributes"| FLOW4
    I1 -.->|"Shapes"| FLOW4
    
    %% ================================
    %% LEGEND STYLING
    %% ================================
    classDef paradigm fill:#e1f5ff,stroke:#0077be,stroke-width:3px
    classDef method fill:#fff4e1,stroke:#ff9800,stroke-width:2px
    classDef model fill:#f1f8e9,stroke:#689f38,stroke-width:2px
    classDef application fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef foundation fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef flow fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    
    class A,B,C,D paradigm
    class E,E2,F3,F5,D4 method
    class D1,D2,D3,D5,D6 method
    class APP1,APP2,APP3,APP4,APP5 application
    class I1,I2,I3 foundation
    class FLOW1,FLOW2,FLOW3,FLOW4 flow