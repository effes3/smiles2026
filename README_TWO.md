### 1. The "ICR Probe" Method (Dynamics instead of Statics)

**Paper:** *ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs* (ACL, July 2025)
**Core idea:** The authors proved that searching for hallucinations solely using "static" hidden states (just taking a single layer) works poorly. When an LLM hallucinates, the process of updating its knowledge in the residual stream (between layers) becomes chaotic. They introduced the **ICR (Information Contribution to Residual Stream)** metric.

### 2. The "INSIDE / Eigenscore" Method (Spectral Token Analysis)

**Papers:** *INSIDE: LLMs' internal states retain the power of hallucination detection* (ICLR 2024) and *Hallucination Detection in LLMs Using Spectral Features* (February 2025).
**Core idea:** The authors examined the hidden state matrix for all generated response tokens. They discovered that during a hallucination, the latent space "spreads out". The best way to measure this is to calculate the covariance matrix of the token vectors and extract its **eigenvalues**.

### 3. "Representation Engineering (RepE)" Method (Truth Direction)

**Papers:** *Taxonomy of Representation Engineering for LLMs* (March 2025) and *SHARP: Steering Hallucination in LVLMs* (November 2025).
**Core idea:** Models contain an internal linear direction (vector) that corresponds to the concept of "truth / falsehood". This vector can be easily calculated using the "Difference-in-Means" method.

### 4. "Semantic Entropy Probes (SEP)" / SAPLMA Method

**Papers:** *Semantic Entropy Probes: Robust and Cheap Hallucination Detection* (2024) and *Do LLMs Know about Hallucination?* (February 2024 / March 2025).
**Core idea:** Studies show that the model "realizes" it doesn't know the answer as early as the prompt reading stage. The $s_1$ token (the last token of the prompt) contains more information about an upcoming hallucination than the generated tokens themselves.