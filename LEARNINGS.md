# Project Learnings: AI Music Mentor

_Reflections on building a multi-modal AI system for music production feedback_
_Developed during Data Science Retreat bootcamp (September 2024)_
_Written: October 7, 2025_

## Overview

This project was an ambitious attempt to combine audio arrangement processing (CRNN), semantic search (RAG), and LLM-generated feedback into a single system that could help music producers overcome writer's block with smart arrangement ideas and production feedback. The one-month timeline meant making pragmatic trade-offs between scope and depth. Here's what I learned.

---

## 1. Data Survey & Initial Decisions

**Challenge:** I conducted a 10-person survey within my local music network to understand how producers engage with AI tools and what they'd want from a production assistant. My initial concept was to base all feedback on one professional producer's style, creating a consistent voice and tone. However, the research revealed that while subjective feedback was helpful, it wasn't the core selling point. Interestingly, branding the tool with different producers did increase perceived value.

The survey identified three key priorities: actionable feedback, creative direction, and technical analysis. Security concerns also emerged, which influenced my decision to use local LLMs rather than OpenAI's API to avoid potential data leakage of users' music. To make the project feasible within a one-month timeline, I scoped it to techno music and limited the musical concepts analyzed to primarily arrangement, rhythm, and frequency characteristics.

**Learning:** The survey validated that the idea was novel and had genuine interest, but executing it well would be challenging. With more time, I would have conducted a larger survey to gather deeper insights into feature prioritization. The research helped me focus on the right problems, even if the solutions needed more iteration than the bootcamp timeline allowed.

---

## 2. Audio Feature Extraction

**Approach:** Feature extraction served three purposes in this project: (1) vector embeddings for cosine similarity search, (2) audio descriptions for LLM prompts (both input and reference tracks), and (3) input features for CRNN training and inference.

For the CRNN, I used the same 15 features from the base model's original training to avoid compatibility issues and leverage its existing chorus-detection capabilities. These features were extracted using Librosa and covered rhythm, harmony, energy, spectral, and frequency characteristics.

For the vector search, I initially tested a minimal approach with just 5 features (one representing each musical dimension: rhythm, harmony, energy, spectral, and frequency). Testing revealed that 15 features produced better similarity matches, and expanding to 19 features improved results further.

**Challenges:**

- **Normalization issues:** Getting feature scaling right was tricky initially. Spectral brightness dominated the similarity calculations due to its larger numerical scale, causing most tracks to match regardless of actual musical similarity. I had to expand my dataset and experiment with different normalization strategies to balance the feature contributions.

- **Global vs. chunked features:** The CRNN used chunked audio for segmentation prediction, but the vector search and prompt descriptions relied on global (track-average) features. This limitation meant structurally different songs could match if their averaged features were similar. For example, a track with a long ambient intro and energetic drop might match a consistently mid-energy track because the averaging masked the structural differences.

**Learning:** The global feature approach was pragmatic for the one-month timeline but has clear limitations. A better architecture would chunk songs into their structural sections (intro, verse/A, chorus/B, bridge/C) and create dedicated embeddings for each section. This would enable more precise matching: "your chorus sounds like this artist's chorus" rather than "your whole track averages out similarly to this other track." I did include the predicted arrangement structure in the main LLM prompt as a workaround to provide some broader structural comparison, but this was crude compared to section-specific embeddings with detailed musical information per section.

---

## 3. CRNN Fine-tuning for Arrangement Classification

**Base Model Acknowledgment:**
This work builds upon Dennis Dang's CRNN architecture originally designed for chorus detection ([original repo](https://github.com/dennisvdang/chorus-detection)). I adapted his base model and fine-tuned it for arrangement section classification, modifying the classification head to support four classes instead of binary classification. The core CNN + Bidirectional LSTM architecture remained largely intact, which provided a solid foundation for this music information retrieval task.

**Data & Labeling:**
I fine-tuned the CRNN using 30 techno tracks that I manually labeled into four arrangement sections: O (intro/outro), A (verse), B (chorus/drop), and C (bridge/breakdown). Each track was segmented into roughly 10-15 sections of 40 seconds or less, yielding approximately 300-450 labeled segments. Since segment lengths varied, I used zero-padding and masking to standardize inputs for batch training.

This was significantly less data than the base model's training set (332 songs), but I labeled mine more thoroughly. The original model used binary classification (chorus vs. non-chorus), while I expanded to four classes to capture more nuanced arrangement structures common in electronic music.

**Augmentation Strategy:**
To address the limited dataset, I experimented with augmentation techniques:

- **Pitch shifting:** Shifted each track up/down by semitones. This _decreased_ accuracy significantly, likely because pitch information is crucial for identifying arrangement sections in techno (e.g., bassline presence, harmonic changes).
- **Time stretching:** Changed playback speed while preserving pitch (±10-15%). This worked much better, improving accuracy by roughly 10% and expanding the dataset 4x to approximately 1,200 augmented segments.

The time stretching success makes sense: arrangement structure is more about rhythm and energy patterns than absolute tempo, so tempo variations didn't confuse the model while still providing useful training diversity.

**Handling Class Imbalance:**
The four classes weren't evenly distributed in my dataset (more A/B sections than O/C sections), so I used class weighting during training to account for this. This improved per-class recall and precision metrics but had minimal effect on overall accuracy.

**3 vs. 4 Classes Decision:**
During final validation, I tested both the 4-class model and a simplified 3-class version (combining O and A sections). While the 3-class model showed slightly better quantitative metrics, practical testing on 5 real tracks revealed the 4-class model provided better structural analysis for feedback generation, even if section boundaries were occasionally off by 5-10 seconds. I chose the 4-class model because accurate section identification mattered more for the end-user experience than marginal metric improvements.

**Architecture Experimentation:**
With the final augmented dataset and 4-class structure, I experimented with three classification head architectures:

1. Simple: 128-node dense layer + softmax
2. Regularized: 128-node dense layer + dropout + softmax
3. Complex: 256 → 128-node dense layers + dropout + softmax

The more complex head (option 3) performed 3% better on average, which makes sense given the increased number of classes (4 vs. the base model's 2) required more representational capacity.

---

## 4. RAG System & LLM Integration

**Similarity Search Approach:**
I experimented with three distance metrics for embedding similarity: dot product, cosine similarity, and Euclidean distance. Through extensive manual testing and subjective evaluation (comparing retrieved tracks to input tracks), cosine similarity consistently selected the most musically similar tracks. While difficult to quantify numerically, my domain knowledge as a producer made the superiority of cosine obvious. The global embedding approach had limitations (discussed in Feature Extraction), but proved sufficient for this demo.

**Initial LLM Struggles:**
The first implementation used Llama 3.2 (smaller model) for feedback generation. As the prompt complexity grew, the model struggled with:

- Hallucinations (inventing arrangement sections that didn't exist)
- Inconsistent output quality
- Poor understanding of arrangement patterns (ABCO structure)
- Difficulty expanding on arrangement concepts

I upgraded to Qwen3, which has nearly 3x the parameters and includes a "thinking" mechanism, paired with a Goal-Oriented Task (GOT) structured output approach.

**Goal-Oriented Task (GOT) Structure:**
I structured the output as a graph where each arrangement section (O, A, B, C) is a node, and the LLM suggests edges representing actions: "add section," "extend section," "adjust energy," "modify groove," "mixing advice," or "suggest classic track for arrangement inspiration." This provided clear structure and made the output more actionable.

**Prompt Engineering Journey:**
Prompt engineering proved more art than science. Key challenges and solutions:

- **Grounding the model:** Added strict rules to only reference the input track (preventing hallucinations about non-existent sections) and included examples of effective ABCO patterns from classic techno tracks directly in the prompt.

- **Feature integration:** Initially, including numerical audio features (spectral brightness, tempo, etc.) resulted in the LLM regurgitating numbers rather than actionable advice. I solved this by mapping feature values to descriptive words (e.g., "high energy" instead of "0.87") and enforcing this mapping strictly in the prompt.

- **Feedback filtering problem:** RAG retrieval returned similar tracks based on cosine similarity, but those tracks didn't necessarily have the same production issues. A track similar in overall features might have completely different feedback needs. I added a filtering step using Llama 3.2 to evaluate each retrieved feedback against the user's query (e.g., "Does this feedback help extend my arrangement?"). Only the top 2 relevant feedbacks were passed to the final prompt. This significantly improved output quality although was not perfect, EQ issues seemed to work the best for global features, including our predicted arrangement patterns in the final prompt also helped a lot here to get around the limitation of global embeddings.

- **Scope limitation:** To prevent confusion, I limited feedback domains to three categories: general production advice, EQ/mixing, and arrangement ideas.

**Evaluation Approach:**

**Personal evaluation:** LLM output quality is inherently subjective, but early iterations had obvious issues (hallucinations, poor structure, excessive examples). I created a rubric assessing: audio description accuracy, practical advice quality, relevance to input track, truthfulness, and technical understanding. Using 5 diverse song pairs as test cases, I evaluated each iteration as I adjusted prompts, data, and LLM choice. Quality improved from roughly 45% to 67% across iterations—far from perfect, but a substantial improvement for helping producers overcome writer's block.

**External validation:** I conducted a blind test with 2 producer friends, comparing our system's output against ChatGPT for 2 song pairs. Our system scored 68% approval vs. ChatGPT's 58%, primarily because ChatGPT provided generic advice without understanding the specific arrangement structure of the input track. I wanted to also test this against human output but ran out of time for this comparison.

**Key Limitation Discovered:**
My RAG dataset included sketches, half-finished tracks, and nearly complete songs. Producers seek different advice at these different stages (e.g., "extend this idea" vs. "polish the mix"), but I initially treated all songs identically. This caused confusion in prompt engineering and should have been addressed with stage-specific prompts or separate workflows.

**Learning:**
Several critical insights emerged: (1) LLM size dramatically impacts reasoning ability and instruction-following—larger models are worth the computational cost for complex tasks. (2) Global cosine similarity is useful but limited, especially for arrangement feedback where section-level detail matters; chunked embeddings comparing intro-to-intro or drop-to-drop would be more effective. (3) Injecting my personal producer "voice" into the feedback proved difficult. While prompting achieved this to some extent, it wasn't strongly apparent. A larger, curated feedback dataset might help, though I'm uncertain whether RAG can truly capture the subjective, stylistic nuances I was aiming for—this remains an open question.

---

## 5. Time Constraints & Trade-offs

This was a one-month bootcamp capstone project, which meant balancing ambition with reality. Several aspects were compressed or simplified due to time pressure:

**What was rushed:**

- **Dataset size:** Only 30 labeled tracks for CRNN fine-tuning. A production system would need 300+ tracks with diverse arrangement styles. Only 50 feedback examples were used in the RAG setup too, this gave severe limitations on which domains of feedback the system could accurately output.
- **Validation methodology:** Used train/test split without proper validation set due to limited labeled data and time constraints. Something I plan to expand on soon when I fine-tune again on a larger dataset.
- **Prompt iteration:** Spent most time on core functionality; could have refined prompts further with more user testing.
- **Feature engineering:** Stuck with global features rather than implementing the more sophisticated chunked embedding approach I knew would work better, I had planned this in but ran out of time given the prompting was a time sink hole.

**What suffered:**

- **Evaluation rigor:** Subjective eval with small sample size (5 songs personally, 2 songs with external reviewers). A proper study would need 50+ tracks with multiple evaluators, but this was enough to have some small albeit flawed benchmark for my demo presentation.
- **Edge cases:** Focused on "typical" techno tracks; didn't test extensively on experimental or genre-bending productions.
- **UI/UX polish:** Built functional interface but didn't have time for refinement or user experience optimization.

**What I'm proud of despite constraints:**

- **End-to-end integration:** Successfully combined three complex AI systems (CRNN, RAG, LLM) into a working pipeline.
- **Practical decision-making:** Chose 4-class model over 3-class based on real-world utility, not just metrics.
- **Domain-driven design:** Leveraged my music production knowledge to make smart architectural choices and evaluate quality effectively.
- **Honest evaluation:** Recognized limitations and measured improvement systematically (45% → 67% quality score).
- **Effective fine-tuning with limited data:** While I didn't hit 70% accuracy across all segmentation classes, the CRNN performed surprisingly well given the small dataset (30 tracks) and produced useful arrangement predictions for feedback generation. This proved that thoughtful augmentation and architecture choices can compensate for data constraints—and made me excited about what's possible with more dedicated training time.

**Key takeaway:** The project successfully demonstrated that AI can provide contextual, personalized feedback to music producers, but each component could be significantly improved with more iteration. The one-month constraint forced me to focus on proving the concept rather than perfecting the execution—a valuable lesson in MVP thinking.

---

## 6. What I'd Do Differently

**Short-term improvements (1-2 weeks of work):**

- **Proper validation split:** Collect more labeled data and implement train/val/test split with cross-validation to better assess model generalization.
- **Section-specific embeddings:** Chunk audio by predicted arrangement sections and create dedicated embeddings for each, enabling "verse-to-verse" comparisons instead of global track averaging.
- **LLM-as-evaluator:** Implement automated feedback quality assessment using an LLM judge to evaluate output before showing to users, reducing manual evaluation burden.
- **Stage-aware prompts:** Create different prompt templates for sketches, half-finished tracks, and nearly-complete productions, addressing the dataset heterogeneity issue.

**Long-term improvements (1-2 months of work):**

- **Larger training dataset:** Collect 200-300 labeled tracks across diverse techno subgenres (minimal, industrial, melodic) to improve CRNN generalization.
- **Custom embedding model:** Fine-tune a music-specific embedding model on production feedback tasks rather than using generic audio features.
- **A/B testing framework:** Build infrastructure to measure whether feedback actually helps producers improve their tracks (the ultimate success metric).
- **Agentic workflow for context-aware feedback:** Implement a multi-step LLM process that assesses track state before generating advice. For nearly-finished tracks, the agent might determine feedback isn't needed, or ask clarifying questions about the producer's goals rather than blindly following prompts. This could also intelligently decide which sections to compare (e.g., matching A-to-A sections when arrangement extension is requested).
- **Enhanced music understanding with Essentia:** Integrate Essentia's advanced music analysis for mood detection and instrument classification. This would provide richer context for LLM feedback generation, enabling more musically-aware and specific production advice.
- **User feedback loop:** Implement thumbs-up/down on AI suggestions to continuously improve the system based on real user preferences.
- **Deploy with secure authentication:** Deploy the system with secure authentication to enable broader user testing and gather real-world feedback for iteration.

**Architectural rethink:**
If starting over, I'd reverse the development order: begin with chunked, section-specific embeddings from day one rather than retrofitting them later. The global feature approach created technical debt that rippled through the RAG and LLM components.

---

## 7. Key Technical Learnings

**Multi-modal systems require careful orchestration:**
Integrating audio ML + semantic search + LLM isn't just bolting three models together. Each component's output becomes another component's input, so errors compound. I spent significant time debugging issues where the CRNN's mislabeled section caused RAG to retrieve irrelevant feedback, which then confused the LLM. Building robust error handling, debugging, and fallbacks for each stage is critical.

**Prompt engineering deserves serious time investment:**
I initially underestimated this, thinking "I'll just write a clear prompt." In reality, I spent more time iterating on prompts than training models. Small changes—like mapping numerical features to words or adding specific examples—had massive impact on output quality. This isn't just "prompt hacking"; it's a core engineering skill for LLM systems.

**Domain knowledge is your superpower:**
My 15+ years of music production made this project possible. I could evaluate whether feedback was useful, understand why certain features mattered for arrangement detection, and design the GOT structure around how producers actually think. Without this domain expertise, I would have optimized for metrics that didn't correlate with real-world usefulness.

**Every architectural decision has trade-offs:**

- Global vs. chunked features: Faster to implement vs. more accurate
- 3 vs. 4 classes: Better metrics vs. better user experience
- Small vs. large LLM: Faster inference vs. better reasoning
- Local vs. cloud LLM: Privacy vs. performance

There are no universally "correct" choices—only choices appropriate for your constraints and goals. Understanding these trade-offs explicitly made my decision-making more confident.

**Evaluation is harder than building:**
For creative AI systems, quantitative metrics don't tell the full story. A model with 80% accuracy might produce useless feedback, while 67% accuracy with thoughtful prompt engineering can be genuinely helpful. I learned to combine quantitative metrics (precision/recall), qualitative assessment (does this make sense?), and user validation (would a producer find this useful?) to get a complete picture.

---

## Conclusion

Despite time constraints and numerous technical challenges, this project achieved its core goal: demonstrating that AI can provide contextual, personalized feedback to music producers struggling with writer's block. The system works—not perfectly, but well enough to validate the concept and identify clear paths for improvement.

More importantly, I learned how to architect, build, and evaluate a complex multi-modal AI system from scratch. I navigated the messy reality of production ML: limited data, imperfect models, subjective evaluation criteria, and constant trade-offs between competing priorities. These are the skills that don't show up in tutorials but define real-world ML engineering.

The project also reinforced that technical sophistication matters less than solving real problems for real users. My producer friends didn't care about CRNN architecture or RAG retrieval metrics—they cared whether the feedback helped them finish their tracks. That user-centric perspective will guide all my future AI work.

Looking back, I'm proud of what I built in one month while learning some of these technologies for the first time. The imperfections aren't failures—they're documented learning opportunities and a roadmap for version 2.
