# Viability of LLMs for Low-Resource NLP  
### Evaluating Zero-Shot Stance Detection and Uncertainty in Catalan vs. Spanish

---

## Abstract

This study extends the CoAnnotating framework to the task of stance detection in political tweets written in Catalan and Spanish. Using the Catalonia Independence Corpus dataset, we investigate how well ChatGPT can produce annotations in a low-resource language such as Catalan, and how its performance compares to a higher-resource language such as Spanish. In addition, we assess whether uncertainty measures ( entropy across different prompt formulations and self-reported confidence) can help identify predictions that are less reliable. Our results show that ChatGPT achieves moderate accuracy (0.47 for Catalan, 0.49 for Spanish), and that both entropy and confidence scores correlate with errors, suggesting they can help flag uncertain annotations. Overall, our findings suggest that LLMs can assist in stance annotation in multilingual, low-resource settings, but the task remains challenging due to the nuanced nature of political discourse and limitations in existing datasets.


---

## Research Questions

1. How accurately can ChatGPT annotate stance in Catalan and Spanish tweets compared to gold-standard labels?  
2. To what extent do uncertainty metrics (entropy and confidence) predict annotation errors and support annotation reliability?  
3. How do different prompt formulations influence annotation quality across languages?

---

## Methodology

- **Model:** GPT-4o-mini (OpenAI API)  
- **Framework:** Adaptation of CoAnnotating (Li et al., 2023)  
- **Dataset:** [Catalonia Independence Corpus](https://aclanthology.org/2020.lrec-1.171/)  
- **Languages:** Catalan 🇨🇦 and Spanish 🇪🇸  
- **Uncertainty Metrics:**  
  - *Entropy* across multiple prompt formulations  
  - *Self-reported confidence scores*  
- **Evaluation:** Accuracy, per-class F1, entropy-error correlation, and per-prompt analysis

---

## Key Results

| Metric | Catalan | Spanish |
|--------|----------|----------|
| Overall Accuracy | 0.47 | 0.49 |
| Mean Entropy | 0.43 | 0.45 |
| Mean Confidence | 0.78 | 0.80 |
| Entropy–Error Correlation | ρ = 0.37 | ρ = 0.38 |

**Findings:**
- ChatGPT shows moderate stance classification accuracy for Catalan and Spanish.  
- Entropy correlates strongly with errors, confirming its utility for identifying unreliable annotations.  
- Confirmation-bias prompts achieved slightly higher accuracy across languages.  
- Performance was consistent across Catalan and Spanish, likely due to linguistic similarity and task complexity.

---

## Repository Structure

```text
├── data/                          
│   ├── annotations/               
|   │   ├── annotations_chatGPT/        # raw ChatGPT annotation outputs       
|   │   └── annotations_parsed/         # annotations parsed into numeric stance label
│   └── tweets/                    
|       ├── CAT_dataset/                # Catalan dataset (original + sampled subset)
|       ├── ES_dataset/                 # Spanish dataset (original + sampled subset)                    
|       └── input_for_chatgpt/          # subsets with added prompt text, ready for ChatGPT annotation
├── notebooks/                          # data retrieval and preprocessing
|   ├── get_dataset.ipynb           
|   └── get_prompts.ipynb 
├── scripts/                       
|   ├── get_annotations_chatGPT.py      # send prompts to the ChatGPT API to generate annotations
|   ├── parse_annotations.py            # parse ChatGPT’s text outputs into structured stance labels (0, 1, 2)
│   └── evaluation/                
|       ├── error_analysis.py           # compare accuracy across different prompt formulations
|       ├── evaluate_confidence_variants.py     # reliability evaluation of confidence score 
|       ├── evaluation.py                       # compute accuracy, F1, entropy correlations, and generates summary plots
|       ├── final_plots.py                      # final figures and summary reports (entropy, per-class metrics, etc.)
|       └── visualize_chatgpt_performance.py    # generate visual comparisons of model performance across classes and prompts
└── README.md              
```

# Usage 
1. Clone the repository:
   ```bash
   git clone https://github.com/eraimu/CoAnnotating-Catalan/git
   cd CoAnnotating-Catalan

2. Install dependencies
   ```bash
   pip install -r requirements.txt

3. Run annotation or analysis scripts. The repository includes several scripts under the scripts/ and notebooks/ directories.
These files contain the functions used for annotation, and analysis, but they are not standalone executables. Users can explore or repurpose the provided functions to reproduce the analysis or extend it with new data.

## Citation 
If you use this repository or refer to this work, please cite it as:

Raimundo Schulz, Emma (2025). Viability of LLMs for Low-Resource NLP: Evaluating Zero-Shot Stance Detection and Uncertainty in Catalan vs. Spanish.
Unpublished manuscript. GitHub repository: https://github.com/eraimu/CoAnnotating-Catalan

```text
@misc{raimundo2025stance,
  author = {Emma Raimundo Schulz},
  title = {Viability of LLMs for Low-Resource NLP: Evaluating Zero-Shot Stance Detection and Uncertainty in Catalan vs. Spanish},
  year = {2025},
  note = {Unpublished manuscript},
  howpublished = {\url{https://github.com/eraimu/CoAnnotating-Catalan}}
}
```
