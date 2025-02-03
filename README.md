# FairFiltering

## Abstract
- Spurious correlations in training data often lead to biased performance across different groups, posing significant challenges in classification tasks. Current approaches, such as group-based fine-tuning or retraining, rely heavily on group annotations or held-out datasets, making them impractical for real-world scenarios. In this work, we propose a novel dimension filtering method that employs a learnable filtering vector to mitigate spurious correlations within a single-step, end-to-end training process. Unlike existing methods, our approach does not require multiple training stages or explicitly identifying spurious features. By leveraging the linear separability of core and spurious features, our method filters out irrelevant dimensions in the representation without needing additional datasets or group information. This ensures competitive group robustness while maintaining overall accuracy, effectively addressing the inherent trade-off between accuracy and robustness. To quantify this trade-off, we introduce a new evaluation metric that balances improvements in Worst-Group Accuracy (WGA) against reductions in overall accuracy. Experimental results on multiple spurious correlation benchmarks demonstrate the effectiveness of our approach in enhancing WGA with minimal impact on overall accuracy.

---

### Method
![Method](figures/Method2.png) 

---

### Datasets
This repository uses the following datasets for training and evaluation:

1. Waterbirds Dataset
- **Description:** A dataset used for studying distributional robustness, consisting of images of birds labeled as landbirds or waterbirds with spurious correlations.
- **Source:** [Official Waterbirds Dataset](https://github.com/kohpangwei/group_DRO)

2. CelebA Dataset
- **Description:**: A large-scale face attributes dataset containing 200,000+ celebrity images with 40 attribute annotations.
- **Source:** [Official CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

3. CivilComments
- **Description:** A dataset of online comments annotated for toxicity, designed for studying bias and fairness in NLP models.
- **Source:** [CivilComments Kaggle Download](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)
  
4. MultiNLI
- **Description:** A large-scale dataset for natural language inference (NLI), covering multiple genres of text.
- **Source:** [Official MultiNLI Dataset](https://cims.nyu.edu/~sbowman/multinli/)
