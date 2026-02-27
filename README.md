**Summary for GLSurv**

GLSurv is a novel framework for glioma survival prediction that leverages genomic information to guide MRI-based model training and incorporates censor modeling to fully utilize censored samples. It outperforms state-of-the-art survival prediction models on public glioma datasets (UCSF-PDGM and BraTS2020) by exploring cross-modal correlations between MRI and genomics, frequency-spatial feature interactions in MRI, and effective utilization of censored data.

**Overview**

Glioma is characterized by poor prognosis, and while genomic-based models enable accurate survival prediction, their clinical application is limited by invasive sampling. MRI serves as a promising non-invasive alternative but lacks sufficient accuracy when used alone. 
Key challenges in glioma survival prediction include:

Sparse paired MRI-genomics data for glioma patients;
Transformer's high computational cost in MRI analysis and underutilization of frequency-domain heuristic information in MRI;
Scarcity of uncensored samples and neglect of valuable censored samples in existing methods.

**GLSurv addresses these challenges:**

A genomics-guided prompt framework to explore complementary information between unpaired MRI and genomic data;

Frequency-Aware Tetra-Orientated Mamba (FTMamba) to capture correlations between frequency and spatial features in MRI;

Event-Conditional with Genomics-Guided Censor Modeling (EGCM) module to filter reliable censored data and assign genomics-guided survival time, increasing the effective proportion of censored samples.

In inference, GLSurv only relies on the MRI-based module, ensuring clinical practicality as a non-invasive tool.

**Key Contributions**

A genomics-guided training framework to explore cross-modal correlations between unpaired MRIs and genomic data, enhancing non-invasive glioma prognosis;

Frequency-Aware Tetra-Orientated Mamba (FTMamba) to exploit complementary frequency and spatial features in MRI for survival prediction;

Event-Conditional with Genomics-Guided Censor Modeling (EGCM) to filter high-confidence censored data and infer their survival time, improving the utility of censored samples.

**Datasets**
We validate GLSurv on two public glioma datasets:

MRI Dataset:

UCSF-PDGM Dataset (https://www.cancerimagingarchive.net/collection/ucsf-pdgm/)

BraTS2020 Dataset (https://www.med.upenn.edu/cbica/brats2020/data.html)

Gene Dataset:

TCGA-GBM and TCGA-LGG Dataset (https://www.cancer.gov/ccg/research/genome-sequencing/tcga)

CGGA Dataset (https://www.cgga.org.cn/)

Ensure the following packages are installed:

torch ≥ 1.10.0
monai ≥ 0.9.0
torchvision ≥ 0.11.0
scipy ≥ 1.7.0
numpy ≥ 1.21.0
pickle5 ≥ 0.0.11

**Pretrained Models**

GLSurv uses the following pretrained models for feature encoding:

BioClinicalBERT : https://github.com/EmilyAlsentzer/clinicalBERT

scFoundation : https://github.com/biomap-research/scFoundation


**Training**

To train GLSurv on the UCSF-PDGM or BraTS2020 dataset, run the following command:

python main.py --dataset ucsf-pdgm --batch-size 4 --mode train
