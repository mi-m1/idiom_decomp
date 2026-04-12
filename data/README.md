## Instructions

Please save a copy of the Bulkes et al., (2017) dataset in `data/bulkes/`. Please refer to the original paper for access.

For the IMPLI dataset, refer to the original paper for access instructions. If you use IMPLI, save it in `data/acl2022-impli/`.

If either dataset is used, please cite the original authors.


```Citation
@Article{Bulkes2017,
author={Bulkes, Nyssa Z.
and Tanner, Darren},
title={``Going to town'': Large-scale norming and statistical analysis of 870 American English idioms},
journal={Behavior Research Methods},
year={2017},
month={Apr},
day={01},
volume={49},
number={2},
pages={772-783},
abstract={An idiom is classically defined as a formulaic sequence whose meaning is comprised of more than the sum of its parts. For this reason, idioms pose a unique problem for models of sentence processing, as researchers must take into account how idioms vary and along what dimensions, as these factors can modulate the ease with which an idiomatic interpretation can be activated. In order to help ensure external validity and comparability across studies, idiom research benefits from the availability of publicly available resources reporting ratings from a large number of native speakers. Resources such as the one outlined in the current paper facilitate opportunities for consensus across studies on idiom processing and help to further our goals as a research community. To this end, descriptive norms were obtained for 870 American English idioms from 2,100 participants along five dimensions: familiarity, meaningfulness, literal plausibility, global decomposability, and predictability. Idiom familiarity and meaningfulness strongly correlated with one another, whereas familiarity and meaningfulness were positively correlated with both global decomposability and predictability. Correlations with previous norming studies are also discussed.},
issn={1554-3528},
doi={10.3758/s13428-016-0747-8},
url={https://doi.org/10.3758/s13428-016-0747-8}
}


```

```Citation
@inproceedings{stowe-etal-2022-impli,
    title = "{IMPLI}: Investigating {NLI} Models' Performance on Figurative Language",
    author = "Stowe, Kevin  and
      Utama, Prasetya  and
      Gurevych, Iryna",
    editor = "Muresan, Smaranda  and
      Nakov, Preslav  and
      Villavicencio, Aline",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.369/",
    doi = "10.18653/v1/2022.acl-long.369",
    pages = "5375--5388",
    abstract = "Natural language inference (NLI) has been widely used as a task to train and evaluate models for language understanding. However, the ability of NLI models to perform inferences requiring understanding of figurative language such as idioms and metaphors remains understudied. We introduce the IMPLI (Idiomatic and Metaphoric Paired Language Inference) dataset, an English dataset consisting of paired sentences spanning idioms and metaphors. We develop novel methods to generate 24k semiautomatic pairs as well as manually creating 1.8k gold pairs. We use IMPLI to evaluate NLI models based on RoBERTa fine-tuned on the widely used MNLI dataset. We then show that while they can reliably detect entailment relationship between figurative phrases with their literal counterparts, they perform poorly on similarly structured examples where pairs are designed to be non-entailing. This suggests the limits of current NLI models with regard to understanding figurative language and this dataset serves as a benchmark for future improvements in this direction."
}```