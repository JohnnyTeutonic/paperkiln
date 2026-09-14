# Literature scoping for the transfer_s2 paper

Scope: does the comparative structure of a six-lane attention panel (exact vs sliding-window 16/32/64/128, with and without a sink token) transfer across width in a tiny GPT-2-style family, and what caught the false positive of Study 1. Target venue: TMLR.

Every item below was checked on 14 September 2026 against the arXiv abstract page, the publisher or proceedings page, or an OpenReview or ICLR proceedings record. Venue is given as it appears on the verified page; where only an arXiv record was checked, the venue is listed as arXiv. Items suspected but not confirmed are listed at the end and must not be cited.

Notation: for each item, one sentence on what it shows, then one sentence on how our paper relates (agrees, extends, contradicts, positions).

---

## 1. Hyperparameter transfer across width and scale

- Yang, Hu, Babuschkin, Sidor, Liu, Farhi, Ryder, Pachocki, Chen and Gao (2022). *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer*. NeurIPS 2021. https://arxiv.org/abs/2203.03466
  Shows that under the Maximal Update Parametrisation (muP) many optimal hyperparameters, the learning rate above all, are stable across width, so they can be tuned on a small model and transferred. Our study trains in standard parametrisation, where the optimal learning rate does shift with width; this paper is the canonical statement of why a single fixed rate across widths is a confounder, and positions our Study 2 per-width rule as the standard-parametrisation alternative.

- Yang and Hu (2021). *Feature Learning in Infinite-Width Neural Networks*. ICML 2021 (Tensor Programs IV). https://arxiv.org/abs/2011.14522
  Establishes the parametrisation dichotomy (feature learning vs kernel regime) that makes muP possible. Cite for the theoretical origin of width-dependence of the optimal learning rate; our paper does not use the theory, only its practical consequence.

- Everett, Xiao, Wortsman, Alemi, Novak, Liu, Gur, Sohl-Dickstein, Kaelbling, Lee and Pennington (2024). *Scaling Exponents Across Parameterizations and Optimizers*. ICML 2024. https://arxiv.org/abs/2407.05872
  Trains tens of thousands of models and shows that all common parametrisations, not only muP, can achieve learning-rate transfer once the per-layer learning-rate exponents are set correctly, and that the optimal rate otherwise moves with width. Agrees with our premise that a fixed learning rate across widths is mistuned for at least one width; our paper is the small-scale, methodology-focused corollary.

- Yang, Yu, Zhu and Hayou (2023). *Tensor Programs VI: Feature Learning in Infinite-Depth Neural Networks*. arXiv. https://arxiv.org/abs/2310.02244
  Extends transfer to depth (Depth-muP) and notes that transformer blocks with multiple layers break the simple depth result. Positions our choice to vary width only, with depth fixed, as the regime where transfer results are best understood.

- Bordelon, Noci, Li, Hanin and Pehlevan (2024). *Depthwise Hyperparameter Transfer in Residual Networks: Dynamics and Scaling Limit*. ICLR 2024. https://arxiv.org/abs/2309.16620
  Shows that muP plus 1/sqrt(depth) residual scaling gives joint width-and-depth transfer of optimal hyperparameters on ResNets and ViTs. Cite alongside TP VI to position why our study holds depth fixed.

- Lingle (2024). *An Empirical Study of muP Learning Rate Transfer*. arXiv (v6, 2025). https://arxiv.org/abs/2404.05728
  Empirically checks muP learning-rate transfer in transformers and finds near-optimal transfer in most settings, with some architectural details where it underperforms. Closest empirical cousin on the transfer side; our paper asks the complementary question (does the comparison transfer, not the hyperparameter) and finds that it does not once the rate is tuned per width.

- Kosson, Welborn, Liu, Jaggi and Chen (2025). *Weight Decay may matter more than muP for Learning Rate Transfer in Practice*. ICLR 2026. https://arxiv.org/abs/2510.19093
  Argues that in practical training the transfer of the optimal learning rate across width is driven more by weight decay than by muP. Must cite as the current critique of the muP account; relevant because our per-width rule is agnostic about the mechanism and simply measures the optimum.

- Dey, Zhang, Noci, Li, Bordelon, Bergsma, Pehlevan, Hanin and Hestness (2025). *Don't be lazy: CompleteP enables compute-efficient deep transformers*. NeurIPS 2025. https://arxiv.org/abs/2505.01618
  Introduces CompleteP, a parametrisation giving width-and-depth transfer in transformers while avoiding lazy learning, with 12 to 34 percent compute gains. Positions the alternative route (adopt a transferable parametrisation) that a follow-up to our study could take instead of per-width tuning.

- Li, Zheng, Wang, Zhang, Wang, Xuyang, Fan, Ding, Wang, Ding, Zhou, Zhang and Jiang (2025). *Predictable Scale: Part I, Step Law: Optimal Hyperparameter Scaling Law in Large Language Model Pretraining*. arXiv. https://arxiv.org/abs/2503.04715
  Fits a power law for the optimal learning rate in model size N and data size D, with optimal batch size depending mainly on D. Directly supports the claim that the optimum shifts with size; our per-width selection is a local, mechanical version of the same observation at toy scale.

- Shallue, Lee, Antognini, Sohl-Dickstein, Frostig and Dahl (2019). *Measuring the Effects of Data Parallelism on Neural Network Training*. JMLR 20. https://arxiv.org/abs/1811.03600
  Shows that the relationship between batch size, learning rate and steps-to-target depends on the model and optimiser, with no universal scaling rule. Cite for the general point that optimal hyperparameters are workload-dependent and that "steps to a target loss" is a legitimate comparison axis (which our loss-matched milestones adopt).

Secondary, verified, not itemised: McCandlish, Kaplan, Amodei and OpenAI Dota Team (2018), *An Empirical Model of Large-Batch Training*, arXiv, https://arxiv.org/abs/1812.06162; Blake et al. (2024), *u-muP: The Unit-Scaled Maximal Update Parametrization*, arXiv, https://arxiv.org/abs/2407.17465; Mlodozeniec et al. (2025), *Completed Hyperparameter Transfer across Modules, Width, Depth, Batch and Duration*, arXiv, https://arxiv.org/abs/2512.22382; Ghosh, Wu and Bietti (2025), *Understanding the Mechanisms of Fast Hyperparameter Transfer*, arXiv, https://arxiv.org/abs/2512.22768; Li et al. (2024), *Surge Phenomenon in Optimal Learning Rate and Batch Size Scaling*, arXiv, https://arxiv.org/abs/2405.14578; DeepSeek-AI (2024), *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism*, arXiv, https://arxiv.org/abs/2401.02954.

---

## 2. Small-scale proxies and whether architecture rankings transfer to larger scale

- Narang, Chung, Tay, Fedus, Fevry, Matena, Malkan, Fiedel, Shazeer, Lan, Zhou, Li, Ding, Marcus, Roberts and Raffel (2021). *Do Transformer Modifications Transfer Across Implementations and Applications?*. EMNLP 2021. https://arxiv.org/abs/2102.11972
  Re-implements dozens of transformer modifications in one codebase and finds most do not meaningfully improve over the baseline, with the gains that survive coming from the same codebase or minor changes. The nearest methodological ancestor: it asks whether a comparison transfers across implementation and task; we ask whether it transfers across width, and find the same fragility.

- Tay, Dehghani, Abnar, Chung, Fedus, Rao, Narang, Tran, Yogatama and Metzler (2022). *Scaling Laws vs Model Architectures: How does Inductive Bias Influence Scaling?*. arXiv. https://arxiv.org/abs/2207.10551
  Trains ten architectures across scales and finds that the best performing model can change with scale, so architecture rankings measured at one size need not hold at another. Our rank reversal (Spearman -0.60 across widths) is a small-scale, pre-registered instance of exactly this phenomenon.

- Yang, Esperança and Carlucci (2020). *NAS evaluation is frustratingly hard*. ICLR 2020. https://arxiv.org/abs/1912.12522
  Shows that NAS method rankings depend heavily on evaluation protocol and training tricks, and that random search is a strong baseline. Cite to position the general finding that architecture rankings are protocol-dependent.

- Yu, Sciuto, Jaggi, Musat and Salzmann (2020). *Evaluating the Search Phase of Neural Architecture Search*. ICLR 2020. https://arxiv.org/abs/1902.08142
  Finds that the search phase of several NAS methods does not beat random sampling once evaluated fairly, and proposes a framework for fair comparison. Same lesson (a comparison protocol can manufacture a winner); our paper adds width as the axis along which the manufactured winner dissolves.

- Abdelfattah, Mehrotra, Dudziak and Lane (2021). *Zero-Cost Proxies for Lightweight NAS*. ICLR 2021. https://arxiv.org/abs/2101.08134
  Introduces cheap proxies and reports their Spearman rank correlation with final accuracy (e.g. 0.82 on NAS-Bench-201). Cite as the standard use of rank correlation to judge whether a proxy preserves a ranking, which is the statistic we report across widths.

- Krishnakumar, White, Zela, Tu, Safari, Hutter et al. (2022). *NAS-Bench-Suite-Zero: Accelerating Research on Zero Cost Proxies*. NeurIPS 2022 Datasets and Benchmarks. https://arxiv.org/abs/2210.03230
  Evaluates zero-cost proxies across 28 tasks and shows their rank correlations vary substantially by benchmark. Supports the point that proxy-to-target rank agreement is not a property of the proxy alone but of the pairing; our width pairing is one such case.

- Wortsman, Liu, Xiao, Everett, Alemi, Adlam, Co-Reyes, Gur, Kumar, Novak, Pennington, Sohl-Dickstein, Xu, Lee, Gilmer and Kornblith (2023). *Small-scale proxies for large-scale Transformer training instabilities*. arXiv. https://arxiv.org/abs/2309.14322
  Shows that large-scale instabilities can be reproduced in small models by raising the learning rate, and that the sensitivity of loss to learning rate changes with scale. Cite for the learning-rate sensitivity that our Study 1 tripped over at d=1024.

- Bhagia, Liu, Wettig, Heineman, Tafjord, Jha, Soldaini, Smith, Groeneveld, Koh, Dodge and Hajishirzi (2025). *Establishing Task Scaling Laws via Compute-Efficient Model Ladders*. COLM 2025. https://arxiv.org/abs/2412.04403
  Uses a ladder of small models (about 1 percent of target compute) to predict task performance of large ones via an intermediate loss. Positions the "train small, extrapolate" practice whose validity for comparisons our paper stress-tests.

- Choshen, Zhang and Andreas (2025). *A Hitchhiker's Guide to Scaling Law Estimation*. ICML 2025. https://arxiv.org/abs/2410.11840
  Analyses over 1,000 scaling-law fits and recommends using intermediate checkpoints and several small models rather than one large one. Agrees with our use of the whole curve (five loss milestones) rather than a single end-of-training number.

- Poli, Thomas, Nguyen, Ponnusamy, Deiseroth, Kersting, Suzuki, Hie, Ermon, Ré, Zhang and Massaroli (2024). *Mechanistic Design and Scaling of Hybrid Architectures*. arXiv. https://arxiv.org/abs/2403.17844
  Proposes small synthetic proxy tasks to rank architectures and checks the ranking against compute-optimal scaling laws. Positions the optimistic view (proxies can rank architectures) against which our negative result is a caution.

Secondary, verified, not itemised: Tay et al. (2022), *Scale Efficiently: Insights from Pre-training and Fine-tuning Transformers*, ICLR 2022, https://arxiv.org/abs/2109.10686; Ruan, Maddison and Hashimoto (2024), *Observational Scaling Laws and the Predictability of Language Model Performance*, NeurIPS 2024, https://arxiv.org/abs/2405.10938; Zela, Siems and Hutter (2020), *NAS-Bench-1Shot1: Benchmarking and Dissecting One-shot Neural Architecture Search*, ICLR 2020, https://arxiv.org/abs/2001.10422; Koh, Suk, Han, Yun and Shin (2026), *Predicting LLM Reasoning Performance with Small Proxy Model*, ICLR 2026, https://arxiv.org/abs/2509.21013.

---

## 3. Hyperparameter tuning changing the ranking of methods

- Choi, Shallue, Nado, Lee, Maddison and Dahl (2019). *On Empirical Comparisons of Optimizers for Deep Learning*. arXiv. https://arxiv.org/abs/1910.05446
  Shows that the hyperparameter search space is the single most important factor explaining optimiser rankings in the literature, and that with inclusive tuning adaptive methods never underperform the methods they generalise. The template for our thesis: the ranking was a property of the tuning protocol; we show the same for an architecture panel across width.

- Sivaprasad, Mai, Vogels, Jaggi and Fleuret (2020). *Optimizer Benchmarking Needs to Account for Hyperparameter Tuning*. ICML 2020. https://arxiv.org/abs/1910.11758
  Argues that a fair optimiser comparison must account for the tuning budget, and shows that rankings change with that budget. Agrees; our Study 1 vs Study 2 contrast is a two-point version of the tuning-budget curve.

- Schmidt, Schneider and Hennig (2021). *Descending through a Crowded Valley: Benchmarking Deep Learning Optimizers*. ICML 2021, PMLR 139. https://proceedings.mlr.press/v139/schmidt21a.html
  Benchmarks fifteen optimisers over 50,000 runs and finds no consistent winner, with results strongly problem-dependent. Cite for the general instability of method rankings across conditions.

- Wen, Hall, Ma and Liang (2025). *Fantastic Pretraining Optimizers and Where to Find Them*. arXiv. https://arxiv.org/abs/2509.02046
  Finds that claimed optimiser speed-ups shrink under rigorous per-scale tuning (from about 1.4x at 0.1B to about 1.1x at 1.2B) and that fair comparisons require tuning each method at each scale. The most recent large-scale statement that untuned or shared hyperparameters manufacture rankings; our paper is the toy-scale, architecture-side analogue.

- Kaddour, Key, Nawrot, Minervini and Kusner (2023). *No Train No Gain: Revisiting Efficient Training Algorithms For Transformer-based Language Models*. NeurIPS 2023. https://arxiv.org/abs/2307.06440
  Shows that the gains of several efficient-training methods vanish against a baseline with a fully decayed learning rate under a matched compute budget. Cite for both the "confound is the baseline's tuning" point and the matched-budget comparison protocol.

- Dahl, Schneider, Nado, Agarwal, Shama Sastry, Hennig et al. (2023). *Benchmarking Neural Network Training Algorithms*. arXiv. https://arxiv.org/abs/2306.07179
  Defines the AlgoPerf time-to-result benchmark with explicit rules on hyperparameter tuning, workload sensitivity and what counts as reaching a target. Cite for the formalisation of "reach a target" as the comparison unit, which our loss milestones mirror.

- Kasimbeg, Schneider, Eschenhagen, Bae, Shama Sastry, Saroufim et al. (2025). *Accelerating Neural Network Training: An Analysis of the AlgoPerf Competition*. ICLR 2025. https://arxiv.org/abs/2502.15015
  Reports the first AlgoPerf competition and shows that under fixed tuning rules some conventional wisdom about optimiser rankings changes. Cite as evidence that the tuning ruleset decides the ranking.

- Zhao, Morwani, Brandfonbrener, Vyas and Kakade (2025). *Deconstructing What Makes a Good Optimizer for Language Models*. ICLR 2025. https://arxiv.org/abs/2407.07972
  Finds that several optimisers perform comparably once tuned and that stability to hyperparameter misspecification, not peak performance, separates them. Relevant to our learning-rate cell result: sensitivity to a shared rate differs across lanes and widths.

- Melis, Dyer and Blunsom (2018). *On the State of the Art of Evaluation in Neural Language Models*. ICLR 2018. https://arxiv.org/abs/1707.05589
  Re-evaluates language-model architectures with large-scale black-box tuning and finds that a well-regularised LSTM beats the newer models that had claimed to surpass it. Early, clean case of a ranking reversing under tuning; we reproduce the pattern across width.

- Lucic, Kurach, Michalski, Gelly and Bousquet (2018). *Are GANs Created Equal? A Large-Scale Study*. NeurIPS 2018. https://arxiv.org/abs/1711.10337
  Finds that with equal tuning budgets and many seeds most GAN variants reach similar scores, and that reported differences were largely budget and seed effects. Cite for the combination of tuning budget and seed variance as joint confounders.

Secondary, verified, not itemised: Semenov, Pagliardini and Jaggi (2025), *Benchmarking Optimizers for Large Language Model Pretraining*, arXiv, https://arxiv.org/abs/2509.01440.

---

## 4. Seed variance and statistical practice in ML benchmarks

- Bouthillier, Delaunay, Bronzi, Trofimov, Nichyporuk, Szeto, Sepah, Raff, Madan, Voleti, Ebrahimi Kahou, Michalski, Serdyuk, Arbel, Pal, Varoquaux and Vincent (2021). *Accounting for Variance in Machine Learning Benchmarks*. MLSys 2021 (Proceedings of Machine Learning and Systems 3). https://arxiv.org/abs/2103.03098
  Models the whole benchmarking pipeline and shows that variance from data sampling, initialisation and hyperparameter choice markedly affects conclusions, recommending randomised multi-trial designs. Our twelve seeds with bootstrap bands follow this recommendation; the paper also motivates why hyperparameter choice must be treated as a variance source rather than fixed.

- Bouthillier, Laurent and Vincent (2019). *Unreproducible Research is Reproducible*. ICML 2019, PMLR 97. https://proceedings.mlr.press/v97/bouthillier19a.html
  Distinguishes reproducing a method from reproducing a finding, and shows that a slightly different experiment can fail to support a finding that is numerically reproducible. Our Study 1 positive was reproducible and wrong in exactly this sense.

- Agarwal, Schwarzer, Castro, Courville and Bellemare (2021). *Deep Reinforcement Learning at the Edge of the Statistical Precipice*. NeurIPS 2021 (outstanding paper). https://arxiv.org/abs/2108.13264
  Shows that point estimates over few runs mislead and recommends stratified bootstrap interval estimates, performance profiles and interquartile means (the rliable library). Our bootstrap band on concordance follows this practice.

- Henderson, Islam, Bachman, Pineau, Precup and Meger (2018). *Deep Reinforcement Learning that Matters*. AAAI 2018. https://arxiv.org/abs/1709.06560
  Documents that seeds, hyperparameters and implementation details change deep RL conclusions, and calls for reporting practices that survive them. Cite for the general case; our paper is the same argument for architecture comparisons across width.

- Dodge, Ilharco, Schwartz, Farhadi, Hajishirzi and Smith (2020). *Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping*. arXiv. https://arxiv.org/abs/2002.06305
  Shows across 2,100 fine-tuning runs that initialisation seed and data order contribute comparably to variance and that best-of-N reporting inflates results. Cite for seed-driven variance in language models specifically.

- Picard (2021). *torch.manual_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision*. arXiv. https://arxiv.org/abs/2109.08203
  Scans up to 10,000 seeds and finds it easy to locate outlier seeds that beat or trail the mean by more than typical claimed improvements. Cite for why seed count, not a lucky seed, is the unit of evidence.

- Colas, Sigaud and Oudeyer (2018). *How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments*. arXiv. https://arxiv.org/abs/1806.08295
  Gives a power-analysis recipe for choosing the number of seeds and discusses t-test vs bootstrap intervals. Cite to justify twelve seeds as a deliberate power choice rather than a convention.

- Reimers and Gurevych (2017). *Reporting Score Distributions Makes a Difference: Performance Study of LSTM-networks for Sequence Tagging*. EMNLP 2017. https://arxiv.org/abs/1707.09861
  Shows over 50,000 runs that seed alone produces statistically significant differences and recommends reporting score distributions. Cite for the distribution-not-point reporting norm.

- Dror, Baumer, Shlomov and Reichart (2018). *The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing*. ACL 2018. https://aclanthology.org/P18-1128/
  Surveys significance testing for NLP comparisons and recommends non-parametric tests, including bootstrap, when metric distributions are unknown. Cite for the choice of bootstrap over parametric tests on our concordance statistic.

- Jordan (2024). *On the Variance of Neural Network Training with respect to Test Sets and Distributions*. arXiv. https://arxiv.org/abs/2304.01910
  Shows that run-to-run variance on standard benchmarks is largely test-set variance and that trained networks make approximately independent errors. Cite when discussing which part of our seed variance is intrinsic to training vs to the evaluation slice.

---

## 5. Pre-registration and falsifiability in ML

Note on the venue: TMLR's editorial policies (https://jmlr.org/tmlr/editorial-policies.html) state that the acceptance test is whether claims are supported by accurate, convincing and clear evidence and whether some of TMLR's audience would want to know the findings, even when the contribution is modest, and that reproducibility studies are in scope. This is the hook for a paper whose main result is a negative and a protocol failure.

- Bertinetto, Henriques, Albanie, Paganini and Varol, eds (2021). *Preface: NeurIPS 2020 Workshop on Pre-registration in Machine Learning*. PMLR 148. https://proceedings.mlr.press/v148/bertinetto21a.html (volume: https://proceedings.mlr.press/v148/)
  The first ML venue to review and accept papers on their experimental protocol before results existed, on the argument that result-only rewards suppress negative findings. Our study followed this model informally (protocol, foil and falsifier committed before any run); cite as the precedent and note the second edition at NeurIPS 2021 (https://neurips.cc/virtual/2021/workshop/21885).

- Nosek, Ebersole, DeHaven and Mellor (2018). *The preregistration revolution*. PNAS 115(11), 2600 to 2606. https://www.pnas.org/doi/10.1073/pnas.1708274114
  Argues that preregistration separates prediction from postdiction and thereby raises the credibility of confirmatory findings. The general-science source for our falsifier-named-in-advance design.

- Hofman, Chatzimparmpas, Sharma, Watts and Hullman (2023). *Pre-registration for Predictive Modeling*. arXiv. https://arxiv.org/abs/2311.18807
  Proposes a lightweight pre-registration template for predictive-modelling research and reports a qualitative study with ML researchers on its use. Closest in spirit to our PREREGISTRATION.md; cite and, if useful, map our template onto theirs.

- Herrmann, Lange, Eggensperger, Casalicchio, Wever, Feurer, Rügamer, Hüllermeier, Boulesteix and Bischl (2024). *Position: Why We Must Rethink Empirical Research in Machine Learning*. ICML 2024. https://arxiv.org/abs/2405.02200
  Argues that most empirical ML is exploratory dressed as confirmatory, and calls for explicit separation of the two. Our two-study structure (exploratory Study 1, confirmatory pre-registered Study 2) is a worked example of what they ask for.

- Karl, Kemeter, Dax and Sierak (2024). *Position: Embracing Negative Results in Machine Learning*. ICML 2024, PMLR 235. https://proceedings.mlr.press/v235/karl24a.html (arXiv: https://arxiv.org/abs/2406.03980)
  Argues that judging papers by predictive performance alone creates perverse incentives and proposes measures to normalise publication of negative results. Positions our negative (rank reversal) and our protocol failure (the fallback clause) as the kind of result they argue should be published.

- Lipton and Steinhardt (2018). *Troubling Trends in Machine Learning Scholarship*. ICML 2018 Debates. https://arxiv.org/abs/1807.03341
  Names four failure modes, including obscuring the source of empirical gains. Cite for "source of gains" since our Study 1 gain came from a shared confounder rather than the architecture.

- Pineau, Vincent-Lamarre, Sinha, Larivière, Beygelzimer, d'Alché-Buc, Fox and Larochelle (2021). *Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)*. JMLR 22. https://arxiv.org/abs/2003.12206
  Describes the reproducibility checklist and code-submission policy and reports their effect. Cite for the checklist literature; our receipts directory is the artefact-level counterpart.

- Sculley, Snoek, Wiltschko and Rahimi (2018). *Winner's Curse? On Pace, Progress, and Empirical Rigor*. ICLR 2018 Workshop Track. https://iclr.cc/virtual/2018/workshop/406
  Argues that the pace of empirical advance has outrun empirical rigour and proposes incentive changes. Cite for the incentive framing in the discussion.

- Forde and Paganini (2019). *The Scientific Method in the Science of Machine Learning*. ICLR 2019 Debugging ML Models workshop. https://arxiv.org/abs/1904.10922
  Argues that ML lacks hypothesis formulation, testing and uncertainty estimation, and points to physics practice. Cite for the falsifiability framing.

- Kapoor, Cantrell, Peng, Pham, Bail, Gundersen et al. (2023). *REFORMS: Reporting Standards for Machine Learning Based Science*. arXiv. https://arxiv.org/abs/2308.07832
  A 32-item consensus checklist for ML-based scientific claims covering validity, reproducibility and generalisability. Cite as the current reporting standard and, if space allows, self-audit the paper against it.

Secondary, verified, not itemised: Gundersen and Kjensmo (2018), *State of the Art: Reproducibility in Artificial Intelligence*, AAAI 2018, https://ojs.aaai.org/index.php/AAAI/article/view/11503; Raff (2019), *A Step Toward Quantifying Independently Reproducible Machine Learning Research*, NeurIPS 2019, https://arxiv.org/abs/1909.06674; Kapoor and Narayanan (2023), *Leakage and the reproducibility crisis in machine-learning-based science*, Patterns 4(9), https://www.cell.com/patterns/fulltext/S2666-3899(23)00159-9; Gencoglu et al. (2019), *HARK Side of Deep Learning: From Grad Student Descent to Automated Machine Learning*, arXiv, https://arxiv.org/abs/1904.07633; Vaccaro (2026), *Preregistration for Experiments with AI Agents*, ICML 2026 position paper, https://arxiv.org/abs/2606.11217; Forde, Ruiz, Pradier and Schein, eds (2020), *Proceedings on "I Can't Believe It's Not Better!" at NeurIPS Workshops*, PMLR 137, https://proceedings.mlr.press/v137/; Sodhani et al. (2020), *Ideas for Improving the Field of Machine Learning: Summarizing Discussion from the NeurIPS 2019 Retrospectives Workshop*, arXiv, https://arxiv.org/abs/2007.10546.

---

## 6. Comparing training runs at matched loss or matched compute rather than matched steps

- Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu and Amodei (2020). *Scaling Laws for Neural Language Models*. arXiv. https://arxiv.org/abs/2001.08361
  Establishes power-law scaling of loss in parameters, data and compute, and frames comparisons in terms of loss reached per unit compute rather than per step. Cite as the origin of compute-matched and loss-matched comparison in language modelling.

- Hoffmann, Borgeaud, Mensch, Buchatskaya, Cai, Rutherford et al. (2022). *Training Compute-Optimal Large Language Models*. NeurIPS 2022. https://arxiv.org/abs/2203.15556
  Uses iso-FLOP curves (many models at fixed compute, varying size and tokens) to show that parameters and tokens should scale together. Cite for the iso-compute methodology and for the point that step-matched comparisons across sizes are not compute-matched.

- Porian, Wortsman, Jitsev, Schmidt and Carmon (2024). *Resolving Discrepancies in Compute-Optimal Scaling of Language Models*. NeurIPS 2024 (spotlight). https://arxiv.org/abs/2406.19146
  Traces the Kaplan vs Chinchilla disagreement to three protocol choices: last-layer cost accounting, warm-up duration, and scale-dependent optimiser tuning. Must cite: a headline finding of the field changed because hyperparameters were not tuned per scale, which is our result in miniature.

- Hägele, Bakouch, Kosson, Ben Allal, von Werra and Jaggi (2024). *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations*. NeurIPS 2024 (spotlight). https://arxiv.org/abs/2405.18392
  Shows that constant learning rate with cooldown scales like cosine and allows one run to be compared at many durations, making scaling studies far cheaper. Positions why a fixed cosine schedule to a fixed step count makes intermediate milestones awkward, and offers a schedule that would suit milestone-matched designs.

- Bjorck, Benhaim, Chaudhary, Wei and Song (2025). *Scaling Optimal LR Across Token Horizons*. ICLR 2025. https://arxiv.org/abs/2409.19913
  Shows that the optimal learning rate depends on the training horizon (token count) and gives a scaling rule for it. Cite because matching by loss rather than by step changes the effective horizon per width, which interacts with the per-width learning-rate optimum.

- Dehghani, Arnab, Beyer, Vaswani and Tay (2022). *The Efficiency Misnomer*. ICLR 2022. https://arxiv.org/abs/2110.12894
  Shows that cost indicators (parameters, FLOPs, wall-clock) disagree and that reporting one can invert an efficiency ranking. Cite for the principle that the choice of matching axis (steps, compute, loss) is itself a modelling decision that can flip a ranking.

- Brandfonbrener, Anand, Vyas, Malach and Kakade (2025). *Loss-to-Loss Prediction: Scaling Laws for All Datasets*. TMLR 2025. https://arxiv.org/abs/2411.12925
  Shows shifted power-law relationships between losses of models paired by compute, and between train and test losses. A TMLR precedent for loss-based pairing of runs across conditions; cite when justifying loss milestones as the comparison unit.

- Liu, Xie, Li and Ma (2023). *Same Pre-training Loss, Better Downstream: Implicit Bias Matters for Language Models*. ICML 2023, PMLR 202. https://proceedings.mlr.press/v202/liu23ao.html
  Shows that models at identical pre-training loss can differ in downstream performance, with flatter minima transferring better. A caution on our matching axis: equal validation loss does not guarantee equal models, so lane differences at matched loss are about the path, not only the endpoint.

- Besiroglu, Erdil, Barnett and You (2024). *Chinchilla Scaling: A replication attempt*. arXiv. https://arxiv.org/abs/2404.10102
  Re-fits Chinchilla's third estimation method and finds inconsistencies and implausibly narrow confidence intervals. Cite for the value of independent re-analysis and for uncertainty reporting on fitted scaling quantities.

- Hestness, Narang, Ardalani, Diamos, Jun, Kianinejad, Patwary, Yang and Zhou (2017). *Deep Learning Scaling is Predictable, Empirically*. arXiv. https://arxiv.org/abs/1712.00409
  Early empirical power-law scaling across domains, with per-size hyperparameter search. Cite for the practice of re-tuning at each scale before fitting curves.

Secondary, verified, not itemised: Muennighoff et al. (2023), *Scaling Data-Constrained Language Models*, arXiv, https://arxiv.org/abs/2305.16264.

---

## 7. Sliding-window attention and attention sinks (panel provenance)

- Xiao, Tian, Chen, Han and Lewis (2024). *Efficient Streaming Language Models with Attention Sinks*. ICLR 2024. https://arxiv.org/abs/2309.17453
  Identifies the attention-sink phenomenon (heavy attention on initial tokens regardless of content) and shows that keeping a few sink tokens with a sliding window stabilises streaming generation. Source of our sink lanes.

- Gu, Pang, Du, Liu, Zhang, Du, Wang and Lin (2025). *When Attention Sink Emerges in Language Models: An Empirical View*. ICLR 2025 (spotlight). https://arxiv.org/abs/2410.10781
  Shows that the sink emerges after sufficient optimisation on sufficient data, acts like a key bias, and that under sliding-window training a smaller window prevents its emergence while larger windows place it on the absolute first token. The one prior work on where window and sink effects appear during training; cite and compare with our window sweep (16 to 128) at small width.

- Beltagy, Peters and Cohan (2020). *Longformer: The Long-Document Transformer*. arXiv. https://arxiv.org/abs/2004.05150
  Introduces sliding-window attention with a few global tokens as a linear-cost replacement for full attention. Source of the sliding-window lanes.

- Jiang, Sablayrolles, Mensch, Bamford, Chaplot, de las Casas et al. (2023). *Mistral 7B*. arXiv. https://arxiv.org/abs/2310.06825
  Uses sliding-window attention in a production 7B model with a rolling KV cache. Cite as the modern motivation for a window-vs-exact panel.

- Gemma Team (2024). *Gemma 2: Improving Open Language Models at a Practical Size*. arXiv. https://arxiv.org/abs/2408.00118
  Interleaves local sliding-window and global attention layers. Cite as evidence that the window-vs-exact choice is live in current models.

- Sun, Chen, Kolter and Liu (2024). *Massive Activations in Large Language Models*. COLM 2024. https://arxiv.org/abs/2402.17762
  Finds a few input-independent massive activations that act as implicit biases and concentrate attention, linking sinks to activation outliers. Cite for the mechanism behind sink tokens.

- Barbero, Arroyo, Gu, Perivolaropoulos, Bronstein, Veličković and Pascanu (2025). *Why do LLMs attend to the first token?*. arXiv. https://arxiv.org/abs/2504.02732
  Proposes that sinks prevent over-mixing and shows the effect depends on context length, depth and data packing. Cite for a functional account of why a sink lane might help or hurt at a given width.

- Child, Gray, Radford and Sutskever (2019). *Generating Long Sequences with Sparse Transformers*. arXiv. https://arxiv.org/abs/1904.10509
  Introduces factorised sparse attention patterns including strided local windows. Historical source of local attention.

- Han, Wang, Peng, Xiong, Chen, Ji and Wang (2024). *LM-Infinite: Zero-Shot Extreme Length Generalization for Large Language Models*. NAACL 2024 (outstanding paper). https://arxiv.org/abs/2308.16137
  Independently proposes a Lambda-shaped mask (initial tokens plus local window) for length generalisation. Cite alongside StreamingLLM as concurrent evidence for sink-plus-window.

- Eldan and Li (2023). *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*. arXiv. https://arxiv.org/abs/2305.07759
  Introduces the synthetic TinyStories corpus and shows that models under 10M parameters produce coherent text on it. Data provenance for our study; also justifies why tiny widths are meaningful on this corpus.

Secondary, verified, not itemised: Zaheer et al. (2020), *Big Bird: Transformers for Longer Sequences*, NeurIPS 2020, https://arxiv.org/abs/2007.14062; Darcet, Oquab, Mairal and Bojanowski (2024), *Vision Transformers Need Registers*, ICLR 2024, https://arxiv.org/abs/2309.16588; Cancedda (2024), *Spectral Filters, Dark Signals, and Attention Sinks*, arXiv, https://arxiv.org/abs/2402.09221; Fu et al. (2025), *Sliding Window Attention Training for Efficient Large Language Models*, arXiv, https://arxiv.org/abs/2502.18845; Radford, Wu, Child, Luan, Amodei and Sutskever (2019), *Language Models are Unsupervised Multitask Learners*, OpenAI technical report, https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf.

---

## 8. Confounding and interaction effects in ablation studies

- Chen, Wu, Wang and Hanin (2024). *Principled Architecture-aware Scaling of Hyperparameters*. arXiv. https://arxiv.org/abs/2402.17440
  Derives initialisation and maximal learning rate as functions of depth, width, kernel size and topology, and shows that architecture-aware tuning changes network rankings on AutoML benchmarks. The clearest statement that the optimal learning rate is a function of the architecture, so a shared rate across lanes and widths is an interaction confounder.

- Bello, Fedus, Du, Cubuk, Srinivas, Lin, Shlens and Zoph (2021). *Revisiting ResNets: Improved Training and Scaling Strategies*. arXiv. https://arxiv.org/abs/2103.07579
  Shows that training and scaling strategy matter more than architectural changes, with improved ResNets matching newer architectures. Cite for "the gain was the recipe, not the architecture".

- Wightman, Touvron and Jégou (2021). *ResNet strikes back: An improved training procedure in timm*. arXiv. https://arxiv.org/abs/2110.00476
  Retrains ResNet-50 with a modern recipe to 80.4 percent ImageNet top-1, showing that many architecture comparisons were recipe comparisons. Same point, widely cited.

- Steiner, Kolesnikov, Zhai, Wightman, Uszkoreit and Beyer (2022). *How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers*. TMLR 2022. https://arxiv.org/abs/2106.10270
  Trains over 50,000 ViTs and shows that data, augmentation and regularisation interact with model size so that the best setting depends on scale. A TMLR precedent for a large factorial study of hyperparameter-by-scale interactions.

- Nado, Gilmer, Shallue, Anil and Dahl (2021). *A Large Batch Optimizer Reality Check: Traditional, Generic Optimizers Suffice Across Batch Sizes*. arXiv. https://arxiv.org/abs/2102.06356
  Shows that with careful tuning standard optimisers match LARS and LAMB at large batch, so the reported advantage was a tuning artefact. Cite as a case where a claimed method effect was a tuning effect.

- Wilson, Roelofs, Stern, Srebro and Recht (2017). *The Marginal Value of Adaptive Gradient Methods in Machine Learning*. arXiv (conference venue not checked; see unverified list). https://arxiv.org/abs/1705.08292
  Argued that adaptive methods generalise worse than SGD; later work (Choi et al. 2019, above) showed the ranking depended on the search space. Cite the pair as the canonical example of a ranking that flipped under tuning.

- Probst, Boulesteix and Bischl (2019). *Tunability: Importance of Hyperparameters of Machine Learning Algorithms*. JMLR 20. https://jmlr.org/papers/v20/18-444.html
  Formalises tunability and measures per-hyperparameter importance across 38 datasets. Cite for the concept that tunability differs by method, which is what our lr-sensitivity cell measures across lanes.

- Dehghani, Tay, Gritsenko, Zhao, Houlsby, Diaz, Metzler and Vinyals (2021). *The Benchmark Lottery*. arXiv. https://arxiv.org/abs/2107.07002
  Shows across domains that the relative ranking of methods changes with the choice of benchmark task. Cite for the general form of the argument: a ranking is a joint property of method and protocol, with width as our protocol axis.

Secondary, verified, not itemised: Meyes, Lu, Waubert de Puiseau and Meisen (2019), *Ablation Studies in Artificial Neural Networks*, arXiv, https://arxiv.org/abs/1901.08644; Liu et al. (2022), *A ConvNet for the 2020s*, CVPR 2022, https://arxiv.org/abs/2201.03545.

---

## Positioning summary

Closest prior works. (1) Narang et al. (2021) ask whether transformer modifications transfer across implementations and tasks and find most do not; we ask whether a six-lane attention comparison transfers across width and find the ranking reverses once each width is tuned. (2) Tay et al. (2022) show that the best architecture can change with scale; our result is a pre-registered, twelve-seed, loss-matched instance at toy scale with a bootstrap band on the concordance. (3) Choi et al. (2019), Sivaprasad et al. (2020) and, most recently, Wen et al. (2025) show that optimiser rankings are a property of the tuning protocol; we show the same for architectures across width, and identify the specific mechanism (a learning rate shared across widths that is mistuned for the larger one). (4) Porian et al. (2024) show that a headline scaling-law disagreement came partly from not tuning the optimiser per scale; our Study 1 vs Study 2 contrast is that mechanism isolated in a controlled panel. The muP line (Yang et al. 2022; Everett et al. 2024; Kosson et al. 2025) explains why the shift exists but does not address whether comparisons survive it.

Gap filled. No prior work, to our verification, runs a pre-registered architecture comparison across width with a committed foil and falsifier, matches runs by loss milestone rather than by step, and reports that the transfer claim which held under a shared learning rate collapsed under per-width tuning, together with a documented failure of one of its own pre-registration clauses.

---

## Items I could not verify

None of the following may be cited without independent confirmation.

- A paper specifically on "rank reversal" of attention variants (sliding window vs full) across model width. Searched; nothing found beyond Gu et al. (2025) on window size and sink emergence, which does not report cross-width rankings.
- A journal or TMLR paper that explicitly analyses step-matched vs loss-matched comparison as a methodological choice. Searches returned only indirect support (Kaplan; Hoffmann; Dahl et al.; Liu et al. 2023). Treat the loss-milestone protocol as our own framing with those citations.
- Any published critique that pre-registration fallback or contingency clauses misbehave in unanticipated regimes. Nosek et al. (2018) and Hofman et al. (2023) discuss deviations from pre-registration in general terms only; I did not find an ML paper reporting a protocol clause failing in practice.
- A follow-up publication from the NeurIPS 2021 pre-registration workshop analogous to PMLR 148. The 2021 workshop page is verified; I did not confirm a proceedings volume.
- The OpenReview record for Sculley et al. (2018) (https://openreview.net/forum?id=rJWF0Fywf) could not be fetched (bot check); the ICLR 2018 workshop page is verified and is the URL given above.
- Kapoor et al., REFORMS: reported as later appearing in Science Advances (2024); only the arXiv record was verified, so the arXiv citation is used.
- Melis et al. (2018), Darcet et al. (2024), Yu et al. (2020), Bordelon et al. (2024) and Probst et al. (2019): venues were confirmed by a second source (dblp, ICLR proceedings or JMLR page) but the arXiv page itself does not state the venue; the citations above use the confirmed venue.
- Semenov et al. (2025) and Wen et al. (2025): arXiv only at time of check; no peer-reviewed venue confirmed.
- Wilson et al. (2017), *The Marginal Value of Adaptive Gradient Methods*: commonly cited as NeurIPS 2017, but only the arXiv record was checked; cited as arXiv in refs.bib with an UNVERIFIED comment.
