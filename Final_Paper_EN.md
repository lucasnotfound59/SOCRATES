**Knowing What You Don't Know: A Symmetric Comparison of Metacognitive
Calibration in Humans and Large Language Models**

\[ Author \] · \[ Affiliation \]

**Abstract**

*\[To be written last --- four sentences: problem → method → (expected)
findings → contribution; 150--250 words.\]*

**1. Introduction**

Large language models (hereafter "LLMs") are increasingly replacing
search engines as a source of knowledge for a growing number of people,
yet they share one fundamental flaw: when they are wrong, they are often
just as confident as when they are right. Scholten et al. (2024)
transplant the methods of human cognitive psychology onto LLMs, and,
drawing on the tradition of cognitive ecology, characterize the
"cognitive psychology" of an LLM as follows: it is adept at exploiting
stimulus information, yet nearly "blind" to the source, history, and
validity of that information. Underlying this are the two core processes
of metacognition---monitoring and control. They argue that the root of
LLM hallucination is not, as is commonly assumed, errors in the labeled
data; rather, it is that the entire training corpus shapes the weights,
so that for an LLM, processing a piece of information is tantamount to
taking it as true. As a result, the model does not assess the veracity
of an information source, and further treats high-frequency and surface
information as if it were reliable. This establishes that LLMs exhibit
so-called "hallucination," which is the background problem motivating
the present study.

The ability to accurately assess the boundaries of one's own
knowledge---to know what one does not know---is a foundation of how
humans come to understand the world: it is precisely because we can
become aware of our ignorance that we are driven to seek knowledge.
Metacognitive calibration, restated, is the degree of correspondence
between subjective metacognitive judgments (confidence) and objective
performance (accuracy) (Flavell, 1979). As Socrates put it, "The only
thing I know is that I know nothing"; metacognitive calibration is a
core human capacity. And insofar as we wish LLMs to be sufficiently
human-like, their metacognitive calibration likewise becomes an
important dimension of evaluation. This is why comparing humans and LLMs
on this capacity matters---and hallucination is a key dimension
affecting the metacognitive calibration of LLMs.

When humans delegate decisions to these systems, a system that does not
know when it is wrong is more dangerous than a weak one that does know
its boundaries---so the central question is: when can we trust the
uncertainty it expresses? Steyvers et al. (2025) offer an instructive
empirical answer. They distinguish two gaps: the calibration gap, the
divergence between a model's internal confidence and users' perception
of the model's accuracy; and the discrimination gap, the difference
between the model and users in their ability to distinguish correct from
incorrect answers. Under the default-explanation condition, users
systematically overestimated LLM accuracy; more tellingly, users'
ability to judge answer correctness from explanations (AUC ≈ 0.59) was
only slightly above chance, whereas the model's own internal confidence
discriminated correct from incorrect answers considerably better (AUC ≈
0.75--0.78). This contrast reveals that the problem often lies not in
the model "not knowing" when it is wrong, but in its knowing while
failing to convey that uncertainty faithfully through natural language.
The study further found that explanation length alone inflated user
confidence without improving discrimination---users mistook a surface
cue such as length for a signal of reliability. The answer to "whether
its expressed uncertainty can be trusted" is therefore conditional: only
when the model's uncertainty language is deliberately aligned to its
internal confidence do the calibration and discrimination gaps narrow
together (Steyvers et al., 2025).

However, research on "whether machines can know what they do not know"
and research on "how humans monitor their own knowledge" have almost
never met within a single framework. On one side, LLM-calibration
research (e.g., Kadavath et al.'s P(IK), Guo et al.'s ECE and
temperature scaling) measures only whether a model's own confidence
matches its accuracy, without a human baseline---no one knows whether
the machine's performance approaches that of humans, how it differs, or
how to make machines more human-like. On the other side, human
metacognition research (Fleming & Lau) has mature sensitivity measures,
yet is never contrasted against LLMs. The closest point of convergence
is Steyvers et al. (2025): they place humans and models on the same
scale, but what they compare is "humans' perception of an LLM's answers
vs. the LLM's internal confidence"---measuring whether users can read
the model's uncertainty, rather than "humans' metacognitive calibration
about themselves vs. LLMs' metacognitive calibration about themselves."
The real gap lies precisely here: directly comparing, on the same set of
tasks and under the same metrics, the ability of humans and LLMs to each
assess the boundaries of their own knowledge.

*The essential challenge of this comparison is that the two confidences
are heterogeneous quantities: a human's confidence arises from
second-order introspection, whereas a model's "confidence" is a product
of its generative process. Placing them on a single, fair
scale---without letting a difference in raw ability masquerade as a
difference in self-knowledge---is the core methodological difficulty
this study must resolve.*

It is precisely to fill this gap that the present study proposes a
symmetric framework: rather than treating humans as readers of the LLM,
it places humans and LLMs as parallel respondents who face exactly the
same questions and each report both an answer and a confidence. Humans
and LLMs are thereby placed on a single scale that measures the same
thing---each party's ability to assess the boundaries of its own
knowledge, not whether a user can detect the accuracy of the model's
information. For metrics, the study centers on the calibration curve,
using the Expected Calibration Error (ECE) to quantify the divergence
between "generated confidence" and "actual accuracy" for both LLMs and
humans; it further introduces meta-d′ from signal detection theory to
control for the difference in first-order accuracy between humans and
models, so that "being accurate about oneself" is not obscured by "being
accurate on the task." The broader backdrop of this design comes from
the Tong Test proposed by Zhu and colleagues, which argues that genuine
general intelligence should possess self-knowledge---precisely the
dimension on which current LLMs remain in doubt (Peng et al., 2024). It
should be noted that this study does not attempt to answer the grand
question of whether LLMs "genuinely possess" metacognition; it only
seeks, at the level of behavior, to compare humans and LLMs through one
quantifiable facet---confidence versus accuracy.

Within this framework, the study tests three competing hypotheses to
determine the state of LLM metacognitive calibration. Under H1, an LLM's
reported confidence is merely a stylistic artifact---it has learned
wording that "sounds certain" but bears little relation to its true
accuracy, so its calibration is poor and its pattern of deviation is not
of the same kind as humans'. Under H2, the LLM possesses near-human
metacognitive calibration---despite different underlying mechanisms, the
two show highly similar calibration curves at the behavioral level.
Under H3, the LLM's calibration is partially human-like: close to humans
on ordinary questions, but systematically collapsing on specific item
types---especially hallucination-triggering questions involving
fictional entities or "unknown unknowns," where it is most confident
exactly where it should hesitate. We expect H3 to be the most likely and
the most informative: rather than the blunt claim that an LLM is "well-"
or "poorly-" calibrated, H3 further locates which kind of knowledge
boundary the difference appears at. It should be emphasized that the
design is intended to adjudicate among these three hypotheses rather
than merely report a set of calibration numbers---the item types are
deliberately constructed so that the three hypotheses' predictions come
apart.

*This study makes three contributions. Methodologically, it provides a
symmetric framework that places humans and LLMs on a single scale of
self-assessment. Empirically, it yields a set of results on where---and
on which item types---human and LLM metacognition diverge. Conceptually,
it clarifies that behavioral calibration is not the same as internal
metacognition, and offers guidance on when an LLM's expressed
uncertainty can be trusted.*

The remainder of this paper is organized as follows. Section 2 reviews
the two research lines of LLM calibration and human metacognition, and
clarifies how this study differs from the closest prior work. Section 3
sets out the symmetric experimental design, the construction of item
types, and the evaluation metrics, with particular attention to the
measures that safeguard human--model comparability. Section 4 reports
the calibration comparison between humans and LLMs. Section 5 discusses
the significance and the boundaries of these results---what a
behavioral-level difference can and cannot show---and returns to the
larger question that awareness of one's ignorance is the starting point
of all inquiry. Section 6 summarizes contributions and limitations and
points to future directions.

**2. Related Work**

**2.1 Confidence Calibration in Large Language Models**

LLM-calibration research concerns one question: whether a model's
reported confidence matches its actual accuracy. This line has developed
two main families of methods. The first calibrates on the basis of
internal model signals: Guo et al. (2017) showed that modern neural
networks, though highly accurate, are generally overconfident,
established the Expected Calibration Error (ECE) as a standard metric,
and proposed post-hoc remedies such as temperature scaling---a framework
later widely transplanted onto LLMs. The second directly probes a
model's prediction about its own knowledge: Kadavath et al. (2022)
trained models to predict P(True) (the probability that an answer is
correct) and P(IK) (the probability that the model "knows" the answer),
finding that larger models are well-calibrated when the format is
appropriate, but that P(IK) calibration degrades markedly on tasks not
covered in training. Geng et al. (2024), in a survey, further map the
methodological spectrum from temperature scaling, through
consistency/ensemble methods, to verbalized confidence (having the model
state its confidence directly in words or numbers). Yet this entire line
shares a common boundary: it measures only how well a model's own
confidence matches its accuracy, without a human baseline---no one knows
whether the machine's calibration is close to or far from human levels,
or where the two differ.

**2.2 Human Metacognition and Its Measurement**

By contrast, human metacognition research has a longer tradition and
more mature measurement tools. Flavell (1979) first systematically
proposed the metacognition framework, distinguishing "cognition about
one's own cognition" into metacognitive knowledge, metacognitive
experience, and cognitive monitoring; the monitoring component
corresponds precisely to "knowing what one knows and does not know." At
the level of measurement, the key advance in this field is decoupling
"metacognitive sensitivity" from "first-order task performance." Fleming
and Lau (2014) systematically reviewed these methods, of which the
signal-detection-theoretic meta-d′ is especially important: it estimates
"the first-order sensitivity a subject should have had, if confidence
perfectly reflected all information available at the time of response,"
and compares it against the subject's actual sensitivity d′; the ratio
M-ratio (meta-d′/d′) then measures metacognitive efficiency---equal to 1
when confidence exhausts all knowledge, and below 1 when there is loss.
The value of this measure is that it controls for first-order ability
differences: even when two subjects differ greatly in accuracy, their
efficiency in "using knowledge through confidence" can be compared
fairly. Yet this toolkit, honed for human psychological experiments, has
almost never been used to evaluate LLMs.

**2.3 The Convergence of Humans and LLMs, and the Remaining Gap**

Recent work has begun to examine humans and models side by side, but
each such effort touches only part of the problem. Griot et al. (2025)
found, on medical-reasoning tasks, that LLMs lack reliable
self-monitoring and may remain overconfident in high-stakes situations;
Scholten et al. (2024) propose, at the mechanistic level, "metacognitive
myopia," noting that LLMs over-rely on surface and high-frequency
information and lack meta-level processing of source reliability---both
provide evidence that LLM metacognition is deficient, yet neither
introduces a human control or a unified calibration quantification. Cash
et al. (2024) transplant a cognitive-psychology confidence-judgment
paradigm onto LLMs, a near neighbor in methodological approach, but
their focus remains on the model side. The closest point of convergence
is Steyvers et al. (2025): they do place humans and models on the same
scale, but what they compare is "humans' perceived confidence in an
LLM's answers" versus "the LLM's internal confidence"---measuring
whether users can read the uncertainty the model expresses, rather than
humans' and models' respective ability to assess the boundaries of their
own knowledge. The overall orientation of this line echoes the argument
of Peng et al. (2024) in the Tong Test: that genuine general
intelligence should possess self-knowledge, a dimension on which current
LLMs remain in doubt. The real gap thus comes clearly into view: to
compare directly, on the same set of tasks and under the same metrics,
the ability of humans and LLMs to each assess the boundaries of their
own knowledge---which is what this study seeks to fill.

**3. Methods**

**3.1 Task and Experimental Design**

This study adopts a symmetric design: human participants and LLMs serve
as parallel respondents, facing exactly the same set of true/false
questions and reporting their confidence alongside each answer.
True/false items are chosen because their correctness is objectively
determinable, so that scoring does not depend on subjective judgment and
humans and models are graded under the same standard.

**Item bank.** The bank comprises 200--320 items spanning four item
types in roughly equal numbers: (1) ordinary items---everyday common
sense and encyclopedic knowledge, with time-sensitive "viral" internet
items folded in as a difficulty variant of the ordinary type rather than
a separate category, since they may already exist in model training
data; (2) discipline-specific items---specialized knowledge from various
academic fields; (3) trap items---statements that appear plausible on
the surface but contain misleading content; and (4)
hallucination-triggering items---built around fictional entities or
non-existent facts to induce "unknown-unknown" situations, in which a
respondent ought to recognize that it cannot possibly know, yet may
fabricate a confident answer. The four types are deliberately
constructed so that the predictions of the three hypotheses (H1--H3)
come apart: if a difference appears only on a particular item type, the
exact locus of metacognitive collapse can be identified.

**Difficulty stratification.** Given the wide capability span of the
model tier (from small open-source models to frontier models), an
unbalanced difficulty would drive one side to answer nearly all items
correctly or incorrectly, distorting or even precluding estimation of d′
and meta-d′. Each item type is therefore difficulty-stratified to
deliberately span a range, ensuring that strong models still err and
weak models still get some items right, thereby preserving an estimable
distribution of correct and incorrect responses for the signal-detection
metrics.

**Confidence elicitation.** Both humans and models report,
retrospectively (after answering), their confidence in that answer on
the same 5-point discrete scale: ① pure guess, ② not very sure, ③
half-and-half, ④ fairly sure, ⑤ very sure. Both sides use identical
scale points and wording. It should be made explicit that this study
defines its object of measurement at the behavioral level---the behavior
of "retrospectively and subjectively reported confidence" itself; the
possibility that a report deviates from any underlying internal state
is, as a behavioral phenomenon, symmetric across humans and models. The
study neither assumes nor needs to assume that equivalent internal
mechanisms underlie the two (see §3.4 and the Discussion).

Although "whether one possesses a given piece of knowledge" may, under
certain epistemological positions, be binary in nature, what this study
measures is not knowledge itself but a respondent's assessment of its
own knowledge state; that assessment is widely documented in psychology
as graded, and the study's core metric, meta-d′, relies on graded
information for estimation. The number of scale points is not an
arbitrary preference but is driven by what meta-d′ requires: meta-d′
treats each confidence level as a point on a Type-2 ROC curve and infers
metacognitive sensitivity from the shape of that curve, so too few
levels (e.g., a binary scale) leave too few sampling points and yield
unstable estimates, whereas too many levels exceed what respondents can
use consistently. A 5-point scale is therefore adopted as the
compromise---it provides one more ROC sampling point than a 4-point
scale, improving estimation stability, while remaining within the range
a human can use reliably. Its two ends correspond to the "know /
do-not-know" extremes, while the middle three points capture the graded
uncertain region of the assessment.

**3.2 Participants and Models**

**Human participants.** 40--60 participants are recruited; each answers
40 items randomly drawn from the bank---10 from each of the four item
types---so that everyone covers all types without excessive fatigue.
Participants provide a true/false judgment and a 4-point confidence for
each item.

**Model tier.** To examine both "whether calibration varies with
capability" and "whether it varies by vendor/architecture," the model
tier spans a range of LLMs from small to frontier across multiple
vendors, including small models from the Gemma and Qwen families, as
well as several frontier models accessed via API (e.g., GPT, Claude
Opus, Qwen, DeepSeek). Unlike humans, models are not subject to fatigue,
so each model answers the entire item bank to obtain high-precision
group-level estimates. All models use a fixed prompt template to control
prompt sensitivity---since the same item under different wording changes
a model's confidence, a fixed template removes this confound.

*\[Note: the exact participant count and item-bank size should be fixed
to definite numbers before submission; ranges are placeholders. Also to
decide: model sampling temperature (temperature = 0 for
determinism/reproducibility, vs. retaining randomness with multiple
samples per item).\]*

**3.3 Evaluation Metrics**

The study uses two complementary metrics, characterizing calibration and
metacognitive efficiency respectively.

**Expected Calibration Error (ECE).** ECE quantifies the divergence
between each respondent's (or group's) stated confidence and actual
accuracy (Guo et al., 2017): trials are binned by confidence, the
absolute difference between mean confidence and actual accuracy is
computed per bin, and these are averaged weighted by the number of
trials in each bin. To compute ECE, the five confidence levels must be
mapped to numerical confidence values. Crucially, because the task
consists of true/false items, the lower bound of confidence is not 0%
but 50%: a pure guess on a binary item is expected to be correct about
half the time. Mapping "pure guess" to 0% would therefore make the
lowest level appear severely under-confident (stated 0%, actual ≈50%)
and inflate ECE spuriously. The five levels are accordingly mapped to
equally spaced values from the guessing floor to full certainty---① pure
guess = 50%, ② not very sure = 62.5%, ③ half-and-half = 75%, ④ fairly
sure = 87.5%, ⑤ very sure = 100%---so that the mapping is consistent
with the probability structure of a binary task. This mapping is stated
explicitly to make ECE reproducible. ECE corresponds intuitively to how
far the calibration curve departs from the diagonal, and serves as the
primary calibration metric and the basis for plotting the calibration
curves of humans and each model.

**Metacognitive efficiency (meta-d′ / M-ratio).** A built-in limitation
of ECE is that it is affected by first-order ability (accuracy)---a
high-ability respondent may show low ECE even with mediocre monitoring.
To decouple "being accurate about oneself" from "being accurate on the
task," the study introduces meta-d′ from signal detection theory
(Fleming & Lau, 2014): it estimates the first-order sensitivity a
respondent should have had if confidence perfectly reflected all
information available at response time, and compares it to actual
sensitivity d′; the ratio M-ratio (meta-d′/d′) measures metacognitive
efficiency---equal to 1 when confidence exhausts all knowledge, below 1
when there is loss. Because M-ratio controls for first-order ability
differences, humans and models with very different accuracy can still be
compared fairly on metacognitive efficiency, which is exactly what this
study's symmetric aim requires.

**Group-level estimation.** Because the binary outcome of true/false
items carries limited information, and meta-d′ requires sufficient
correct and incorrect trials within each confidence level,
per-participant item counts are insufficient for stable individual-level
estimation. The study therefore does not estimate meta-d′ at the
individual level, but uses a hierarchical Bayesian approach (HMeta-d) to
estimate at the group level: pooling all trials within a group (the
human group, or a given model) to estimate that group's M-ratio. This
fits naturally with the study's "humans vs. each model" between-group
comparison; the reported values are the group-level M-ratios of the
human group and each model, not individual values. Estimation uses an
existing implementation (e.g., metadPy / HMeta-d) rather than a bespoke
one.

**3.4 Safeguarding Comparability**

Since the study's central claim is a "symmetric comparison" of humans
and models, comparability is the crux of the design and is controlled in
the following respects: (1) same items---humans and models answer
exactly the same questions; (2) same modality---all items are presented
as text, introducing no visual or other modality the model cannot
process; (3) same confidence elicitation---both report on the same
4-point scale, retrospectively; and (4) fixed prompting---the model side
uses a unified template, ruling out the confound of prompt wording.

The boundary of this design must be stated explicitly: the above
measures safeguard behavioral-level comparability---humans and models
are elicited for the same behavior (retrospectively reporting subjective
confidence on the same scale). But the mechanisms producing that report
are not symmetric: a human's confidence arises from genuine second-order
introspection, whereas a model's confidence is produced by its
generative process. What this study can therefore claim is a
behavioral-level difference in metacognitive calibration, not the
equivalence or absence of an internal metacognitive mechanism; this
boundary is developed further in the Discussion.

**3.5 Auxiliary Analysis: Word--Deed Consistency**

As a supplement to the main analysis, for models that can return token
probabilities (logprobs), the study additionally collects a
probability-based confidence, to examine whether the same model's
"verbalized subjective confidence" and "internal token probability"
agree. This analysis does not enter the main human--model comparison; it
answers a separate question: does a model's "expressed confidence"
reflect its internal signal? If the two agree, the credibility of the
verbalized data in the main analysis is strengthened; if they
disagree---that is, the model is internally "hesitant" yet
linguistically confident---this itself constitutes a noteworthy finding,
echoing Steyvers et al.'s (2025) observation that the model "knows but
does not convey." Because some APIs do not expose logprobs, this
auxiliary analysis is performed only for supported models.

**4. Results**

*\[Outline --- to be written once data are collected. Experiment design
should meet the argument in the Introduction.\]*

- Overview of results: brief summary of the key findings (human vs. LLM
  calibration).

- Detailed analysis:

  - Calibration curves and ECE for the human group and each model; which
    of H1/H2/H3 is supported.

  - Group-level M-ratio comparison (humans vs. each model).

  - Breakdown by item type --- locating where divergence appears (esp.
    hallucination-triggering items).

- Comparison across the model tier: does calibration vary with
  capability and/or vendor?

- Auxiliary: verbalized vs. logprob confidence (word--deed consistency)
  for supported models.

- Anomalies or unexpected findings, with possible explanations.

*\[If no data yet, use a clearly labeled illustrative figure
("Conceptual illustration --- not measured data") to show the analysis
pipeline, as in the mid-term slides.\]*

**5. Discussion**

*\[Outline --- interpretation, honest layering of claims, and return to
the larger significance.\]*

- Interpret the results in dialogue with prior work (Steyvers, Griot,
  Scholten).

- What can and cannot be shown: behavioral-level difference (yes) vs.
  internal metacognition (no --- the black-box ceiling); frame this
  boundary itself as a contribution.

- Return to "metacognition is a precondition for inquiry" (Flavell; Tong
  Test).

- Creativity / breaking with common sense --- confined to this section
  as an extension, not part of the measured scope.

**6. Conclusion, Limitations, and Future Work**

*\[Outline for conclusion and limitations --- to be expanded.\]*

- Contributions on three levels: methodological / empirical /
  conceptual.

- Limitations: comparability; behavioral not internal; sample size;
  prompt sensitivity.

**6.1 Future Work: From Calibration to Knowing What One Does Not Know**

A conceptual gap should be stated plainly. The construct this study
measures---metacognitive calibration, the correspondence between
confidence and accuracy---is not identical to the construct that
ultimately motivates the work: knowing what one does not know.
Calibration is a graded, continuous correspondence measured over items a
respondent does attempt; "knowing what one does not know" is closer to
boundary awareness---the capacity to recognize that a question falls
outside one's knowledge altogether. A respondent can be well-calibrated
on items within its competence yet have no awareness of an "unknown
unknown." These two are closely related but not the same, and the
present study does not claim to have collapsed the distance between
them.

The hallucination-triggering items are a first step toward narrowing
this gap. By embedding fictional entities among real but obscure ones
(see §3.1), they probe whether a respondent---human or model---becomes
alert to an entity it could not possibly have encountered, i.e., whether
it registers "I have never heard of this." This targets boundary
awareness more directly than overall ECE does, and the results on this
item type are therefore the most relevant evidence bearing on the larger
question. Nonetheless, this remains an indirect proxy: a correct "false"
judgment on a fictional-entity item does not fully distinguish genuine
boundary awareness from a mere bias toward answering "false."

Two directions are left to future work. First, a more rigorous,
dedicated measurement of boundary awareness itself---rather than
inferring it from calibration on a subset of items---would more
faithfully capture "knowing what one does not know." Second, the
construction of hallucination-triggering items could be made empirical:
a survey could be used to identify the conditions under which models are
most prone to hallucination, and items authored to target those
conditions. This survey-driven item construction was not carried out
here owing to time constraints, but it is a natural extension for
strengthening the validity of the boundary-probing items.

**References**

*\[APA 7, alphabetical --- the 11 sources from the annotated
bibliography plus additions. Verify full author lists,
volume/issue/pages, and DOIs before submission.\]*

---------

***Note on formatting:** sentences shown in blue italics are the two
passages I added to strengthen the "essential challenge" and "core
contribution" per the tutorial (steps 5 and 9). They are additions, not
translations of your existing text --- delete or keep as you see fit.
Grey italics are placeholders/outline for sections awaiting data.*
