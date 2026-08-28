# Candidate Pool Design for Safer Preference-Based Post-Training of LLM Companion Chatbots

Method code for an Engaged-Neutral candidate-pool example in preference-based post-training for companion chatbots.

## Research Question

This project asks whether candidate-pool design can shape the geometry between Supportiveness and Social Risk before preference optimization begins. In this setting, high-support responses often validate the user, take the user's side, and provide concrete escalation plans; lower-risk responses preserve uncertainty, boundaries, and de-escalation even when they are less immediately satisfying.

## Method

The code follows the research pipeline at a method level. `part-1-simulation` generates structured Chinese-language personas using a simplified schema adapted from the core TinyPerson persona fields in TinyTroupe, compresses them into minibios, and turns them into realistic first-message user questions. The prompts in `common/prompts.py` define the persona schema, question variants, answer sources, and the independent Supportiveness / Social Risk judge rubric. The appendix prompts in the paper are English renderings of the Chinese prompt templates implemented in `common/prompts.py`.

This code release is centered on the Engaged-Neutral construction example. The answer sources are named directly in code: `engaged` is the support-conditioned companion response, `neutral` is the ordinary general-assistant response, and `guarded` is the lower-risk boundary-preserving response used as a contrastive source. `part-2-preprocess` scores candidate answers, selects winners and losers under `support_maximizing`, `risk_minimizing`, and `risk_bounded_support` objectives. For `risk_bounded_support`, the intended rule is to select the highest-S response among candidates with R≤3, and to select the lowest-R response when no candidate satisfies R≤3. The selected examples are exported into SFT, DPO, and KTO formats. `part-3-training` contains the corresponding TRL training wrappers, while `part-4-judgement` contains the checkpoint-level generation and judging flow.

## API

The omitted `common.api` module was only a minimal OpenAI-compatible chat-completions wrapper. It sent chat messages, requested JSON output when needed, retried failed calls, and recorded model names. There is no special API mechanism in the method; any OpenAI-compatible chat-completion endpoint can replace it.

## Local Configuration

Machine-specific runtime settings such as directory paths, shell initialization, queue locations, and local evaluation output paths are intentionally left to the user. In particular, `common/config.py` and `part-4-judgement/paths.py` should be adapted to the local environment; these files only define local execution settings and do not encode method-specific logic.
