# FinServ Compliance Assistant: Evaluation Report

## Run details

| Item | Value |
|---|---|
| Date | 09 November 2026, 21:29 UTC |
| Setup | `offline` profile |
| Answer model | `extractive-v1` |
| Embedding model | `wordllama-l2-supercat-256` |
| Re-ranker | `feature-v1` |
| Documents | 86 chunks (0 from official documents) |
| Test set | 20 questions, 2 of them deliberately unanswerable |

## Summary

- **6 of 6 quality targets met.**
- **This is the offline baseline.** No LLM was used, and the documents are paraphrased digests, not official text. Answers are quoted sentences, so faithfulness is 1.00 by design and says nothing about LLM behaviour. The retrieval scores are real. Run `make full` on a machine with Ollama for LLM results.
- **Re-ranking did not help.** Best retrieval: Hybrid (RRF) (precision 0.91). Adding the re-ranker changed precision from 0.91 to 0.87.
- **5 of 20 questions need work.** Details are in "Questions that need work" below.

## Scores

| Metric | What it checks | Score | Target | Result |
|---|---|---:|---:|---|
| Faithfulness | Each claim in the answer is supported by a source | 1.00 | 0.85 | Pass |
| Answer relevance | The answer addresses the question | 0.63 | 0.50 | Pass |
| Context precision | The right paragraphs are ranked near the top | 0.87 | 0.70 | Pass |
| Context recall | The right paragraphs are found at all | 1.00 | 0.85 | Pass |
| Required facts present | The answer states the facts it should | 0.81 | 0.80 | Pass |
| Unanswerable questions refused | Out-of-scope questions get a refusal, not a guess | 1.00 | 1.00 | Pass |
| Answerable questions wrongly refused | Lower is better | 0.00 | n/a | n/a |

Response time: median 7 ms, p95 11 ms. Degraded answers: 0.

## Retrieval comparison

Each retrieval step was switched on in turn to see whether it improves results.

| Retrieval method | Context recall | Context precision | Right paragraph ranked first |
|---|---:|---:|---:|
| Dense only | 1.00 | 0.86 | 0.78 |
| BM25 only | 1.00 | 0.90 | 0.83 |
| Hybrid (RRF) | 1.00 | 0.91 | 0.83 |
| Hybrid + rerank (feature) | 1.00 | 0.87 | 0.78 |

## Questions that need work

| ID | Topic | Problem |
|---|---|---|
| Q03 | Capital | Found the right paragraph, but the answer left out the key fact. |
| Q04 | Capital | Found the right paragraph, but the answer left out the key fact. |
| Q11 | KYC | Found the right paragraph, but ranked it below less relevant ones. |
| Q14 | KYC | Found the right paragraph, but the answer left out the key fact. |
| Q18 | Cross-regulation | Found the right paragraph, but the answer left out the key fact. |

### Q03: How does the output floor determine total risk-weighted assets?

- **Problem:** Found the right paragraph, but the answer left out the key fact.
- **Expected paragraphs:** RBC20.4
- **Found, in rank order:** RBC20.4, RBC20.13, RBC20.1, RBC20.3, RBC30.1, Art25(3)
- **Answer given:** This second element is the output floor, and it is subject to the transitional arrangements in RBC90. [S1] Banks must meet three minimum risk-based capital ratios at all times, each measured against risk-weighted assets (RWA). [S3] Approaches used only for the output floor are not counted among a bank's nominated approaches. [S4]

### Q04: How much of earnings must a bank retain if its CET1 ratio is 5.5%?

- **Problem:** Found the right paragraph, but the answer left out the key fact.
- **Expected paragraphs:** RBC30.4
- **Found, in rank order:** RBC30.4, RBC20.1, RBC30.1, LEX10.6, LEX20.1, CR-AMD-Para 116
- **Answer given:** A bank must recalculate its conservation ratio at each distribution date. [S1] With a CET1 ratio of 4.5% to 5.125%, 100% of earnings must be retained. [S1]

### Q11: Can a bank begin a transaction with a customer if customer due diligence has not been followed?

- **Problem:** Found the right paragraph, but ranked it below less relevant ones.
- **Expected paragraphs:** KYC-Para 17(3)
- **Found, in rank order:** KYC-Para 5(2)(v), KYC-Para 61, KYC-Para 17(3), KYC-Para 17(2), KYC-WIRE-Para 74(1)(i), KYC-Para 19
- **Answer given:** Customer due diligence is required when opening an account-based relationship, for occasional transactions of Rs 50,000 or more whether single or connected, and for international money transfers. [S1] Banks must use FATF statements circulated by RBI and public information to identify countries that do not or insufficiently apply the FATF Recommendations, and must apply risk-proportionate enhanced due diligence to relationships and transactions with them. [S2] No transaction or account relationship may begin without following CDD. [S3]

### Q14: Which countries are on the FATF call for action list as circulated by RBI in July 2026?

- **Problem:** Found the right paragraph, but the answer left out the key fact.
- **Expected paragraphs:** FATF-CFA
- **Found, in rank order:** KYC-Para 61, FATF-CFA, FATF-IM, KYC-Para 58, KYC-Para 17(9), FATF-CHG
- **Answer given:** Banks must use FATF statements circulated by RBI and public information to identify countries that do not or insufficiently apply the FATF Recommendations, and must apply risk-proportionate enhanced due diligence to relationships and transactions with them. [S1]

### Q18: For an Upper Layer NBFC, how does the single counterparty limit compare with the Basel large exposure limit?

- **Problem:** Found the right paragraph, but the answer left out the key fact.
- **Expected paragraphs:** NBFC-CR-Para 33, LEX20.1
- **Found, in rank order:** NBFC-CR-Para 33, NBFC-CR-Para 4(5), LEX20.1, NBFC-CR-Para 45, LEX10.1, LEX10.9
- **Answer given:** An Upper Layer NBFC's total exposure to a single counterparty must stay at or below 20% of its eligible capital base at all times. [S1] For an Upper Layer NBFC, a large exposure is the total exposure value to a counterparty or connected group that equals or exceeds 10% of the eligible capital base, which is Tier 1 capital. [S2]

## Limits of these results

- The test set is small (20 questions), so one question moves an average by about 5 points.
- The refusal threshold and the feature re-ranker were tuned on this same test set, so the scores are optimistic. A separate test set written by compliance officers is needed before any release decision.
- Finding the right paragraph is easier in a small corpus with few similar paragraphs. Expect lower recall on full official documents.
- The answer metrics are automatic checks, not a legal review by a person.

## Scores by question

| ID | Topic | Recall | Precision | Faithful | Relevance | Facts |
|---|---|---:|---:|---:|---:|---:|
| Q01 | Capital | 1.00 | 1.00 | 1.00 | 0.47 | 1.00 |
| Q02 | Capital | 1.00 | 1.00 | 1.00 | 0.74 | 1.00 |
| Q03 | Capital | 1.00 | 1.00 | 1.00 | 0.66 | 0.00 |
| Q04 | Capital | 1.00 | 1.00 | 1.00 | 0.68 | 0.00 |
| Q05 | Large exposures | 1.00 | 1.00 | 1.00 | 0.68 | 1.00 |
| Q06 | Large exposures | 1.00 | 0.50 | 1.00 | 0.49 | 1.00 |
| Q07 | Large exposures | 1.00 | 1.00 | 1.00 | 0.62 | 1.00 |
| Q08 | MiFID II | 1.00 | 1.00 | 1.00 | 0.66 | 1.00 |
| Q09 | MiFID II | 1.00 | 1.00 | 1.00 | 0.59 | 1.00 |
| Q10 | MiFID II | 1.00 | 0.50 | 1.00 | 0.71 | 1.00 |
| Q11 | KYC | 1.00 | 0.33 | 1.00 | 0.59 | 1.00 |
| Q12 | KYC | 1.00 | 1.00 | 1.00 | 0.41 | 1.00 |
| Q13 | KYC | 1.00 | 1.00 | 1.00 | 0.56 | 1.00 |
| Q14 | KYC | 1.00 | 0.50 | 1.00 | 0.47 | 0.00 |
| Q15 | Priority sector | 1.00 | 1.00 | 1.00 | 0.79 | 1.00 |
| Q16 | Priority sector | 1.00 | 1.00 | 1.00 | 0.74 | 1.00 |
| Q17 | Priority sector | 1.00 | 1.00 | 1.00 | 0.83 | 1.00 |
| Q18 | Cross-regulation | 1.00 | 0.83 | 1.00 | 0.69 | 0.50 |
| Q19 | Unanswerable | Refused (correct) | | | | |
| Q20 | Unanswerable | Refused (correct) | | | | |

## How each metric is calculated

- **Context recall:** share of expected paragraphs that were retrieved.
- **Context precision:** how high the expected paragraphs rank (average precision).
- **Faithfulness:** share of answer claims where one source contains at least 60% of the claim's words and all of its numbers.
- **Answer relevance:** average of question-answer similarity and the share of question words used in the answer.
- **Required facts present:** share of the labelled facts that appear in the answer.
- **Unanswerable questions refused:** share of out-of-scope questions the system refused.
