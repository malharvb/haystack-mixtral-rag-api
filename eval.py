from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu


# def bleu(ref, gen):
#     """
#     calculate pair wise bleu score. uses nltk implementation
#     Args:
#         references : a list of reference sentences
#         candidates : a list of candidate(generated) sentences
#     Returns:
#         bleu score(float)
#     """
#     ref_bleu = []
#     gen_bleu = []
#     for l in gen:
#         gen_bleu.append(l.split())
#     for i, l in enumerate(ref):
#         ref_bleu.append([l.split()])
#     cc = SmoothingFunction()
#     score_bleu = corpus_bleu(
#         ref_bleu, gen_bleu, weights=(0, 1, 0, 0), smoothing_function=cc.method4
#     )
#     return score_bleu


# rouge scores for a reference/generated sentence pair
# source google seq2seq source code.

import itertools


# supporting function
def _split_into_words(sentences):
    """Splits multiple sentences into words and flattens the result"""
    return list(itertools.chain(*[_.split(" ") for _ in sentences]))


# supporting function
def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences."""
    assert len(sentences) > 0
    assert n > 0

    words = _split_into_words(sentences)
    return _get_ngrams(n, words)


# supporting function
def _get_ngrams(n, text):
    """Calcualtes n-grams.
    Args:
      n: which n-grams to calculate
      text: An array of tokens
    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i : i + n]))
    return ngram_set


def rouge_n(reference_sentences, evaluated_sentences, n=2):
    """
    Computes ROUGE-N of two text collections of sentences.
    Source: http://research.microsoft.com/en-us/um/people/cyl/download/
    papers/rouge-working-note-v1.3.1.pdf
    Args:
      evaluated_sentences: The sentences that have been picked by the summarizer
      reference_sentences: The sentences from the reference set
      n: Size of ngram.  Defaults to 2.
    Returns:
      recall rouge score(float)
    Raises:
      ValueError: raises exception if a param has len <= 0
    """
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences)
    reference_ngrams = _get_word_ngrams(n, reference_sentences)
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    # Gets the overlapping ngrams between evaluated and reference
    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    # Handle edge case. This isn't mathematically correct, but it's good enough
    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

    # just returning recall count in rouge, useful for our purpose
    return recall, precision, f1_score


def main():
    # Assignment and Assumption - Indemnification, No Amendment, Waiver
    # NDA - Disclosure to Company Representatives, Indemnity, Compelled Disclosure to Authorities
    # Arbitration Agreement - Severability, Waiver, Enforceability, Interpretation, Amendments

    ref_list = [
        "Each Party shall indemnify and hold harmless other Party and Licensor against any and all loss, liability, damage or expenses which may be incurred by other Party and Licensor due to any claims of a third party in connection with the breach, default or non-performance of the SLA by a Party on or after the date of execution of this Agreement.",
        "Parties agree that they may not amend this Agreement in any way that will materially affect the Licensor’s rights or provisions of the SLA.",
        "No failure to enforce any of its right by a Party shall not constitute waiver of such right.",
        "The Company, in its discretion, may disclose the Confidential Information to its employees, officers, agents and third party consultants on a need to know basis in furtherance of the Project.",
        "The Corporation shall protect, indemnify, and hold the Company harmless from any loss suffered by the Company due to the authorized use of the Confidential Information by the Company.",
        "To the extent absolutely necessary, the Company may disclose Confidential Information, if required by any judicial or governmental request, requirement or order; provided that the Company will provide with sufficient prior written notice of such request, requirement or order to the Corporation so that the Corporation shall contest such request, requirement or order to oppose the disclosure of Confidential Information. The Company shall cooperate with the Corporation in its attempts to oppose such disclosure.",
        "If any provision of this Agreement is or becomes illegal, invalid or unenforceable in any jurisdiction, the illegality, invalidity or unenforceability of that provision will not affect the validity of any other provision of this Agreement.",
        "No failure to enforce any of its right by a Party shall not constitute waiver of such right.",
        "This Agreement contains a binding arbitration provision that may be enforced by either Party.",
        "This Agreement shall be interpreted in favor of arbitration. No rule of construction shall be applied to undermine any presumption in favor of enforcement of this Agreement or in favor of arbitration.",
        "No amendment, supplement, modification or restatement of any provision of this Agreement shall be binding unless it is in writing and signed by each Party."
    ]

    gen_list = [
        "The Assignee agrees to indemnify and hold harmless the Assignor, its affiliates, and their respective officers, directors, and employees from any and all claims, losses, liabilities, damages, expenses, and costs (including attorneys' fees and court costs) arising out of or related to the Assignee's breach of this Agreement.",
        "This Agreement may not be amended or modified except in writing executed by both Parties.",
        "Any waiver of any provision of this Agreement must be in writing and signed by the Party against whom the waiver is sought to be enforced. No waiver of any provision shall be deemed a further or continuing waiver of such provision or a waiver of any other provision.",
        "The company may disclose Confidential Information to its employees, agents, and representatives who need to know the information for the purposes of this Agreement, provided that such individuals are bound by confidentiality obligations no less restrictive than those set forth herein. The company remains responsible for any breach of confidentiality by its representatives.",
        "Each party agrees to indemnify and hold harmless the other party, its affiliates, directors, officers, and employees from any claims, liabilities, damages, or expenses, including reasonable attorneys' fees, arising out of their breach of this agreement.",
        "In the event that a Party is required by law to disclose Confidential Information to a governmental or regulatory authority, it shall promptly notify the other Party and use reasonable efforts to limit the disclosure to the extent required by law",
        "If any provision of this Agreement is held invalid or unenforceable, such provision shall be struck and the remaining provisions shall be enforced",
        "Failure to enforce any provision of this Agreement shall not constitute a waiver of any other provision, and the waiver of any right or remedy on any occasion shall not be construed as a bar to or waiver of any right or remedy on any future occasion.",
        "The Parties agree that any award rendered by the arbitrator shall be final and binding on the Parties and enforceable in any court of competent jurisdiction.",
        "In this Arbitration Agreement, unless the context otherwise requires, \"Parties\" refers to the Employee and the Company, \"Arbitration\" refers to the process of resolving disputes through final and binding arbitration conducted under the Indian Arbitration and Conciliation Act, 1996, and \"Arbitrator\" refers to a mutually-agreeable neutral arbitrator who shall be a retired judge of Chennai High Court.",
        "Any amendment to this Agreement must be in writing and signed by both Parties."
    ]

    # bleu_scores = []
    rouge_recalls = []
    rouge_precisions = []
    rouge_f1_scores = []

    for i, (ref, gen) in enumerate(zip(ref_list, gen_list), 1):
        # bleu_score = bleu(ref, gen)
        rouge_recall, rouge_precision, rouge_f1 = rouge_n(ref, gen, 2)
        # print(f"BLEU Score {i}: {bleu_score}")
        print(f"ROUGE-N Recall {i}: {rouge_recall}")
        print(f"ROUGE-N Precision {i}: {rouge_precision}")
        print(f"ROUGE-N F1 Score {i}: {rouge_f1}")

        # bleu_scores.append(bleu_score)
        rouge_recalls.append(rouge_recall)
        rouge_precisions.append(rouge_precision)
        rouge_f1_scores.append(rouge_f1)

    # print(f"Average BLEU Score: {sum(bleu_scores) / len(bleu_scores)}")
    print(f"Average ROUGE-N Recall: {sum(rouge_recalls) / len(rouge_recalls)}")
    print(f"Average ROUGE-N Precision: {sum(rouge_precisions) / len(rouge_precisions)}")
    print(f"Average ROUGE-N F1 Score: {sum(rouge_f1_scores) / len(rouge_f1_scores)}")


if __name__ == "__main__":
    main()
