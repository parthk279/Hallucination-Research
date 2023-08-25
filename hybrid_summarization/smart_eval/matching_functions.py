# coding=utf-8
# Copyright 2022 The SMART Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Matching functions."""

import functools

from nltk.tokenize import word_tokenize
from nltk.translate import meteor_score
import sacrebleu

# THIS DIDNT WORK, not sure why. trying to replace with rouge_score
# from rouge import rouge_scorer
from rouge_score import rouge_scorer

_ROUGE = rouge_scorer.RougeScorer(rouge_types=['rouge1', 'rouge2', 'rougeL'])

# Matching functions should follow the format below and returns a score
#   between 0 and 1:
# def matcher(reference: str, candidate: str):
#   return 0


def _rouge(reference, candidate, rouge_type):
  return _ROUGE.score(reference, candidate)[rouge_type].fmeasure


rouge_1_matcher = functools.partial(_rouge, rouge_type='rouge1')
rouge_2_matcher = functools.partial(_rouge, rouge_type='rouge2')
rouge_l_matcher = functools.partial(_rouge, rouge_type='rougeL')


def meteor_matcher(reference, candidate):
  return meteor_score.single_meteor_score(
      word_tokenize(reference), word_tokenize(candidate))


def chrf_matcher(reference, candidate):
  # return sacrebleu.sentence_chrf(candidate, reference)
  # I think something happened with the version of sacrebleu, gets mad if reference is NOT a list  
  return sacrebleu.sentence_chrf(candidate, [reference])


def bleu_matcher(reference, candidate):
  return sacrebleu.sentence_bleu(candidate, reference)
