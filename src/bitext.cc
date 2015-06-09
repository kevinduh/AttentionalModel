#include <fstream>
#include "bitext.h"

using namespace std;

unsigned Bitext::size() const {
  assert(source_sentences.size() == target_sentences.size());
  return source_sentences.size();
}

bool ReadCorpus(string filename, Bitext& bitext, bool add_bos_eos) {
  ifstream f(filename);
  if (!f.is_open()) {
    return false;
  }

  WordId sBOS, sEOS, tBOS, tEOS;
  if (add_bos_eos) {
    sBOS = bitext.source_vocab.Convert("<s>");
    sEOS = bitext.source_vocab.Convert("</s>");
    tBOS = bitext.target_vocab.Convert("<s>");
    tEOS = bitext.target_vocab.Convert("</s>");
  }

  for (string line; getline(f, line);) {
    vector<WordId> source;
    vector<WordId> target;
    if (add_bos_eos) {
      source.push_back(sBOS);
      target.push_back(tBOS);
    }
    ReadSentencePair(line, &source, &bitext.source_vocab, &target, &bitext.target_vocab);
    if (add_bos_eos) {
      source.push_back(sEOS);
      target.push_back(tEOS);
    }
    bitext.source_sentences.push_back(source);
    bitext.target_sentences.push_back(target);
  }
  return true;
}
