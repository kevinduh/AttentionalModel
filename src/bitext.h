#pragma once
#include <vector>
#include "cnn/dict.h"

using namespace std;
using namespace cnn;

typedef int WordId;

struct Bitext {
  vector<vector<WordId> > source_sentences;
  vector<vector<WordId> > target_sentences;
  Dict source_vocab;
  Dict target_vocab;

  unsigned size() const;
};

bool ReadCorpus(string filename, Bitext& bitext, bool add_bos_eos);
