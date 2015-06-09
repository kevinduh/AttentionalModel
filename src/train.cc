#include "cnn/cnn.h"
#include "cnn/training.h"

#include <boost/archive/text_oarchive.hpp>

#include <iostream>
#include <fstream>
#include <csignal>

#include "bitext.h"
#include "attentional.h"

using namespace cnn;
using namespace std;

bool ctrlc_pressed = false;
void ctrlc_handler(int signal) {
  if (ctrlc_pressed) {
    exit(1);
  }
  else {
    ctrlc_pressed = true;
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " corpus.txt" << endl;
    cerr << endl;
    exit(1);
  }
  signal (SIGINT, ctrlc_handler);

  const string corpus_filename = argv[1];
  Bitext bitext;
  ReadCorpus(corpus_filename, bitext, true);
  cerr << "Read " << bitext.size() << " lines from " << corpus_filename << endl;
  cerr << "Vocab size: " << bitext.source_vocab.size() << "/" << bitext.target_vocab.size() << endl;

  const double learning_rate = 1.0e-1;

  cnn::Initialize(argc, argv);
  Model model;
  AttentionalModel attentional_model(model, bitext.source_vocab.size(), bitext.target_vocab.size());
  SimpleSGDTrainer sgd(&model);

  cerr << "Training model...\n";
  for (unsigned iteration = 0; iteration < 1000 || true; iteration++) {
    //shuffle(corpus.begin(), corpus.end(), *rndeng);
    double loss = 0.0;
    for (unsigned i = 0; i < bitext.size(); ++i) {
      //cerr << "Reading sentence pair #" << i << endl;
      vector<WordId> source_sentence = bitext.source_sentences[i];
      vector<WordId> target_sentence = bitext.target_sentences[i];
      ComputationGraph hg;
      attentional_model.BuildGraph(source_sentence, target_sentence, hg);
      loss += as_scalar(hg.forward());
      hg.backward();
      sgd.update(learning_rate);
      if (ctrlc_pressed) {
        break;
      }
    }
    if (ctrlc_pressed) {
      break;
    }
    cerr << "Iteration " << iteration << " loss: " << loss << endl;
    sgd.update_epoch();
  }

  boost::archive::text_oarchive oa(cout);
  oa & bitext.source_vocab;
  oa & bitext.target_vocab;
  oa << attentional_model;
  oa << model;

  return 0;
}
