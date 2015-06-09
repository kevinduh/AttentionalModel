#include "cnn/cnn.h"
#include "cnn/training.h"

#include <boost/archive/text_iarchive.hpp>

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
    cerr << "Usage: cat source.txt | " << argv[0] << " model" << endl;
    cerr << endl;
    exit(1);
  }
  signal (SIGINT, ctrlc_handler);

  const string model_filename = argv[1];
  ifstream model_file(model_filename);
  if (!model_file.is_open()) {
    cerr << "ERROR: Unable to open " << model_filename << endl;
    exit(1);
  }
  boost::archive::text_iarchive ia(model_file);

  cnn::Initialize(argc, argv);
  Dict source_vocab;
  Dict target_vocab;
  ia & source_vocab;
  ia & target_vocab;

  Model model;
  AttentionalModel attentional_model(model, source_vocab.size(), target_vocab.size());

  ia & attentional_model;
  ia & model;

  cout << source_vocab.Convert(3) << endl;

  /*for (unsigned i = 0; i < bitext.size(); ++i) { 
    vector<WordId> source_sentence = bitext.source_sentences[i];
    ComputationGraph hg;
    attentional_model.BuildGraph(source_sentence, target_sentence, hg);
    if (ctrlc_pressed) {
      break;
    } 
  }*/

  return 0;
}
