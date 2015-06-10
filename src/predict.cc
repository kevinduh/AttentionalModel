#include "cnn/cnn.h"
#include "cnn/training.h"

#include <boost/archive/text_iarchive.hpp>

#include <iostream>
#include <fstream>
#include <csignal>

#include "bitext.h"
#include "attentional.h"
#include "utils.h"

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
  source_vocab.Freeze();
  target_vocab.Freeze();

  Model model;
  AttentionalModel attentional_model(model, source_vocab.size(), target_vocab.size());

  ia & attentional_model;
  ia & model;

  WordId ktSOS = target_vocab.Convert("<s>");
  WordId ktEOS = target_vocab.Convert("</s>");

  string line;
  for (; getline(cin, line);) {
    vector<string> tokens = tokenize(line, " ");
    vector<WordId> source(tokens.size());
    for (unsigned i = 0; i < tokens.size(); ++i) {
      source[i] = source_vocab.Convert(tokens[i]);
    }
    cerr << "Read source sentence: ";
    for (unsigned i = 0; i < tokens.size(); ++i) {
      if (i != 0) {
        cerr << " ";
      }
      cerr << tokens[i];
    }
    cerr << endl;

    ComputationGraph hg;
    vector<WordId> target = attentional_model.Translate(source, ktSOS, ktEOS, 10, hg);
    cout << "Translation: ";
    for (unsigned i = 0; i < target.size(); ++i) {
      if (i != 0) {
        cout << " ";
      }
      cout << target_vocab.Convert(target[i]);
    }
    cout << endl;

    if (ctrlc_pressed) {
      break;
    }
  }

  return 0;
}
