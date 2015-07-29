#include "cnn/cnn.h"
#include "cnn/training.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include <iostream>
#include <fstream>
#include <csignal>
#include <algorithm>

#include "bitext.h"
#include "attentional.h"

using namespace cnn;
using namespace std;

bool ctrlc_pressed = false;
void ctrlc_handler(int signal) {
  if (ctrlc_pressed) {
    cerr << "Exiting..." << endl;
    exit(1);
  }
  else {
    cerr << "Ctrl-c pressed!" << endl;
    ctrlc_pressed = true;
  }
}

template <class RNG>
void shuffle(Bitext& bitext, RNG& g) {
  vector<unsigned> indices(bitext.size(), 0);
  for (unsigned i = 0; i < bitext.size(); ++i) {
    indices[i] = i;
  }
  shuffle(indices.begin(), indices.end(), g);
  vector<vector<WordId> > source(bitext.size());
  vector<vector<WordId> > target(bitext.size());
  for (unsigned i = 0; i < bitext.size(); ++i) {
    source[i] = bitext.source_sentences[i];
    target[i] = bitext.target_sentences[i];
  }
  bitext.source_sentences = source;
  bitext.target_sentences = target;
}

int main(int argc, char** argv) {

  namespace po = boost::program_options;
  po::variables_map vm;
  po::options_description opts("Allowed options");
  opts.add_options()
    ("help", "print help message")
    ("lstm_layer_count,l", po::value<unsigned>()->default_value(1), "LSTM layer count")
    ("embedding_dim,e", po::value<unsigned>()->default_value(51), "Dimensionality of both source and target word embeddings. For now these are the same.")
    ("half_annotation_dim,h", po::value<unsigned>()->default_value(251), "Dimensionality of h_forward and h_backward. The full h has twice this dimension.")
    ("output_state_dim,o", po::value<unsigned>()->default_value(53), "Dimensionality of s_j, the state just before outputing target word y_j")
    ("alignment_hidden_dim,a", po::value<unsigned>()->default_value(47), "Dimensionality of the hidden layer in the alignment FFNN")
    ("final_hidden_dim", po::value<unsigned>()->default_value(57), "Dimensionality of the hidden layer in the final FFNN")
    ;
  po::store(po::parse_command_line(argc, argv, opts), vm);
  po::notify(vm);

  if (vm.count("help") || (argc < 2)){
    cout << opts << endl;
    exit(1);
  }
  signal (SIGINT, ctrlc_handler);

  const string corpus_filename = argv[1];
  Bitext bitext;
  ReadCorpus(corpus_filename, bitext, true);
  cerr << "Read " << bitext.size() << " lines from " << corpus_filename << endl;
  cerr << "Vocab size: " << bitext.source_vocab.size() << "/" << bitext.target_vocab.size() << endl; 

  cnn::Initialize(argc, argv);
  std::mt19937 rndeng(42);

  Model model;
  AttentionalModel attentional_model;
  attentional_model.SetParams(vm);
  attentional_model.Initialize(model, bitext.source_vocab.size(), bitext.target_vocab.size());
  //SimpleSGDTrainer sgd(&model, 0.0, 0.1);
  //AdagradTrainer sgd(&model, 0.0, 0.1);
  AdadeltaTrainer sgd(&model, 0.0);
  //AdadeltaTrainer sgd(&model, 0.0, 1e-6, 0.992);
  //RmsPropTrainer sgd(&model, 0.0, 0.1);
  sgd.eta_decay = 0.05;

  cerr << "Training model...\n";
  unsigned minibatch_count = 0;
  const unsigned minibatch_size = 1;
  for (unsigned iteration = 0; iteration < 1000 || false; iteration++) {
    unsigned word_count = 0;
    shuffle(bitext, rndeng);
    double loss = 0.0;
    for (unsigned i = 0; i < bitext.size(); ++i) {
      //cerr << "Reading sentence pair #" << i << endl;
      vector<WordId> source_sentence = bitext.source_sentences[i];
      vector<WordId> target_sentence = bitext.target_sentences[i];
      word_count += bitext.target_sentences[i].size() - 1; // Minus one for <s>
      ComputationGraph hg;
      attentional_model.BuildGraph(source_sentence, target_sentence, hg);
      loss += as_scalar(hg.forward());
      hg.backward();
      if (++minibatch_count == minibatch_size) {
        sgd.update(1.0 / minibatch_size);
        minibatch_count = 0;
      }
      if (ctrlc_pressed) {
        break;
      }
    }
    if (ctrlc_pressed) {
      break;
    }
    cerr << "Iteration " << iteration << " loss: " << loss << " (perp=" << loss/word_count << ")" << endl;
    sgd.update_epoch();
  }

  boost::archive::text_oarchive oa(cout);
  oa & bitext.source_vocab;
  oa & bitext.target_vocab;
  oa << attentional_model;
  oa << model;

  return 0;
}
