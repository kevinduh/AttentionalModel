#include "cnn/dict.h"
#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/lstm.h"
#include "utils.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/map.hpp>

#include <iostream>
#include <fstream>
#include <unordered_set>
#include <climits>
#include <csignal>

#define NONLINEAR
#define FAST

using namespace std;
using namespace cnn;

bool ctrlc_pressed = false;
void ctrlc_handler(int signal) {
  if (ctrlc_pressed) {
    exit(1);
  }
  else {
    ctrlc_pressed = true;
  }
}

bool ReadCorpus(string filename, vector<vector<unsigned> >* corpus, Dict* vocab) {
  ifstream f(filename);
  if (!f.is_open()) {
    return false;
  }

  for (string line; getline(f, line);) {
    vector<string> tokens;
    tokens.reserve(line.size() + 2);
    tokens.push_back("<s>");
    string spaced;
    unsigned i = 0;
    while (i < line.size()) {
      if (i != 0) {
        spaced += " ";
      }
      unsigned size = UTF8Len(line[i]);
      tokens.push_back(line.substr(i, size));
      i += size;
    }
    tokens.push_back("</s>");

    vector<unsigned> word_ids;
    word_ids.reserve(tokens.size());
    for (string token : tokens) {
      word_ids.push_back(vocab->Convert(token));
    }
    corpus->push_back(word_ids);
  }
  f.close();
  return true;
}

class LSTMLanguageModel {
public:
  LSTMLanguageModel(Model& model, unsigned layer_count, unsigned input_dim, unsigned hidden_dim, unsigned vocab_size);
  VariableIndex BuildGraph(const vector<unsigned>& sentence, ComputationGraph& hg);
  vector<unsigned> SampleSentence(unsigned kSOS, unsigned kEOS, unsigned max_length);
private:
  LSTMBuilder builder;
  LookupParameters* p_c; //input word vectors
  Parameters* p_R; // hidden layer -> output layer weights
  Parameters* p_b; // output layer bias
};

LSTMLanguageModel::LSTMLanguageModel(Model& model, unsigned layer_count, unsigned input_dim, unsigned hidden_dim, unsigned vocab_size)
    : builder(layer_count, input_dim, hidden_dim, &model) {
  p_c = model.add_lookup_parameters(vocab_size, {input_dim});
  p_R = model.add_parameters({vocab_size, hidden_dim});
  p_b = model.add_parameters({vocab_size});
}

VariableIndex LSTMLanguageModel::BuildGraph(const vector<unsigned>& sentence, ComputationGraph& hg) {
  builder.new_graph(&hg);
  builder.start_new_sequence();
  VariableIndex i_R = hg.add_parameters(p_R);
  VariableIndex i_b = hg.add_parameters(p_b);
  vector<VariableIndex> errors;
  for (unsigned t = 0; t < sentence.size() - 1; ++t) {
    VariableIndex i_x_t = hg.add_lookup(p_c, sentence[t]);
    VariableIndex i_y_t = builder.add_input(i_x_t, &hg);
    VariableIndex i_r_t = hg.add_function<AffineTransform>({i_b, i_R, i_y_t});
    VariableIndex error = hg.add_function<PickNegLogSoftmax>({i_r_t}, sentence[t + 1]);
    errors.push_back(error);
  }

  VariableIndex i_total_error = hg.add_function<Sum>(errors);
  return i_total_error;
}

vector<unsigned> LSTMLanguageModel::SampleSentence(unsigned kSOS, unsigned kEOS, unsigned max_length) {
  ComputationGraph hg;
  builder.new_graph(&hg);
  builder.start_new_sequence();
  VariableIndex i_R = hg.add_parameters(p_R);
  VariableIndex i_b = hg.add_parameters(p_b);
  vector<unsigned> sentence;
  sentence.push_back(kSOS);
  unsigned prev_word = kSOS;
  while (prev_word != kEOS && sentence.size() < max_length) {
    VariableIndex i_x_t = hg.add_lookup(p_c, prev_word);
    VariableIndex i_y_t = builder.add_input(i_x_t, &hg);
    VariableIndex i_r_t = hg.add_function<AffineTransform>({i_b, i_R, i_y_t});
    hg.add_function<Softmax>({i_r_t});
    vector<float> dist = as_vector(hg.incremental_forward()); 

    unsigned w = 0;
    while (w == kSOS) {
      double r = rand01();
     //cerr << "r=" << r << "; prob[kSOS]=" << dist[kSOS] << "; prob[kEOS]=" << dist[kEOS] << "; d=" << dist.size() << "\n";
      while (true) {
        r -= dist[w];
        if (r < 0.0) {
          break;
        }
        ++w;
      }
    }
    sentence.push_back(w);
    prev_word = w;
  }
  return sentence;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " corpus.txt" << endl;
    cerr << endl;
    exit(1);
  }
  signal (SIGINT, ctrlc_handler);

  const string corpus_filename = argv[1];
  Dict vocabulary;
  const unsigned kSOS = vocabulary.Convert("<s>");
  const unsigned kEOS = vocabulary.Convert("</s>");
  vector<vector<unsigned> > corpus;
  ReadCorpus(corpus_filename, &corpus, &vocabulary);
  cerr << "Read " << corpus.size() << " lines from " << corpus_filename << endl;
  cerr << "Vocab size: " << vocabulary.size() << endl;

  const unsigned layer_count = 2;
  const unsigned input_dim = 32;
  const unsigned hidden_dim = 96;
  const unsigned vocab_size = vocabulary.size();
  const double learning_rate = 1.0e-1;

  cnn::Initialize(argc, argv);
  Model model;
  LSTMLanguageModel lm(model, layer_count, input_dim, hidden_dim, vocab_size);
  SimpleSGDTrainer sgd(&model);

  cerr << "Training model...\n";
  for (unsigned iteration = 0; iteration < 1000 || true; iteration++) {
    shuffle(corpus.begin(), corpus.end(), *rndeng);
    double loss = 0.0;
    for (vector<unsigned> sentence : corpus) {
      ComputationGraph hg;
      lm.BuildGraph(sentence, hg);
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

  for (unsigned i = 0; i < 10; ++i) {
    vector<unsigned> sentence = lm.SampleSentence(kSOS, kEOS, 100);
    for (unsigned j = 0; j < sentence.size(); j++) {
      unsigned w = sentence[j];
      cerr << vocabulary.Convert(w);
    }
    cerr << "\n";
  }

  //boost::archive::text_oarchive oa(cout);
  //oa << num_dimensions << hidden_size << p_w1 << p_w2 << p_b;

  return 0;
}
