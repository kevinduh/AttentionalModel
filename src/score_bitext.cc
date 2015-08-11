#include "cnn/cnn.h"
#include "cnn/training.h"

#include <boost/archive/text_iarchive.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>

#include <iostream>
#include <fstream>

#include "bitext.h"
#include "attentional.h"
#include "utils.h"
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace cnn;
using namespace std;


void trim(vector<string>& tokens, bool removeEmpty) {
  for (unsigned i = 0; i < tokens.size(); ++i) {
    boost::algorithm::trim(tokens[i]);
    if (tokens[i].length() == 0 && removeEmpty) {
      tokens.erase(tokens.begin() + i);
      --i;
    }
  }
}

int main(int argc, char** argv) {
 
  namespace po = boost::program_options;
  po::variables_map vm;
  po::options_description opts("Usage: ./score_bitext modelfile < input \n Allowed options");
  opts.add_options()
    ("help","print help message")
    ("reverse,r", po::value<bool>()->default_value(false), "reverse source/target in input")
    ;
  po::store(po::parse_command_line(argc, argv, opts), vm);
  po::notify(vm);

  if (vm.count("help") || (argc < 2)) {
    cerr << opts << endl;
    exit(1);
  }


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
  AttentionalModel attentional_model;
  ia & attentional_model;
  attentional_model.Initialize(model, source_vocab.size(), target_vocab.size());
  ia & model;

  WordId ksSOS = source_vocab.Convert("<s>");
  WordId ksEOS = source_vocab.Convert("</s>");
  WordId ktSOS = target_vocab.Convert("<s>");
  WordId ktEOS = target_vocab.Convert("</s>");

  unsigned column_source=0;
  unsigned column_reference=1;
  if (vm["reverse"].as<bool>()) {
    column_source=1;
    column_reference=0;
  }
  
  string line;
  unsigned line_id = 0;
  unsigned word_count = 0;
  double loss = 0.0;
  for (; getline(cin, line);) {
    vector<string> parts = tokenize(line, "|||");
    trim(parts, false);

    vector<string> tokens = tokenize(parts[column_source], " ");
    trim(tokens, true);
    cerr << line_id << " : " << boost::algorithm::join(tokens, " ") << " ||| ";
    vector<WordId> source(tokens.size());
    for (unsigned i = 0; i < tokens.size(); ++i) {
      source[i] = source_vocab.Convert(tokens[i]);
    }
    source.insert(source.begin(), ksSOS);
    source.insert(source.end(), ksEOS);

    tokens = tokenize(parts[column_reference], " ");
    trim(tokens, true);
    cerr << boost::algorithm::join(tokens, " ") << " ||| ";
    vector<WordId> reference(tokens.size());
    for (unsigned i = 0; i < tokens.size(); ++i) {
      reference[i] = target_vocab.Convert(tokens[i]);
    }
    reference.insert(reference.begin(), ksSOS);
    reference.insert(reference.end(), ksEOS);


    unsigned wc = reference.size() - 1; // Minus one for <s>
    ComputationGraph hg;
    attentional_model.BuildGraph(source, reference, hg);
    double l = as_scalar(hg.forward());
    loss += l;
    word_count += wc;
    cerr << " loss: " << l << " perp: " << exp(l/wc) << endl;
    cout << line_id << " " << l << " " << exp(l/wc) << endl;
    ++line_id;
  }
  cerr << "TOTAL loss: " << loss << " perp: " << exp(loss/word_count) << endl;
  return 0;
}
