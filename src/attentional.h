#include <vector>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options/variables_map.hpp>
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/lstm.h"
#include "kbestlist.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

struct OutputState {
  // State is the LSTM state
  Expression state;
  // Context is a weighted sum of annotation vectors
  Expression context;
};

// A simple 2 layer MLP
struct MLP {
  Expression i_IH;
  Expression i_Hb;
  Expression i_HO;
  Expression i_Ob;
};

struct PartialHypothesis {
  vector<WordId> hyp;
  OutputState os;
};

class AttentionalModel {
public:
  AttentionalModel(){}
  void Initialize(Model& model, unsigned src_vocab_size, unsigned tgt_vocab_size);
  void SetParams(boost::program_options::variables_map vm);
  vector<Expression> BuildForwardAnnotations(const vector<WordId>& sentence, ComputationGraph& hg);
  vector<Expression> BuildReverseAnnotations(const vector<WordId>& sentence, ComputationGraph& hg);
  vector<Expression> BuildAnnotationVectors(const vector<Expression>& forward_contexts, const vector<Expression>& reverse_contexts, ComputationGraph& hg);
  OutputState GetNextOutputState(const Expression& context, const vector<Expression>& annotations, const MLP& aligner, ComputationGraph& hg, vector<float>* out_alignment = NULL);
  Expression ComputeOutputDistribution(const WordId prev_word, const Expression state, const Expression context, const MLP& final, ComputationGraph& hg);
  Expression BuildGraph(const vector<WordId>& source, const vector<WordId>& target, ComputationGraph& hg);
  void GetParams() const;

  vector<vector<float> > Align(const vector<WordId>& source, const vector<WordId>& target);
  vector<WordId> SampleTranslation(const vector<WordId>& source, WordId kSOS, WordId kEOS, unsigned max_length);
  vector<WordId> Translate(const vector<WordId>& source, WordId kSOS, WordId kEOS, unsigned beam_size, unsigned max_length);
  KBestList<vector<WordId> > TranslateKBest(const vector<WordId>& source, WordId kSOS, WordId kEOS, unsigned k, unsigned beam_size, unsigned max_length);

private:

  unsigned lstm_layer_count;
  unsigned embedding_dim; // Dimensionality of both source and target word embeddings. For now these are the same.
  unsigned half_annotation_dim; // Dimensionality of h_forward and h_backward. The full h has twice this dimension.
  unsigned output_state_dim; // Dimensionality of s_j, the state just before outputing target word y_j
  unsigned alignment_hidden_dim; // Dimensionality of the hidden layer in the alignment FFNN
  unsigned final_hidden_dim; // Dimensionality of the hidden layer in the "final" FFNN

  LSTMBuilder forward_builder, reverse_builder, output_builder;
  LookupParameters* p_Es; // source language word embedding matrix
  LookupParameters* p_Et; // target language word embedding matrix
  Parameters* p_aIH; // Alignment NN weight matrix between the input and hidden layers
  Parameters* p_aHb; // Alignment NN hidden layer bias
  Parameters* p_aHO; // Alignment NN weight matrix between the hidden and output layers
  Parameters* p_aOb; // Alignment NN output layer bias;
  Parameters* p_Ws; // Used to compute p_0 from h_backwards_0
  Parameters* p_bs; // Used to compute p_0 from h_backwars_0
  Parameters* p_fIH; // "Final" NN (from the tuple (y_{i-1}, s_i, c_i) to the distribution over output words y_i), input->hidden weights
  Parameters* p_fHb; // Same, hidden bias
  Parameters* p_fHO; // Same, hidden->output weights
  Parameters* p_fOb; // Same, output bias

  friend class boost::serialization::access;
  template<class Archive> void serialize(Archive& ar, const unsigned int) {
    ar & lstm_layer_count;
    ar & embedding_dim;
    ar & half_annotation_dim;
    ar & output_state_dim;
    ar & alignment_hidden_dim;
    ar & final_hidden_dim;
  }
};
