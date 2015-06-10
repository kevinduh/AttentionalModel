#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/lstm.h"

#include "bitext.h"
#include "attentional.h"

using namespace std;
using namespace cnn;

AttentionalModel::AttentionalModel(Model& model, unsigned src_vocab_size, unsigned tgt_vocab_size) {
  forward_builder = LSTMBuilder(lstm_layer_count, embedding_dim, half_annotation_dim, &model);
  reverse_builder = LSTMBuilder(lstm_layer_count, embedding_dim, half_annotation_dim, &model);
  output_builder = LSTMBuilder(lstm_layer_count, 2 * half_annotation_dim, output_state_dim, &model);
  p_Es = model.add_lookup_parameters(src_vocab_size, {embedding_dim});
  p_Et = model.add_lookup_parameters(tgt_vocab_size, {embedding_dim});
  p_aIH = model.add_parameters({alignment_hidden_dim, output_state_dim + 2 * half_annotation_dim});
  p_aHb = model.add_parameters({alignment_hidden_dim, 1});
  p_aHO = model.add_parameters({1, alignment_hidden_dim});
  p_aOb = model.add_parameters({1, 1});
  // The paper says s_0 = tanh(Ws * h1_reverse), and that Ws is an N x N matrix, but somehow below implies Ws is 2N x N.
  p_Ws = model.add_parameters({2 * half_annotation_dim, half_annotation_dim});
  p_bs = model.add_parameters({2 * half_annotation_dim});

  p_fIH = model.add_parameters({final_hidden_dim, embedding_dim + 2 * half_annotation_dim + output_state_dim});
  p_fHb = model.add_parameters({final_hidden_dim});
  p_fHO = model.add_parameters({tgt_vocab_size, final_hidden_dim});
  p_fOb = model.add_parameters({tgt_vocab_size});
}

vector<VariableIndex> AttentionalModel::BuildForwardAnnotations(const vector<WordId>& sentence, ComputationGraph& hg) {
  forward_builder.new_graph(&hg);
  forward_builder.start_new_sequence();
  vector<VariableIndex> forward_annotations(sentence.size());
  for (unsigned t = 0; t < sentence.size(); ++t) {
    VariableIndex i_x_t = hg.add_lookup(p_Es, sentence[t]);
    VariableIndex i_y_t = forward_builder.add_input(i_x_t, &hg);
    forward_annotations[t] = i_y_t;
  }
  return forward_annotations;
}

vector<VariableIndex> AttentionalModel::BuildReverseAnnotations(const vector<WordId>& sentence, ComputationGraph& hg) {
  reverse_builder.new_graph(&hg);
  reverse_builder.start_new_sequence();
  vector<VariableIndex> reverse_annotations(sentence.size());
  for (unsigned t = sentence.size(); t > 0; ) {
    t--;
    VariableIndex i_x_t = hg.add_lookup(p_Es, sentence[t]);
    VariableIndex i_y_t = reverse_builder.add_input(i_x_t, &hg);
    reverse_annotations[t] = i_y_t;
  }
  return reverse_annotations;
}

vector<VariableIndex> AttentionalModel::BuildAnnotationVectors(const vector<VariableIndex>& forward_annotations, const vector<VariableIndex>& reverse_annotations, ComputationGraph& hg) {
  vector<VariableIndex> annotations(forward_annotations.size());
  for (unsigned t = 0; t < forward_annotations.size(); ++t) {
    const VariableIndex& i_f = forward_annotations[t];
    const VariableIndex& i_r = reverse_annotations[t];
    VariableIndex i_h = hg.add_function<Concatenate>({i_f, i_r});
    annotations[t] = i_h;
  }
  return annotations;
}

OutputState AttentionalModel::GetNextOutputState(const VariableIndex& prev_state, const vector<VariableIndex>& annotations, const MLP& aligner, ComputationGraph& hg) {
  const unsigned source_size = annotations.size();

  VariableIndex new_state = output_builder.add_input(prev_state, &hg);
  vector<VariableIndex> unnormalized_alignments(source_size); // e_ij
  for (unsigned s = 0; s < source_size; ++s) {
    VariableIndex a_input = hg.add_function<Concatenate>({new_state, annotations[s]});
    VariableIndex a_hidden1 = hg.add_function<AffineTransform>({aligner.i_Hb, aligner.i_IH, a_input});
    VariableIndex a_hidden2 = hg.add_function<Tanh>({a_hidden1});
    VariableIndex a_output = hg.add_function<AffineTransform>({aligner.i_Ob, aligner.i_HO, a_hidden2});
    unnormalized_alignments[s] = a_output;
  }

  VariableIndex unnormalized_alignment_vector = hg.add_function<Concatenate>(unnormalized_alignments);
  VariableIndex normalized_alignment_vector = hg.add_function<Softmax>({unnormalized_alignment_vector});
  VariableIndex annotation_matrix = hg.add_function<ConcatenateColumns>(annotations);
  VariableIndex context = hg.add_function<SumColumns>({annotation_matrix, normalized_alignment_vector});

  OutputState os;
  os.state = new_state;
  os.context = context;
  return os;
}

VariableIndex AttentionalModel::ComputeOutputDistribution(const WordId prev_word, const VariableIndex state, const VariableIndex context, const MLP& final, ComputationGraph& hg) {
  VariableIndex prev_target_embedding = hg.add_lookup(p_Et, prev_word);
  VariableIndex final_input = hg.add_function<Concatenate>({prev_target_embedding, state, context});
  VariableIndex final_hidden1 = hg.add_function<AffineTransform>({final.i_Hb, final.i_IH, final_input}); 
  VariableIndex final_hidden2 = hg.add_function<Tanh>({final_hidden1});
  VariableIndex final_output = hg.add_function<AffineTransform>({final.i_Ob, final.i_HO, final_hidden2});
  return final_output;
}

vector<WordId> AttentionalModel::Translate(const vector<WordId>& source, WordId kSOS, WordId kEOS, unsigned max_length, ComputationGraph& hg) {
  output_builder.new_graph(&hg);
  output_builder.start_new_sequence();

  vector<VariableIndex> forward_annotations = BuildForwardAnnotations(source, hg);
  vector<VariableIndex> reverse_annotations = BuildReverseAnnotations(source, hg);
  vector<VariableIndex> annotations = BuildAnnotationVectors(forward_annotations, reverse_annotations, hg);

  VariableIndex i_aIH = hg.add_parameters(p_aIH);
  VariableIndex i_aHb = hg.add_parameters(p_aHb);
  VariableIndex i_aHO = hg.add_parameters(p_aHO);
  VariableIndex i_aOb = hg.add_parameters(p_aOb);
  MLP aligner = {i_aIH, i_aHb, i_aHO, i_aOb};

  VariableIndex i_fIH = hg.add_parameters(p_fIH);
  VariableIndex i_fHb = hg.add_parameters(p_fHb);
  VariableIndex i_fHO = hg.add_parameters(p_fHO);
  VariableIndex i_fOb = hg.add_parameters(p_fOb);
  MLP final = {i_fIH, i_fHb, i_fHO, i_fOb};

  VariableIndex i_bs = hg.add_parameters(p_bs);
  VariableIndex i_Ws = hg.add_parameters(p_Ws);

  VariableIndex first_input = hg.add_function<AffineTransform>({i_bs, i_Ws, reverse_annotations[0]});
  VariableIndex input = first_input;
  vector<WordId> output;
  unsigned prev_word = kSOS;
  while (prev_word != kEOS && output.size() < max_length) {
    OutputState os = GetNextOutputState(input, annotations, aligner, hg);
    VariableIndex log_output_distribution = ComputeOutputDistribution(prev_word, os.state, os.context, final, hg);
    VariableIndex output_distribution = hg.add_function<Softmax>({log_output_distribution});
    vector<float> dist = as_vector(hg.incremental_forward());
    double r = rand01();
    unsigned w = 0;
    while (true) {
      r -= dist[w];
      if (r < 0.0) {
        break;
      }
      ++w;
    }
    output.push_back(w);
    prev_word = w;
    input = os.context;
  }
  return output;
}

VariableIndex AttentionalModel::BuildGraph(const vector<WordId>& source, const vector<WordId>& target, ComputationGraph& hg) {
  output_builder.new_graph(&hg);
  output_builder.start_new_sequence();

  vector<VariableIndex> forward_annotations = BuildForwardAnnotations(source, hg);
  vector<VariableIndex> reverse_annotations = BuildReverseAnnotations(source, hg);
  vector<VariableIndex> annotations = BuildAnnotationVectors(forward_annotations, reverse_annotations, hg);

  VariableIndex i_aIH = hg.add_parameters(p_aIH);
  VariableIndex i_aHb = hg.add_parameters(p_aHb);
  VariableIndex i_aHO = hg.add_parameters(p_aHO);
  VariableIndex i_aOb = hg.add_parameters(p_aOb);
  MLP aligner = {i_aIH, i_aHb, i_aHO, i_aOb};

  VariableIndex i_fIH = hg.add_parameters(p_fIH);
  VariableIndex i_fHb = hg.add_parameters(p_fHb);
  VariableIndex i_fHO = hg.add_parameters(p_fHO);
  VariableIndex i_fOb = hg.add_parameters(p_fOb);
  MLP final = {i_fIH, i_fHb, i_fHO, i_fOb};

  VariableIndex i_bs = hg.add_parameters(p_bs);
  VariableIndex i_Ws = hg.add_parameters(p_Ws);

  vector<VariableIndex> output_states(target.size());
  vector<VariableIndex> contexts(target.size());

  // TODO: Verify that this crap corresponds to the comment below and is sane
  VariableIndex first_input = hg.add_function<AffineTransform>({i_bs, i_Ws, reverse_annotations[0]});
  VariableIndex input = first_input;
  for (unsigned t = 0; t < target.size(); ++t) {
    OutputState os = GetNextOutputState(input, annotations, aligner, hg);
    output_states[t] = os.state;
    contexts[t] = os.context;
    input = contexts[t];
  }

  /*
  first input: W_s * h1_backwards
  afterwards, input: c_i
  c_i = sum_j(alpha_ij * h_j)
  alpha_ij = exp(e_ij) / sum_k(exp(e_ik))
  e_ij = a(s_i-1, h_j) where a is a FFNN
  */
  vector<VariableIndex> output_distributions(target.size() - 1);
  for (unsigned t = 1; t < target.size(); ++t) {
    WordId prev_word = target[t - 1];
    output_distributions[t - 1] = ComputeOutputDistribution(prev_word, output_states[t], contexts[t], final, hg);
  }

  vector<VariableIndex> errors(target.size() - 1);
  for (unsigned t = 1; t < target.size(); ++t) {
    VariableIndex output_distribution = output_distributions[t - 1];
    VariableIndex error = hg.add_function<PickNegLogSoftmax>({output_distribution}, target[t]);
    errors[t - 1] = error;
  }
  VariableIndex total_error = hg.add_function<Sum>(errors);
  return total_error;
}
