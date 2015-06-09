#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/lstm.h"
#include "utils.h"

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
  vector<VariableIndex> forward_contexts(sentence.size());
  for (unsigned t = 0; t < sentence.size(); ++t) {
    VariableIndex i_x_t = hg.add_lookup(p_Es, sentence[t]);
    VariableIndex i_y_t = forward_builder.add_input(i_x_t, &hg);
    forward_contexts[t] = i_y_t;
  }
  return forward_contexts;
}

vector<VariableIndex> AttentionalModel::BuildReverseAnnotations(const vector<WordId>& sentence, ComputationGraph& hg) {
  reverse_builder.new_graph(&hg);
  reverse_builder.start_new_sequence();
  vector<VariableIndex> reverse_contexts(sentence.size());
  for (unsigned t = sentence.size(); t > 0; ) {
    t--;
    VariableIndex i_x_t = hg.add_lookup(p_Es, sentence[t]);
    VariableIndex i_y_t = reverse_builder.add_input(i_x_t, &hg);
    reverse_contexts[t] = i_y_t;
  }
  return reverse_contexts;
}

vector<VariableIndex> AttentionalModel::BuildAnnotationVectors(const vector<VariableIndex>& forward_contexts, const vector<VariableIndex>& reverse_contexts, ComputationGraph& hg) {
  vector<VariableIndex> contexts(forward_contexts.size());
  for (unsigned t = 0; t < forward_contexts.size(); ++t) {
    const VariableIndex& i_f = forward_contexts[t];
    const VariableIndex& i_r = reverse_contexts[t];
    VariableIndex i_h = hg.add_function<Concatenate>({i_f, i_r});
    contexts[t] = i_h;
  }
  return contexts;
}

VariableIndex AttentionalModel::BuildGraph(const vector<WordId>& source, const vector<WordId>& target, ComputationGraph& hg) {
  output_builder.new_graph(&hg);
  output_builder.start_new_sequence();

  vector<VariableIndex> forward_annotations = BuildForwardAnnotations(source, hg);
  vector<VariableIndex> reverse_annotations = BuildReverseAnnotations(source, hg);
  vector<VariableIndex> annotations = BuildAnnotationVectors(forward_annotations, reverse_annotations, hg);

  vector<VariableIndex> output_states(target.size());
  vector<VariableIndex> contexts(target.size());
  VariableIndex i_aIH = hg.add_parameters(p_aIH);
  VariableIndex i_aHb = hg.add_parameters(p_aHb);
  VariableIndex i_aHO = hg.add_parameters(p_aHO);
  VariableIndex i_aOb = hg.add_parameters(p_aOb);

  VariableIndex i_fIH = hg.add_parameters(p_fIH);
  VariableIndex i_fHb = hg.add_parameters(p_fHb);
  VariableIndex i_fHO = hg.add_parameters(p_fHO);
  VariableIndex i_fOb = hg.add_parameters(p_fOb);

  VariableIndex i_bs = hg.add_parameters(p_bs);
  VariableIndex i_Ws = hg.add_parameters(p_Ws);

  VariableIndex first_input = hg.add_function<AffineTransform>({i_bs, i_Ws, reverse_annotations[0]});
  VariableIndex state_input = first_input;
  // TODO: Verify that this crap corresponds to the comment below and is sane
  for (unsigned t = 0; t < target.size(); ++t) {
    VariableIndex state = output_builder.add_input(state_input, &hg);
    output_states[t] = state;

    vector<VariableIndex> unnormalized_alignments(source.size()); // e_ij
    for (unsigned s = 0; s < source.size(); ++s) {
      VariableIndex a_input = hg.add_function<Concatenate>({state, annotations[s]});
      VariableIndex a_hidden1 = hg.add_function<AffineTransform>({i_aHb, i_aIH, a_input});
      VariableIndex a_hidden2 = hg.add_function<Tanh>({a_hidden1});
      VariableIndex a_output = hg.add_function<AffineTransform>({i_aOb, i_aHO, a_hidden2});
      unnormalized_alignments[s] = a_output;
    }

    VariableIndex unnormalized_alignment_vector = hg.add_function<Concatenate>(unnormalized_alignments);
    VariableIndex normalized_alignment_vector = hg.add_function<Softmax>({unnormalized_alignment_vector});
    //vector<float> asdf = as_vector(hg.incremental_forward());
    //for (float f : asdf) { cerr << f << " "; } cerr << endl;
    VariableIndex annotation_matrix = hg.add_function<ConcatenateColumns>(annotations);
    VariableIndex context = hg.add_function<SumColumns>({annotation_matrix, normalized_alignment_vector});
    contexts[t] = context;
  }
/*
  first input: W_s * h1_backwards
  afterwards, input: c_i
  c_i = sum_j(alpha_ij * h_j)
  alpha_ij = exp(e_ij) / sum_k(exp(e_ik))
  e_ij = a(s_i-1, h_j) where a is a FFNN
*/

  vector<VariableIndex> errors(target.size() - 1);
  for (unsigned t = 1; t < target.size(); ++t) {
    VariableIndex prev_target_embedding = hg.add_lookup(p_Et, target[t  - 1]);
    VariableIndex final_input = hg.add_function<Concatenate>({prev_target_embedding, output_states[t], contexts[t]});
    VariableIndex final_hidden1 = hg.add_function<AffineTransform>({i_fHb, i_fIH, final_input});
    VariableIndex final_hidden2 = hg.add_function<Tanh>({final_hidden1});
    VariableIndex final_output = hg.add_function<AffineTransform>({i_fOb, i_fHO, final_hidden2});
    VariableIndex error = hg.add_function<PickNegLogSoftmax>({final_output}, target[t]);
    errors[t - 1] = error;
  }
  VariableIndex total_error = hg.add_function<Sum>(errors);
  return total_error;
}
